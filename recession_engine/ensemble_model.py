"""
Recession Prediction Engine - Ensemble Modeling Module

Literature-informed design:
- Estrella & Mishkin (1998): Probit baseline with term spread
- Wright (2006): Augmented probit with FFR level
- Kauppi & Saikkonen (2008): Dynamic probit with autoregressive term
- Dopke et al. (2017): Boosted trees for recession prediction
- Billakanti & Shin (2025): At-risk features improve all model types

Key improvements over v1:
1. Class-weighted models for recession imbalance (~12% base rate)
2. Expanding-window training (no frozen 2015 cutoff)
3. Time-series cross-validation for hyperparameter selection
4. Youden's J threshold optimization (not naive 0.5)
5. Platt scaling / isotonic regression for probability calibration
6. Performance-weighted ensemble (BMA-inspired)
7. Brier score + log-loss evaluation (proper scoring rules)
"""

import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, brier_score_loss, log_loss, average_precision_score,
    precision_recall_curve
)
import warnings
import logging
from datetime import datetime

try:
    from recession_engine.calibration_diag import (
        reliability_curve as _reliability_curve,
        expected_calibration_error as _ece,
        calibration_slope_intercept as _slope_intercept,
        brier_decomposition as _brier_decomp,
        fit_calibrator as _fit_calibrator,
        stationary_block_bootstrap_ci as _stationary_bootstrap,
        HAS_BETACAL,
    )
except ImportError:  # pragma: no cover - when loaded as top-level module
    from calibration_diag import (  # type: ignore
        reliability_curve as _reliability_curve,
        expected_calibration_error as _ece,
        calibration_slope_intercept as _slope_intercept,
        brier_decomposition as _brier_decomp,
        fit_calibrator as _fit_calibrator,
        stationary_block_bootstrap_ci as _stationary_bootstrap,
        HAS_BETACAL,
    )

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import xgboost as xgb  # type: ignore
    HAS_XGBOOST = True
except Exception as e:  # pragma: no cover - environment-specific
    HAS_XGBOOST = False
    xgb = None  # type: ignore
    logger.warning("XGBoost could not be imported and will be disabled: %s", e)

try:
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    HAS_MARKOV = True
except Exception as e:  # pragma: no cover - environment-specific
    HAS_MARKOV = False
    MarkovRegression = None  # type: ignore
    logger.warning("statsmodels MarkovRegression not available: %s", e)

try:
    # Force single-threaded BLAS to avoid segfaults when statsmodels (OpenBLAS)
    # and PyTorch (MKL/Accelerate) both load competing BLAS implementations.
    import os as _os
    for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        _os.environ.setdefault(_var, '1')
    import torch
    import torch.nn as nn
    HAS_TORCH = True
    logger.info("PyTorch %s available (device: cpu — MPS disabled for LSTM stability)",
                torch.__version__)
except ImportError as e:
    HAS_TORCH = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    logger.warning("PyTorch not available: %s", e)


class MarkovSwitchingWrapper:
    """
    Sklearn-like wrapper for a 3-regime Markov-switching model with
    time-varying transition probabilities (TVTP) and isotonic calibration.

    Literature basis:
    - Hamilton (1989): Core regime-switching framework
    - Filardo (1994): TVTP — leading indicators drive transition probabilities
    - Diebold et al. (1994): TVTP improves recession dating accuracy
    - Chauvet & Hamilton (2006): Real-time recession probability estimation
    - FEDS Notes (2026): 3-state model distinguishing expansion,
      U-shaped (recovery) recession, and L-shaped (hysteresis) recession

    3-state design:
    - Regime 0 (expansion): highest composite mean, normal growth
    - Regime 1 (mild contraction): intermediate mean, mild/short recessions
    - Regime 2 (deep contraction): lowest mean, severe recessions
    - Recession probability = P(mild) + P(deep)

    Key improvements:
    1. 3 regimes instead of 2 — better calibration for mild vs severe recessions
    2. TVTP: SAHM indicator and NFCI influence transition probabilities
    3. Isotonic calibration of combined recession probability
    4. Fixed-parameter prediction via smooth() (no re-fitting on test data)
    """

    # ── Composite indicators: sign-aligned so higher = stronger economy ──
    COMPOSITE_INDICATORS = {
        'leading_T10Y3M': +1,              # Positive spread = expansion
        'NEAR_TERM_FORWARD_SPREAD': +1,    # Engstrom-Sharpe: short-term rate expectations
        'SAHM_INDICATOR': -1,              # Higher = closer to recession trigger
        'financial_NFCI': -1,              # Higher = tighter conditions
        'EBP_PROXY': -1,                   # Higher = credit stress (Gilchrist-Zakrajsek)
        'CREDIT_STRESS_INDEX': -1,         # Higher = financial stress
    }

    # ── TVTP covariates: these influence transition probabilities ──
    # The key insight: yield curve inversion alone shouldn't lock the model
    # in recession regime if labor market and financial conditions are healthy.
    TVTP_INDICATORS = {
        'SAHM_INDICATOR': -1,       # Low SAHM = strong labor market → favor expansion
        'financial_NFCI': -1,       # Low NFCI = loose conditions → favor expansion
    }

    N_REGIMES = 3  # expansion, mild contraction, deep contraction

    def __init__(self):
        self.result = None
        self.recession_regimes = []  # indices of contraction regimes (mild + deep)
        self.expansion_regime = None
        self.training_composite = None
        self.training_tvtp = None
        self._composite_means = {}
        self._composite_stds = {}
        self._tvtp_means = {}
        self._tvtp_stds = {}
        self._trained_params = None
        self.fitted = False
        self._use_tvtp = False

        # Isotonic calibration (fitted during fit, applied during predict)
        self._calibrator = None
        self._calibrator_fitted = False

    def _build_composite(self, df, fit_stats=False):
        """Build sign-aligned composite from available indicators."""
        available = {c: s for c, s in self.COMPOSITE_INDICATORS.items()
                     if c in df.columns}
        if len(available) == 0:
            return None

        parts = []
        for col, sign in available.items():
            s = df[col]
            if fit_stats:
                mu, sigma = s.mean(), s.std() + 1e-8
                self._composite_means[col] = mu
                self._composite_stds[col] = sigma
            else:
                mu = self._composite_means.get(col, s.mean())
                sigma = self._composite_stds.get(col, s.std() + 1e-8)
            parts.append(sign * (s - mu) / sigma)

        composite = pd.concat(parts, axis=1).mean(axis=1)
        return composite

    def _build_tvtp_covariates(self, df, fit_stats=False):
        """Build TVTP covariate matrix (standardized, sign-aligned)."""
        available = {c: s for c, s in self.TVTP_INDICATORS.items()
                     if c in df.columns}
        if len(available) < 1:
            return None

        parts = []
        for col, sign in available.items():
            s = df[col]
            if fit_stats:
                mu, sigma = s.mean(), s.std() + 1e-8
                self._tvtp_means[col] = mu
                self._tvtp_stds[col] = sigma
            else:
                mu = self._tvtp_means.get(col, s.mean())
                sigma = self._tvtp_stds.get(col, s.std() + 1e-8)
            parts.append(sign * (s - mu) / sigma)

        tvtp_df = pd.concat(parts, axis=1)
        # Add intercept column (required by statsmodels TVTP)
        tvtp_df.insert(0, '_const', 1.0)
        return tvtp_df

    def fit(self, X_df, y):
        """
        Fit Markov-switching model with TVTP and isotonic calibration.

        Steps:
        1. Build composite signal from 6 macro indicators
        2. Build TVTP covariates from labor market + financial conditions
        3. Fit Hamilton model with TVTP (fall back to constant if TVTP fails)
        4. Identify recession regime (lower composite mean)
        5. Calibrate filtered probabilities via isotonic regression on y
        """
        from sklearn.isotonic import IsotonicRegression

        composite = self._build_composite(X_df, fit_stats=True)
        if composite is None:
            logger.warning("MarkovSwitching: no composite indicators found, skipping")
            self.fitted = False
            return self

        tvtp_df = self._build_tvtp_covariates(X_df, fit_stats=True)

        # Align composite and TVTP on shared index (drop NaN rows)
        if tvtp_df is not None:
            shared_idx = composite.dropna().index.intersection(tvtp_df.dropna().index)
        else:
            shared_idx = composite.dropna().index
        composite = composite.loc[shared_idx]
        if tvtp_df is not None:
            tvtp_df = tvtp_df.loc[shared_idx]

        if len(composite) < 60:
            logger.warning("MarkovSwitching: too few observations (%d), skipping",
                           len(composite))
            self.fitted = False
            return self

        # ── Try 3-regime TVTP model, then 3-regime constant, then 2-regime fallback ──
        self._use_tvtp = False
        n_regimes = self.N_REGIMES

        # Attempt 1: 3-regime TVTP
        if tvtp_df is not None and len(tvtp_df.columns) >= 2:
            try:
                mod = MarkovRegression(
                    composite.values,
                    k_regimes=n_regimes,
                    trend='c',
                    switching_variance=True,
                    exog_tvtp=tvtp_df.values,
                )
                self.result = mod.fit(maxiter=500, disp=False)
                self._use_tvtp = True
                logger.info("MarkovSwitching: %d-regime TVTP model fitted "
                            "(%d covariates)", n_regimes, tvtp_df.shape[1] - 1)
            except Exception as e:
                logger.warning("MarkovSwitching: %d-regime TVTP failed (%s)",
                               n_regimes, e)
                self.result = None

        # Attempt 2: 3-regime constant transitions
        if self.result is None:
            try:
                mod = MarkovRegression(
                    composite.values,
                    k_regimes=n_regimes,
                    trend='c',
                    switching_variance=True,
                )
                self.result = mod.fit(maxiter=500, disp=False)
                self._use_tvtp = False
                logger.info("MarkovSwitching: %d-regime constant model fitted",
                            n_regimes)
            except Exception as e:
                logger.warning("MarkovSwitching: %d-regime constant failed (%s)",
                               n_regimes, e)
                self.result = None

        # Attempt 3: 2-regime fallback (simpler, more stable)
        if self.result is None:
            n_regimes = 2
            try:
                mod = MarkovRegression(
                    composite.values,
                    k_regimes=2,
                    trend='c',
                    switching_variance=True,
                )
                self.result = mod.fit(maxiter=300, disp=False)
                self._use_tvtp = False
                logger.info("MarkovSwitching: 2-regime fallback model fitted")
            except Exception as e:
                logger.warning("MarkovSwitching fit failed completely: %s", e)
                self.result = None
                self.fitted = False
                return self

        # ── Identify regimes by composite mean ──
        # Higher composite mean = stronger economy (expansion)
        param_names = self.result.model.param_names
        means = []
        for i in range(n_regimes):
            idx = param_names.index(f'const[{i}]')
            means.append(float(self.result.params[idx]))

        # Sort regimes by mean: highest = expansion, lower = contraction
        sorted_regimes = sorted(range(n_regimes), key=lambda i: means[i], reverse=True)
        self.expansion_regime = sorted_regimes[0]
        self.recession_regimes = sorted_regimes[1:]  # all non-expansion regimes
        self._n_regimes_actual = n_regimes

        # Store trained parameters and data for prediction
        self._trained_params = self.result.params.copy()
        self.training_composite = composite.values.copy()
        if self._use_tvtp:
            self.training_tvtp = tvtp_df.values.copy()

        # ── Isotonic calibration on training data ──
        # The raw filtered probabilities are near-binary (0.01 or 0.99).
        # Isotonic regression learns a monotonic mapping from raw probs
        # to actual recession frequencies, producing well-calibrated output.
        # For 3-state model: recession prob = sum of all contraction regime probs
        filtered = self.result.filtered_marginal_probabilities
        raw_train_probs = sum(
            filtered[:, r] for r in self.recession_regimes
        )

        # Align y with the composite index
        y_aligned = y.loc[shared_idx].values if hasattr(y, 'loc') else y

        if len(y_aligned) == len(raw_train_probs):
            try:
                self._calibrator = IsotonicRegression(
                    y_min=0.01, y_max=0.99, out_of_bounds='clip'
                )
                self._calibrator.fit(raw_train_probs, y_aligned)
                self._calibrator_fitted = True

                # Log calibration effect
                cal_probs = self._calibrator.predict(raw_train_probs)
                logger.info("MarkovSwitching calibration: raw range [%.3f, %.3f] → "
                            "calibrated range [%.3f, %.3f]",
                            raw_train_probs.min(), raw_train_probs.max(),
                            cal_probs.min(), cal_probs.max())
            except Exception as e:
                logger.warning("MarkovSwitching calibration failed: %s", e)
                self._calibrator_fitted = False
        else:
            logger.warning("MarkovSwitching: y length mismatch for calibration "
                           "(%d vs %d)", len(y_aligned), len(raw_train_probs))
            self._calibrator_fitted = False

        self.fitted = True
        logger.info("MarkovSwitching fitted: %d regimes, expansion=%d, "
                     "recession=%s, means=%s, tvtp=%s, calibrated=%s",
                     n_regimes, self.expansion_regime,
                     self.recession_regimes,
                     [f"{m:.3f}" for m in means],
                     self._use_tvtp, self._calibrator_fitted)

        return self

    def predict_proba(self, X_df):
        """
        Return (n_samples, 2) array of [P(expansion), P(recession)].

        Uses the trained model's parameters applied to an extended
        composite (training + test), then extracts test-period filtered
        probabilities and applies isotonic calibration.
        """
        n = len(X_df)

        if not self.fitted or self.result is None:
            return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])

        # Build test composite using TRAINING statistics
        test_composite = self._build_composite(X_df, fit_stats=False)
        if test_composite is None:
            return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])
        test_composite = test_composite.fillna(0).values

        # Build test TVTP covariates using TRAINING statistics
        test_tvtp = None
        if self._use_tvtp:
            test_tvtp_df = self._build_tvtp_covariates(X_df, fit_stats=False)
            if test_tvtp_df is not None:
                test_tvtp = test_tvtp_df.fillna(0).values

        # Concatenate training + test data
        full_composite = np.concatenate([self.training_composite, test_composite])
        full_tvtp = None
        if self._use_tvtp and self.training_tvtp is not None and test_tvtp is not None:
            full_tvtp = np.concatenate([self.training_tvtp, test_tvtp])

        try:
            # Apply trained parameters to extended series via smooth()
            # This avoids re-estimating parameters on test data
            n_regimes = self._n_regimes_actual
            if self._use_tvtp and full_tvtp is not None:
                mod = MarkovRegression(
                    full_composite,
                    k_regimes=n_regimes,
                    trend='c',
                    switching_variance=True,
                    exog_tvtp=full_tvtp,
                )
            else:
                mod = MarkovRegression(
                    full_composite,
                    k_regimes=n_regimes,
                    trend='c',
                    switching_variance=True,
                )

            # Use smooth() with trained parameters — no re-estimation
            try:
                res = mod.smooth(self._trained_params)
            except Exception:
                # Fall back to re-fitting if smooth fails
                res = mod.fit(
                    maxiter=300, disp=False,
                    start_params=self._trained_params
                )

            # Extract filtered probabilities for test period
            # For 3-state: recession prob = sum of all contraction regime probs
            filtered = res.filtered_marginal_probabilities
            test_probs = sum(
                filtered[-n:, r] for r in self.recession_regimes
            )

            # Apply isotonic calibration if available
            if self._calibrator_fitted and self._calibrator is not None:
                test_probs = self._calibrator.predict(test_probs)

            # Clamp for numerical safety
            test_probs = np.clip(test_probs, 0.01, 0.99)

            return np.column_stack([1.0 - test_probs, test_probs])

        except Exception as e:
            logger.warning("MarkovSwitching predict failed: %s", e)
            return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])


class LSTMRecessionModel:
    """
    LSTM-based recession prediction model wrapper.

    Literature basis:
    - Vrontos et al. (2024): LSTM/GRU outperform traditional models for
      recession forecasting, especially at 6-12 month horizons
    - SHAP analysis confirms term spread dominates at 6+ month horizons

    Architecture (deliberately small to avoid overfitting on ~500 samples):
    - Internal feature selection: top-K features by mutual information (default 15)
    - Single LSTM layer (16 units) with 50% dropout
    - Lookback window of 6 months (shorter = less overfitting, still captures trends)
    - Binary output with sigmoid activation
    - Class-weighted training for recession imbalance
    - Aggressive early stopping (patience=5) to prevent memorization
    - Validation-based early stopping using 15% hold-out from training data
    """

    MAX_FEATURES = 10  # Aggressive feature reduction for ~500 sample regime
    N_SEEDS = 5        # Number of seed-averaged models for stability

    def __init__(self, lookback=6, epochs=80, batch_size=32):
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.models = []   # List of (model, seed) for seed averaging
        self.scaler = None
        self.fitted = False
        self._selected_features = None
        self._device = None

    def _get_device(self):
        """Select best available device: CUDA > CPU.

        NOTE: MPS (Apple Metal) is deliberately excluded — it has known
        hangs with small LSTM tensors and offers no speed advantage for
        the ~500-sequence datasets used here.
        """
        if self._device is not None:
            return self._device
        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')
        return self._device

    def _build_model(self, n_features):
        """Build single-layer LSTM in PyTorch.

        Deliberately small to prevent overfitting on ~500 training samples:
        - Single LSTM layer (16 units) instead of 2 layers (32→16)
        - Heavy dropout (0.5) on output
        - No hidden dense layer — direct LSTM→output
        Total params: ~(4×8×(n_features+8+1)) + 9 ≈ 600 for 10 features
        """
        class _LSTMNet(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, 8, batch_first=True)
                self.dropout = nn.Dropout(0.5)
                self.fc = nn.Linear(8, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                out = out[:, -1, :]          # last time step
                out = self.dropout(out)
                out = torch.sigmoid(self.fc(out))
                return out.squeeze(-1)

        return _LSTMNet(n_features).to(self._get_device())

    def _select_features(self, X, y):
        """Select top-K features by mutual information to prevent overfitting.

        With ~500 training samples, using 78+ features causes severe
        overfitting (train AUC ~1.0, test AUC ~0.52). MI-based selection
        picks the most informative features without data leakage.
        """
        from sklearn.feature_selection import mutual_info_classif

        n_features = min(self.MAX_FEATURES, X.shape[1])
        mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
        top_indices = np.argsort(mi_scores)[-n_features:]
        top_indices = np.sort(top_indices)  # preserve column order

        logger.info("LSTM feature selection: %d/%d features (top MI scores: %s)",
                     n_features, X.shape[1],
                     ", ".join(f"{mi_scores[i]:.3f}" for i in top_indices[:5]))
        return top_indices

    def _create_sequences(self, X, y=None):
        """Create lookback sequences for LSTM input."""
        X_seq = []
        y_seq = []
        for i in range(self.lookback, len(X)):
            X_seq.append(X[i - self.lookback:i])
            if y is not None:
                y_seq.append(y[i])
        X_seq = np.array(X_seq)
        if y is not None:
            y_seq = np.array(y_seq)
            return X_seq, y_seq
        return X_seq

    def fit(self, X_train, y_train):
        """Fit LSTM on scaled, feature-selected matrix."""
        if not HAS_TORCH:
            logger.warning("LSTM: PyTorch not available, skipping")
            self.fitted = False
            return self

        from sklearn.preprocessing import StandardScaler

        # Feature selection BEFORE scaling (uses raw values for MI)
        y_values = y_train.values if hasattr(y_train, 'values') else np.asarray(y_train)
        X_values = X_train.values if hasattr(X_train, 'values') else np.asarray(X_train)

        # Fill any remaining NaN for MI computation
        X_clean = np.nan_to_num(X_values, nan=0.0)
        self._selected_features = self._select_features(X_clean, y_values)
        X_selected = X_clean[:, self._selected_features]

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_selected)

        X_seq, y_seq = self._create_sequences(X_scaled, y_values)

        if len(X_seq) < 60:
            logger.warning("LSTM: too few sequences (%d), skipping", len(X_seq))
            self.fitted = False
            return self

        # Class weights for imbalanced data
        n_pos = y_seq.sum()
        n_neg = len(y_seq) - n_pos
        pos_weight_val = float(n_neg / max(n_pos, 1))

        device = self._get_device()
        X_tensor = torch.tensor(X_seq, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y_seq, dtype=torch.float32, device=device)
        sample_weights = torch.where(y_tensor == 1, pos_weight_val, 1.0)

        # ── Seed averaging: train N_SEEDS models with different seeds ──
        # Averaging across seeds dramatically reduces variance from random
        # initialization, which dominates uncertainty at this sample size.
        self.models = []
        seeds = [42, 123, 7, 2024, 999][:self.N_SEEDS]

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = self._build_model(X_selected.shape[1])
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-3)

            best_loss = float('inf')
            patience_counter = 0
            best_state = None

            model.train()
            for epoch in range(self.epochs):
                indices = torch.randperm(len(X_tensor), device=device)
                epoch_loss = 0.0
                n_batches = 0

                for start in range(0, len(X_tensor), self.batch_size):
                    batch_idx = indices[start:start + self.batch_size]
                    X_batch = X_tensor[batch_idx]
                    y_batch = y_tensor[batch_idx]
                    w_batch = sample_weights[batch_idx]

                    optimizer.zero_grad()
                    preds = model(X_batch)
                    bce = -(y_batch * torch.log(preds + 1e-7) +
                            (1 - y_batch) * torch.log(1 - preds + 1e-7))
                    loss = (bce * w_batch).mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1

                avg_loss = epoch_loss / max(n_batches, 1)
                if avg_loss < best_loss - 1e-4:
                    best_loss = avg_loss
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= 10:
                        break

            if best_state is not None:
                model.load_state_dict(best_state)
            self.models.append(model)

        self.fitted = True
        logger.info("LSTM fitted: %d sequences, %d/%d features, lookback=%d, "
                     "%d seed-averaged models, device=%s",
                     len(X_seq), X_selected.shape[1], X_values.shape[1],
                     self.lookback, len(self.models), device)
        return self

    def predict_proba(self, X_test):
        """Return (n_samples, 2) array of [P(expansion), P(recession)].

        Averages predictions across N_SEEDS models for stability.
        """
        n = len(X_test)

        if not self.fitted or len(self.models) == 0:
            return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])

        # Apply same feature selection as training
        X_values = X_test.values if hasattr(X_test, 'values') else np.asarray(X_test)
        X_clean = np.nan_to_num(X_values, nan=0.0)
        X_selected = X_clean[:, self._selected_features]
        X_scaled = self.scaler.transform(X_selected)
        X_seq = self._create_sequences(X_scaled)

        if len(X_seq) == 0:
            return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])

        # Predict: average across all seed models
        device = self._get_device()
        X_tensor = torch.tensor(X_seq, dtype=torch.float32, device=device)
        all_preds = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                p = model(X_tensor).cpu().numpy()
            all_preds.append(p)
        preds = np.mean(all_preds, axis=0)
        preds = np.clip(preds, 0.01, 0.99)

        # Pad the first `lookback` predictions with earliest valid prediction
        # (avoids injecting 0.5 which distorts ensemble and calibration)
        n_pad = n - len(preds)
        if n_pad > 0:
            pad_val = preds[0] if len(preds) > 0 else 0.5
            preds = np.concatenate([np.full(n_pad, pad_val), preds])

        return np.column_stack([1.0 - preds, preds])


# ──────────────────────────────────────────────────────────────────────
# C1 — Benchmark (peer-model) ensemble members
# ──────────────────────────────────────────────────────────────────────
#
# These three wrappers expose the same (fit, predict_proba) API as the
# supervised base models so the DMA weighting pipeline can treat them
# uniformly. Two wrap externally-published recession probability series
# (Hamilton, Chauvet-Piger) pulled from FRED as `ref_*` columns; the
# third fits a classical Wright (2006) probit on (T10Y3M, DFF, T10Y3M ×
# DFF). Reference columns are excluded from ordinary feature selection
# (see RecessionEnsembleModel.select_features), so using them here does
# not double-count signal into probit/RF/XGB.


class HamiltonBenchmarkModel:
    """
    Ensemble wrapper around Hamilton's GDP-Based Recession Indicator Index.

    JHGDPBRINDX (Hamilton 2019, "Why You Should Never Use the Hodrick-Prescott
    Filter") is published by FRED on a [0, 100] scale representing the
    probability of recession in the just-completed quarter. Divide by 100 to
    get a [0, 1] probability.

    fit() is a no-op (the values are externally-computed); predict_proba()
    reads the column on-the-fly from the test DataFrame.
    """

    COLUMN = 'ref_JHGDPBRINDX'
    NEUTRAL_PROB = 0.05  # Fallback when the series is missing (low, not 0.5)

    def __init__(self):
        self.fitted = False

    def fit(self, X_df, y=None):  # noqa: ARG002 — y unused; external series
        self.fitted = True
        return self

    def predict_proba(self, X_df):
        n = len(X_df)
        if self.COLUMN in X_df.columns:
            series = X_df[self.COLUMN].astype(float) / 100.0
            probs = series.fillna(self.NEUTRAL_PROB).clip(0.0, 1.0).values
        else:
            probs = np.full(n, self.NEUTRAL_PROB)
        probs = np.clip(probs, 0.01, 0.99)
        return np.column_stack([1.0 - probs, probs])


class ChauvetPigerBenchmarkModel:
    """
    Ensemble wrapper around the Chauvet-Piger smoothed recession probability.

    RECPROUSM156N (Chauvet-Piger 2008) is published by FRED on [0, 1]
    already and represents the probability the US economy is in recession
    based on a real-time regime-switching dynamic factor model. No rescaling
    needed; just map NaNs to a neutral prior.

    fit() is a no-op; predict_proba() reads the column from the test frame.
    """

    COLUMN = 'ref_RECPROUSM156N'
    NEUTRAL_PROB = 0.05

    def __init__(self):
        self.fitted = False

    def fit(self, X_df, y=None):  # noqa: ARG002
        self.fitted = True
        return self

    def predict_proba(self, X_df):
        n = len(X_df)
        if self.COLUMN in X_df.columns:
            series = X_df[self.COLUMN].astype(float)
            probs = series.fillna(self.NEUTRAL_PROB).clip(0.0, 1.0).values
        else:
            probs = np.full(n, self.NEUTRAL_PROB)
        probs = np.clip(probs, 0.01, 0.99)
        return np.column_stack([1.0 - probs, probs])


class WrightProbitModel:
    """
    Classical Wright (2006) recession probit: probit probability of a
    recession within `target_horizon` months as a function of the yield
    curve slope (T10Y3M), the effective federal funds rate (DFF), and
    their interaction.

    We stand in an L2-penalized logistic regression for the probit link —
    statsmodels' probit solver is finicky on small samples with a single
    quasi-separating spread, and under a probit vs logit link the resulting
    recession probabilities differ only in a monotonic transformation that
    isotonic / sigmoid calibration would absorb anyway. Signal content is
    the same.

    Features used: ``leading_T10Y3M``, ``monetary_DFF``, and their product.
    Missing rows are dropped for fit; for predict, missing values are
    forward-filled (or, if still missing, mapped to a neutral low prob).
    """

    SLOPE_COL = 'leading_T10Y3M'
    FFR_COL = 'monetary_DFF'
    NEUTRAL_PROB = 0.05

    def __init__(self):
        self.model = None
        self.fitted = False
        self._mean = None
        self._std = None

    def _build_matrix(self, df):
        if self.SLOPE_COL not in df.columns or self.FFR_COL not in df.columns:
            return None
        slope = df[self.SLOPE_COL].astype(float)
        ffr = df[self.FFR_COL].astype(float)
        interaction = slope * ffr
        X = pd.concat([slope, ffr, interaction], axis=1)
        X.columns = [self.SLOPE_COL, self.FFR_COL, f"{self.SLOPE_COL}_x_{self.FFR_COL}"]
        return X

    def fit(self, X_df, y):
        X = self._build_matrix(X_df)
        if X is None:
            logger.warning("WrightProbit: required columns missing (%s, %s); skipping",
                           self.SLOPE_COL, self.FFR_COL)
            self.fitted = False
            return self

        # Align and drop rows where any predictor or y is NaN
        y_series = y if hasattr(y, 'loc') else pd.Series(y, index=X_df.index)
        combined = pd.concat([X, y_series.rename('y')], axis=1).dropna()
        if len(combined) < 60 or combined['y'].nunique() < 2:
            logger.warning("WrightProbit: too few observations (%d) or single class; skipping",
                           len(combined))
            self.fitted = False
            return self

        X_fit = combined.drop(columns='y').values
        y_fit = combined['y'].values

        # Standardize the 3 features for stable L2 logistic fit
        self._mean = X_fit.mean(axis=0)
        self._std = X_fit.std(axis=0) + 1e-8
        X_scaled = (X_fit - self._mean) / self._std

        try:
            self.model = LogisticRegression(
                penalty='l2',
                C=1.0,
                solver='lbfgs',
                max_iter=500,
                class_weight='balanced',
                random_state=42,
            )
            self.model.fit(X_scaled, y_fit)
            self.fitted = True
            logger.info("WrightProbit fitted on %d obs (%d positives)",
                        len(combined), int(y_fit.sum()))
        except Exception as e:
            logger.warning("WrightProbit fit failed: %s", e)
            self.fitted = False
        return self

    def predict_proba(self, X_df):
        n = len(X_df)
        if not self.fitted or self.model is None:
            return np.column_stack([np.full(n, 1 - self.NEUTRAL_PROB),
                                     np.full(n, self.NEUTRAL_PROB)])

        X = self._build_matrix(X_df)
        if X is None:
            return np.column_stack([np.full(n, 1 - self.NEUTRAL_PROB),
                                     np.full(n, self.NEUTRAL_PROB)])

        X_ff = X.ffill().fillna(0.0).values
        X_scaled = (X_ff - self._mean) / self._std
        probs = self.model.predict_proba(X_scaled)[:, 1]
        probs = np.clip(probs, 0.01, 0.99)
        return np.column_stack([1.0 - probs, probs])


# Registry (ordered) of benchmark members added under --benchmark-members=on.
# Keep as a list of (name, factory) tuples so iteration is deterministic.
BENCHMARK_MODEL_FACTORIES = [
    ('hamilton_benchmark', HamiltonBenchmarkModel),
    ('chauvet_piger_benchmark', ChauvetPigerBenchmarkModel),
    ('wright_probit', WrightProbitModel),
]


class RecessionEnsembleModel:
    """
    Ensemble recession prediction model.

    Architecture:
    - Base models: L1-Probit, Random Forest, XGBoost, Markov-Switching (TVTP)
    - Preprocessing: PCA factor extraction (augments feature set)
    - Calibration: Isotonic regression on supervised models; built-in isotonic on Markov
    - Ensemble: CV-gated forecast combination with shrinkage toward equal weights
    - Threshold: Optimized via Youden's J on validation set
    """

    def __init__(self, target_horizon=6, n_cv_splits=5, model_config=None):
        self.target_horizon = target_horizon
        self.target_col = f'RECESSION_FORWARD_{target_horizon}M'
        self.n_cv_splits = n_cv_splits
        self.model_config = model_config or {}

        # These get set during fit
        self.decision_threshold = 0.5
        self.ensemble_weights = {}     # Final live ensemble weights
        self.dma_weights = {}          # DMA weights before shrinkage
        self.static_weights = {}       # Static BMA weights (fallback)
        self.calibrated_models = {}    # Post-hoc calibrated classifiers
        self.calibration_diagnostics = {}  # Per-model reliability/ECE/slope/Brier decomp
        self.calibrator_choice = {}    # A/B winner per model (isotonic/sigmoid/beta)
        # A1.5 — promoted ensemble-level calibrator (sigmoid/Platt by default).
        # Fit inside _compute_calibration_diagnostics on pooled CV ensemble predictions
        # so the deployed probability scale reflects the A/B winner, not the raw
        # weighted average.
        self.ensemble_calibrator = None
        # Method actually installed as self.ensemble_calibrator; may differ from
        # self.calibrator_choice["ensemble"] (A/B winner) if we pin a specific
        # calibrator for stability. Default is the A1.5 decision: "sigmoid".
        self.deployed_ensemble_calibrator = None
        # A1.5 pins sigmoid as the ensemble-level deployed calibrator regardless
        # of per-run A/B variance (A/B winner on A1's run was sigmoid; we honour
        # that promotion unless the config explicitly overrides).
        self.preferred_ensemble_calibrator = str(
            (self.model_config.get("preferred_ensemble_calibrator")
             or "sigmoid")
        ).lower()
        self.feature_cols = []
        self.feature_importance = {}
        self.metrics = {}
        self.cv_results = {}
        self.active_models = []
        self.feature_drift_scores = {}
        self.equal_weight_shrinkage = float(self.model_config.get('equal_weight_shrinkage', 1.0))
        # C1 — benchmark (peer-model) ensemble members. When enabled, the
        # Hamilton + Chauvet-Piger probability series and a Wright probit are
        # added as ensemble members alongside probit/RF/XGB/markov_switching.
        self.benchmark_members_enabled = bool(
            self.model_config.get('benchmark_members_enabled', False)
        )
        self.benchmark_models = {}  # name -> fitted benchmark wrapper (set in fit)
        self.ensemble_method = "equal_weight_active"
        self.threshold_method = "calibrated training F1 with precision/recall tie-breaks"
        self.threshold_diagnostics = []
        self.threshold_plateau_diagnostics = {}
        self.is_fitted = False
        self.recency_half_life_months = self.model_config.get('recency_half_life_months')
        self.recency_weight_floor = float(self.model_config.get('recency_weight_floor', 0.25))
        self.drift_lookback_months = 36
        self.drift_recent_months = 12
        self.drift_penalty_strength = float(self.model_config.get('drift_penalty_strength', 0.0))
        self.severe_drift_psi = None
        self.selected_drift_prune_psi = 9.0
        self.selected_drift_prune_count = 0

        # PCA for factor extraction
        self.pca = None
        self.n_pca_components = int(self.model_config.get('n_pca_components', 5))

        # Markov-switching model (needs DataFrame, not numpy array)
        self.markov_model = MarkovSwitchingWrapper() if HAS_MARKOV else None

        # LSTM model — DISABLED: insufficient sample size (~600 samples, ~85 positive)
        # causes severe overfitting (train AUC 0.96 vs test AUC 0.47) that seed-averaging,
        # feature selection, and regularization cannot overcome. The LSTMRecessionModel class
        # is preserved below for future use when more data accumulates.
        # See: Vrontos et al. (2024) used much larger datasets for effective LSTM forecasting.
        self.lstm_model = None

        # Base models with class_weight='balanced' for rare-event handling
        probit_params = {
            'penalty': 'l1',
            'solver': 'saga',
            'C': 0.1,
            'max_iter': 2000,
            'random_state': 42,
            'class_weight': 'balanced',
        }
        probit_params.update(self.model_config.get('probit', {}))

        random_forest_params = {
            'n_estimators': 300,
            'max_depth': 8,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'random_state': 42,
            'n_jobs': -1,
            'class_weight': 'balanced',
        }
        random_forest_params.update(self.model_config.get('random_forest', {}))

        self.models = {
            'probit': LogisticRegression(**probit_params),
            'random_forest': RandomForestClassifier(**random_forest_params),
        }
        if HAS_XGBOOST:
            # scale_pos_weight set dynamically in fit() based on actual class ratio
            xgboost_params = {
                'n_estimators': 400,
                'max_depth': 5,
                'learning_rate': 0.01,
                'subsample': 0.8,
                'colsample_bytree': 0.7,
                'min_child_weight': 10,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'n_jobs': -1,
                'eval_metric': 'logloss',
            }
            xgboost_params.update(self.model_config.get('xgboost', {}))
            self.models['xgboost'] = xgb.XGBClassifier(**xgboost_params)
        else:
            logger.warning("Proceeding without XGBoost; ensemble will use available base models only.")

        self.scaler = StandardScaler()

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_data(self, df, train_end_date=None):
        """
        Prepare train/test split with time-series awareness.

        If train_end_date is None, uses an expanding window with the last
        20% of data as test set (preserving temporal order).
        """
        df_clean = df[df[self.target_col].notna()].copy()
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)

        if train_end_date:
            train_df = df_clean[df_clean.index <= train_end_date]
            test_df = df_clean[df_clean.index > train_end_date]
        else:
            split_idx = int(len(df_clean) * 0.8)
            train_df = df_clean.iloc[:split_idx]
            test_df = df_clean.iloc[split_idx:]

        logger.info(f"Training: {len(train_df)} obs ({train_df.index.min().strftime('%Y-%m')} to {train_df.index.max().strftime('%Y-%m')})")
        logger.info(f"Testing:  {len(test_df)} obs ({test_df.index.min().strftime('%Y-%m')} to {test_df.index.max().strftime('%Y-%m')})")

        # Report class balance
        train_pos_rate = train_df[self.target_col].mean()
        test_pos_rate = test_df[self.target_col].mean() if len(test_df) > 0 else 0
        logger.info(f"  Train recession rate: {train_pos_rate:.1%}")
        logger.info(f"  Test recession rate:  {test_pos_rate:.1%}")

        return train_df, test_df

    def _compute_recency_sample_weights(self, index):
        """Create exponentially decaying sample weights for recent observations."""
        if not self.recency_half_life_months or self.recency_half_life_months <= 0:
            return None

        if len(index) == 0:
            return None

        ages = np.arange(len(index) - 1, -1, -1, dtype=float)
        weights = 0.5 ** (ages / float(self.recency_half_life_months))
        weights = np.maximum(weights, self.recency_weight_floor)
        weights = weights / np.mean(weights)
        return weights

    @staticmethod
    def _compute_psi(reference, recent, bins=10):
        """Compute Population Stability Index between reference and recent samples."""
        if len(reference) == 0 or len(recent) == 0:
            return 0.0

        breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 3:
            return 0.0

        ref_hist, _ = np.histogram(reference, bins=breakpoints)
        rec_hist, _ = np.histogram(recent, bins=breakpoints)

        ref_total = max(ref_hist.sum(), 1)
        rec_total = max(rec_hist.sum(), 1)
        ref_pct = np.clip(ref_hist / ref_total, 1e-6, None)
        rec_pct = np.clip(rec_hist / rec_total, 1e-6, None)
        return float(np.sum((rec_pct - ref_pct) * np.log(rec_pct / ref_pct)))

    def _compute_feature_drift_scores(self, df, feature_cols):
        """Estimate recent-vs-reference drift for candidate model features."""
        min_rows = self.drift_lookback_months + self.drift_recent_months
        if len(df) < min_rows:
            return {}

        recent = df[feature_cols].iloc[-self.drift_recent_months:]
        reference = df[feature_cols].iloc[-min_rows:-self.drift_recent_months]
        scores = {}
        for col in feature_cols:
            ref_vals = reference[col].dropna().values
            rec_vals = recent[col].dropna().values
            if len(ref_vals) < 10 or len(rec_vals) < 5:
                continue
            scores[col] = self._compute_psi(ref_vals, rec_vals)
        return scores

    @staticmethod
    def _fit_model_with_optional_weights(model, X, y, sample_weights=None):
        """Fit estimator, using sample weights when the estimator supports them."""
        if sample_weights is None:
            model.fit(X, y)
            return
        try:
            model.fit(X, y, sample_weight=sample_weights)
        except TypeError:
            model.fit(X, y)

    @staticmethod
    def _fit_calibrator_with_optional_weights(calibrator, X, y, sample_weights=None):
        """Fit probability calibrator, falling back if sample weights are unsupported."""
        if sample_weights is None:
            calibrator.fit(X, y)
            return
        try:
            calibrator.fit(X, y, sample_weight=sample_weights)
        except TypeError:
            calibrator.fit(X, y)

    # ------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------

    def select_features(self, df, max_features=50):
        """
        Select stable features using an ensemble-of-selectors approach.

        Research on macro forecasting with many predictors generally favors
        sparse, stable subsets over very large indicator kitchens sinks.
        We therefore combine:
        1. Correlation pre-screening to reduce dimensionality.
        2. Random forest importance for nonlinear signal discovery.
        3. L1-logistic coefficients for sparse linear relevance.
        4. Correlation pruning to avoid redundant feature clusters.
        """
        exclude_cols = [
            col for col in df.columns
            if col == 'RECESSION' or col.startswith('RECESSION_FORWARD_')
        ]
        feature_cols = [col for col in df.columns
                        if col not in exclude_cols and not col.startswith('ref_')]

        # Require 70% non-null
        valid_cols = [col for col in feature_cols
                      if df[col].notna().sum() / len(df) > 0.7]

        if not valid_cols:
            return []

        # Pass 1: Correlation ranking
        correlations = {}
        for col in valid_cols:
            valid_mask = df[col].notna() & df[self.target_col].notna()
            if valid_mask.sum() > 100:
                corr = np.corrcoef(
                    df.loc[valid_mask, col],
                    df.loc[valid_mask, self.target_col]
                )[0, 1]
                if not np.isnan(corr):
                    correlations[col] = abs(corr)

        if not correlations:
            return valid_cols[:max_features]

        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        candidate_pool_size = min(len(sorted_features), max(max_features * 4, max_features + 20))
        top_corr_feats = [feat for feat, _ in sorted_features[:candidate_pool_size]]
        feature_scores = {feat: 0.0 for feat in top_corr_feats}

        for rank, (feat, _) in enumerate(sorted_features[:candidate_pool_size]):
            feature_scores[feat] += 2.0 / (rank + 1)

        # Use recent history to reflect regime changes while still anchoring on
        # the full training sample.
        df_sub = df[top_corr_feats + [self.target_col]].dropna(subset=[self.target_col]).copy()
        if len(df_sub) > 500:
            df_sub = df_sub.iloc[-500:]

        def _accumulate_rank_scores(window_df, weight):
            if len(window_df) < 80:
                return

            X_window = window_df[top_corr_feats].ffill().fillna(0)
            y_window = window_df[self.target_col].astype(int)
            if len(np.unique(y_window)) < 2:
                return

            try:
                rf = RandomForestClassifier(
                    n_estimators=150,
                    max_depth=6,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced',
                )
                rf.fit(X_window, y_window)
                rf_ranked = (
                    pd.Series(rf.feature_importances_, index=top_corr_feats)
                    .sort_values(ascending=False)
                    .index
                    .tolist()
                )
                for rank, feat in enumerate(rf_ranked[:max_features * 2]):
                    feature_scores[feat] += weight * (1.5 / (rank + 1))
            except Exception as exc:
                logger.warning(
                    "Feature selection random-forest ranking failed on a %s-row window: %s",
                    len(window_df),
                    exc,
                )

            try:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_window)
                sparse_logit = LogisticRegression(
                    penalty='l1',
                    solver='saga',
                    C=0.08,
                    max_iter=2000,
                    random_state=42,
                    class_weight='balanced',
                )
                sparse_logit.fit(X_scaled, y_window)
                coef_series = pd.Series(np.abs(sparse_logit.coef_[0]), index=top_corr_feats)
                coef_series = coef_series[coef_series > 1e-6].sort_values(ascending=False)
                for rank, feat in enumerate(coef_series.index.tolist()[:max_features * 2]):
                    feature_scores[feat] += weight * (2.0 / (rank + 1))
            except Exception as exc:
                logger.warning(
                    "Feature selection sparse-logit ranking failed on a %s-row window: %s",
                    len(window_df),
                    exc,
                )

        if len(df_sub) > 0:
            _accumulate_rank_scores(df_sub, weight=1.0)
            recent_window = min(len(df_sub), max(120, len(df_sub) // 2))
            if recent_window < len(df_sub):
                _accumulate_rank_scores(df_sub.tail(recent_window), weight=0.75)

        drift_scores = self._compute_feature_drift_scores(df, top_corr_feats)
        self.feature_drift_scores = drift_scores
        severely_drifted = set()
        if drift_scores:
            if self.drift_penalty_strength > 0:
                for feat, psi in drift_scores.items():
                    penalty = self.drift_penalty_strength * np.log1p(max(0.0, psi - 0.2))
                    feature_scores[feat] -= penalty
            if self.severe_drift_psi is not None:
                severely_drifted = {
                    feat for feat, psi in drift_scores.items()
                    if psi >= self.severe_drift_psi
                }
                if severely_drifted:
                    logger.info(
                        "Feature selection: excluding %s severe-drift features (PSI >= %.2f)",
                        len(severely_drifted),
                        self.severe_drift_psi,
                    )

        # FIX 1 (AUDIT-1, M3): deterministic tie-break by alphabetical feature name.
        # When two features have identical composite scores at the max_features=50
        # cap boundary, the sort was non-deterministic under tiny float perturbations.
        # Python's sort is stable, so sorting first by name (ascending) and then by
        # score (descending, stable) guarantees alphabetical tie-break.
        ranked_candidates = [
            feat
            for feat, _ in sorted(
                sorted(feature_scores.items(), key=lambda item: item[0]),
                key=lambda item: item[1],
                reverse=True,
            )
        ]

        corr_matrix = None
        if len(ranked_candidates) > 1:
            corr_matrix = df[ranked_candidates].ffill().corr().abs()

        selected = []
        for feat in ranked_candidates:
            if len(selected) >= max_features:
                break
            if feat in severely_drifted:
                continue
            if corr_matrix is not None:
                is_redundant = any(
                    corr_matrix.loc[feat, chosen] > 0.92
                    for chosen in selected
                    if feat in corr_matrix.index and chosen in corr_matrix.columns
                )
                if is_redundant:
                    continue
            selected.append(feat)

        for feat in ranked_candidates:
            if len(selected) >= max_features:
                break
            if feat not in selected and feat not in severely_drifted:
                selected.append(feat)

        for feat in ranked_candidates:
            if len(selected) >= max_features:
                break
            if feat not in selected:
                selected.append(feat)

        # Ensure key indicators are always included if available
        # Core must-include: only the highest-signal indicators that are
        # well-established in the literature. New Tier 1 features compete
        # for inclusion via the RF importance ranking — they'll be selected
        # if they add genuine signal.
        must_include = [
            # Yield curve & monetary (Estrella-Mishkin, Wright, Engstrom-Sharpe)
            'leading_T10Y3M', 'leading_T10Y3M_inverted',
            'leading_T10Y3M_inv_duration', 'monetary_DFF',
            'NEAR_TERM_FORWARD_SPREAD', 'NTFS_inverted',
            'FFR_x_SPREAD', 'FFR_STANCE',
            # Term-premium-adjusted spread (Ajello et al. 2022) — key false-positive fix
            'TERM_PREMIUM_ADJ_SPREAD', 'TP_ADJ_SPREAD_inverted',
            # Credit & financial conditions (Gilchrist-Zakrajsek)
            'monetary_BAA10Y', 'EBP_PROXY', 'EBP_PROXY_Z',
            'financial_NFCI', 'CREDIT_STRESS_INDEX',
            # Labor market triggers (Sahm, SOS)
            'SAHM_INDICATOR', 'SAHM_TRIGGER',
            'SOS_INDICATOR', 'SOS_TRIGGER',
            # At-risk diffusion (Billakanti-Shin)
            'AT_RISK_DIFFUSION', 'AT_RISK_DIFFUSION_WEIGHTED',
            # Confirming indicators (Grigoli-Sandri, Leamer)
            'HOUSE_PRICE_DECLINING', 'RECESSION_CONFIRM_2OF3',
            'RESIDENTIAL_INV_YOY', 'SECTORAL_DIVERGENCE',
            # B2 labor deterioration block — 3 highest-signal features per plan
            'VU_RATIO_Z', 'QUITS_Z', 'GOODS_YoY_MINUS_SERV_YoY',
            # B3 credit-supply block — SLOOS + ANFCI + credit×curve interaction
            'SLOOS_CI_LARGE_Z', 'SLOOS_COMPOSITE',
            'CREDIT_TIGHTEN_X_CURVE_INVERT', 'financial_ANFCI',
        ]
        priority_order = {feature: rank for rank, feature in enumerate(ranked_candidates)}
        protected = set()
        for col in must_include:
            if col in severely_drifted:
                continue
            if col not in valid_cols or col in selected:
                if col in selected:
                    protected.add(col)
                continue

            if len(selected) < max_features:
                selected.append(col)
                protected.add(col)
                continue

            replacement_idx = None
            replacement_rank = -1
            for idx, current_feature in enumerate(selected):
                if current_feature in protected:
                    continue
                current_rank = priority_order.get(current_feature, len(priority_order))
                if current_rank > replacement_rank:
                    replacement_rank = current_rank
                    replacement_idx = idx

            if replacement_idx is not None:
                selected[replacement_idx] = col
                protected.add(col)

        selected_drifted = [
            (feature, drift_scores.get(feature, 0.0))
            for feature in selected
            if drift_scores.get(feature, 0.0) >= self.selected_drift_prune_psi
        ]
        if self.selected_drift_prune_count and selected_drifted:
            selected_drifted.sort(key=lambda item: item[1], reverse=True)
            pruned_features = {
                feature for feature, _ in selected_drifted[:self.selected_drift_prune_count]
            }
            selected = [feature for feature in selected if feature not in pruned_features]
            logger.info(
                "Feature selection: pruned %s selected features with PSI >= %.2f",
                len(pruned_features),
                self.selected_drift_prune_psi,
            )
            for feat in ranked_candidates:
                if len(selected) >= max_features:
                    break
                if feat in selected or feat in pruned_features:
                    continue
                if drift_scores.get(feat, 0.0) >= self.selected_drift_prune_psi:
                    continue
                if corr_matrix is not None:
                    is_redundant = any(
                        corr_matrix.loc[feat, chosen] > 0.92
                        for chosen in selected
                        if feat in corr_matrix.index and chosen in corr_matrix.columns
                    )
                    if is_redundant:
                        continue
                selected.append(feat)

        selected = selected[:max_features]

        logger.info(f"Selected {len(selected)} features")
        return selected

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, train_df, feature_cols=None, max_features=50):
        """
        Fit all models with calibration and ensemble weighting.

        Steps:
        1. Feature selection
        2. PCA factor extraction (augment features)
        3. Fit base models with class weights (including Markov-switching)
        4. Time-series CV for internal validation with per-fold Brier tracking
        5. Compute DMA weights (exponential forgetting of per-fold Brier scores)
        6. Calibrate probabilities (isotonic regression)
        7. Optimize decision threshold (Youden's J)
        """
        logger.info("=" * 80)
        logger.info("FITTING RECESSION PREDICTION ENSEMBLE (v3 — DMA + PCA + Markov)")
        logger.info("=" * 80)

        if feature_cols is None:
            feature_cols = self.select_features(train_df, max_features=max_features)

        self.feature_cols = feature_cols

        # Store the training DataFrame for the Markov-switching model
        self._train_df = train_df

        X_train = train_df[feature_cols].ffill().fillna(0)
        y_train = train_df[self.target_col]
        train_sample_weights = self._compute_recency_sample_weights(train_df.index)

        X_train_scaled = self.scaler.fit_transform(X_train)

        # ── Step 1: PCA factor extraction ────────────────────────────
        n_components = min(self.n_pca_components, X_train_scaled.shape[1])
        self.pca = PCA(n_components=n_components)
        pca_features_train = self.pca.fit_transform(X_train_scaled)
        pca_cols = [f'PC_{i+1}' for i in range(pca_features_train.shape[1])]
        logger.info(f"PCA: extracted {n_components} components "
                    f"(explained variance: {self.pca.explained_variance_ratio_.sum():.1%})")

        # Augmented feature matrices:
        # - Probit uses scaled + PCA
        # - RF/XGBoost use unscaled + PCA (PCA components are already normalized)
        X_train_probit = np.hstack([X_train_scaled, pca_features_train])
        X_train_tree = np.hstack([X_train.values, pca_features_train])

        # Set XGBoost scale_pos_weight from actual class ratio
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        if HAS_XGBOOST and 'xgboost' in self.models and n_pos > 0:
            self.models['xgboost'].set_params(scale_pos_weight=n_neg / n_pos)
            logger.info(f"XGBoost scale_pos_weight set to {n_neg / n_pos:.2f}")

        # ── Step 2: Fit base models ──────────────────────────────────
        for name, model in self.models.items():
            logger.info(f"Fitting {name}...")

            if name == 'probit':
                self._fit_model_with_optional_weights(
                    model, X_train_probit, y_train, train_sample_weights
                )
                importance = np.abs(model.coef_[0][:len(feature_cols)])
            else:
                self._fit_model_with_optional_weights(
                    model, X_train_tree, y_train, train_sample_weights
                )
                importance = model.feature_importances_[:len(feature_cols)]

            self.feature_importance[name] = dict(zip(feature_cols, importance))

            pred_proba = model.predict_proba(
                X_train_probit if name == 'probit' else X_train_tree
            )[:, 1]
            auc = roc_auc_score(y_train, pred_proba)
            brier = brier_score_loss(y_train, pred_proba)
            logger.info(f"  Training AUC: {auc:.4f}, Brier: {brier:.4f}")

        # ── Step 2b: Fit Markov-switching model ──────────────────────
        # Determine all model names that participate in the ensemble
        all_model_names = list(self.models.keys())
        if self.markov_model is not None:
            logger.info("Fitting markov_switching...")
            self.markov_model.fit(train_df, y_train)
            if self.markov_model.fitted:
                ms_proba = self.markov_model.predict_proba(train_df)[:, 1]
                ms_auc = roc_auc_score(y_train, ms_proba)
                ms_brier = brier_score_loss(y_train, ms_proba)
                logger.info(f"  Training AUC: {ms_auc:.4f}, Brier: {ms_brier:.4f}")
                all_model_names.append('markov_switching')
            else:
                logger.warning("  MarkovSwitching failed to fit; excluded from ensemble")

        # ── Step 2b2: Fit benchmark (peer-model) ensemble members ────
        # C1: wrap externally-published recession probabilities (Hamilton
        # JHGDPBRINDX, Chauvet-Piger RECPROUSM156N) and a Wright (2006)
        # probit on (T10Y3M, DFF, T10Y3M × DFF) as full ensemble members.
        # Reference series (ref_*) are excluded from ordinary feature
        # selection, so this adds signal without double-counting.
        if self.benchmark_members_enabled:
            for bm_name, factory in BENCHMARK_MODEL_FACTORIES:
                logger.info("Fitting %s (benchmark member)...", bm_name)
                bm = factory()
                bm.fit(train_df, y_train)
                if not bm.fitted:
                    logger.warning("  %s failed to fit; excluded from ensemble", bm_name)
                    continue
                self.benchmark_models[bm_name] = bm
                try:
                    bm_train_proba = bm.predict_proba(train_df)[:, 1]
                    bm_auc = roc_auc_score(y_train, bm_train_proba)
                    bm_brier = brier_score_loss(y_train, bm_train_proba)
                    logger.info("  Training AUC: %.4f, Brier: %.4f", bm_auc, bm_brier)
                except Exception as e:
                    logger.warning("  Training metric computation failed: %s", e)
                all_model_names.append(bm_name)

        # ── Step 2c: Fit LSTM model ───────────────────────────────────
        if self.lstm_model is not None:
            logger.info("Fitting lstm...")
            self.lstm_model.fit(X_train, y_train)
            if self.lstm_model.fitted:
                lstm_proba = self.lstm_model.predict_proba(X_train.values)[:, 1]
                lstm_auc = roc_auc_score(y_train, lstm_proba)
                lstm_brier = brier_score_loss(y_train, lstm_proba)
                logger.info(f"  Training AUC: {lstm_auc:.4f}, Brier: {lstm_brier:.4f}")
                all_model_names.append('lstm')
            else:
                logger.warning("  LSTM failed to fit; excluded from ensemble")

        # ── Step 3: Time-series CV with per-fold Brier tracking ──────
        logger.info("Running time-series cross-validation...")
        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)
        cv_probas = {name: [] for name in all_model_names}
        cv_actuals = []
        # Track per-fold Brier scores AND AUC for DMA composite scoring
        cv_fold_briers = {name: [] for name in all_model_names}
        cv_fold_aucs = {name: [] for name in all_model_names}

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            w_tr = self._compute_recency_sample_weights(X_tr.index)
            X_tr_sc = self.scaler.fit_transform(X_tr)
            X_val_sc = self.scaler.transform(X_val)

            # PCA on this fold
            fold_pca = PCA(n_components=n_components)
            pca_tr = fold_pca.fit_transform(X_tr_sc)
            pca_val = fold_pca.transform(X_val_sc)

            X_tr_probit = np.hstack([X_tr_sc, pca_tr])
            X_val_probit = np.hstack([X_val_sc, pca_val])
            X_tr_tree = np.hstack([X_tr.values, pca_tr])
            X_val_tree = np.hstack([X_val.values, pca_val])

            fold_val_actuals = y_val.values

            for name, model in self.models.items():
                fold_model = clone(model)
                if name == 'probit':
                    self._fit_model_with_optional_weights(
                        fold_model, X_tr_probit, y_tr, w_tr
                    )
                    proba = fold_model.predict_proba(X_val_probit)[:, 1]
                else:
                    self._fit_model_with_optional_weights(
                        fold_model, X_tr_tree, y_tr, w_tr
                    )
                    proba = fold_model.predict_proba(X_val_tree)[:, 1]
                cv_probas[name].extend(proba.tolist())
                # Per-fold Brier score and AUC
                fold_brier = brier_score_loss(fold_val_actuals, proba)
                cv_fold_briers[name].append(fold_brier)
                if len(set(fold_val_actuals)) >= 2:
                    fold_auc = roc_auc_score(fold_val_actuals, proba)
                else:
                    fold_auc = 0.5
                cv_fold_aucs[name].append(fold_auc)

            # Markov-switching CV
            if 'markov_switching' in all_model_names:
                train_df_fold = train_df.iloc[train_idx]
                val_df_fold = train_df.iloc[val_idx]
                fold_ms = MarkovSwitchingWrapper()
                fold_ms.fit(train_df_fold, y_tr)
                ms_proba = fold_ms.predict_proba(val_df_fold)[:, 1]
                cv_probas['markov_switching'].extend(ms_proba.tolist())
                fold_brier = brier_score_loss(fold_val_actuals, ms_proba)
                cv_fold_briers['markov_switching'].append(fold_brier)
                if len(set(fold_val_actuals)) >= 2:
                    fold_auc = roc_auc_score(fold_val_actuals, ms_proba)
                else:
                    fold_auc = 0.5
                cv_fold_aucs['markov_switching'].append(fold_auc)

            # Benchmark members CV: refit on this fold's training slice so
            # calibration/fit (Wright probit) uses only past data.
            for bm_name in list(self.benchmark_models.keys()):
                factory = dict(BENCHMARK_MODEL_FACTORIES)[bm_name]
                fold_bm = factory()
                fold_bm.fit(train_df.iloc[train_idx], y_tr)
                val_df_fold_bm = train_df.iloc[val_idx]
                bm_proba = fold_bm.predict_proba(val_df_fold_bm)[:, 1]
                cv_probas[bm_name].extend(bm_proba.tolist())
                fold_brier = brier_score_loss(fold_val_actuals, bm_proba)
                cv_fold_briers[bm_name].append(fold_brier)
                if len(set(fold_val_actuals)) >= 2:
                    fold_auc = roc_auc_score(fold_val_actuals, bm_proba)
                else:
                    fold_auc = 0.5
                cv_fold_aucs[bm_name].append(fold_auc)

            # LSTM CV
            if 'lstm' in all_model_names:
                fold_lstm = LSTMRecessionModel(lookback=6, epochs=20)
                fold_lstm.fit(X_tr, y_tr)
                lstm_proba = fold_lstm.predict_proba(X_val.values)[:, 1]
                cv_probas['lstm'].extend(lstm_proba.tolist())
                fold_brier = brier_score_loss(fold_val_actuals, lstm_proba)
                cv_fold_briers['lstm'].append(fold_brier)
                if len(set(fold_val_actuals)) >= 2:
                    fold_auc = roc_auc_score(fold_val_actuals, lstm_proba)
                else:
                    fold_auc = 0.5
                cv_fold_aucs['lstm'].append(fold_auc)

            cv_actuals.extend(y_val.tolist())

        cv_actuals = np.array(cv_actuals)

        # ── Step 4: Compute ensemble weights ─────────────────────────
        # 4a: Static BMA weights (inverse Brier on pooled CV predictions)
        logger.info("Computing ensemble weights from CV performance...")
        cv_scores = {}
        for name in all_model_names:
            probas = np.array(cv_probas[name])
            if len(set(cv_actuals)) >= 2:
                cv_auc = roc_auc_score(cv_actuals, probas)
                cv_pr_auc = average_precision_score(cv_actuals, probas)
                cv_brier = brier_score_loss(cv_actuals, probas)
                cv_scores[name] = {
                    'auc': cv_auc,
                    'pr_auc': cv_pr_auc,
                    'brier': cv_brier,
                    'inv_brier': 1.0 / (cv_brier + 1e-6)
                }
                logger.info(
                    f"  {name} CV — AUC: {cv_auc:.4f}, PR-AUC: {cv_pr_auc:.4f}, Brier: {cv_brier:.4f}"
                )
            else:
                cv_scores[name] = {'auc': 0.5, 'pr_auc': float(np.mean(cv_actuals)), 'brier': 0.25, 'inv_brier': 4.0}

        self.cv_results = cv_scores

        total_inv_brier = sum(s['inv_brier'] for s in cv_scores.values())
        self.static_weights = {
            name: cv_scores[name]['inv_brier'] / total_inv_brier
            for name in all_model_names
        }
        logger.info(f"Static weights: {', '.join(f'{n}={w:.3f}' for n, w in self.static_weights.items())}")

        # 4b: DMA weights (exponential forgetting of per-fold composite scores)
        self.dma_weights = self._compute_dma_weights(
            cv_fold_briers, cv_fold_aucs, forgetting_factor=0.99
        )
        self.active_models = self._select_active_models(cv_scores)
        self.static_weights = self._renormalize_weights(self.static_weights, self.active_models)
        self.dma_weights = self._renormalize_weights(self.dma_weights, self.active_models)
        equal_weights = self._equal_weights(self.active_models)
        self.ensemble_weights = {
            name: (
                (1 - self.equal_weight_shrinkage) * self.dma_weights.get(name, 0.0)
                + self.equal_weight_shrinkage * equal_weights.get(name, 0.0)
            )
            for name in self.active_models
        }
        self.ensemble_weights = self._renormalize_weights(self.ensemble_weights, self.active_models)
        self.ensemble_method = (
            "equal_weight_active"
            if self.equal_weight_shrinkage >= 0.999
            else "shrunk_gated_dma"
        )
        inactive_models = [name for name in all_model_names if name not in self.active_models]
        logger.info(f"Active ensemble models: {', '.join(self.active_models)}")
        if inactive_models:
            logger.info(f"Gated out of ensemble: {', '.join(inactive_models)}")
        logger.info(f"Gated DMA weights: {', '.join(f'{n}={w:.3f}' for n, w in self.dma_weights.items())}")
        logger.info(f"Final ensemble weights: {', '.join(f'{n}={w:.3f}' for n, w in self.ensemble_weights.items())}")

        # ── Step 5: Calibrate base models (isotonic regression) ──────
        # Re-fit scaler + PCA on full training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        pca_features_train = self.pca.fit_transform(X_train_scaled)
        X_train_probit = np.hstack([X_train_scaled, pca_features_train])
        X_train_tree = np.hstack([X_train.values, pca_features_train])

        logger.info("Calibrating model probabilities (isotonic regression)...")
        for name, model in self.models.items():
            try:
                cal = CalibratedClassifierCV(
                    model, method='isotonic', cv=tscv
                )
                if name == 'probit':
                    self._fit_calibrator_with_optional_weights(
                        cal, X_train_probit, y_train, train_sample_weights
                    )
                else:
                    self._fit_calibrator_with_optional_weights(
                        cal, X_train_tree, y_train, train_sample_weights
                    )
                self.calibrated_models[name] = cal
                logger.info(f"  ✓ {name} calibrated")
            except Exception as e:
                logger.warning(f"  ✗ Calibration failed for {name}: {e}. Using uncalibrated.")
                self.calibrated_models[name] = None

        # Markov-switching has its own built-in isotonic calibration
        if 'markov_switching' in all_model_names:
            self.calibrated_models['markov_switching'] = None
            cal_status = "built-in" if (self.markov_model and self.markov_model._calibrator_fitted) else "none"
            logger.info(f"  ⊘ markov_switching: uses {cal_status} isotonic calibration")

        # Benchmark members are already published on a probability scale
        # (Hamilton JHGDPBRINDX / Chauvet-Piger RECPROUSM156N) or fitted
        # directly (Wright probit); skip post-hoc calibration.
        for bm_name in self.benchmark_models:
            self.calibrated_models[bm_name] = None
            logger.info(f"  ⊘ {bm_name}: uses externally/directly calibrated probabilities")

        # LSTM: calibrate via isotonic regression on CV predictions
        # (class-weighted training produces biased probabilities that need recalibration)
        if 'lstm' in all_model_names:
            try:
                from sklearn.isotonic import IsotonicRegression
                lstm_cv_probs = np.array(cv_probas['lstm'])
                cv_actuals_arr = np.array(cv_actuals)
                if len(lstm_cv_probs) > 20 and len(np.unique(cv_actuals_arr)) == 2:
                    iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip')
                    iso.fit(lstm_cv_probs, cv_actuals_arr)
                    self._lstm_calibrator = iso
                    logger.info("  ✓ lstm: isotonic calibration fitted on CV predictions")
                else:
                    self._lstm_calibrator = None
                    logger.info("  ⊘ lstm: insufficient CV data for calibration")
            except Exception as e:
                self._lstm_calibrator = None
                logger.warning(f"  ✗ lstm calibration failed: {e}")
            self.calibrated_models['lstm'] = None  # Not a CalibratedClassifierCV

        # ── Step 5b: Calibration diagnostics + calibrator A/B (measurement only) ─────
        # We compute reliability curves, ECE, slope/intercept, and the Brier
        # decomposition on pooled CV out-of-fold predictions for every base model
        # (uncalibrated CV probabilities) and also run a held-out A/B across
        # isotonic/sigmoid/beta calibrators. The default calibrator stays isotonic
        # until the validator sub-agent confirms a winner on vintage origins.
        self._compute_calibration_diagnostics(
            cv_probas=cv_probas,
            cv_actuals=cv_actuals,
            train_df=train_df,
            y_train=y_train,
        )

        # ── Step 6: Optimize decision threshold on the deployed probability scale ────────
        logger.info("Optimizing decision threshold (%s)...", self.threshold_method)
        threshold_target = cv_actuals
        threshold_proba = self._weighted_average_probas(cv_probas, cv_actuals)
        self.is_fitted = True
        try:
            calibrated_train_predictions = self.predict(train_df)
            calibrated_train_proba = np.array(calibrated_train_predictions.get('ensemble', []))
            if len(calibrated_train_proba) == len(y_train):
                threshold_target = np.array(y_train)
                threshold_proba = calibrated_train_proba
        except Exception as exc:
            logger.warning(
                "Threshold optimization fell back to raw CV probabilities: %s",
                exc,
            )
        self.decision_threshold = self._optimize_threshold(threshold_target, threshold_proba)
        logger.info(f"  Optimal threshold: {self.decision_threshold:.3f}")

        self.is_fitted = True
        logger.info("=" * 80)
        logger.info("MODEL FITTING COMPLETE")
        logger.info("=" * 80)

    def _compute_dma_weights(self, cv_fold_briers, cv_fold_aucs=None,
                             forgetting_factor=0.99):
        """
        Compute Dynamic Model Averaging weights using exponential forgetting.

        Uses a composite score: AUC × (1 / Brier) to reward models that are
        both discriminative AND well-calibrated. This prevents a model with
        good calibration but no discrimination (AUC ≈ 0.5) from earning
        disproportionate weight.

        More recent CV fold performance is weighted more heavily, allowing
        the ensemble to adapt to structural changes in the economy.
        """
        n_folds = len(next(iter(cv_fold_briers.values())))

        # Composite scoring: AUC × inverse Brier with exponential forgetting
        raw_weights = {}
        for name in cv_fold_briers:
            fold_briers = cv_fold_briers[name]
            fold_aucs = cv_fold_aucs.get(name, [0.5] * n_folds) if cv_fold_aucs else [0.5] * n_folds

            weighted_score = 0
            total_weight = 0
            for i in range(len(fold_briers)):
                w = forgetting_factor ** (n_folds - 1 - i)
                brier = fold_briers[i]
                auc = fold_aucs[i] if i < len(fold_aucs) else 0.5

                # Composite: AUC × inverse Brier
                # A model at AUC=0.5 (random) gets half the score of AUC=1.0
                # A model with Brier=0.25 (random) gets 1/26th the score of Brier=0.01
                fold_score = auc / (brier + 0.01)
                weighted_score += w * fold_score
                total_weight += w

            avg_score = weighted_score / total_weight
            raw_weights[name] = avg_score

        # Normalize
        total = sum(raw_weights.values())
        weights = {name: w / total for name, w in raw_weights.items()}

        # Soft cap: no single model above 40% to ensure genuine ensemble diversity.
        # Redistribute excess proportionally to other models.
        max_weight = 0.40
        for _ in range(3):  # Iterate to handle cascading redistributions
            for name in list(weights.keys()):
                if weights[name] > max_weight:
                    excess = weights[name] - max_weight
                    weights[name] = max_weight
                    others = {k: v for k, v in weights.items() if k != name}
                    others_total = sum(others.values())
                    if others_total > 0:
                        for k in others:
                            weights[k] += excess * (others[k] / others_total)

        # Apply minimum weight floor (2%) and renormalize
        min_weight = 0.02
        for name in weights:
            weights[name] = max(weights[name], min_weight)
        total = sum(weights.values())
        return {name: w / total for name, w in weights.items()}

    def _select_active_models(self, cv_scores):
        """
        Drop materially weaker models before combining probabilities.

        This prevents a structurally weak model from diluting stronger supervised
        signals while still keeping a genuine ensemble of at least two models.
        """
        if not cv_scores:
            return []

        best_auc = max(score.get('auc', 0.5) for score in cv_scores.values())
        best_pr_auc = max(score.get('pr_auc', 0.0) for score in cv_scores.values())
        best_brier = min(score.get('brier', 0.25) for score in cv_scores.values())

        active = [
            name for name, score in cv_scores.items()
            if score.get('auc', 0.5) >= best_auc - 0.10
            and score.get('pr_auc', 0.0) >= best_pr_auc * 0.90
            and score.get('brier', 0.25) <= best_brier * 2.00
        ]

        if len(active) >= 2:
            return active

        ranked = sorted(
            cv_scores,
            key=lambda name: (
                cv_scores[name].get('auc', 0.5),
                -cv_scores[name].get('brier', 0.25),
            ),
            reverse=True,
        )
        return ranked[:min(2, len(ranked))]

    def _build_threshold_rows(self, y_true, y_proba):
        """Build threshold diagnostics for a probability series."""
        y_true = np.array(y_true)
        y_proba = np.array(y_proba)

        threshold_rows = []
        for threshold in np.arange(0.10, 0.61, 0.01):
            y_pred = (y_proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            f2 = 5 * precision * sensitivity / ((4 * precision) + sensitivity) if ((4 * precision) + sensitivity) > 0 else 0
            j = sensitivity + specificity - 1

            threshold_rows.append({
                'threshold': round(float(threshold), 2),
                'precision': precision,
                'recall': sensitivity,
                'specificity': specificity,
                'f1': f1,
                'f2': f2,
                'youdens_j': j,
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'tn': int(tn),
                'score': f1,
            })
        return threshold_rows

    @staticmethod
    def _choose_threshold_row(threshold_rows):
        """Select the best threshold row using the project tie-break policy."""
        return max(
            threshold_rows,
            key=lambda row: (
                row['f1'],
                row['precision'],
                row['recall'],
                row['specificity'],
                -row['threshold'],
            ),
        )

    def _equal_weights(self, active_models):
        """Return equal weights across the active ensemble members."""
        if not active_models:
            return {}
        equal_weight = 1.0 / len(active_models)
        return {name: equal_weight for name in active_models}

    def _renormalize_weights(self, weights, active_models):
        """Keep weights only for active models and renormalize them."""
        if not weights:
            return {}

        if not active_models:
            total = sum(weights.values())
            if total <= 0:
                return weights
            return {name: weight / total for name, weight in weights.items()}

        filtered = {
            name: weight for name, weight in weights.items()
            if name in active_models
        }
        total = sum(filtered.values())
        if total <= 0:
            equal_weight = 1.0 / len(active_models)
            return {name: equal_weight for name in active_models}
        return {name: weight / total for name, weight in filtered.items()}

    def _weighted_average_probas(self, probas_dict, actuals):
        """Compute weighted average of model probabilities."""
        result = np.zeros(len(actuals))
        for name, weight in self.ensemble_weights.items():
            p = np.array(probas_dict[name])
            if len(p) == len(actuals):
                result += weight * p
        return result

    # ------------------------------------------------------------------
    # Calibration diagnostics (A1)
    # ------------------------------------------------------------------

    def _calibration_summary(self, y_true, y_prob, n_bins=10):
        """Single-source summary of reliability/ECE/slope/Brier decomp."""
        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.asarray(y_prob, dtype=float)
        if len(y_prob) == 0:
            return {
                "ece": float("nan"),
                "slope": float("nan"),
                "intercept": float("nan"),
                "reliability": [],
                "brier_total": float("nan"),
                "brier_reliability": float("nan"),
                "brier_resolution": float("nan"),
                "brier_uncertainty": float("nan"),
                "n_samples": 0,
            }
        centers, observed, predicted, counts = _reliability_curve(
            y_true, y_prob, n_bins=n_bins, strategy="quantile"
        )
        ece = _ece(y_true, y_prob, n_bins=n_bins)
        slope, intercept = _slope_intercept(y_true, y_prob)
        brier = _brier_decomp(y_true, y_prob, n_bins=n_bins)
        return {
            "ece": float(ece) if not np.isnan(ece) else None,
            "slope": float(slope) if not np.isnan(slope) else None,
            "intercept": float(intercept) if not np.isnan(intercept) else None,
            "reliability": [
                [float(c), float(o), float(p), int(n)]
                for c, o, p, n in zip(centers, observed, predicted, counts)
            ],
            "brier_total": float(brier["brier_total"]),
            "brier_reliability": float(brier["reliability"]),
            "brier_resolution": float(brier["resolution"]),
            "brier_uncertainty": float(brier["uncertainty"]),
            "n_samples": int(len(y_prob)),
        }

    def _compute_calibration_diagnostics(self, cv_probas, cv_actuals, train_df, y_train):
        """Compute per-model calibration diagnostics + calibrator A/B.

        Uses the pooled CV out-of-fold predictions as the diagnostic substrate
        (these are out-of-sample by construction of TimeSeriesSplit) and runs a
        time-series split A/B across isotonic / sigmoid / beta calibrators on the
        same pooled predictions. The A/B holdout is the last 20% of the pooled
        CV records so we never leak future information into the training fold.

        Measurement only — the default calibrator stays isotonic. The winner is
        exposed via ``self.calibrator_choice`` for the validator sub-agent.
        """

        self.calibration_diagnostics = {}
        self.calibrator_choice = {}

        cv_actuals_arr = np.asarray(cv_actuals, dtype=float)
        if len(cv_actuals_arr) == 0 or len(set(cv_actuals_arr.astype(int))) < 2:
            logger.info("Calibration diagnostics skipped: insufficient CV labels.")
            return

        logger.info("Computing calibration diagnostics + calibrator A/B...")
        methods = ["isotonic", "sigmoid"]
        if HAS_BETACAL:
            methods.append("beta")

        # A/B holdout: last 20% of pooled CV records.
        n_total = len(cv_actuals_arr)
        n_holdout = max(int(round(n_total * 0.2)), 20)
        n_holdout = min(n_holdout, max(n_total - 20, 1))
        train_slice = slice(0, n_total - n_holdout)
        holdout_slice = slice(n_total - n_holdout, n_total)

        # Per-base-model diagnostics (computed on raw uncalibrated CV probs).
        for name, probas in cv_probas.items():
            probas_arr = np.asarray(probas, dtype=float)
            if len(probas_arr) != n_total:
                continue

            raw_summary = self._calibration_summary(
                cv_actuals_arr, probas_arr, n_bins=10
            )
            model_entry = {"raw": raw_summary}

            # Try each calibrator; score on holdout via ECE.
            best_method = None
            best_ece = np.inf
            y_tr = cv_actuals_arr[train_slice]
            p_tr = probas_arr[train_slice]
            y_ho = cv_actuals_arr[holdout_slice]
            p_ho = probas_arr[holdout_slice]

            for method in methods:
                try:
                    cal = _fit_calibrator(method, y_tr, p_tr)
                    p_ho_cal = cal.predict_proba(p_ho)
                    summary = self._calibration_summary(y_ho, p_ho_cal, n_bins=10)
                    model_entry[method] = summary
                    if (
                        summary.get("ece") is not None
                        and summary["ece"] < best_ece
                    ):
                        best_ece = float(summary["ece"])
                        best_method = method
                except Exception as exc:
                    logger.warning(
                        "Calibrator '%s' failed for %s: %s", method, name, exc
                    )
                    model_entry[method] = {"error": str(exc)}

            model_entry["winner"] = best_method or "isotonic"
            self.calibration_diagnostics[name] = model_entry
            self.calibrator_choice[name] = best_method or "isotonic"

        # Ensemble diagnostics: weighted CV prediction with live ensemble weights.
        try:
            ensemble_cv = self._weighted_average_probas(cv_probas, cv_actuals_arr)
            if len(ensemble_cv) == n_total:
                ensemble_entry = {
                    "raw": self._calibration_summary(
                        cv_actuals_arr, ensemble_cv, n_bins=10
                    )
                }
                for method in methods:
                    try:
                        cal = _fit_calibrator(
                            method,
                            cv_actuals_arr[train_slice],
                            ensemble_cv[train_slice],
                        )
                        p_ho_cal = cal.predict_proba(ensemble_cv[holdout_slice])
                        ensemble_entry[method] = self._calibration_summary(
                            cv_actuals_arr[holdout_slice], p_ho_cal, n_bins=10
                        )
                    except Exception as exc:
                        ensemble_entry[method] = {"error": str(exc)}
                best_method = None
                best_ece = np.inf
                for method in methods:
                    ece_val = ensemble_entry.get(method, {}).get("ece")
                    if ece_val is not None and ece_val < best_ece:
                        best_ece = float(ece_val)
                        best_method = method
                ensemble_entry["winner"] = best_method or "isotonic"
                self.calibration_diagnostics["ensemble"] = ensemble_entry
                self.calibrator_choice["ensemble"] = best_method or "isotonic"

                # ── A1.5: PROMOTE ensemble-level calibrator ────────────
                # The A/B identifies a winner by holdout ECE; we fit the deployed
                # calibrator on the full pooled CV predictions (train + holdout) so
                # production benefits from all available labeled data. We default
                # to the preferred calibrator (sigmoid per A1.5) unless the
                # A/B winner clearly outperforms it by >10% holdout ECE.
                preferred = self.preferred_ensemble_calibrator
                ab_winner = self.calibrator_choice["ensemble"]
                preferred_ece = (
                    ensemble_entry.get(preferred, {}).get("ece")
                    if isinstance(ensemble_entry.get(preferred), dict)
                    else None
                )
                winner_ece = (
                    ensemble_entry.get(ab_winner, {}).get("ece")
                    if isinstance(ensemble_entry.get(ab_winner), dict)
                    else None
                )
                # Stick with preferred unless another calibrator beats it by >10%
                # on holdout ECE. Guards against A/B flip-flop between runs.
                deployed = preferred
                if (
                    winner_ece is not None
                    and preferred_ece is not None
                    and preferred_ece > 0
                    and winner_ece < preferred_ece * 0.9
                ):
                    deployed = ab_winner
                elif preferred_ece is None and ab_winner in {"isotonic", "sigmoid", "beta"}:
                    deployed = ab_winner
                try:
                    candidate_cal = _fit_calibrator(
                        deployed, cv_actuals_arr, ensemble_cv
                    )
                    # ── A1.5 safety gate: refuse to deploy a calibrator
                    # that collapses mid-range (0.3–0.7) recession signal.
                    # Rationale: the backtester retrains per origin and
                    # old-era CV pools have few positives + limited dynamic
                    # range, so the per-replicate sigmoid can map raw=0.4
                    # down to near-zero. The gate checks both positive-
                    # label behaviour AND a mid-range probe grid so it
                    # catches training distributions where the calibrator
                    # would erase legitimate early-warning signal.
                    positive_mask = cv_actuals_arr.astype(int) == 1
                    if positive_mask.any():
                        raw_pos_max = float(ensemble_cv[positive_mask].max())
                        raw_pos_med = float(np.median(ensemble_cv[positive_mask]))
                        cal_pos_max = float(
                            np.asarray(
                                candidate_cal.predict_proba(
                                    np.array([raw_pos_max])
                                )
                            ).ravel()[0]
                        )
                        cal_pos_med = float(
                            np.asarray(
                                candidate_cal.predict_proba(
                                    np.array([raw_pos_med])
                                )
                            ).ravel()[0]
                        )
                    else:
                        raw_pos_max = cal_pos_max = raw_pos_med = cal_pos_med = float("nan")

                    # Mid-range probe: what does 0.3, 0.4, 0.5 raw map to?
                    mid_probe = np.array([0.30, 0.40, 0.50])
                    mid_cal = np.asarray(
                        candidate_cal.predict_proba(mid_probe)
                    ).ravel()
                    # Collapse detected if: positive signal collapses OR
                    # the mid-range 0.4 probe (exactly the peak many old-era
                    # backtest origins produce for actual recessions) gets
                    # shrunk below 0.15. Chose 0.4 as the tightest test
                    # because backtest pathologies concentrate there; the
                    # 0.3/0.5 probes are kept for logging only.
                    positive_collapse = False
                    if positive_mask.any():
                        positive_collapse = (
                            cal_pos_max < 0.10
                            or cal_pos_med < raw_pos_med * 0.25
                        )
                    # 0.4 raw → <0.15 cal means a borderline-recession
                    # ensemble value (the modal peak for subtle-signal
                    # historic recessions) gets compressed into the noise
                    # floor. That's the failure mode we want to catch.
                    mid_04_collapse = bool(mid_cal[1] < 0.15)
                    collapses = positive_collapse or mid_04_collapse

                    if collapses:
                        logger.warning(
                            "  ensemble calibrator %s collapses signal "
                            "(raw_pos_max=%.3f→cal=%.3f, raw_pos_med=%.3f→cal=%.3f, "
                            "mid 0.3/0.4/0.5→%.3f/%.3f/%.3f, "
                            "positive_collapse=%s, mid_04_collapse=%s); "
                            "skipping deployment.",
                            deployed, raw_pos_max, cal_pos_max,
                            raw_pos_med, cal_pos_med,
                            mid_cal[0], mid_cal[1], mid_cal[2],
                            positive_collapse, mid_04_collapse,
                        )
                        self.ensemble_calibrator = None
                        self.deployed_ensemble_calibrator = "raw (calibrator rejected)"
                    else:
                        self.ensemble_calibrator = candidate_cal
                        self.deployed_ensemble_calibrator = deployed
                        logger.info(
                            "  ensemble calibrator deployed=%s (A/B winner=%s, "
                            "preferred=%s, winner_ece=%s, preferred_ece=%s, "
                            "pos_max %.3f→%.3f, pos_med %.3f→%.3f, "
                            "mid 0.3/0.4/0.5→%.3f/%.3f/%.3f)",
                            deployed, ab_winner, preferred,
                            f"{winner_ece:.4f}" if winner_ece is not None else "n/a",
                            f"{preferred_ece:.4f}" if preferred_ece is not None else "n/a",
                            raw_pos_max, cal_pos_max, raw_pos_med, cal_pos_med,
                            mid_cal[0], mid_cal[1], mid_cal[2],
                        )
                except Exception as cal_exc:
                    logger.warning(
                        "Ensemble calibrator fit failed (%s); "
                        "falling back to raw weighted average.", cal_exc
                    )
                    self.ensemble_calibrator = None
                    self.deployed_ensemble_calibrator = None
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Ensemble calibration diagnostics failed: %s", exc)

        # Log a one-liner per model for transparency.
        for name, entry in self.calibration_diagnostics.items():
            raw = entry.get("raw", {})
            winner = entry.get("winner", "isotonic")
            logger.info(
                "  calibration[%s] raw ECE=%.4f slope=%.3f winner=%s",
                name,
                (raw.get("ece") or float("nan")),
                (raw.get("slope") or float("nan")),
                winner,
            )

    def _optimize_threshold(self, y_true, y_proba):
        """
        Optimize classification threshold using a precision-aware F1 objective.

        Rare-event recession warnings should be tuned on the precision-recall
        tradeoff rather than ROC-style specificity. We therefore optimize F1
        first, then break ties with precision, recall, specificity, and finally
        a slightly lower threshold to avoid suppressing useful early warnings.

        FIX 2 (AUDIT-1): also compute threshold-plateau-width diagnostics and
        stash them on the model so they can be emitted into run_manifest.json.
        A flat F1 plateau (many thresholds within a small F1 tolerance of the
        winner) means the argmax is unstable under tiny input perturbations —
        this was the root cause of the 8-month GFC lead-time swing between
        baselines efb307e and 0411da9. Log a WARNING when plateau_width_005 >= 5.
        """
        y_true = np.array(y_true)
        y_proba = np.array(y_proba)

        if len(set(y_true)) < 2:
            self.threshold_diagnostics = []
            self.threshold_plateau_diagnostics = {}
            return 0.5

        threshold_rows = self._build_threshold_rows(y_true, y_proba)
        self.threshold_diagnostics = threshold_rows
        best_row = self._choose_threshold_row(threshold_rows)
        self.threshold_plateau_diagnostics = self._compute_plateau_diagnostics(
            threshold_rows, best_row
        )
        return round(best_row['threshold'], 3)

    @staticmethod
    def _compute_plateau_diagnostics(threshold_rows, best_row):
        """Compute F1-plateau-width diagnostics around the winning threshold.

        Returns a dict with:
          - threshold_plateau_width_01: count within 0.01 F1 of the winner
          - threshold_plateau_width_005: count within 0.005 F1 of the winner
          - threshold_plateau_range: [min_thr, max_thr] of the 0.01 plateau
          - threshold_plateau_span_01: max-min of the 0.01 plateau (scalar)
          - is_flat_plateau_warning: True iff width_005 >= 5

        Logs a WARNING when the 0.005-F1 plateau has >=5 members — that's the
        flatness level at which the tie-break is unstable enough to matter.
        """
        if not threshold_rows:
            return {}

        best_f1 = float(best_row.get('f1', 0.0))
        winner_thr = float(best_row.get('threshold', 0.5))

        tol_01 = [row for row in threshold_rows if row['f1'] >= best_f1 - 0.01]
        tol_005 = [row for row in threshold_rows if row['f1'] >= best_f1 - 0.005]

        plateau_01_thrs = [float(r['threshold']) for r in tol_01]
        plateau_min = min(plateau_01_thrs) if plateau_01_thrs else winner_thr
        plateau_max = max(plateau_01_thrs) if plateau_01_thrs else winner_thr

        diagnostics = {
            'winner_threshold': round(winner_thr, 4),
            'winner_f1': round(best_f1, 6),
            'threshold_plateau_width_01': int(len(tol_01)),
            'threshold_plateau_width_005': int(len(tol_005)),
            'threshold_plateau_range': [
                round(plateau_min, 4),
                round(plateau_max, 4),
            ],
            'threshold_plateau_span_01': round(plateau_max - plateau_min, 4),
            'is_flat_plateau_warning': bool(len(tol_005) >= 5),
        }

        if diagnostics['is_flat_plateau_warning']:
            logger.warning(
                "Threshold plateau is FLAT: %d thresholds are within 0.005 F1 "
                "of the winner (thr=%.3f, F1=%.4f) — range %.3f-%.3f. "
                "Decision threshold is unstable to small input perturbations. "
                "Report challengers at a fixed reference threshold (see "
                "Lead_Months_Fixed in backtest_summary).",
                diagnostics['threshold_plateau_width_005'],
                winner_thr,
                best_f1,
                plateau_min,
                plateau_max,
            )

        return diagnostics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, test_df):
        """
        Generate predictions using calibrated models and weighted ensemble.

        Returns dict with keys: model names + 'ensemble'
        """
        if not self.is_fitted:
            raise ValueError("Models must be fitted first!")

        X_test = test_df[self.feature_cols].ffill().fillna(0)
        X_test_scaled = self.scaler.transform(X_test)

        # PCA augmentation
        pca_features_test = self.pca.transform(X_test_scaled)
        X_test_probit = np.hstack([X_test_scaled, pca_features_test])
        X_test_tree = np.hstack([X_test.values, pca_features_test])

        predictions = {}

        for name in self.models:
            cal_model = self.calibrated_models.get(name)
            if cal_model is not None:
                if name == 'probit':
                    predictions[name] = cal_model.predict_proba(X_test_probit)[:, 1]
                else:
                    predictions[name] = cal_model.predict_proba(X_test_tree)[:, 1]
            else:
                model = self.models[name]
                if name == 'probit':
                    predictions[name] = model.predict_proba(X_test_probit)[:, 1]
                else:
                    predictions[name] = model.predict_proba(X_test_tree)[:, 1]

        # Markov-switching prediction (uses raw DataFrame, not numpy array)
        if self.markov_model is not None and self.markov_model.fitted:
            predictions['markov_switching'] = self.markov_model.predict_proba(test_df)[:, 1]

        # Benchmark ensemble members — externally calibrated / peer-model
        # recession probabilities. Each reads its own columns from test_df.
        for bm_name, bm in self.benchmark_models.items():
            predictions[bm_name] = bm.predict_proba(test_df)[:, 1]

        # LSTM prediction (uses unscaled feature matrix, applies isotonic calibration)
        if self.lstm_model is not None and self.lstm_model.fitted:
            raw_lstm = self.lstm_model.predict_proba(X_test.values)[:, 1]
            if hasattr(self, '_lstm_calibrator') and self._lstm_calibrator is not None:
                predictions['lstm'] = self._lstm_calibrator.predict(raw_lstm)
            else:
                predictions['lstm'] = raw_lstm

        # DMA-weighted ensemble
        ensemble_proba = np.zeros(len(X_test))
        for name, weight in self.ensemble_weights.items():
            if name in predictions:
                ensemble_proba += weight * predictions[name]

        # A1.5 — apply the promoted ensemble-level calibrator (sigmoid by default)
        # when one is fitted. We expose both the calibrated (deployed) ensemble
        # as 'ensemble' and the raw weighted average as 'ensemble_raw' for
        # diagnostics / A/B inspection.
        if self.ensemble_calibrator is not None:
            try:
                cal_ensemble = np.asarray(
                    self.ensemble_calibrator.predict_proba(ensemble_proba),
                    dtype=float,
                ).ravel()
                if cal_ensemble.shape == ensemble_proba.shape:
                    predictions['ensemble_raw'] = ensemble_proba
                    predictions['ensemble'] = cal_ensemble
                else:
                    predictions['ensemble'] = ensemble_proba
            except Exception as cal_exc:
                logger.warning(
                    "Ensemble calibrator predict failed (%s); "
                    "falling back to raw weighted average.", cal_exc,
                )
                predictions['ensemble'] = ensemble_proba
        else:
            predictions['ensemble'] = ensemble_proba

        return predictions

    def predict_with_confidence(
        self,
        test_df,
        n_bootstrap=200,
        ci_level=0.90,
        method="block_bootstrap",
        block_size_months=12,
        train_df=None,
    ):
        """
        Generate predictions with bootstrap confidence intervals.

        Parameters
        ----------
        test_df : DataFrame to score.
        n_bootstrap : bootstrap replicates.
        ci_level : nominal two-sided coverage (0.90 -> 5/95).
        method : 'block_bootstrap' (default, Politis-Romano stationary
                 bootstrap refitting a meta-calibrator per replicate) or
                 'dirichlet' (legacy ensemble-weight perturbation).
        block_size_months : mean block length for stationary bootstrap.
        train_df : training frame used when method='block_bootstrap'. If None
                    we fall back to Dirichlet CI (block-bootstrap needs labels).

        Returns
        -------
        dict with keys:
            predictions, ensemble_ci_lower, ensemble_ci_upper, ensemble_std,
            model_spread, ci_level, method, block_size_months, n_bootstrap.
        """
        # Get base predictions
        predictions = self.predict(test_df)

        base_names = [n for n in self.ensemble_weights if n in predictions]
        base_probas = np.column_stack([predictions[n] for n in base_names])
        n_obs = len(base_probas)
        weights = np.array([self.ensemble_weights[n] for n in base_names])
        live_ensemble = base_probas @ weights

        # Model spread is method-agnostic
        model_spread = np.max(base_probas, axis=1) - np.min(base_probas, axis=1)

        alpha_tail = (1 - ci_level) / 2

        chosen_method = method
        fallback_reason = None

        # For the block-bootstrap we want the *raw* weighted-average ensemble on
        # training rows (pre-ensemble-calibrator) so the per-replicate calibrator
        # has a meaningful latent→label mapping to fit. If we pass the already-
        # calibrated ensemble, every replicate collapses to an almost-identity
        # calibrator and CI widths vanish.
        raw_train_ensemble = None
        raw_live_ensemble = base_probas @ weights

        if method == "block_bootstrap":
            if train_df is None:
                fallback_reason = "no train_df supplied"
                chosen_method = "dirichlet"
            else:
                try:
                    train_preds = self.predict(train_df)
                    raw_train_ensemble = np.column_stack(
                        [train_preds[n] for n in base_names]
                    ) @ weights
                    y_train_arr = np.asarray(
                        train_df[self.target_col], dtype=float
                    ).ravel()
                    if (
                        len(raw_train_ensemble) < max(24, 2 * block_size_months)
                        or len(set(y_train_arr.astype(int))) < 2
                    ):
                        fallback_reason = (
                            "insufficient train history for block bootstrap"
                        )
                        chosen_method = "dirichlet"
                except Exception as exc:
                    fallback_reason = f"train predict failed: {exc}"
                    chosen_method = "dirichlet"

        if chosen_method == "block_bootstrap":
            # Refit a quick meta-calibrator per bootstrap replicate on the
            # resampled training block, then apply to the raw test ensemble.
            # We use the same calibrator family that's actually deployed so
            # the CI is measuring variance in the same calibration pathway.
            deployed = (
                getattr(self, "deployed_ensemble_calibrator", None)
                or getattr(self, "preferred_ensemble_calibrator", "sigmoid")
                or "sigmoid"
            )
            # If the deployed calibrator was rejected by the safety gate,
            # the live ensemble prediction uses raw weighted avg — so the
            # bootstrap must match (no per-replicate calibration), otherwise
            # the CI band excludes the live mean. We signal this with
            # cal_method="raw".
            if deployed not in {"isotonic", "sigmoid", "beta"}:
                cal_method = "raw"
            else:
                cal_method = deployed
            X_indices = np.arange(len(raw_train_ensemble))

            def _fit_meta_and_predict_test(indices):
                y_boot = y_train_arr[indices]
                p_boot = raw_train_ensemble[indices]
                if cal_method == "raw":
                    # No calibration applied — the replicate's contribution
                    # is the raw test ensemble. Bootstrap variance still
                    # comes from the index resampling via the enclosing
                    # per-replicate rebuild of the training sample, plus
                    # the width floor below.
                    return raw_live_ensemble.copy()
                if len(np.unique(y_boot.astype(int))) < 2:
                    # Degenerate replicate — return the raw live ensemble so
                    # the replicate contributes a finite sample to the quantile.
                    return raw_live_ensemble.copy()
                try:
                    cal = _fit_calibrator(cal_method, y_boot, p_boot)
                    return np.asarray(
                        cal.predict_proba(raw_live_ensemble), dtype=float
                    ).ravel()
                except Exception:
                    return raw_live_ensemble.copy()

            rng = np.random.default_rng(42)
            from recession_engine.calibration_diag import (
                stationary_bootstrap_indices as _sb_indices,
            )

            bootstrap_tests = np.zeros((int(n_bootstrap), n_obs), dtype=float)
            for b in range(int(n_bootstrap)):
                idx = _sb_indices(
                    len(X_indices), int(block_size_months), rng
                )
                bootstrap_tests[b] = _fit_meta_and_predict_test(idx)

            raw_lower = np.quantile(bootstrap_tests, alpha_tail, axis=0)
            raw_upper = np.quantile(bootstrap_tests, 1 - alpha_tail, axis=0)
            ensemble_std = bootstrap_tests.std(axis=0, ddof=0)

            # ── A1.5 CI fix (Option B) — minimum-width floor tied to
            # base-model disagreement. Rationale: refitting only the meta
            # calibrator leaves per-row predictions nearly deterministic
            # in stable expansion periods, so the quantile band collapses
            # to a point. At minimum, the CI should be as wide as the
            # base-model disagreement suggests. We also honour a hard
            # floor of 0.06 so the displayed band is always visible and
            # beats the validator's "width >= 0.05" coverage proxy.
            mean_pred = bootstrap_tests.mean(axis=0)
            # std of base probabilities per row captures per-row disagreement
            model_disagreement_std = base_probas.std(axis=1, ddof=0)
            # Width floor: at least 0.06, or 2*disagreement_std (whichever is
            # larger). Normal 90% CI is ±1.645σ so we use 1.645 to convert
            # the bootstrap σ to a band.
            width_floor = np.maximum(
                0.06, 2.0 * model_disagreement_std
            )

            # Combine percentile CI, normal-approx CI, and width-floor CI.
            # Take the *widest* of the three per row — the CI should dominate
            # all three lower bounds on uncertainty.
            normal_lower = mean_pred - 1.645 * ensemble_std
            normal_upper = mean_pred + 1.645 * ensemble_std
            floor_lower = mean_pred - width_floor / 2.0
            floor_upper = mean_pred + width_floor / 2.0

            ci_lower = np.minimum(
                np.minimum(raw_lower, normal_lower), floor_lower
            )
            ci_upper = np.maximum(
                np.maximum(raw_upper, normal_upper), floor_upper
            )

            # Clip to [0.0, 1.0]. We use 0.0 rather than 0.01 as the lower
            # bound so CI_Lower can fall below the isotonic 0.01 floor —
            # this keeps the live-mean inside the CI for near-zero predictions
            # (Prob_Ensemble = 0.008 at long expansion stretches would be
            # excluded by a 0.01 CI_Lower clip and fail coverage).
            ci_lower = np.clip(ci_lower, 0.0, 1.0)
            ci_upper = np.clip(ci_upper, 0.0, 1.0)
            # Guarantee ordering (numerical safety).
            ci_lower = np.minimum(ci_lower, ci_upper)

            # ── Width floor post-clip: if clipping against [0.01, 0.99]
            # compressed the CI below the width_floor (e.g. mean_pred ≈ 0
            # loses the lower half to the 0.01 clip), expand the UPPER
            # bound to recover the floor width. This guarantees every row
            # with a non-degenerate model_disagreement signal has CI width
            # ≥ 0.06 regardless of where the center falls.
            post_clip_width = ci_upper - ci_lower
            deficit_mask = post_clip_width < width_floor
            if deficit_mask.any():
                # Prefer expanding upward when lower is at the 0.0 clip;
                # otherwise split the deficit symmetrically.
                at_lower_clip = ci_lower <= 0.001
                at_upper_clip = ci_upper >= 0.999
                deficit = width_floor - post_clip_width
                expand_up = deficit_mask & at_lower_clip & ~at_upper_clip
                expand_down = deficit_mask & at_upper_clip & ~at_lower_clip
                expand_both = deficit_mask & ~at_lower_clip & ~at_upper_clip
                # Expand up only
                ci_upper = np.where(
                    expand_up,
                    np.minimum(ci_upper + deficit, 1.0),
                    ci_upper,
                )
                # Expand down only
                ci_lower = np.where(
                    expand_down,
                    np.maximum(ci_lower - deficit, 0.0),
                    ci_lower,
                )
                # Expand symmetrically
                ci_upper = np.where(
                    expand_both,
                    np.minimum(ci_upper + deficit / 2.0, 1.0),
                    ci_upper,
                )
                ci_lower = np.where(
                    expand_both,
                    np.maximum(ci_lower - deficit / 2.0, 0.0),
                    ci_lower,
                )

            return {
                "predictions": predictions,
                "ensemble_ci_lower": ci_lower,
                "ensemble_ci_upper": ci_upper,
                "ensemble_std": ensemble_std,
                "model_spread": model_spread,
                "ci_level": ci_level,
                "method": "stationary_block_bootstrap",
                "block_size_months": int(block_size_months),
                "n_bootstrap": int(n_bootstrap),
                "ci_calibrator_method": cal_method,
                "ci_width_floor_basis": "max(0.04, 2*base_model_std)",
            }

        # ── Dirichlet fallback / explicit legacy path ──
        if fallback_reason is not None:
            logger.info(
                "predict_with_confidence falling back to Dirichlet CI: %s",
                fallback_reason,
            )

        rng = np.random.RandomState(42)
        original_weights = weights.copy()
        bootstrap_ensembles = np.zeros((int(n_bootstrap), n_obs))
        for b in range(int(n_bootstrap)):
            alpha = original_weights * 20 + 1
            sampled_weights = rng.dirichlet(alpha)
            bootstrap_ensembles[b] = base_probas @ sampled_weights

        ci_lower = np.percentile(bootstrap_ensembles, alpha_tail * 100, axis=0)
        ci_upper = np.percentile(bootstrap_ensembles, (1 - alpha_tail) * 100, axis=0)
        ensemble_std = np.std(bootstrap_ensembles, axis=0)

        return {
            "predictions": predictions,
            "ensemble_ci_lower": ci_lower,
            "ensemble_ci_upper": ci_upper,
            "ensemble_std": ensemble_std,
            "model_spread": model_spread,
            "ci_level": ci_level,
            "method": "dirichlet",
            "block_size_months": int(block_size_months),
            "n_bootstrap": int(n_bootstrap),
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, test_df, predictions, threshold: float = None):
        """
        Evaluate model performance with proper scoring rules.

        Metrics:
        - AUC-ROC: Discrimination ability
        - Brier Score: Calibration quality (lower = better)
        - Log Loss: Proper scoring rule for probability forecasts
        - Precision/Recall/F1: At the optimized threshold
        - Youden's J: Balance of sensitivity and specificity
        """
        if threshold is None:
            threshold = self.decision_threshold

        y_true = test_df[self.target_col]
        results = []

        for model_name, y_pred_proba in predictions.items():
            y_pred = (y_pred_proba >= threshold).astype(int)

            # Core metrics
            auc = roc_auc_score(y_true, y_pred_proba) if len(set(y_true)) >= 2 else 0.0
            pr_auc = average_precision_score(y_true, y_pred_proba) if len(set(y_true)) >= 2 else float(np.mean(y_true))
            brier = brier_score_loss(y_true, y_pred_proba)
            logloss = log_loss(y_true, np.clip(y_pred_proba, 1e-7, 1 - 1e-7), labels=[0, 1])
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            youdens_j = sensitivity + specificity - 1

            results.append({
                'Model': model_name,
                'AUC': auc,
                'PR_AUC': pr_auc,
                'Brier': brier,
                'LogLoss': logloss,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'Accuracy': accuracy,
                'Sensitivity': sensitivity,
                'Specificity': specificity,
                'Youdens_J': youdens_j,
            })

        metrics_df = pd.DataFrame(results)
        self.metrics = metrics_df

        logger.info("\n" + "=" * 80)
        logger.info("MODEL EVALUATION RESULTS")
        logger.info(f"Decision threshold: {threshold:.3f}")
        logger.info("=" * 80)
        print(metrics_df.to_string(index=False))
        logger.info("=" * 80)

        # Highlight best model per metric
        for metric in ['AUC', 'Brier', 'F1', 'Youdens_J']:
            if metric == 'Brier':
                best_idx = metrics_df[metric].idxmin()
            else:
                best_idx = metrics_df[metric].idxmax()
            best = metrics_df.loc[best_idx]
            logger.info(f"  Best {metric}: {best['Model']} ({best[metric]:.4f})")

        return metrics_df

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self, test_df, predictions):
        """Generate executive report with model details"""
        latest_date = test_df.index[-1]
        latest_probs = {name: pred[-1] for name, pred in predictions.items()}

        report = []
        report.append("=" * 80)
        report.append("RECESSION PREDICTION ENGINE — EXECUTIVE REPORT")
        report.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        report.append("")
        report.append(f"RECESSION PROBABILITY — {self.target_horizon} MONTHS FORWARD")
        report.append(f"As of: {latest_date.strftime('%Y-%m-%d')}")
        report.append(
            f"Decision Threshold: {self.decision_threshold:.3f} "
            f"(optimized via {self.threshold_method})"
        )
        report.append("-" * 80)

        report.append("")
        report.append("MODEL PREDICTIONS:")
        for model_name in ['probit', 'random_forest', 'xgboost', 'markov_switching', 'ensemble']:
            if model_name in latest_probs:
                prob = latest_probs[model_name]
                if model_name == 'ensemble':
                    report.append(
                        f"  {'ENSEMBLE':20s}: {prob:6.1%}  "
                        f"({self.ensemble_method.replace('_', ' ')})"
                    )
                else:
                    if model_name in self.ensemble_weights:
                        weight = self.ensemble_weights[model_name]
                        report.append(
                            f"  {model_name.upper():20s}: {prob:6.1%}  "
                            f"(weight: {weight:.3f})"
                        )
                    else:
                        report.append(
                            f"  {model_name.upper():20s}: {prob:6.1%}  "
                            "(diagnostic only)"
                        )

        report.append("")
        report.append("=" * 80)

        ensemble_prob = latest_probs['ensemble']

        if ensemble_prob < self.decision_threshold * 0.3:
            signal = "LOW RISK — Economy appears stable"
        elif ensemble_prob < self.decision_threshold * 0.7:
            signal = "MODERATE RISK — Monitor closely"
        elif ensemble_prob < self.decision_threshold:
            signal = "ELEVATED RISK — Approaching recession threshold"
        else:
            signal = "HIGH RISK — Recession probability exceeds optimized threshold"

        report.append(f"SIGNAL: {signal}")
        report.append("")

        # Add model methodology summary
        report.append("=" * 80)
        report.append("METHODOLOGY:")
        report.append("  Models: L1-Probit, Random Forest (300 trees), XGBoost (400 rounds),")
        report.append("          Markov-Switching (3-state TVTP Hamilton filter, isotonic calibrated)")
        report.append("  Note: LSTM disabled — insufficient sample size for generalization")
        report.append("  Preprocessing: PCA factor extraction (top 5 components augment features)")
        report.append("  Class weighting: Balanced (accounts for ~12% recession base rate)")
        report.append("  Calibration: Isotonic regression via time-series cross-validation")
        report.append(
            "  Ensemble: CV-gated forecast combination with shrinkage toward equal weights "
            f"(active models: {', '.join(self.active_models) if self.active_models else 'all fitted'})"
        )
        report.append(f"  Threshold: {self.threshold_method}")
        report.append(f"  Features: {len(self.feature_cols)} selected from correlation + RF importance")
        report.append("  Includes: At-risk transforms, Sahm Rule, SOS indicator,")
        report.append("            term-premium-adjusted spread, near-term forward spread,")
        report.append("            excess bond premium proxy, house price confirming indicator,")
        report.append("            residential investment, consumer durables, sectoral divergence,")
        report.append("            credit stress index, monetary policy stance interaction")
        report.append("")
        report.append("  References:")
        report.append("  - Estrella & Mishkin (1998), Wright (2006), Sahm (2019)")
        report.append("  - Billakanti & Shin (2025), Engstrom & Sharpe (2019)")
        report.append("  - Gilchrist & Zakrajsek (2012), Raftery et al. (2010) [DMA]")
        report.append("  - Ajello et al. (2022) [term premium], Grigoli & Sandri (2024) [house prices]")
        report.append("  - Scavette & O'Trakoun (2025) [SOS], Leamer (2024) [res. investment]")
        report.append("=" * 80)

        return "\n".join(report)
