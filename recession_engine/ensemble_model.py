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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, brier_score_loss, log_loss, precision_recall_curve
)
import warnings
import logging
from datetime import datetime

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
    import torch
    import torch.nn as nn
    HAS_TORCH = True
    logger.info("PyTorch %s available (MPS: %s)", torch.__version__,
                torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)
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

    Architecture:
    - 2 LSTM layers (32, 16 units) with dropout
    - Lookback window of 12 months (captures medium-term trends)
    - Binary output with sigmoid activation
    - Class-weighted training for recession imbalance
    """

    def __init__(self, lookback=12, epochs=50, batch_size=32):
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = None
        self.fitted = False
        self._feature_cols = None
        self._device = None

    def _get_device(self):
        """Select best available device: MPS (Apple GPU) > CUDA > CPU."""
        if self._device is not None:
            return self._device
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self._device = torch.device('mps')
        elif torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')
        return self._device

    def _build_model(self, n_features):
        """Build 2-layer LSTM in PyTorch."""
        class _LSTMNet(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.lstm1 = nn.LSTM(input_dim, 32, batch_first=True, dropout=0.2)
                self.lstm2 = nn.LSTM(32, 16, batch_first=True)
                self.fc1 = nn.Linear(16, 16)
                self.dropout = nn.Dropout(0.3)
                self.fc2 = nn.Linear(16, 1)

            def forward(self, x):
                out, _ = self.lstm1(x)
                out, _ = self.lstm2(out)
                out = out[:, -1, :]          # last time step
                out = torch.relu(self.fc1(out))
                out = self.dropout(out)
                out = torch.sigmoid(self.fc2(out))
                return out.squeeze(-1)

        return _LSTMNet(n_features).to(self._get_device())

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
        """Fit LSTM on scaled feature matrix."""
        if not HAS_TORCH:
            logger.warning("LSTM: PyTorch not available, skipping")
            self.fitted = False
            return self

        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        X_seq, y_seq = self._create_sequences(X_scaled, y_train.values)

        if len(X_seq) < 60:
            logger.warning("LSTM: too few sequences (%d), skipping", len(X_seq))
            self.fitted = False
            return self

        # Class weights for imbalanced data
        n_pos = y_seq.sum()
        n_neg = len(y_seq) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32,
                                   device=self._get_device())

        self.model = self._build_model(X_scaled.shape[1])

        # Convert to tensors
        device = self._get_device()
        X_tensor = torch.tensor(X_seq, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y_seq, dtype=torch.float32, device=device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        # Per-sample weights: recession samples get higher weight
        sample_weights = torch.where(y_tensor == 1, pos_weight.item(), 1.0)

        # Training loop with early stopping
        best_loss = float('inf')
        patience_counter = 0
        best_state = None

        self.model.train()
        for epoch in range(self.epochs):
            # Mini-batch training
            indices = torch.randperm(len(X_tensor), device=device)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(X_tensor), self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                X_batch = X_tensor[batch_idx]
                y_batch = y_tensor[batch_idx]
                w_batch = sample_weights[batch_idx]

                optimizer.zero_grad()
                preds = self.model(X_batch)
                # Weighted BCE loss
                bce = -(y_batch * torch.log(preds + 1e-7) +
                        (1 - y_batch) * torch.log(1 - preds + 1e-7))
                loss = (bce * w_batch).mean()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            # Early stopping
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.fitted = True
        logger.info("LSTM fitted: %d sequences, %d features, lookback=%d, device=%s",
                     len(X_seq), X_scaled.shape[1], self.lookback, device)
        return self

    def predict_proba(self, X_test):
        """Return (n_samples, 2) array of [P(expansion), P(recession)]."""
        n = len(X_test)

        if not self.fitted or self.model is None:
            return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])

        X_scaled = self.scaler.transform(X_test)
        X_seq = self._create_sequences(X_scaled)

        if len(X_seq) == 0:
            return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])

        # Predict
        device = self._get_device()
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_seq, dtype=torch.float32, device=device)
            preds = self.model(X_tensor).cpu().numpy()
        preds = np.clip(preds, 0.01, 0.99)

        # Pad the first `lookback` predictions with 0.5 (no sequence available)
        n_pad = n - len(preds)
        if n_pad > 0:
            preds = np.concatenate([np.full(n_pad, 0.5), preds])

        return np.column_stack([1.0 - preds, preds])


class RecessionEnsembleModel:
    """
    Ensemble recession prediction model.

    Architecture:
    - Base models: L1-Probit, Random Forest, XGBoost, LSTM, Markov-Switching (TVTP)
    - Preprocessing: PCA factor extraction (augments feature set)
    - Calibration: Isotonic regression on supervised models; built-in isotonic on Markov
    - Ensemble: Dynamic Model Averaging weights (AUC × inverse Brier, exponential forgetting)
    - Threshold: Optimized via Youden's J on validation set
    """

    def __init__(self, target_horizon=6, n_cv_splits=5):
        self.target_horizon = target_horizon
        self.target_col = f'RECESSION_FORWARD_{target_horizon}M'
        self.n_cv_splits = n_cv_splits

        # These get set during fit
        self.decision_threshold = 0.5  # Updated by Youden's J optimization
        self.ensemble_weights = {}     # DMA weights (primary)
        self.static_weights = {}       # Static BMA weights (fallback)
        self.calibrated_models = {}    # Post-hoc calibrated classifiers
        self.feature_cols = []
        self.feature_importance = {}
        self.metrics = {}
        self.cv_results = {}
        self.is_fitted = False

        # PCA for factor extraction
        self.pca = None
        self.n_pca_components = 5

        # Markov-switching model (needs DataFrame, not numpy array)
        self.markov_model = MarkovSwitchingWrapper() if HAS_MARKOV else None

        # LSTM model (needs sequences, handles its own scaling)
        self.lstm_model = LSTMRecessionModel(lookback=12, epochs=50) if HAS_TORCH else None

        # Base models with class_weight='balanced' for rare-event handling
        self.models = {
            'probit': LogisticRegression(
                penalty='l1',
                solver='saga',
                C=0.1,
                max_iter=2000,
                random_state=42,
                class_weight='balanced',
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced',
            ),
        }
        if HAS_XGBOOST:
            # scale_pos_weight set dynamically in fit() based on actual class ratio
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.7,
                min_child_weight=10,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss',
            )
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

    # ------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------

    def select_features(self, df, max_features=60):
        """
        Select most predictive features using correlation + model-based importance.

        Two-pass approach:
        1. Filter by correlation with target (top 2x candidates)
        2. Rank by Random Forest importance on the candidates
        """
        exclude_cols = [self.target_col, 'RECESSION']
        feature_cols = [col for col in df.columns
                        if col not in exclude_cols and not col.startswith('ref_')]

        # Require 70% non-null
        valid_cols = [col for col in feature_cols
                      if df[col].notna().sum() / len(df) > 0.7]

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

        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        top_corr_feats = [feat for feat, _ in sorted_features[:max_features * 2]]

        # Pass 2: Random Forest importance
        df_sub = df[top_corr_feats + [self.target_col]].dropna()
        if len(df_sub) > 500:
            df_sub = df_sub.iloc[-500:]

        if len(df_sub) > 0:
            X_sub = df_sub[top_corr_feats]
            y_sub = df_sub[self.target_col]
            rf = RandomForestClassifier(
                n_estimators=100, max_depth=6, min_samples_leaf=5,
                random_state=42, n_jobs=-1, class_weight='balanced'
            )
            rf.fit(X_sub, y_sub)
            imp_series = pd.Series(rf.feature_importances_, index=top_corr_feats)
            imp_series = imp_series.sort_values(ascending=False)
            selected = imp_series.index.tolist()[:max_features]
        else:
            selected = top_corr_feats[:max_features]

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
        ]
        for col in must_include:
            if col in valid_cols and col not in selected:
                selected.append(col)

        logger.info(f"Selected {len(selected)} features")
        return selected

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, train_df, feature_cols=None, max_features=60):
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
                model.fit(X_train_probit, y_train)
                importance = np.abs(model.coef_[0][:len(feature_cols)])
            else:
                model.fit(X_train_tree, y_train)
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
                from sklearn.base import clone
                fold_model = clone(model)
                if name == 'probit':
                    fold_model.fit(X_tr_probit, y_tr)
                    proba = fold_model.predict_proba(X_val_probit)[:, 1]
                else:
                    fold_model.fit(X_tr_tree, y_tr)
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

            # LSTM CV
            if 'lstm' in all_model_names:
                fold_lstm = LSTMRecessionModel(lookback=12, epochs=30)
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
                cv_brier = brier_score_loss(cv_actuals, probas)
                cv_scores[name] = {
                    'auc': cv_auc,
                    'brier': cv_brier,
                    'inv_brier': 1.0 / (cv_brier + 1e-6)
                }
                logger.info(f"  {name} CV — AUC: {cv_auc:.4f}, Brier: {cv_brier:.4f}")
            else:
                cv_scores[name] = {'auc': 0.5, 'brier': 0.25, 'inv_brier': 4.0}

        self.cv_results = cv_scores

        total_inv_brier = sum(s['inv_brier'] for s in cv_scores.values())
        self.static_weights = {
            name: cv_scores[name]['inv_brier'] / total_inv_brier
            for name in all_model_names
        }
        logger.info(f"Static weights: {', '.join(f'{n}={w:.3f}' for n, w in self.static_weights.items())}")

        # 4b: DMA weights (exponential forgetting of per-fold composite scores)
        dma_weights = self._compute_dma_weights(
            cv_fold_briers, cv_fold_aucs, forgetting_factor=0.99
        )
        self.ensemble_weights = dma_weights
        logger.info(f"DMA weights: {', '.join(f'{n}={w:.3f}' for n, w in self.ensemble_weights.items())}")

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
                    cal.fit(X_train_probit, y_train)
                else:
                    cal.fit(X_train_tree, y_train)
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

        # LSTM: no CalibratedClassifierCV (not an sklearn estimator)
        if 'lstm' in all_model_names:
            self.calibrated_models['lstm'] = None
            logger.info("  ⊘ lstm: sigmoid output (no additional calibration)")

        # ── Step 6: Optimize decision threshold (Youden's J) ────────
        logger.info("Optimizing decision threshold (Youden's J)...")
        cv_ensemble_proba = self._weighted_average_probas(cv_probas, cv_actuals)
        self.decision_threshold = self._optimize_threshold(cv_actuals, cv_ensemble_proba)
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

    def _weighted_average_probas(self, probas_dict, actuals):
        """Compute weighted average of model probabilities."""
        result = np.zeros(len(actuals))
        for name, weight in self.ensemble_weights.items():
            p = np.array(probas_dict[name])
            if len(p) == len(actuals):
                result += weight * p
        return result

    def _optimize_threshold(self, y_true, y_proba):
        """
        Optimize classification threshold using Youden's J statistic.

        J = Sensitivity + Specificity - 1

        This balances the cost of missing a recession (false negative)
        against false alarms (false positive).
        """
        y_true = np.array(y_true)
        y_proba = np.array(y_proba)

        if len(set(y_true)) < 2:
            return 0.5

        best_j = -1
        best_threshold = 0.5

        # Search range bounded to [0.10, 0.40] — prevents extreme thresholds
        # that cause backtest instability. Recession base rate is ~12%, so
        # thresholds above 0.40 are too conservative (miss mild recessions).
        for threshold in np.arange(0.10, 0.41, 0.01):
            y_pred = (y_proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            j = sensitivity + specificity - 1

            if j > best_j:
                best_j = j
                best_threshold = threshold

        return round(best_threshold, 3)

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

        # LSTM prediction (uses unscaled feature matrix)
        if self.lstm_model is not None and self.lstm_model.fitted:
            predictions['lstm'] = self.lstm_model.predict_proba(X_test.values)[:, 1]

        # DMA-weighted ensemble
        ensemble_proba = np.zeros(len(X_test))
        for name, weight in self.ensemble_weights.items():
            if name in predictions:
                ensemble_proba += weight * predictions[name]
        predictions['ensemble'] = ensemble_proba

        return predictions

    def predict_with_confidence(self, test_df, n_bootstrap=200, ci_level=0.90):
        """
        Generate predictions with bootstrap confidence intervals.

        Bootstraps the ensemble weight vector to produce a distribution of
        ensemble probabilities at each time step. This captures uncertainty
        from model combination (which model to trust) without the cost of
        refitting base models.

        Also includes model-spread-based uncertainty (disagreement among base
        models as a natural measure of epistemic uncertainty).

        Returns:
            dict with keys:
                'predictions': standard predictions dict
                'ensemble_ci_lower': lower bound of CI
                'ensemble_ci_upper': upper bound of CI
                'ensemble_std': standard deviation across bootstrap samples
                'model_spread': max - min across base model predictions
        """
        # Get base predictions
        predictions = self.predict(test_df)

        base_names = [n for n in self.ensemble_weights if n in predictions]
        base_probas = np.column_stack([predictions[n] for n in base_names])
        n_obs = len(base_probas)

        # Method 1: Bootstrap ensemble weights
        rng = np.random.RandomState(42)
        original_weights = np.array([self.ensemble_weights[n] for n in base_names])

        bootstrap_ensembles = np.zeros((n_bootstrap, n_obs))
        for b in range(n_bootstrap):
            # Dirichlet perturbation of weights (concentrated around original)
            # Higher alpha = tighter around original weights
            alpha = original_weights * 20 + 1  # concentration parameter
            sampled_weights = rng.dirichlet(alpha)
            bootstrap_ensembles[b] = base_probas @ sampled_weights

        # Compute CI bounds
        alpha_tail = (1 - ci_level) / 2
        ci_lower = np.percentile(bootstrap_ensembles, alpha_tail * 100, axis=0)
        ci_upper = np.percentile(bootstrap_ensembles, (1 - alpha_tail) * 100, axis=0)
        ensemble_std = np.std(bootstrap_ensembles, axis=0)

        # Method 2: Model spread (simpler epistemic uncertainty)
        model_spread = np.max(base_probas, axis=1) - np.min(base_probas, axis=1)

        return {
            'predictions': predictions,
            'ensemble_ci_lower': ci_lower,
            'ensemble_ci_upper': ci_upper,
            'ensemble_std': ensemble_std,
            'model_spread': model_spread,
            'ci_level': ci_level,
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
            brier = brier_score_loss(y_true, y_pred_proba)
            logloss = log_loss(y_true, np.clip(y_pred_proba, 1e-7, 1 - 1e-7))
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
        report.append(f"Decision Threshold: {self.decision_threshold:.3f} (optimized via Youden's J)")
        report.append("-" * 80)

        report.append("")
        report.append("MODEL PREDICTIONS:")
        for model_name in ['probit', 'random_forest', 'xgboost', 'lstm', 'markov_switching', 'ensemble']:
            if model_name in latest_probs:
                prob = latest_probs[model_name]
                weight = self.ensemble_weights.get(model_name, 'N/A')
                if model_name == 'ensemble':
                    report.append(f"  {'ENSEMBLE':20s}: {prob:6.1%}  (weighted combination)")
                else:
                    report.append(f"  {model_name.upper():20s}: {prob:6.1%}  (weight: {weight:.3f})")

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
        report.append("          LSTM (2-layer, 12-month lookback),")
        report.append("          Markov-Switching (3-state TVTP Hamilton filter, isotonic calibrated)")
        report.append("  Preprocessing: PCA factor extraction (top 5 components augment features)")
        report.append("  Class weighting: Balanced (accounts for ~12% recession base rate)")
        report.append("  Calibration: Isotonic regression via time-series cross-validation")
        report.append("  Ensemble: Dynamic Model Averaging (exponential forgetting, lambda=0.99)")
        report.append("  Threshold: Youden's J statistic (maximizes sensitivity + specificity)")
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
