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


class MarkovSwitchingWrapper:
    """
    Sklearn-like wrapper for a 2-regime Markov-switching model.

    Instead of predicting from a high-dimensional feature matrix, this model
    builds a composite signal from a few key macro indicators and infers
    latent expansion/recession regimes via Hamilton filtering.

    During fit(): fits the MS model on the training-period composite and
    stores the training composite values.

    During predict_proba(): concatenates the stored training composite with
    a new composite built from the test features, re-runs the Hamilton filter
    on the extended series, and returns filtered regime probabilities for the
    test period only.
    """

    # Indicators and their sign direction: +1 = higher means stronger economy,
    # -1 = higher means weaker economy (recession signal)
    KEY_INDICATORS = {
        'leading_T10Y3M': +1,      # Positive spread = expansion
        'SAHM_INDICATOR': -1,      # Higher = closer to recession trigger
        'financial_NFCI': -1,      # Higher = tighter financial conditions (stress)
    }

    def __init__(self):
        self.result = None
        self.recession_regime = None
        self.training_composite = None
        self._train_means = {}
        self._train_stds = {}
        self.fitted = False

    def fit(self, X_df, y):
        """Fit on composite of key indicators (X_df must be a DataFrame)."""
        available = {c: s for c, s in self.KEY_INDICATORS.items() if c in X_df.columns}
        if len(available) == 0:
            logger.warning("MarkovSwitching: no key indicators found, skipping fit")
            self.fitted = False
            return self

        # Standardize each indicator, flip sign so higher = stronger economy,
        # then average into a composite. This ensures the "recession" regime
        # always has the lower mean.
        parts = []
        for col, sign in available.items():
            s = X_df[col]
            mu, sigma = s.mean(), s.std() + 1e-8
            self._train_means[col] = mu
            self._train_stds[col] = sigma
            parts.append(sign * (s - mu) / sigma)
        composite = pd.concat(parts, axis=1).mean(axis=1).dropna()

        if len(composite) < 60:
            logger.warning("MarkovSwitching: too few observations (%d), skipping", len(composite))
            self.fitted = False
            return self

        try:
            mod = MarkovRegression(
                composite.values,
                k_regimes=2,
                trend='c',
                switching_variance=True,
            )
            self.result = mod.fit(maxiter=300, disp=False)

            # Identify the recession regime (lower composite mean = weaker economy)
            # params is a numpy array; use param_names to find const indices
            param_names = self.result.model.param_names
            means = []
            for i in range(2):
                idx = param_names.index(f'const[{i}]')
                means.append(float(self.result.params[idx]))
            self.recession_regime = int(np.argmin(means))

            # Store training composite for later extension
            self.training_composite = composite.values.copy()
            self.fitted = True
            logger.info("MarkovSwitching fitted: recession_regime=%d, means=%s",
                        self.recession_regime, [f"{m:.3f}" for m in means])

        except Exception as e:
            logger.warning("MarkovSwitching fit failed: %s", e)
            self.result = None
            self.fitted = False

        return self

    def predict_proba(self, X_df):
        """
        Return (n_samples, 2) array of [P(expansion), P(recession)].

        Extends the training composite with test-period data and runs the
        Hamilton filter on the full series, returning only the test-period
        filtered probabilities.
        """
        n = len(X_df)

        if not self.fitted or self.result is None:
            return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])

        available = {c: s for c, s in self.KEY_INDICATORS.items() if c in X_df.columns}
        if len(available) == 0:
            return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])

        # Build test composite using TRAINING means/stds (same scale as fit)
        parts = []
        for col, sign in available.items():
            mu = self._train_means.get(col, X_df[col].mean())
            sigma = self._train_stds.get(col, X_df[col].std() + 1e-8)
            parts.append(sign * (X_df[col] - mu) / sigma)
        test_composite = pd.concat(parts, axis=1).mean(axis=1).fillna(0).values

        # Concatenate training + test composites
        full_composite = np.concatenate([self.training_composite, test_composite])

        try:
            # Re-fit on the extended series to get filtered probabilities
            mod = MarkovRegression(
                full_composite,
                k_regimes=2,
                trend='c',
                switching_variance=True,
            )
            res = mod.fit(maxiter=300, disp=False)

            # Extract filtered probabilities for the test period
            # filtered_marginal_probabilities shape: (n_obs, n_regimes)
            filtered = res.filtered_marginal_probabilities
            test_probs = filtered[-n:, self.recession_regime]

            # Clamp to [0.01, 0.99] for numerical safety
            test_probs = np.clip(test_probs, 0.01, 0.99)

            return np.column_stack([1.0 - test_probs, test_probs])

        except Exception as e:
            logger.warning("MarkovSwitching predict failed: %s", e)
            return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])


class RecessionEnsembleModel:
    """
    Ensemble recession prediction model.

    Architecture:
    - Base models: L1-Probit, Random Forest, XGBoost, Markov-Switching
    - Preprocessing: PCA factor extraction (augments feature set)
    - Calibration: Isotonic regression on each base model
    - Ensemble: Dynamic Model Averaging weights (exponential forgetting of CV Brier scores)
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
        must_include = [
            'leading_T10Y3M', 'leading_T10Y3M_inverted',
            'leading_T10Y3M_inv_duration', 'monetary_DFF',
            'monetary_BAA10Y', 'SAHM_INDICATOR', 'SAHM_TRIGGER',
            'AT_RISK_DIFFUSION', 'AT_RISK_DIFFUSION_WEIGHTED',
            'CREDIT_STRESS_INDEX', 'FFR_x_SPREAD', 'FFR_STANCE',
            'financial_NFCI', 'NFCI_Z', 'FINANCIAL_STRESS_COMPOSITE',
            'NEAR_TERM_FORWARD_SPREAD', 'NTFS_inverted', 'NTFS_momentum',
            'EBP_PROXY', 'EBP_PROXY_Z', 'EBP_AT_RISK',
        ]
        for col in must_include:
            if col in valid_cols and col not in selected:
                selected.append(col)

        logger.info(f"Selected {len(selected)} features")
        return selected

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, train_df, feature_cols=None):
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
            feature_cols = self.select_features(train_df, max_features=60)

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

        # ── Step 3: Time-series CV with per-fold Brier tracking ──────
        logger.info("Running time-series cross-validation...")
        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)
        cv_probas = {name: [] for name in all_model_names}
        cv_actuals = []
        # Track per-fold Brier scores for DMA
        cv_fold_briers = {name: [] for name in all_model_names}

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
                # Per-fold Brier score
                fold_brier = brier_score_loss(fold_val_actuals, proba)
                cv_fold_briers[name].append(fold_brier)

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

        # 4b: DMA weights (exponential forgetting of per-fold Brier scores)
        dma_weights = self._compute_dma_weights(cv_fold_briers, forgetting_factor=0.99)
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

        # Markov-switching is not calibrated via isotonic (regime model, not classifier)
        if 'markov_switching' in all_model_names:
            self.calibrated_models['markov_switching'] = None
            logger.info("  ⊘ markov_switching: skipped isotonic (regime model)")

        # ── Step 6: Optimize decision threshold (Youden's J) ────────
        logger.info("Optimizing decision threshold (Youden's J)...")
        cv_ensemble_proba = self._weighted_average_probas(cv_probas, cv_actuals)
        self.decision_threshold = self._optimize_threshold(cv_actuals, cv_ensemble_proba)
        logger.info(f"  Optimal threshold: {self.decision_threshold:.3f}")

        self.is_fitted = True
        logger.info("=" * 80)
        logger.info("MODEL FITTING COMPLETE")
        logger.info("=" * 80)

    def _compute_dma_weights(self, cv_fold_scores, forgetting_factor=0.99):
        """
        Compute Dynamic Model Averaging weights using exponential forgetting.

        More recent CV fold performance is weighted more heavily.
        This allows the ensemble to adapt to structural changes in the economy.

        Uses negative log-Brier as the score (avoids extreme ratios from
        1/brier when Brier scores are very small) and softmax normalization
        to prevent weight collapse onto a single model.
        """
        n_folds = len(next(iter(cv_fold_scores.values())))
        scores = {}

        for name in cv_fold_scores:
            fold_briers = cv_fold_scores[name]
            # Compute time-weighted average Brier score (more recent folds weighted more)
            weighted_brier = 0
            total_weight = 0
            for i, brier in enumerate(fold_briers):
                w = forgetting_factor ** (n_folds - 1 - i)  # More recent = higher weight
                weighted_brier += w * brier
                total_weight += w
            avg_brier = weighted_brier / total_weight
            # Use negative Brier as score (lower Brier = higher score)
            scores[name] = -avg_brier

        # Inverse Brier weighting with minimum weight floor
        # This gives appropriate weight to performance differences while
        # preventing complete collapse onto a single model
        raw_weights = {}
        for name in cv_fold_scores:
            fold_briers = cv_fold_scores[name]
            weighted_brier = 0
            total_weight = 0
            for i, brier in enumerate(fold_briers):
                w = forgetting_factor ** (n_folds - 1 - i)
                weighted_brier += w * brier
                total_weight += w
            avg_brier = weighted_brier / total_weight
            raw_weights[name] = 1.0 / (avg_brier + 0.01)  # +0.01 prevents extreme ratios

        # Normalize
        total = sum(raw_weights.values())
        weights = {name: w / total for name, w in raw_weights.items()}

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

        for threshold in np.arange(0.10, 0.90, 0.01):
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
        for model_name in ['probit', 'random_forest', 'xgboost', 'markov_switching', 'ensemble']:
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
        report.append("          Markov-Switching Regime Model (2-state Hamilton filter)")
        report.append("  Preprocessing: PCA factor extraction (top 5 components augment features)")
        report.append("  Class weighting: Balanced (accounts for ~12% recession base rate)")
        report.append("  Calibration: Isotonic regression via time-series cross-validation")
        report.append("  Ensemble: Dynamic Model Averaging (exponential forgetting, lambda=0.99)")
        report.append("  Threshold: Youden's J statistic (maximizes sensitivity + specificity)")
        report.append(f"  Features: {len(self.feature_cols)} selected from correlation + RF importance")
        report.append("  Includes: At-risk transforms, Sahm Rule, term spread dynamics,")
        report.append("            near-term forward spread, excess bond premium proxy,")
        report.append("            credit stress index, monetary policy stance interaction")
        report.append("")
        report.append("  References:")
        report.append("  - Estrella & Mishkin (1998), Wright (2006), Sahm (2019)")
        report.append("  - Billakanti & Shin (2025), Engstrom & Sharpe (2019)")
        report.append("  - Gilchrist & Zakrajsek (2012), Raftery et al. (2010) [DMA]")
        report.append("=" * 80)

        return "\n".join(report)
