"""
Goldman Sachs Recession Prediction Engine
Ensemble Modeling Module
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
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


class RecessionEnsembleModel:
    """Ensemble recession prediction model"""
    
    def __init__(self, target_horizon=6, decision_threshold: float = 0.5):
        self.target_horizon = target_horizon
        self.target_col = f'RECESSION_FORWARD_{target_horizon}M'
        self.decision_threshold = decision_threshold
        
        # Initialize models
        self.models = {
            'probit': LogisticRegression(
                penalty='l1',
                solver='saga',
                C=0.1,
                max_iter=1000,
                random_state=42
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            ),
        }
        if HAS_XGBOOST:
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
        else:
            logger.warning("Proceeding without XGBoost model; ensemble will use available base models only.")
        
        self.meta_model = LogisticRegression(random_state=42)
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.metrics = {}
        self.is_fitted = False
        
    def prepare_data(self, df, train_end_date=None):
        """Prepare train/test split"""
        df_clean = df[df[self.target_col].notna()].copy()
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        
        if train_end_date:
            train_df = df_clean[df_clean.index <= train_end_date]
            test_df = df_clean[df_clean.index > train_end_date]
        else:
            split_idx = int(len(df_clean) * 0.8)
            train_df = df_clean.iloc[:split_idx]
            test_df = df_clean.iloc[split_idx:]
        
        logger.info(f"Training: {len(train_df)} obs, Testing: {len(test_df)} obs")
        return train_df, test_df
    
    def select_features(self, df, max_features=50):
        """Select most predictive features using correlation + model-based importance"""
        exclude_cols = [self.target_col, 'RECESSION']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        valid_cols = []
        for col in feature_cols:
            if df[col].notna().sum() / len(df) > 0.7:
                valid_cols.append(col)
        
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
        
        # First-pass ranking by correlation
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        top_corr_feats = [feat for feat, corr in sorted_features[: max_features * 2]]

        # Second-pass: quick RandomForest importance on training-style subset
        df_sub = df[top_corr_feats + [self.target_col]].dropna()
        if len(df_sub) > 500:
            df_sub = df_sub.iloc[-500:]

        if len(df_sub) > 0:
            X_sub = df_sub[top_corr_feats]
            y_sub = df_sub[self.target_col]
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )
            rf.fit(X_sub, y_sub)
            importances = rf.feature_importances_
            imp_series = pd.Series(importances, index=top_corr_feats).sort_values(ascending=False)
            selected = imp_series.index.tolist()[:max_features]
        else:
            selected = top_corr_feats[:max_features]

        logger.info(f"Selected {len(selected)} features")
        return selected
    
    def fit(self, train_df, feature_cols=None):
        """Fit all models"""
        logger.info("="*80)
        logger.info("FITTING RECESSION PREDICTION ENSEMBLE")
        logger.info("="*80)
        
        if feature_cols is None:
            feature_cols = self.select_features(train_df, max_features=50)
        
        self.feature_cols = feature_cols
        
        X_train = train_df[feature_cols].fillna(method='ffill').fillna(0)
        y_train = train_df[self.target_col]
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Fit base models
        for name, model in self.models.items():
            logger.info(f"Fitting {name}...")
            
            if name == 'probit':
                model.fit(X_train_scaled, y_train)
                importance = np.abs(model.coef_[0])
                self.feature_importance[name] = dict(zip(feature_cols, importance))
            else:
                model.fit(X_train, y_train)
                importance = model.feature_importances_
                self.feature_importance[name] = dict(zip(feature_cols, importance))
            
            pred_proba = model.predict_proba(X_train_scaled if name == 'probit' else X_train)[:, 1]
            auc = roc_auc_score(y_train, pred_proba)
            logger.info(f"  Training AUC: {auc:.4f}")
        
        # Fit meta-model using whatever base models are available
        logger.info("Fitting meta-model...")
        base_names = [name for name in ['probit', 'random_forest', 'xgboost'] if name in self.models]
        meta_cols = []
        for name in base_names:
            if name == 'probit':
                meta_cols.append(self.models[name].predict_proba(X_train_scaled)[:, 1])
            else:
                meta_cols.append(self.models[name].predict_proba(X_train)[:, 1])
        meta_features = np.column_stack(meta_cols)

        self.meta_model.fit(meta_features, y_train)
        meta_pred = self.meta_model.predict_proba(meta_features)[:, 1]
        meta_auc = roc_auc_score(y_train, meta_pred)
        logger.info(f"  Meta-model AUC: {meta_auc:.4f}")
        
        self.is_fitted = True
        logger.info("="*80)
    
    def predict(self, test_df):
        """Generate predictions"""
        if not self.is_fitted:
            raise ValueError("Models must be fitted first!")
        
        X_test = test_df[self.feature_cols].fillna(method='ffill').fillna(0)
        X_test_scaled = self.scaler.transform(X_test)
        
        predictions = {}
        predictions['probit'] = self.models['probit'].predict_proba(X_test_scaled)[:, 1]
        predictions['random_forest'] = self.models['random_forest'].predict_proba(X_test)[:, 1]
        if 'xgboost' in self.models:
            predictions['xgboost'] = self.models['xgboost'].predict_proba(X_test)[:, 1]
        
        # Meta-features built from whichever base predictions exist
        base_names = [name for name in ['probit', 'random_forest', 'xgboost'] if name in predictions]
        meta_cols = [predictions[name] for name in base_names]
        meta_features = np.column_stack(meta_cols)
        predictions['ensemble'] = self.meta_model.predict_proba(meta_features)[:, 1]
        
        return predictions
    
    def evaluate(self, test_df, predictions, threshold: float = None):
        """Evaluate model performance at a given probability threshold"""
        if threshold is None:
            threshold = self.decision_threshold
        y_true = test_df[self.target_col]
        results = []
        
        for model_name, y_pred_proba in predictions.items():
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            auc = roc_auc_score(y_true, y_pred_proba)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Ensure a full 2x2 confusion matrix even if only one class is present
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            
            results.append({
                'Model': model_name,
                'AUC': auc,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'Accuracy': (tp + tn) / (tp + tn + fp + fn)
            })
        
        metrics_df = pd.DataFrame(results)
        self.metrics = metrics_df
        
        logger.info("\n" + "="*80)
        logger.info("MODEL EVALUATION RESULTS")
        logger.info("="*80)
        print(metrics_df.to_string(index=False))
        logger.info("="*80)
        
        return metrics_df
    
    def generate_report(self, test_df, predictions):
        """Generate executive report"""
        latest_date = test_df.index[-1]
        latest_probs = {name: pred[-1] for name, pred in predictions.items()}
        
        report = []
        report.append("="*80)
        report.append("GOLDMAN SACHS RECESSION PREDICTION ENGINE")
        report.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*80)
        report.append("")
        report.append(f"RECESSION PROBABILITY - {self.target_horizon} MONTHS FORWARD")
        report.append(f"As of: {latest_date.strftime('%Y-%m-%d')}")
        report.append("-"*80)
        
        for model_name in ['probit', 'random_forest', 'xgboost', 'ensemble']:
            if model_name in latest_probs:
                prob = latest_probs[model_name]
                report.append(f"  {model_name.upper():20s}: {prob:6.1%}")
        
        report.append("")
        report.append("="*80)
        
        ensemble_prob = latest_probs['ensemble']
        
        if ensemble_prob < 0.15:
            signal = "🟢 LOW RISK - Economy appears stable"
        elif ensemble_prob < 0.35:
            signal = "🟡 MODERATE RISK - Monitor closely"
        elif ensemble_prob < 0.60:
            signal = "🟠 ELEVATED RISK - Recession likely"
        else:
            signal = "🔴 HIGH RISK - Recession highly probable"
        
        report.append(signal)
        report.append("")
        report.append("="*80)
        
        return "\n".join(report)