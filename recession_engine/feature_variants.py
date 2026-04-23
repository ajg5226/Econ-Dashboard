"""
Feature variant filters for the recession-prediction at-risk bake-off (B1).

Four variants are supported:

* ``hybrid`` (default)
    Pass the engineered feature frame through unchanged. Reproduces the
    production pipeline used since Tier 4 was added.

* ``continuous_only``
    Drop the Tier-4 at-risk binary indicators and both diffusion indices.
    Also drops the Tier-2c EBP_AT_RISK flag so every ``*_AT_RISK`` column
    is removed. Keeps continuous transforms (pct-changes, rolling stats,
    z-scores) and every other engineered tier.

* ``at_risk_only``
    Keep only Tier-4 at-risk columns (``*_AT_RISK`` + ``AT_RISK_DIFFUSION``
    + ``AT_RISK_DIFFUSION_WEIGHTED``), plus the Tier-2c ``EBP_AT_RISK``
    flag. Drops every other engineered column. Meta columns needed by the
    training pipeline (``RECESSION``, ``RECESSION_FORWARD_*``, ``ref_*``)
    are always preserved.

* ``pca_on_binarized``
    Continuous variant plus ``N`` PCA components fit on the raw at-risk
    binary matrix. Tests whether the Tier-4 signal is low-rank. PCA is
    fit once on the full history available at call time (this is the same
    concession made in ``ensemble_model.py``'s PCA augmentation step).

The filter is implemented as a runtime post-processor so the core
``engineer_features`` behavior in :mod:`recession_engine.data_acquisition`
stays identical and unbranched.

Usage::

    from recession_engine.feature_variants import apply_feature_variant, AT_RISK_COLUMN_SUFFIX

    df_variant = apply_feature_variant(df_features, variant_name, n_pca=5)
"""

from __future__ import annotations

from typing import Iterable, Optional

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SUPPORTED_VARIANTS = (
    "continuous_only",
    "at_risk_only",
    "hybrid",
    "pca_on_binarized",
)

# Columns that must survive every variant — the model pipeline depends on them.
ALWAYS_KEEP_PREFIXES = ("RECESSION_FORWARD_", "ref_")
ALWAYS_KEEP_EXACT = {"RECESSION"}

# Diffusion indices produced in Tier 4.
DIFFUSION_COLUMNS = ("AT_RISK_DIFFUSION", "AT_RISK_DIFFUSION_WEIGHTED")

# Tier 4 uses the ``<raw_indicator>_AT_RISK`` naming convention. The
# Tier-2c ``EBP_AT_RISK`` column is also a binary at-risk flag derived
# from a z-score cutoff and therefore belongs with the at-risk layer.
AT_RISK_COLUMN_SUFFIX = "_AT_RISK"


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def is_at_risk_column(col: str) -> bool:
    """Return True if ``col`` is part of the at-risk (binary) layer."""
    if col in DIFFUSION_COLUMNS:
        return True
    # ``<col>_AT_RISK`` covers the Tier-4 layer plus Tier-2c EBP_AT_RISK.
    return col.endswith(AT_RISK_COLUMN_SUFFIX)


def classify_columns(columns: Iterable[str]) -> dict:
    """Split feature columns into at-risk vs continuous buckets.

    Meta columns (target, reference series, ``RECESSION``) are returned in
    their own ``meta`` bucket and should not be filtered.
    """
    at_risk = []
    continuous = []
    meta = []
    for col in columns:
        if col in ALWAYS_KEEP_EXACT or any(col.startswith(p) for p in ALWAYS_KEEP_PREFIXES):
            meta.append(col)
        elif is_at_risk_column(col):
            at_risk.append(col)
        else:
            continuous.append(col)
    return {"at_risk": at_risk, "continuous": continuous, "meta": meta}


def describe_classification(df: pd.DataFrame) -> dict:
    """Return a small JSON-friendly summary of the at-risk classification."""
    buckets = classify_columns(df.columns)
    at_risk_sorted = sorted(buckets["at_risk"])
    sample_head = at_risk_sorted[:10]
    sample_tail = at_risk_sorted[-10:] if len(at_risk_sorted) > 10 else []
    return {
        "at_risk_column_count": len(buckets["at_risk"]),
        "continuous_column_count": len(buckets["continuous"]),
        "meta_column_count": len(buckets["meta"]),
        "at_risk_column_list_sample_head": sample_head,
        "at_risk_column_list_sample_tail": sample_tail,
        "diffusion_columns_present": [
            c for c in DIFFUSION_COLUMNS if c in df.columns
        ],
    }


# ---------------------------------------------------------------------------
# PCA-on-binarized helper
# ---------------------------------------------------------------------------

def _pca_on_at_risk(
    df: pd.DataFrame,
    at_risk_cols: Iterable[str],
    n_components: int,
    prefix: str = "AR_PC_",
) -> pd.DataFrame:
    """Return a DataFrame of PCA components fit on the at-risk binary matrix.

    Any NaN values in the at-risk block are filled with 0 (absence of the
    flag is the natural default for a binary indicator). PCA is fit once
    on the full input frame — a concession consistent with how
    ``RecessionEnsembleModel.fit`` already handles its own PCA augmentation.
    """
    from sklearn.decomposition import PCA  # local import to keep the module light

    ar_cols = [c for c in at_risk_cols if c in df.columns]
    if not ar_cols:
        logger.warning("pca_on_binarized variant: no at-risk columns found to fit PCA on")
        return pd.DataFrame(index=df.index)

    block = df[ar_cols].copy().fillna(0.0).astype(float)
    # Drop rows where every column is zero / NaN — PCA needs variance.
    usable_mask = block.sum(axis=1) != 0
    if usable_mask.sum() < max(n_components * 4, 60):
        logger.warning(
            "pca_on_binarized variant: only %s usable rows, falling back to mean-only PCA.",
            int(usable_mask.sum()),
        )

    effective_n = min(int(n_components), len(ar_cols), max(int(usable_mask.sum()), 1))
    if effective_n < 1:
        return pd.DataFrame(index=df.index)

    pca = PCA(n_components=effective_n, random_state=42)
    try:
        transformed = pca.fit_transform(block.values)
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("pca_on_binarized variant: PCA fit failed (%s); skipping components.", exc)
        return pd.DataFrame(index=df.index)

    cols = [f"{prefix}{i + 1}" for i in range(effective_n)]
    pca_df = pd.DataFrame(transformed, index=df.index, columns=cols)
    explained = getattr(pca, "explained_variance_ratio_", None)
    if explained is not None:
        logger.info(
            "pca_on_binarized: fit %d components on %d at-risk features (explained var=%.1f%%).",
            effective_n,
            len(ar_cols),
            float(np.sum(explained)) * 100.0,
        )
    return pca_df


# ---------------------------------------------------------------------------
# Main variant dispatch
# ---------------------------------------------------------------------------

def apply_feature_variant(
    df: pd.DataFrame,
    variant: Optional[str],
    *,
    n_pca: int = 5,
) -> pd.DataFrame:
    """Filter an engineered feature frame according to ``variant``.

    Args:
        df: Engineered feature frame (output of
            ``RecessionDataAcquisition.engineer_features`` or that frame
            after ``create_forecast_target``).
        variant: One of :data:`SUPPORTED_VARIANTS`. ``None`` or ``'hybrid'``
            returns the input unchanged (other than a defensive copy).
        n_pca: Number of PCA components for the ``pca_on_binarized`` variant.

    Returns:
        A new DataFrame; the input is not mutated.
    """
    if variant is None or variant == "hybrid":
        return df.copy()

    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(
            f"Unknown feature variant {variant!r}. Supported: {SUPPORTED_VARIANTS}"
        )

    buckets = classify_columns(df.columns)
    at_risk_cols = buckets["at_risk"]
    continuous_cols = buckets["continuous"]
    meta_cols = buckets["meta"]

    if variant == "continuous_only":
        keep = meta_cols + continuous_cols
        out = df[keep].copy()
        logger.info(
            "Variant continuous_only: kept %d continuous + %d meta cols, dropped %d at-risk cols.",
            len(continuous_cols),
            len(meta_cols),
            len(at_risk_cols),
        )
        return out

    if variant == "at_risk_only":
        keep = meta_cols + at_risk_cols
        out = df[keep].copy()
        logger.info(
            "Variant at_risk_only: kept %d at-risk + %d meta cols, dropped %d continuous cols.",
            len(at_risk_cols),
            len(meta_cols),
            len(continuous_cols),
        )
        return out

    if variant == "pca_on_binarized":
        pca_df = _pca_on_at_risk(df, at_risk_cols, n_components=n_pca)
        base = df[meta_cols + continuous_cols].copy()
        # Attach the PCA columns; they replace the raw at-risk block.
        if not pca_df.empty:
            # Ensure no collisions between base and PCA column names.
            clash = [c for c in pca_df.columns if c in base.columns]
            if clash:
                pca_df = pca_df.drop(columns=clash)
            out = pd.concat([base, pca_df], axis=1)
        else:
            out = base
        logger.info(
            "Variant pca_on_binarized: kept %d continuous + %d meta cols, "
            "added %d PCA components from %d at-risk cols.",
            len(continuous_cols),
            len(meta_cols),
            len(pca_df.columns),
            len(at_risk_cols),
        )
        return out

    # Safety net — should never fall through because of the whitelist above.
    raise ValueError(f"Unhandled variant {variant!r}")


# ---------------------------------------------------------------------------
# Must-include collision auditing
# ---------------------------------------------------------------------------

CORE_MUST_INCLUDE = (
    # Yield curve & monetary
    "leading_T10Y3M", "leading_T10Y3M_inverted",
    "leading_T10Y3M_inv_duration", "monetary_DFF",
    "NEAR_TERM_FORWARD_SPREAD", "NTFS_inverted",
    "FFR_x_SPREAD", "FFR_STANCE",
    # Term-premium-adjusted spread
    "TERM_PREMIUM_ADJ_SPREAD", "TP_ADJ_SPREAD_inverted",
    # Credit & financial conditions
    "monetary_BAA10Y", "EBP_PROXY", "EBP_PROXY_Z",
    "financial_NFCI", "CREDIT_STRESS_INDEX",
    # Labor market triggers
    "SAHM_INDICATOR", "SAHM_TRIGGER",
    "SOS_INDICATOR", "SOS_TRIGGER",
    # At-risk diffusion
    "AT_RISK_DIFFUSION", "AT_RISK_DIFFUSION_WEIGHTED",
    # Confirming indicators
    "HOUSE_PRICE_DECLINING", "RECESSION_CONFIRM_2OF3",
    "RESIDENTIAL_INV_YOY", "SECTORAL_DIVERGENCE",
)


def must_include_collisions(df_variant: pd.DataFrame) -> list:
    """Return the subset of :data:`CORE_MUST_INCLUDE` that is NOT in the variant frame."""
    return [col for col in CORE_MUST_INCLUDE if col not in df_variant.columns]
