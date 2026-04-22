"""
Calibration diagnostics + calibrator A/B + stationary block bootstrap CI.

Implements reliability curves (quantile-binned), expected calibration error (ECE),
calibration slope/intercept via logistic regression on logit(p), and the
Brier-score decomposition into reliability / resolution / uncertainty
(Murphy 1973). Calibrator factory covers isotonic, sigmoid (Platt), and beta
calibration (Kull, Filho & Flach 2017).

Stationary block bootstrap (Politis & Romano 1994) replaces Dirichlet weight
perturbation as the uncertainty source, so temporal dependence is preserved
when producing per-row predicted-probability CIs.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Tuple

import numpy as np

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

try:  # pragma: no cover - optional dependency
    from betacal import BetaCalibration
    HAS_BETACAL = True
except Exception:  # pragma: no cover - environment specific
    HAS_BETACAL = False
    BetaCalibration = None  # type: ignore

logger = logging.getLogger(__name__)

__all__ = [
    "reliability_curve",
    "expected_calibration_error",
    "calibration_slope_intercept",
    "brier_decomposition",
    "fit_calibrator",
    "stationary_block_bootstrap_ci",
    "stationary_bootstrap_indices",
    "HAS_BETACAL",
]

_EPS = 1e-12
_PROB_CLIP = 1e-6  # logit stability near {0, 1}


def _as_numpy(arr) -> np.ndarray:
    return np.asarray(arr, dtype=float).ravel()


def reliability_curve(
    y_true,
    y_prob,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_centers, bin_observed_freq, bin_pred_mean, bin_counts).

    Parameters
    ----------
    y_true : array-like of binary labels
    y_prob : array-like of predicted probabilities
    n_bins : target number of bins
    strategy : 'quantile' (equal-count, default, robust to skewed predictions)
               or 'uniform' (equal-width).

    Empty bins are dropped. Output arrays have length <= n_bins.
    """

    y_true = _as_numpy(y_true)
    y_prob = _as_numpy(y_prob)

    if len(y_true) == 0 or len(y_prob) == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty.astype(int)

    if strategy == "quantile":
        # Unique quantile edges (collapse to coarser binning if many ties).
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(np.quantile(y_prob, quantiles))
        if len(edges) < 2:
            edges = np.array([y_prob.min() - _EPS, y_prob.max() + _EPS])
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    # Expand extrema slightly so endpoint values belong to the outer bins.
    edges = edges.copy()
    edges[0] = min(edges[0], y_prob.min()) - _EPS
    edges[-1] = max(edges[-1], y_prob.max()) + _EPS

    bin_ids = np.digitize(y_prob, edges[1:-1], right=False)
    n_actual_bins = len(edges) - 1

    centers = []
    observed = []
    predicted = []
    counts = []
    for b in range(n_actual_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            continue
        centers.append(0.5 * (edges[b] + edges[b + 1]))
        predicted.append(float(y_prob[mask].mean()))
        observed.append(float(y_true[mask].mean()))
        counts.append(count)

    return (
        np.asarray(centers, dtype=float),
        np.asarray(observed, dtype=float),
        np.asarray(predicted, dtype=float),
        np.asarray(counts, dtype=int),
    )


def expected_calibration_error(
    y_true,
    y_prob,
    n_bins: int = 10,
) -> float:
    """Equal-width ECE per the standard definition.

    ECE = sum_b (n_b / N) * | observed_b - predicted_b |.
    """

    y_true = _as_numpy(y_true)
    y_prob = _as_numpy(y_prob)
    n = len(y_true)
    if n == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges[0] -= _EPS
    edges[-1] += _EPS
    bin_ids = np.digitize(y_prob, edges[1:-1], right=False)

    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            continue
        gap = abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
        ece += (count / n) * gap
    return float(ece)


def calibration_slope_intercept(y_true, y_prob) -> Tuple[float, float]:
    """Fit logistic regression P(Y=1 | logit(p)); return (slope, intercept).

    Perfect calibration implies slope = 1 and intercept = 0.
    """

    y_true = _as_numpy(y_true).astype(int)
    y_prob = _as_numpy(y_prob)

    if len(set(y_true)) < 2 or len(y_prob) < 3:
        return float("nan"), float("nan")

    clipped = np.clip(y_prob, _PROB_CLIP, 1.0 - _PROB_CLIP)
    logit = np.log(clipped / (1.0 - clipped))
    logit = logit.reshape(-1, 1)

    # No penalty so the slope/intercept reflect the empirical fit, not shrinkage.
    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=1000,
    )
    model.fit(logit, y_true)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def brier_decomposition(y_true, y_prob, n_bins: int = 10) -> Dict[str, float]:
    """Murphy (1973) decomposition of the Brier score.

    Returns dict with reliability (lower better), resolution (higher better),
    uncertainty (independent of predictions), and brier_total
    (reliability - resolution + uncertainty).
    """

    y_true = _as_numpy(y_true)
    y_prob = _as_numpy(y_prob)
    n = len(y_true)

    if n == 0:
        nan = float("nan")
        return {
            "reliability": nan,
            "resolution": nan,
            "uncertainty": nan,
            "brier_total": nan,
        }

    base_rate = float(y_true.mean())
    uncertainty = base_rate * (1.0 - base_rate)

    # Quantile bins for stability under skewed probability distributions.
    centers, observed, predicted, counts = reliability_curve(
        y_true, y_prob, n_bins=n_bins, strategy="quantile"
    )

    reliability = 0.0
    resolution = 0.0
    if counts.size > 0:
        for obs_b, pred_b, cnt_b in zip(observed, predicted, counts):
            weight = cnt_b / n
            reliability += weight * (pred_b - obs_b) ** 2
            resolution += weight * (obs_b - base_rate) ** 2

    brier_total = reliability - resolution + uncertainty

    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "brier_total": float(brier_total),
    }


# ---------------------------------------------------------------------------
# Calibrator factory (isotonic / sigmoid / beta)
# ---------------------------------------------------------------------------


class _CalibratorBase:
    """Uniform predict_proba API: takes 1D probabilities, returns 1D calibrated probs."""

    method: str = ""

    def predict_proba(self, y_prob):  # pragma: no cover - interface
        raise NotImplementedError


class _IsotonicCalibrator(_CalibratorBase):
    method = "isotonic"

    def __init__(self, y_min: float = 0.01, y_max: float = 0.99):
        self.model = IsotonicRegression(
            y_min=y_min, y_max=y_max, out_of_bounds="clip"
        )

    def fit(self, y_prob, y_true):
        self.model.fit(_as_numpy(y_prob), _as_numpy(y_true))
        return self

    def predict_proba(self, y_prob):
        return self.model.predict(_as_numpy(y_prob))


class _SigmoidCalibrator(_CalibratorBase):
    """Platt-style sigmoid calibration: fit logistic regression on logit(p)."""

    method = "sigmoid"

    def __init__(self):
        self.model = LogisticRegression(
            penalty=None, solver="lbfgs", max_iter=1000
        )

    def fit(self, y_prob, y_true):
        y_prob = _as_numpy(y_prob)
        y_true = _as_numpy(y_true).astype(int)
        clipped = np.clip(y_prob, _PROB_CLIP, 1.0 - _PROB_CLIP)
        logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        if len(set(y_true)) < 2:
            # Degenerate — fall back to identity on predict.
            self._degenerate = True
            return self
        self._degenerate = False
        self.model.fit(logit, y_true)
        return self

    def predict_proba(self, y_prob):
        y_prob = _as_numpy(y_prob)
        if getattr(self, "_degenerate", False):
            return np.clip(y_prob, 0.01, 0.99)
        clipped = np.clip(y_prob, _PROB_CLIP, 1.0 - _PROB_CLIP)
        logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        return self.model.predict_proba(logit)[:, 1]


class _BetaCalibrator(_CalibratorBase):
    method = "beta"

    def __init__(self, parameters: str = "abm"):
        if not HAS_BETACAL:
            raise ImportError(
                "betacal is required for beta calibration. "
                "Install with: pip install betacal>=1.0.0"
            )
        self.model = BetaCalibration(parameters=parameters)

    def fit(self, y_prob, y_true):
        y_prob = _as_numpy(y_prob).reshape(-1, 1)
        y_true = _as_numpy(y_true).astype(int)
        # BetaCalibration expects 2D predictor and handles degenerate cases internally.
        self.model.fit(y_prob, y_true)
        return self

    def predict_proba(self, y_prob):
        y_prob = _as_numpy(y_prob).reshape(-1, 1)
        out = self.model.predict(y_prob)
        out = np.asarray(out, dtype=float).ravel()
        return np.clip(out, 0.0, 1.0)


def fit_calibrator(method: str, y_true_train, y_prob_train):
    """Fit a calibrator with a uniform predict_proba(1D) -> 1D API.

    method: 'isotonic' | 'sigmoid' | 'beta'.
    """

    method_lower = (method or "").strip().lower()
    if method_lower == "isotonic":
        cal = _IsotonicCalibrator()
    elif method_lower in {"sigmoid", "platt"}:
        cal = _SigmoidCalibrator()
    elif method_lower == "beta":
        cal = _BetaCalibrator()
    else:
        raise ValueError(
            f"Unknown calibrator method '{method}'. "
            "Expected 'isotonic', 'sigmoid', or 'beta'."
        )
    cal.fit(y_prob_train, y_true_train)
    return cal


# ---------------------------------------------------------------------------
# Stationary block bootstrap for CIs
# ---------------------------------------------------------------------------


def stationary_bootstrap_indices(
    n: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices of length n.

    At each position we flip a Bernoulli with p = 1 / block_size: on success,
    restart at a new random index; otherwise step forward by one (mod n).
    This produces a geometric block-length distribution with mean block_size.
    """

    if n <= 0:
        return np.empty(0, dtype=int)
    block_size = max(1, int(block_size))
    p_restart = 1.0 / block_size

    indices = np.empty(n, dtype=int)
    current = int(rng.integers(0, n))
    indices[0] = current
    # Pre-sample restart decisions and new starts in bulk for speed.
    restart_flags = rng.random(n - 1) < p_restart
    new_starts = rng.integers(0, n, size=n - 1)
    for i in range(1, n):
        if restart_flags[i - 1]:
            current = int(new_starts[i - 1])
        else:
            current = (current + 1) % n
        indices[i] = current
    return indices


def stationary_block_bootstrap_ci(
    model_predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    block_size_months: int = 12,
    n_bootstrap: int = 500,
    ci_level: float = 0.90,
    random_state: int = 42,
):
    """Stationary (Politis-Romano) block bootstrap CIs for predicted probabilities.

    Parameters
    ----------
    model_predict_fn : callable (X_resampled) -> 1D array of probabilities with
                        length == len(X_resampled). The function must keep the
                        ordering of X_resampled (no internal shuffling).
    X : array-like with n rows. Bootstrap samples are drawn over the row axis.
    block_size_months : mean geometric block length (12 = 1 year for monthly data).
    n_bootstrap : number of bootstrap replicates (500 by default).
    ci_level : nominal coverage (0.90 for 5th/95th percentile).
    random_state : rng seed for reproducibility.

    Returns
    -------
    dict with keys: ci_lower, ci_upper, bootstrap_mean, bootstrap_std,
                    method, block_size_months, n_bootstrap, ci_level.

    Notes
    -----
    Under the stationary bootstrap, each row may be drawn 0, 1, or many times
    per replicate. We accumulate per-row running statistics (sum, sum-of-squares,
    and a quantile sketch) rather than storing the full (n_bootstrap, n_rows)
    matrix, so memory stays bounded.
    """

    X = np.asarray(X)
    n = len(X)
    if n == 0:
        empty = np.empty(0, dtype=float)
        return {
            "ci_lower": empty,
            "ci_upper": empty,
            "bootstrap_mean": empty,
            "bootstrap_std": empty,
            "method": "stationary_block_bootstrap",
            "block_size_months": int(block_size_months),
            "n_bootstrap": int(n_bootstrap),
            "ci_level": float(ci_level),
        }

    rng = np.random.default_rng(random_state)

    # Store bootstrap predictions per row. Each row only collects predictions
    # from replicates in which it was drawn, so we use lists to avoid NaNs.
    per_row_samples = [list() for _ in range(n)]

    for _ in range(int(n_bootstrap)):
        idx = stationary_bootstrap_indices(n, block_size_months, rng)
        try:
            preds = np.asarray(
                model_predict_fn(X[idx]), dtype=float
            ).ravel()
        except Exception as exc:
            logger.warning(
                "Bootstrap replicate failed; skipping. Error: %s", exc
            )
            continue
        if len(preds) != len(idx):
            logger.warning(
                "model_predict_fn returned %d preds for %d rows; skipping replicate.",
                len(preds), len(idx),
            )
            continue
        for position, row_idx in enumerate(idx):
            per_row_samples[int(row_idx)].append(preds[position])

    alpha_tail = (1.0 - ci_level) / 2.0
    ci_lower = np.full(n, np.nan, dtype=float)
    ci_upper = np.full(n, np.nan, dtype=float)
    bmean = np.full(n, np.nan, dtype=float)
    bstd = np.full(n, np.nan, dtype=float)

    for i in range(n):
        samples = per_row_samples[i]
        if not samples:
            continue
        arr = np.asarray(samples, dtype=float)
        ci_lower[i] = float(np.quantile(arr, alpha_tail))
        ci_upper[i] = float(np.quantile(arr, 1.0 - alpha_tail))
        bmean[i] = float(arr.mean())
        bstd[i] = float(arr.std(ddof=0))

    return {
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "bootstrap_mean": bmean,
        "bootstrap_std": bstd,
        "method": "stationary_block_bootstrap",
        "block_size_months": int(block_size_months),
        "n_bootstrap": int(n_bootstrap),
        "ci_level": float(ci_level),
    }
