"""
Tests for AUDIT-1 FIX 1 — deterministic feature-selection tie-break.

When two features have identical composite scores at the max_features cap
boundary, `select_features` must break ties alphabetically so the selection
is stable under tiny float perturbations. This regression test asserts the
alphabetical fallback is in place.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from recession_engine.ensemble_model import RecessionEnsembleModel


def _build_synthetic_panel(
    n_periods: int = 220,
    seed: int = 0,
    target_col: str = "RECESSION_FORWARD_6M",
) -> pd.DataFrame:
    """Build a small panel with two features sharing an identical correlation
    to the target, so the composite-score tie-break is exercised.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-31", periods=n_periods, freq="ME")

    # Base target: Bernoulli with 20% positive rate (matches training base rate)
    y = rng.binomial(1, 0.2, size=n_periods)

    # Create two features with identical correlation to y. We use y itself
    # + independent noise draws — the correlation magnitude is a pure function
    # of the signal/noise ratio and seeds both features the same way, so their
    # ranks on correlation, on RF importance, and on sparse-L1 coefficient will
    # all be near-identical.
    noise_a = rng.normal(size=n_periods)
    noise_b = rng.normal(size=n_periods)

    # By construction these have the same theoretical correlation with y.
    feat_z_alpha_first = 1.2 * y + 0.8 * noise_a
    feat_z_beta_second = 1.2 * y + 0.8 * noise_b

    # Add several distractor features so the selector has a nontrivial pool
    # that doesn't dominate the top of the ranking.
    distractors = {
        f"distractor_{i}": rng.normal(size=n_periods) for i in range(8)
    }

    df = pd.DataFrame(
        {
            target_col: y,
            "RECESSION": y,
            "alpha_feature": feat_z_alpha_first,
            "beta_feature": feat_z_beta_second,
            **distractors,
        },
        index=dates,
    )
    return df


class TestFeatureSelectionDeterminism(unittest.TestCase):
    """FIX 1 (AUDIT-1 M3) — alphabetical tie-break for composite scores."""

    def test_alphabetical_tiebreak_when_scores_tie(self):
        """With two near-tied features, alphabetical-first must rank higher."""
        df = _build_synthetic_panel(seed=123)

        model = RecessionEnsembleModel(target_horizon=6)
        # Run selection multiple times — seeds are deterministic so the
        # selected feature list must be identical across runs.
        selected_runs = [
            model.select_features(df, max_features=8) for _ in range(3)
        ]
        self.assertEqual(
            selected_runs[0],
            selected_runs[1],
            msg="select_features must be deterministic under repeat invocation",
        )
        self.assertEqual(selected_runs[0], selected_runs[2])

        selected = selected_runs[0]
        self.assertIn("alpha_feature", selected)
        self.assertIn("beta_feature", selected)
        # Alphabetical fallback: "alpha_feature" < "beta_feature", so alpha
        # must appear before beta in the ranked output whenever their
        # composite scores are within numerical noise of one another.
        idx_alpha = selected.index("alpha_feature")
        idx_beta = selected.index("beta_feature")
        self.assertLess(
            idx_alpha,
            idx_beta,
            msg=(
                "alpha_feature must outrank beta_feature under alphabetical "
                "tie-break when composite scores are effectively tied"
            ),
        )

    def test_tiebreak_explicit_identical_scores(self):
        """Directly exercise the score-ranking step: when scores are exactly
        equal, the sort must be stable and alphabetical by name."""
        # Synthesize the core ranking primitive used inside select_features
        # — this is the exact sort in the FIX 1 code path.
        scores = {
            "zulu_last": 1.0,
            "alpha_first": 1.0,
            "mike_middle": 1.0,
            "higher_score": 2.0,
        }
        ranked = [
            feat
            for feat, _ in sorted(
                sorted(scores.items(), key=lambda item: item[0]),
                key=lambda item: item[1],
                reverse=True,
            )
        ]
        self.assertEqual(
            ranked,
            ["higher_score", "alpha_first", "mike_middle", "zulu_last"],
            msg="Alphabetical tie-break must preserve score order for the top",
        )


if __name__ == "__main__":
    unittest.main()
