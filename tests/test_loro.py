"""Unit tests for the LORO evaluator orchestration logic.

The integration check (LORO output matches the existing backtester) runs
manually as Gate A on production state — it requires real model fits and
takes ~10-15 minutes. These unit tests cover the wrapper's orchestration
(scope tagging, summary aggregation, schema preservation) which is where
off-by-one and lookup bugs would actually live.
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts import loro_evaluator  # noqa: E402


class TestScopeTagging(unittest.TestCase):
    def setUp(self):
        self.scope_lookup = {
            "Oil Crisis": "informational",
            "Volcker I": "informational",
            "Volcker II": "informational",
            "S&L Crisis": "in_scope",
            "Dot-com": "in_scope",
            "Great Financial Crisis": "in_scope",
            "COVID": "in_scope",
        }

    def test_tag_scope_matches_in_scope_recessions(self):
        df = pd.DataFrame({"Recession": [
            "S&L Crisis (1990-91)",
            "Dot-com (2001)",
            "COVID (2020)",
        ]})
        tagged = loro_evaluator.tag_scope(df, self.scope_lookup)
        self.assertTrue((tagged["Evaluation_Scope"] == "in_scope").all())

    def test_tag_scope_handles_gfc_alias(self):
        # The backtester labels GFC as "GFC (2007-09)" while eval_origins.json
        # uses "Great Financial Crisis". load_scope_lookup() seeds both forms
        # so the substring match resolves either way.
        lookup = loro_evaluator.load_scope_lookup()
        if not lookup:
            self.skipTest("eval_origins.json missing in this repo state")
        df = pd.DataFrame({"Recession": ["GFC (2007-09)"]})
        tagged = loro_evaluator.tag_scope(df, lookup)
        self.assertEqual(tagged.loc[0, "Evaluation_Scope"], "in_scope")

    def test_tag_scope_matches_informational_recessions(self):
        df = pd.DataFrame({"Recession": [
            "Oil Crisis (1973-75)",
            "Volcker I (1980)",
            "Volcker II (1981-82)",
        ]})
        tagged = loro_evaluator.tag_scope(df, self.scope_lookup)
        self.assertTrue((tagged["Evaluation_Scope"] == "informational").all())

    def test_tag_scope_defaults_unknown_to_informational(self):
        df = pd.DataFrame({"Recession": ["Unknown Recession (2099)"]})
        tagged = loro_evaluator.tag_scope(df, self.scope_lookup)
        self.assertEqual(tagged.loc[0, "Evaluation_Scope"], "informational")

    def test_tag_scope_handles_non_string_label(self):
        df = pd.DataFrame({"Recession": [None]})
        tagged = loro_evaluator.tag_scope(df, self.scope_lookup)
        self.assertEqual(tagged.loc[0, "Evaluation_Scope"], "informational")


class TestComputeSummaries(unittest.TestCase):
    def _build_results(self):
        # Mix of in_scope and informational rows; one row with Error to
        # confirm the summary skips failures.
        return pd.DataFrame([
            {"Recession": "S&L Crisis", "Evaluation_Scope": "in_scope",
             "AUC": 0.90, "Brier": 0.08, "Peak_Prob": 0.7,
             "Lead_Months": 6.0, "Lead_Months_Fixed": 4.0,
             "Crossed_Threshold_Fixed": True, "Error": np.nan},
            {"Recession": "Dot-com", "Evaluation_Scope": "in_scope",
             "AUC": 0.60, "Brier": 0.20, "Peak_Prob": 0.25,
             "Lead_Months": np.nan, "Lead_Months_Fixed": np.nan,
             "Crossed_Threshold_Fixed": False, "Error": np.nan},
            {"Recession": "Oil Crisis", "Evaluation_Scope": "informational",
             "AUC": 0.85, "Brier": 0.10, "Peak_Prob": 0.6,
             "Lead_Months": 5.0, "Lead_Months_Fixed": 2.0,
             "Crossed_Threshold_Fixed": True, "Error": np.nan},
            {"Recession": "Failed Fit", "Evaluation_Scope": "in_scope",
             "AUC": np.nan, "Brier": np.nan, "Peak_Prob": np.nan,
             "Lead_Months": np.nan, "Lead_Months_Fixed": np.nan,
             "Crossed_Threshold_Fixed": np.nan, "Error": "single-class fit"},
        ])

    def test_summaries_split_by_scope(self):
        results = self._build_results()
        summaries = loro_evaluator.compute_summaries(results)
        self.assertEqual(summaries["in_scope"]["n"], 2)
        self.assertEqual(summaries["informational"]["n"], 1)

    def test_summaries_skip_error_rows(self):
        results = self._build_results()
        summaries = loro_evaluator.compute_summaries(results)
        # Failed fit should not contribute to the in-scope mean.
        # In-scope clean rows have AUCs of 0.90 and 0.60 → mean 0.75.
        self.assertAlmostEqual(summaries["in_scope"]["mean_auc"], 0.75)

    def test_summaries_count_crossed_threshold_fixed(self):
        results = self._build_results()
        summaries = loro_evaluator.compute_summaries(results)
        self.assertEqual(summaries["in_scope"]["n_crossed_threshold_fixed"], 1)
        self.assertEqual(summaries["informational"]["n_crossed_threshold_fixed"], 1)

    def test_summaries_handle_empty_scope(self):
        results = pd.DataFrame([
            {"Recession": "S&L Crisis", "Evaluation_Scope": "in_scope",
             "AUC": 0.9, "Brier": 0.1, "Peak_Prob": 0.5,
             "Lead_Months": 1.0, "Lead_Months_Fixed": 1.0,
             "Crossed_Threshold_Fixed": True, "Error": np.nan},
        ])
        summaries = loro_evaluator.compute_summaries(results)
        self.assertEqual(summaries["informational"]["n"], 0)


class TestLoadScopeLookup(unittest.TestCase):
    def test_load_scope_lookup_uses_real_eval_origins(self):
        lookup = loro_evaluator.load_scope_lookup()
        # eval_origins.json ships with the repo; we expect the canonical
        # 4 in_scope and 3 informational recessions.
        if not lookup:
            self.skipTest("eval_origins.json missing in this repo state")
        in_scope = {k for k, v in lookup.items() if v == "in_scope"}
        informational = {k for k, v in lookup.items() if v == "informational"}
        self.assertIn("S&L Crisis", in_scope)
        self.assertIn("Dot-com", in_scope)
        self.assertIn("COVID", in_scope)
        self.assertIn("Oil Crisis", informational)
        self.assertIn("Volcker I", informational)


class TestRenderSummaryText(unittest.TestCase):
    def test_render_summary_text_includes_scope_blocks(self):
        results = pd.DataFrame([
            {"Recession": "S&L Crisis", "Evaluation_Scope": "in_scope",
             "AUC": 0.9, "Brier": 0.08, "Peak_Prob": 0.7,
             "Crossed_Threshold_Fixed": True,
             "Lead_Months": 6.0, "Lead_Months_Fixed": 4.0,
             "Error": np.nan},
        ])
        summaries = loro_evaluator.compute_summaries(results)

        class _Args:
            horizon = 6
            max_features = 50
            skip_markov = False
            start_date = None
        text = loro_evaluator.render_summary_text(results, summaries, _Args())
        self.assertIn("LORO EVALUATION SUMMARY", text)
        self.assertIn("IN_SCOPE", text)
        self.assertIn("INFORMATIONAL", text)
        self.assertIn("S&L Crisis", text)


if __name__ == "__main__":
    unittest.main()
