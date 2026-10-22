"""Verify fixed-K evaluation with small, fully specified candidate pools."""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path


RUNTIME_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "numpy", "pandas", "sklearn")
)
if RUNTIME_AVAILABLE:
    import numpy as np
    import pandas as pd

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
    from virobind import base
    from virobind import model as v5


@unittest.skipUnless(RUNTIME_AVAILABLE, "CPU model dependencies are not installed")
class FixedKRankingTests(unittest.TestCase):
    def pool(self, protein, labels):
        return pd.DataFrame({
            "drug": [f"{protein}_{i}" for i in range(len(labels))],
            "protein": protein,
            "label": labels,
            "prob": np.linspace(0.99, 0.01, len(labels)),
        })

    def dual_head(self, frame):
        frame = frame.copy()
        frame["cls_prob"] = frame["prob"]
        frame["rank_prob"] = frame["prob"]
        frame["cls_logit"] = np.log(frame["prob"] / (1 - frame["prob"]))
        frame["rank_logit"] = frame["cls_logit"]
        for name in (
            "drug_weight_ecfp", "drug_weight_maccs", "drug_weight_rdkit2d",
            "drug_aux_gate", "protein_desc_gate",
        ):
            frame[name] = np.nan
        return frame

    def test_small_pool_is_excluded_and_each_k_has_its_own_proteins(self):
        frame = pd.concat([
            self.pool("small", [1] * 8),
            self.pool("ten", [1, 1, 0, 0, 1, 0, 0, 1, 0, 0]),
            self.pool("fifty", [1] * 7 + [0] * 3 + [1] * 3 + [0] * 37),
        ], ignore_index=True)
        metrics = base.compute_metrics(frame, threshold=0.5)
        self.assertAlmostEqual(metrics["p_at_10_mean"], (0.4 + 0.7) / 2)
        self.assertAlmostEqual(metrics["p_at_50_mean"], 10 / 50)
        self.assertEqual(metrics["n_proteins_p_at_10"], 2)
        self.assertEqual(metrics["n_proteins_p_at_50"], 1)
        self.assertEqual(metrics["n_proteins_with_both_labels"], 2)

    def test_exact_eligibility_boundaries_at_ten_and_fifty(self):
        frame = pd.concat([
            self.pool(str(size), ([1, 0] * size)[:size])
            for size in (9, 10, 49, 50)
        ], ignore_index=True)
        metrics = base.compute_metrics(frame, threshold=0.5)
        self.assertEqual(metrics["n_proteins_p_at_10"], 3)
        self.assertEqual(metrics["n_proteins_p_at_50"], 1)
        self.assertAlmostEqual(metrics["p_at_10_mean"], 0.5)
        self.assertAlmostEqual(metrics["p_at_50_mean"], 0.5)

    def test_single_class_pools_enter_precision_but_not_auc_or_ap(self):
        frame = pd.concat([
            self.pool("positive", [1] * 10),
            self.pool("negative", [0] * 10),
            self.pool("mixed", [1] * 2 + [0] * 8),
        ], ignore_index=True)
        metrics = base.compute_metrics(frame, threshold=0.5)
        self.assertAlmostEqual(metrics["p_at_10_mean"], (1 + 0 + 0.2) / 3)
        self.assertAlmostEqual(metrics["hit_at_10_mean"], 2 / 3)
        self.assertEqual(metrics["n_proteins_p_at_10"], 3)
        self.assertEqual(metrics["n_proteins_with_both_labels"], 1)
        self.assertEqual(metrics["perprotein_auc_mean"], 1.0)
        self.assertAlmostEqual(metrics["perprotein_aupr_mean"], 1.0)

    def test_macro_average_gives_each_eligible_protein_equal_weight(self):
        frame = pd.concat([
            self.pool("ten", [1] * 10),
            self.pool("hundred", [0] * 100),
        ], ignore_index=True)
        metrics = base.compute_metrics(frame, threshold=0.5)
        self.assertEqual(metrics["p_at_10_mean"], 0.5)
        self.assertEqual(metrics["p_at_50_mean"], 0.0)
        self.assertEqual(metrics["n_proteins_p_at_50"], 1)
        self.assertTrue(math.isnan(metrics["perprotein_auc_mean"]))

    def test_gray_and_nonbinary_records_do_not_fill_candidate_pools(self):
        frame = pd.concat([
            self.pool("nine", [1] * 4 + [0] * 5 + [-1, -1]),
            self.pool("ten", [1] * 4 + [0] * 6 + [-1, 0.2, np.nan]),
        ], ignore_index=True)
        metrics = base.compute_metrics(frame, threshold=0.5)
        self.assertEqual(metrics["n"], 19)
        self.assertEqual(metrics["pos"], 8)
        self.assertEqual(metrics["neg"], 11)
        self.assertEqual(metrics["n_proteins_p_at_10"], 1)
        self.assertEqual(metrics["p_at_10_mean"], 0.4)
        dual_metrics = v5.compute_metrics(self.dual_head(frame), threshold=None)
        self.assertEqual(dual_metrics["n"], 19)
        self.assertEqual(dual_metrics["rank_n_proteins_p_at_10"], 1)
        self.assertEqual(dual_metrics["n_proteins_p_at_10"], 1)
        self.assertEqual(dual_metrics["rank_p_at_10_mean"], 0.4)

    def test_no_eligible_proteins_returns_nan_and_zero_coverage(self):
        metrics = base.compute_metrics(self.pool("eight", [1, 0] * 4), threshold=0.5)
        for k in (10, 50):
            self.assertTrue(math.isnan(metrics[f"p_at_{k}_mean"]))
            self.assertTrue(math.isnan(metrics[f"hit_at_{k}_mean"]))
            self.assertEqual(metrics[f"n_proteins_p_at_{k}"], 0)

    def test_dual_head_uses_ranking_logits_even_when_sigmoid_saturates(self):
        # Negative rows come first, but every positive has a higher rank logit.
        frame = self.dual_head(self.pool("twenty", [0] * 10 + [1] * 10))
        frame["cls_prob"] = [0.9] * 10 + [0.1] * 10
        frame["rank_logit"] = np.arange(1000, 1020, dtype=float)
        frame["rank_prob"] = 1.0
        metrics = v5.compute_metrics(frame, threshold=0.5)
        self.assertEqual(metrics["p_at_10_mean"], 1.0)
        self.assertEqual(metrics["rank_p_at_10_mean"], 1.0)
        self.assertEqual(metrics["perprotein_auc_mean"], 1.0)
        self.assertAlmostEqual(metrics["perprotein_aupr_mean"], 1.0)
        self.assertEqual(metrics["cls_p_at_10_mean"], 0.0)
        self.assertEqual(metrics["auc"], 0.0)
        self.assertEqual(metrics["n_proteins_p_at_10"], 1)
        self.assertEqual(metrics["n_proteins_p_at_50"], 0)

    def test_candidate_order_does_not_change_results_for_unique_scores(self):
        frame = pd.concat([
            self.pool("twenty", [1] * 5 + [0] * 15),
            self.pool("sixty", [0] * 10 + [1] * 20 + [0] * 30),
        ], ignore_index=True)
        original = base.compute_metrics(frame, threshold=0.5)
        shuffled = base.compute_metrics(frame.sample(frac=1, random_state=9), threshold=0.5)
        for key in ("p_at_10_mean", "p_at_50_mean", "perprotein_auc_mean", "perprotein_aupr_mean"):
            self.assertEqual(original[key], shuffled[key])

    def test_all_gray_or_empty_evaluation_is_explicitly_rejected(self):
        for labels in ([], [-1, -1]):
            with self.subTest(labels=labels):
                frame = self.pool("gray", labels)
                for evaluator in (base.compute_metrics, v5.compute_metrics):
                    with self.assertRaisesRegex(ValueError, "at least one definitive label"):
                        evaluator(frame, threshold=0.5)


if __name__ == "__main__":
    unittest.main()
