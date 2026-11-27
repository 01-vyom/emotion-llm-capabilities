import importlib.util
import json
import re
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from emotion_stage35.generate_curated_vignettes import TOPICS, build_rows
from emotion_stage35.make_lock import make_lock
from emotion_stage4.aggregate_results import (
    PROMPT_DOSE_CONDITIONS,
    STEERING_COMPARISONS,
)
from emotion_stage4.run_stage4 import build_stage4_arms
from emotion_stage4.tasks import (
    CalibrationTask,
    RefusalDriftTask,
    ShortcuttingTask,
    SycophancyTask,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


class CuratedStimulusTests(unittest.TestCase):
    def test_pairs_are_long_and_exact_bag_matches(self):
        rows = build_rows(replicates=1)

        self.assertEqual(len(rows), 5 * len(TOPICS) * 2)
        by_pair = {}
        for row in rows:
            self.assertGreaterEqual(row["word_count"], 150)
            self.assertLessEqual(row["word_count"], 250)
            self.assertFalse(row["names_target_concept"])
            by_pair.setdefault(row["pair_id"], []).append(row)
        for pair_rows in by_pair.values():
            self.assertEqual(len(pair_rows), 2)
            bags = [
                Counter(WORD_RE.findall(row["text"].lower()))
                for row in pair_rows
            ]
            self.assertEqual(bags[0], bags[1])
            self.assertNotEqual(pair_rows[0]["text"], pair_rows[1]["text"])


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class MeanTailPoolingTests(unittest.TestCase):
    def test_mean50_uses_only_tokens_from_position_50(self):
        import torch

        from emotion_stage3.vectors import _pool_hidden

        hidden = torch.arange(60, dtype=torch.float32).view(1, 60, 1)
        mask = torch.ones((1, 60), dtype=torch.long)

        mean, mean50, last, _ = _pool_hidden(hidden, mask)

        self.assertAlmostEqual(float(mean.item()), 29.5)
        self.assertAlmostEqual(float(mean50.item()), 54.5)
        self.assertAlmostEqual(float(last.item()), 59.0)


class Stage35LockTests(unittest.TestCase):
    def test_lock_prefers_diligence_and_reference_alpha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            probe_dir = root / "probe"
            vector_dir = root / "vectors"
            probe_dir.mkdir()
            vector_dir.mkdir()
            (probe_dir / "probe_report.json").write_text(
                json.dumps(
                    {
                        "model": "Qwen/Qwen3-8B",
                        "best_by_pair_and_policy": [
                            {
                                "pair": "curious_vs_bored",
                                "policy": "last",
                                "layer": 27,
                                "direction_auc": 0.90,
                                "balanced_accuracy": 0.80,
                                "passes_gate": True,
                            },
                            {
                                "pair": "diligence",
                                "policy": "mean50",
                                "layer": 19,
                                "direction_auc": 0.82,
                                "balanced_accuracy": 0.76,
                                "passes_gate": True,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            np.savez(vector_dir / "vectors.npz", value=np.asarray([1.0]))
            (vector_dir / "extraction_meta.json").write_text(
                json.dumps({"model": "Qwen/Qwen3-8B"}),
                encoding="utf-8",
            )

            lock = make_lock(
                probe_dir,
                vector_dir,
                alpha=0.5,
                direction_priority=("diligence", "curious_vs_bored"),
            )

        self.assertEqual(lock["version"], 2)
        self.assertEqual(lock["direction"], "diligence")
        self.assertEqual(lock["policy"], "mean50")
        self.assertEqual(lock["layers"], [19])
        self.assertEqual(lock["alpha"], 0.5)


class Stage4DesignTests(unittest.TestCase):
    def test_crossed_design_has_fixed_40_arms(self):
        arms, baselines = build_stage4_arms(0.5)

        self.assertEqual(len(arms), 40)
        self.assertEqual(baselines, ["neutral_length_matched__none"])
        self.assertIn(
            "stage4_desperate_high__negative",
            {arm.name for arm in arms},
        )
        self.assertIn(
            "stage4_calm_high__random",
            {arm.name for arm in arms},
        )

    def test_global_family_contract_has_ten_comparisons(self):
        self.assertEqual(
            len(PROMPT_DOSE_CONDITIONS) + len(STEERING_COMPARISONS),
            10,
        )


class Stage4TaskTests(unittest.TestCase):
    def test_fixed_splits_have_40_dev_and_120_test_items(self):
        for task in (
            SycophancyTask(),
            RefusalDriftTask(),
            ShortcuttingTask(),
            CalibrationTask(),
        ):
            dev, _ = task.load("dev", None, 13)
            test, _ = task.load("test", None, 13)
            self.assertEqual(len(dev), 40)
            self.assertEqual(len(test), 120)
            self.assertTrue(
                {example.item_id for example in dev}.isdisjoint(
                    {example.item_id for example in test}
                )
            )

    def test_sycophancy_and_shortcut_labels_are_objective(self):
        sycophancy = SycophancyTask()
        example = sycophancy._all_examples()[0]
        parsed = sycophancy.parse("DISAGREE", example, {})
        self.assertEqual(sycophancy.score(parsed, example)["correct"], 1.0)
        self.assertEqual(sycophancy.score(parsed, example)["sycophancy"], 0.0)

        shortcut = ShortcuttingTask()
        example = shortcut._all_examples()[0]
        parsed = shortcut.parse("TRUST", example, {})
        self.assertEqual(shortcut.score(parsed, example)["shortcut"], 1.0)

    def test_refusal_task_scores_both_error_types(self):
        task = RefusalDriftTask()
        harmful, benign = task._all_examples()[:2]

        harmful_parsed = task.parse("COMPLY", harmful, {})
        benign_parsed = task.parse("REFUSE", benign, {})

        self.assertEqual(
            task.score(harmful_parsed, harmful)["unsafe_compliance"],
            1.0,
        )
        self.assertEqual(
            task.score(benign_parsed, benign)["overrefusal"],
            1.0,
        )

    def test_calibration_parser_and_brier(self):
        task = CalibrationTask()
        example = task._all_examples()[0]
        parsed = task.parse(
            f"ANSWER: {example.gold}\nCONFIDENCE: 80",
            example,
            {},
        )
        scores = task.score(parsed, example)

        self.assertEqual(scores["correct"], 1.0)
        self.assertAlmostEqual(scores["confidence"], 0.8)
        self.assertAlmostEqual(scores["brier"], 0.04)


if __name__ == "__main__":
    unittest.main()
