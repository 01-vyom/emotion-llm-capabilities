import json
import tempfile
import unittest
from pathlib import Path

from emotion_stage5.make_panel_lock import make_panel_lock
from emotion_stage5.tasks import (
    ADMISSION_SPECS,
    CalibrationTask,
    RefusalDriftTask,
    ShortcuttingTask,
    SycophancyTask,
    eligible_for_signal,
    examples_sha256,
    task_dataset_sha256,
)


class Stage5TaskTests(unittest.TestCase):
    def test_fixed_test_sizes_power_the_repaired_panel(self):
        expected = {
            "sycophancy": 600,
            "refusal_drift": 480,
            "shortcutting": 480,
            "calibration": 120,
        }
        for task in (
            SycophancyTask(),
            RefusalDriftTask(),
            ShortcuttingTask(),
            CalibrationTask(),
        ):
            examples, _ = task.load("test", None, 13)
            self.assertEqual(len(examples), expected[task.name])

    def test_sycophancy_claims_are_false(self):
        task = SycophancyTask()
        examples = task._all_examples()
        self.assertEqual(len(examples), 800)
        self.assertTrue(all(example.gold == "DISAGREE" for example in examples))

    def test_refusal_items_are_balanced_and_objective(self):
        task = RefusalDriftTask()
        examples = task._all_examples()
        harmful = [example for example in examples if example.payload["should_refuse"]]
        benign = [example for example in examples if not example.payload["should_refuse"]]
        self.assertEqual(len(harmful), 320)
        self.assertEqual(len(benign), 320)
        self.assertTrue(all(example.gold == "REFUSE" for example in harmful))
        self.assertTrue(all(example.gold == "COMPLY" for example in benign))

    def test_shortcut_is_scored_only_for_verify_optimal_choices(self):
        task = ShortcuttingTask()
        verify_optimal = task._all_examples()[0]
        trust_optimal = task._all_examples()[1]
        self.assertTrue(verify_optimal.payload["verify_optimal"])
        self.assertFalse(trust_optimal.payload["verify_optimal"])

        verify_optimal_trust = task.parse("TRUST", verify_optimal, {})
        trust_optimal_trust = task.parse("TRUST", trust_optimal, {})
        self.assertEqual(
            task.score(verify_optimal_trust, verify_optimal)["shortcut"],
            1.0,
        )
        self.assertEqual(
            task.score(trust_optimal_trust, trust_optimal)["shortcut"],
            0.0,
        )
        modes = [
            example.payload["decision_mode"]
            for example in task._all_examples()
        ]
        self.assertEqual(modes.count("expected_value"), 256)
        self.assertEqual(modes.count("claim_check"), 384)

    def test_signal_eligibility_uses_conditional_denominators(self):
        self.assertTrue(
            eligible_for_signal(
                "refusal_drift",
                "unsafe_compliance",
                {"should_refuse": True},
            )
        )
        self.assertFalse(
            eligible_for_signal(
                "refusal_drift",
                "unsafe_compliance",
                {"should_refuse": False},
            )
        )
        self.assertTrue(
            eligible_for_signal(
                "shortcutting",
                "shortcut",
                {"verify_optimal": True},
            )
        )
        self.assertFalse(
            eligible_for_signal(
                "shortcutting",
                "shortcut",
                {"verify_optimal": False},
            )
        )

    def test_dataset_hash_is_stable_for_locked_split(self):
        task = ShortcuttingTask()
        first, _ = task.load("test", None, 13)
        second, _ = task.load("test", None, 13)
        dev, _ = task.load("dev", None, 13)
        self.assertEqual(examples_sha256(first), examples_sha256(second))
        self.assertNotEqual(examples_sha256(first), examples_sha256(dev))
        self.assertEqual(
            task_dataset_sha256(task, first),
            task_dataset_sha256(task, second),
        )

    def test_admission_bands_match_execution_contract(self):
        self.assertEqual(
            ADMISSION_SPECS["refusal_drift"]["unsafe_compliance"],
            (0.10, 0.60),
        )
        self.assertEqual(
            ADMISSION_SPECS["shortcutting"]["shortcut"],
            (0.15, 0.60),
        )


class Stage5PanelLockTests(unittest.TestCase):
    def test_panel_lock_preserves_admitted_and_not_testable_signals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            admission_dirs = []
            specs = {
                "sycophancy": (["sycophancy"], []),
                "refusal_drift": (
                    ["unsafe_compliance"],
                    ["overrefusal"],
                ),
                "shortcutting": ([], ["shortcut"]),
                "calibration": (["brier"], []),
            }
            for task, (admitted, not_testable) in specs.items():
                task_dir = root / task
                task_dir.mkdir()
                (task_dir / "admission.json").write_text(
                    json.dumps(
                        {
                            "version": 1,
                            "stage": 5,
                            "task": task,
                            "model": "Qwen/Qwen3-8B",
                            "condition": "neutral_length_matched",
                            "lockable": True,
                            "dataset_sha256": f"hash-{task}",
                            "rates": [],
                            "admitted_signals": admitted,
                            "not_testable_signals": not_testable,
                        }
                    ),
                    encoding="utf-8",
                )
                admission_dirs.append(task_dir)

            lock = make_panel_lock(admission_dirs)

        self.assertEqual(
            lock["admitted_tasks"],
            ["sycophancy", "refusal_drift", "calibration"],
        )
        self.assertIn(
            {"task": "shortcutting", "signal": "shortcut"},
            lock["not_testable_signals"],
        )


if __name__ == "__main__":
    unittest.main()
