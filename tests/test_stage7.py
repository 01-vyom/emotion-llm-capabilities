import unittest

from emotion_stage7.aggregate_results import (
    planned_accuracy_cells,
    planned_overrefusal_cells,
    planned_safety_cells,
)
from emotion_stage7.admission import task_requires_held_out_admission
from emotion_stage7.arms import ifeval_arms, safety_arms
from emotion_stage7.constants import IFEVAL_MAX_NEW_TOKENS
from emotion_stage7.make_task_lock import select_tier_mix
from emotion_stage7.run_campaign import _format_optional_rate
from emotion_stage7.tasks import RefusalDriftTask, SycophancyTask


class Stage7TaskTests(unittest.TestCase):
    def test_sycophancy_pool_is_tiered_and_powered(self):
        task = SycophancyTask()
        dev, _ = task.load("dev", None, 13)
        test, _ = task.load("test", None, 13)

        self.assertEqual(len(dev), 1200)
        self.assertEqual(len(test), 3600)
        self.assertTrue(all(example.gold == "DISAGREE" for example in dev + test))
        for tier in range(6):
            self.assertEqual(
                sum(
                    int(example.payload["difficulty_tier"]) == tier
                    for example in dev
                ),
                200,
            )
            self.assertEqual(
                sum(
                    int(example.payload["difficulty_tier"]) == tier
                    for example in test
                ),
                600,
            )

    def test_refusal_pool_is_unique_balanced_and_tiered(self):
        task = RefusalDriftTask()
        all_examples = task._all_examples()
        dev, _ = task.load("dev", None, 13)
        test, _ = task.load("test", None, 13)

        self.assertEqual(len(all_examples), 6400)
        self.assertEqual(len({example.item_id for example in all_examples}), 6400)
        self.assertEqual(
            len({example.payload["request"] for example in all_examples}),
            6400,
        )
        self.assertEqual(len(dev), 1600)
        self.assertEqual(len(test), 4800)
        for tier in range(5):
            for harmful in (True, False):
                self.assertEqual(
                    sum(
                        int(example.payload["difficulty_tier"]) == tier
                        and bool(example.payload["should_refuse"]) is harmful
                        for example in dev
                    ),
                    160,
                )
                self.assertEqual(
                    sum(
                        int(example.payload["difficulty_tier"]) == tier
                        and bool(example.payload["should_refuse"]) is harmful
                        for example in test
                    ),
                    480,
                )


class Stage7LockTests(unittest.TestCase):
    def test_mixture_targets_rate_on_fixed_grid(self):
        selected = select_tier_mix([0.0, 0.2, 0.4, 0.8])

        self.assertEqual(sum(selected["units"]), 20)
        self.assertAlmostEqual(sum(selected["weights"]), 1.0)
        self.assertAlmostEqual(selected["predicted_rate"], 0.30)

    def test_mixture_tie_break_prefers_higher_entropy(self):
        selected = select_tier_mix([0.2, 0.4])

        self.assertEqual(selected["units"], [10, 10])

    def test_held_out_admission_requires_a_calibrated_dev_signal(self):
        self.assertFalse(
            task_requires_held_out_admission(
                {"signals": {"sycophancy": {"dev_status": "not_testable_dev"}}}
            )
        )
        self.assertTrue(
            task_requires_held_out_admission(
                {
                    "signals": {
                        "unsafe_compliance": {"dev_status": "calibrated"},
                        "overrefusal": {"dev_status": "not_testable_dev"},
                    }
                }
            )
        )


class Stage7ArmTests(unittest.TestCase):
    def test_safety_and_ifeval_arm_contracts(self):
        self.assertEqual(len(safety_arms("sycophancy")), 8)
        self.assertEqual(len(safety_arms("refusal_drift")), 32)
        self.assertEqual(len(ifeval_arms()), 12)
        self.assertEqual(len(ifeval_arms(prompt_only=True)), 8)
        self.assertIn(
            "neutral_length_matched__negative_a1.0",
            {arm.name for arm in safety_arms("refusal_drift")},
        )
        self.assertEqual(IFEVAL_MAX_NEW_TOKENS, 2048)


class Stage7FamilyTests(unittest.TestCase):
    def test_fixed_family_denominators(self):
        models = ["8b", "14b", "32b", "frontier"]

        self.assertEqual(len(planned_safety_cells(models)), 11)
        self.assertEqual(len(planned_overrefusal_cells(models)), 7)
        self.assertEqual(len(planned_accuracy_cells(models)), 30)

    def test_manifest_formats_unevaluated_held_out_rate(self):
        self.assertEqual(_format_optional_rate(None), "not_evaluated")
        self.assertEqual(_format_optional_rate(0.3), "0.3")


if __name__ == "__main__":
    unittest.main()
