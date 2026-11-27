import unittest

from emotion_stage1.banking77 import parse_prediction
from emotion_stage1.metrics import macro_f1, paired_accuracy_deltas


class ParsingTests(unittest.TestCase):
    def test_parses_label_prefix(self):
        label, method = parse_prediction(
            "Intent: card_arrival",
            ["activate_my_card", "card_arrival"],
        )

        self.assertEqual(label, "card_arrival")
        self.assertEqual(method, "first_line_exact")


class MetricsTests(unittest.TestCase):
    def test_macro_f1_uses_explicit_label_space(self):
        rows = [
            {
                "gold_label": "a",
                "pred_label": "a",
            }
        ]

        self.assertEqual(macro_f1(rows, labels=["a", "b"]), 0.5)

    def test_paired_accuracy_deltas_preserve_item_pairing(self):
        rows = []
        correctness = {
            "neutral_empty": [True, True, False],
            "neutral_length_matched": [True, False, False],
            "positive_pride": [True, True, True],
        }
        for condition, values in correctness.items():
            for item_id, correct in enumerate(values):
                rows.append(
                    {
                        "condition": condition,
                        "item_id": item_id,
                        "correct": correct,
                    }
                )

        deltas = paired_accuracy_deltas(
            rows,
            baselines=("neutral_empty", "neutral_length_matched"),
            reps=100,
        )

        self.assertEqual(len(deltas), 2)
        by_baseline = {delta.baseline: delta for delta in deltas}
        self.assertAlmostEqual(
            by_baseline["neutral_empty"].accuracy_delta,
            1 / 3,
        )
        self.assertEqual(by_baseline["neutral_empty"].wins, 1)
        self.assertEqual(by_baseline["neutral_empty"].losses, 0)
        self.assertAlmostEqual(
            by_baseline["neutral_length_matched"].accuracy_delta,
            2 / 3,
        )

    def test_paired_accuracy_deltas_reject_missing_items(self):
        rows = [
            {"condition": "neutral_empty", "item_id": 0, "correct": True},
            {"condition": "positive_pride", "item_id": 1, "correct": True},
        ]

        with self.assertRaisesRegex(ValueError, "identical item IDs"):
            paired_accuracy_deltas(
                rows,
                baselines=("neutral_empty",),
                reps=10,
            )


if __name__ == "__main__":
    unittest.main()
