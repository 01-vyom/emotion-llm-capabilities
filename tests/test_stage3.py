import importlib.util
import unittest

import numpy as np

from emotion_stage1.affect import (
    EMOTION_CONCEPT_PAIRS,
    EMOTION_STORY_TOPICS,
    EMOTION_VECTOR_CONCEPTS,
    iter_matched_emotion_story_prompts,
)
from emotion_stage3.compare_backends import compare_backend_runs
from emotion_stage3.io import write_jsonl
from emotion_stage3.probe import (
    AFFECT_MASK,
    _tokens,
    balanced_accuracy,
    gate_decision,
    lexical_scores,
    roc_auc,
)
from emotion_stage3.run_stage3 import build_arms, load_steering_spec
from emotion_stage3.vectors import (
    COMBINED_PAIR_NAME,
    PAIR_NAMES,
    build_vector_arrays,
    default_layer_grid,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class VectorExtractionTests(unittest.TestCase):
    def test_default_layer_grid_includes_final_layer(self):
        self.assertEqual(default_layer_grid(10, stride=4), (3, 7, 9))

    def test_vector_arrays_include_pairs_and_combined_direction(self):
        rows = []
        emotions = []
        replicates = []
        for emotion_index in range(len(EMOTION_VECTOR_CONCEPTS)):
            for replicate in range(3):
                value = np.zeros((2, 8), dtype=np.float32)
                value[..., emotion_index % 8] = 1 + emotion_index
                rows.append(value)
                emotions.append(emotion_index)
                replicates.append(replicate)
        arrays = build_vector_arrays(
            np.asarray(rows),
            np.asarray(emotions),
            np.asarray(replicates),
            train_replicates=2,
            seed=13,
        )

        expected_pairs = len(PAIR_NAMES) + 1
        self.assertEqual(
            arrays["unit"].shape,
            (expected_pairs, 2, 8),
        )
        np.testing.assert_allclose(
            np.linalg.norm(arrays["unit"], axis=-1),
            1.0,
            atol=1e-6,
        )
        self.assertEqual(COMBINED_PAIR_NAME, "diligence")


class ProbeMetricTests(unittest.TestCase):
    def test_roc_auc_handles_ties(self):
        scores = np.asarray([0.0, 0.0, 1.0, 1.0])
        labels = np.asarray([False, True, False, True])

        self.assertEqual(roc_auc(scores, labels), 0.5)

    def test_balanced_accuracy_uses_training_orientation(self):
        train_scores = np.asarray([-2.0, -1.0, 1.0, 2.0])
        train_labels = np.asarray([False, False, True, True])
        validation_scores = np.asarray([-3.0, 3.0])
        validation_labels = np.asarray([False, True])

        self.assertEqual(
            balanced_accuracy(
                train_scores,
                train_labels,
                validation_scores,
                validation_labels,
            ),
            1.0,
        )

    def test_lexical_control_masks_exact_concept_words(self):
        scores = lexical_scores(
            ["calm blue", "desperate red"],
            np.asarray([True, False]),
            ["calm blue", "desperate red"],
        )

        self.assertGreater(scores[0], scores[1])

    def test_lexicon_mask_removes_affect_synonyms(self):
        tokens = _tokens("no rush, quiet assurance")

        self.assertTrue({"rush", "quiet", "assurance"} <= AFFECT_MASK)
        self.assertTrue(
            {"rush", "quiet", "assurance"}.isdisjoint(tokens)
        )

    def test_lexical_control_degrades_when_only_signal_is_masked(self):
        labels = np.asarray([True, True, False, False])
        scores = lexical_scores(
            [
                "steady quiet",
                "relaxed assurance",
                "frantic panic",
                "urgent hopeless",
            ],
            labels,
            [
                "steady assurance",
                "quiet relaxed",
                "panic urgent",
                "frantic hopeless",
            ],
        )

        self.assertEqual(roc_auc(scores, labels), 0.5)

    def test_saturated_lexical_control_abstains_without_clean_pass(self):
        decision = gate_decision(
            direction_auc=0.99,
            lexical_auc=1.0,
            shuffled_control=0.6,
            random_control=0.55,
            min_auc=0.75,
            gate_margin=0.02,
            lexical_cap=0.90,
        )

        self.assertEqual(decision["lexical_verdict"], "abstain")
        self.assertTrue(decision["pass_with_lexical_abstain"])
        self.assertFalse(decision["passes_gate"])


class MatchedStimulusTests(unittest.TestCase):
    def test_matched_pairs_share_topic_replicate_and_cover_both_poles(self):
        rows = list(iter_matched_emotion_story_prompts(replicates=2))
        grouped = {}
        for row in rows:
            grouped.setdefault(row["pair_id"], []).append(row)

        expected_pairs = (
            len(EMOTION_CONCEPT_PAIRS)
            * len(EMOTION_STORY_TOPICS)
            * 2
        )
        self.assertEqual(len(grouped), expected_pairs)
        self.assertEqual(len(rows), expected_pairs * 2)
        for pair_rows in grouped.values():
            self.assertEqual(
                {row["pole"] for row in pair_rows},
                {"positive", "negative"},
            )
            self.assertEqual(len({row["topic"] for row in pair_rows}), 1)
            self.assertEqual(
                len({row["replicate"] for row in pair_rows}),
                1,
            )


class Stage3DesignTests(unittest.TestCase):
    def test_confirmatory_design_has_crossed_controls(self):
        arms, baselines = build_arms("confirmatory", alpha=0.05)

        self.assertEqual(len(arms), 7)
        self.assertEqual(baselines, ["no_steer", "random_a+0.0500"])
        self.assertEqual(
            {arm.name for arm in arms},
            {
                "no_steer",
                "positive_a+0.0500",
                "negative_a-0.0500",
                "random_a+0.0500",
                "shuffled_a+0.0500",
                "affect_opposite_a-0.0500",
                "affect_matching_a+0.0500",
            },
        )

    def test_load_spec_uses_layer_specific_vector_and_norm(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.savez(
                path / "vectors.npz",
                pair_names=np.asarray(["diligence"]),
                layers=np.asarray([3, 7]),
                mean_unit=np.asarray(
                    [[
                        [1.0, 0.0],
                        [0.0, 1.0],
                    ]],
                    dtype=np.float32,
                ),
                residual_norm=np.asarray([2.0, 4.0]),
            )

            spec = load_steering_spec(
                path,
                pair_name="diligence",
                policy="mean",
                layers=(7,),
                vector_kind="unit",
                alpha=0.1,
            )

        self.assertEqual(spec.layers, (7,))
        self.assertEqual(spec.residual_norms, {7: 4.0})
        np.testing.assert_array_equal(spec.vectors[7], [0.0, 1.0])


class BackendParityTests(unittest.TestCase):
    def test_matching_metrics_pass_when_raw_outputs_differ(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transformers = root / "transformers"
            vllm = root / "vllm"
            transformers.mkdir()
            vllm.mkdir()
            common = {
                "item_id": 7,
                "source_prompt": "same prompt",
                "strict_prompt": 1.0,
                "parse_success": 1.0,
                "output_truncated": 0.0,
                "output_tokens": 10,
            }
            write_jsonl(
                transformers / "predictions.jsonl",
                [{**common, "raw_output": "transformers output"}],
            )
            write_jsonl(
                vllm / "predictions.jsonl",
                [{**common, "raw_output": "vllm output"}],
            )

            report, rows = compare_backend_runs(transformers, vllm)

        self.assertTrue(report["parity_passed"])
        self.assertEqual(report["exact_output_rate"], 0.0)
        self.assertEqual(rows[0]["strict_prompt_match"], True)

    def test_mismatched_item_ids_are_rejected(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transformers = root / "transformers"
            vllm = root / "vllm"
            transformers.mkdir()
            vllm.mkdir()
            row = {
                "source_prompt": "prompt",
                "raw_output": "output",
                "strict_prompt": 1.0,
                "parse_success": 1.0,
                "output_truncated": 0.0,
                "output_tokens": 1,
            }
            write_jsonl(
                transformers / "predictions.jsonl",
                [{**row, "item_id": 1}],
            )
            write_jsonl(
                vllm / "predictions.jsonl",
                [{**row, "item_id": 2}],
            )

            with self.assertRaisesRegex(ValueError, "item IDs differ"):
                compare_backend_runs(transformers, vllm)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class SteeringHookTests(unittest.TestCase):
    def test_hook_scales_delta_and_cleans_up(self):
        import torch

        from emotion_stage3.steering import SteeringController, SteeringSpec

        class IdentityLayer(torch.nn.Module):
            def forward(self, hidden):
                return hidden

        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([IdentityLayer()])

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Decoder()

        model = Model()
        hidden = torch.zeros((1, 3, 2))
        spec = SteeringSpec(
            vectors={0: np.asarray([1.0, 0.0], dtype=np.float32)},
            residual_norms={0: 4.0},
            alpha=0.5,
        )

        with SteeringController(model, spec):
            steered = model.model.layers[0](hidden)
        unsteered = model.model.layers[0](hidden)

        torch.testing.assert_close(
            steered,
            torch.tensor([[[0.0, 0.0], [0.0, 0.0], [2.0, 0.0]]]),
        )
        torch.testing.assert_close(unsteered, hidden)

    def test_tuple_output_structure_is_preserved(self):
        import torch

        from emotion_stage3.steering import add_steering_to_output

        hidden = torch.zeros((1, 2, 2))
        cache = object()
        output = add_steering_to_output(
            (hidden, cache),
            torch.tensor([0.0, 1.0]),
        )

        self.assertIs(output[1], cache)
        torch.testing.assert_close(
            output[0],
            torch.tensor([[[0.0, 0.0], [0.0, 1.0]]]),
        )

    def test_alpha_zero_is_identity(self):
        import torch

        from emotion_stage3.steering import add_steering_to_output

        hidden = torch.randn((2, 4, 3))
        output = add_steering_to_output(hidden, torch.zeros(3))

        torch.testing.assert_close(output, hidden)


if __name__ == "__main__":
    unittest.main()
