import csv
import json
import os
import re
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from emotion_stage1.affect import build_affect_block, get_condition_cues
from emotion_stage1.generation import GenerationConfig, build_vllm_engine_kwargs
from emotion_stage2.aggregate_results import _write_csv
from emotion_stage2.metrics import holm_adjust, paired_deltas, summarize_signals
from emotion_stage2.run_stage2 import (
    _load_prompt_lock,
    _validate_rendered_prompt_lengths,
)
from emotion_stage2.select_prompt_variants import select_variants
from emotion_stage2.tasks.base import Example
from emotion_stage2.tasks.ruler_niah import RulerNIAHTask

try:
    import absl  # noqa: F401
    import immutabledict  # noqa: F401
    import langdetect  # noqa: F401
    import nltk  # noqa: F401

    IFEVAL_DEPS_AVAILABLE = True
except ModuleNotFoundError:
    IFEVAL_DEPS_AVAILABLE = False


class AffectVariantTests(unittest.TestCase):
    def test_variant_zero_preserves_canonical_cue(self):
        block = build_affect_block("positive_pride")

        self.assertIn(get_condition_cues("positive_pride")[0], block)

    def test_unknown_variant_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot use index"):
            build_affect_block("neutral_empty", variant_index=1)


class GenericMetricsTests(unittest.TestCase):
    def test_summarizes_numeric_and_binary_signals(self):
        rows = [
            {"condition": "a", "correct": 1, "output_tokens": 4},
            {"condition": "a", "correct": 0, "output_tokens": 8},
        ]

        summary = summarize_signals(
            rows,
            signal_keys=("correct", "output_tokens"),
            reps=20,
        )[0]

        self.assertEqual(summary["correct_mean"], 0.5)
        self.assertEqual(summary["output_tokens_mean"], 6.0)

    def test_paired_deltas_and_holm_adjustment(self):
        rows = []
        values = {
            "neutral_empty": [0, 0, 0, 1],
            "positive_pride": [1, 1, 0, 1],
        }
        for condition, condition_values in values.items():
            for item_id, value in enumerate(condition_values):
                rows.append(
                    {
                        "condition": condition,
                        "item_id": item_id,
                        "correct": value,
                    }
                )

        results = paired_deltas(
            rows,
            signal_keys=("correct",),
            baselines=("neutral_empty",),
            primary_signals=("correct",),
            reps=50,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].wins, 2)
        self.assertEqual(results[0].losses, 0)
        self.assertEqual(results[0].delta, 0.5)
        self.assertIsNotNone(results[0].p_value_holm)
        self.assertEqual(holm_adjust([0.01, 0.04]), [0.02, 0.04])


@unittest.skipUnless(IFEVAL_DEPS_AVAILABLE, "IFEval verifier dependencies missing")
class IFEvalAdapterTests(unittest.TestCase):
    def test_official_verifier_scores_fixture(self):
        from emotion_stage2.tasks.ifeval import IFEvalTask

        task = IFEvalTask()
        examples, context = task.load_fixture()
        parsed = task.parse(
            "A cedar stands quietly.",
            examples[0],
            context,
        )
        score = task.score(parsed, examples[0])

        self.assertEqual(score["strict_prompt"], 1.0)
        self.assertEqual(score["loose_prompt"], 1.0)
        self.assertEqual(score["refusal"], 0.0)

    def test_verifier_exception_scores_instruction_as_failed(self):
        from emotion_stage2.tasks.ifeval import IFEvalTask

        task = IFEvalTask()
        examples, context = task.load_fixture()
        with mock.patch(
            "instruction_following_eval.evaluation_lib."
            "test_instruction_following_strict",
            side_effect=RecursionError("malformed nested JSON"),
        ):
            parsed = task.parse(
                "A cedar stands quietly.",
                examples[0],
                context,
            )
        score = task.score(parsed, examples[0])

        self.assertEqual(score["strict_prompt"], 0.0)
        self.assertEqual(score["loose_prompt"], 1.0)
        self.assertEqual(
            parsed.details["verifier_errors"],
            {"strict": "RecursionError"},
        )


class IFEvalConfigurationTests(unittest.TestCase):
    def test_generation_cap_covers_long_form_constraints(self):
        from emotion_stage2.tasks.ifeval import IFEvalTask

        self.assertEqual(IFEvalTask().max_new_tokens(), 4096)


class GenerationConfigurationTests(unittest.TestCase):
    def test_vllm_kwargs_include_yarn_and_capacity_controls(self):
        rope_scaling = {
            "rope_type": "yarn",
            "factor": 3.2,
            "original_max_position_embeddings": 40960,
        }

        kwargs = build_vllm_engine_kwargs(
            GenerationConfig(
                model="Qwen/Qwen3-8B",
                max_model_len=131072,
                gpu_memory_utilization=0.95,
                enforce_eager=True,
                tensor_parallel_size=2,
                rope_scaling=rope_scaling,
            )
        )

        self.assertEqual(kwargs["max_model_len"], 131072)
        self.assertEqual(kwargs["gpu_memory_utilization"], 0.95)
        self.assertTrue(kwargs["enforce_eager"])
        self.assertEqual(kwargs["tensor_parallel_size"], 2)
        self.assertEqual(kwargs["hf_overrides"]["rope_scaling"], rope_scaling)
        self.assertEqual(kwargs["hf_overrides"]["rope_parameters"], rope_scaling)

    def test_vllm_long_window_opt_in_is_limited_to_rope_scaled_runs(self):
        from unittest.mock import patch

        from emotion_stage1.generation import VLLMGenerator

        fake_vllm = type(
            "FakeVLLM",
            (),
            {
                "LLM": lambda **kwargs: kwargs,
                "SamplingParams": lambda **kwargs: kwargs,
            },
        )
        rope_scaling = {
            "rope_type": "yarn",
            "factor": 3.2,
            "original_max_position_embeddings": 40960,
        }
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.dict("sys.modules", {"vllm": fake_vllm}),
        ):
            VLLMGenerator(
                GenerationConfig(
                    model="model",
                    max_model_len=131072,
                    rope_scaling=rope_scaling,
                )
            )
            self.assertEqual(
                os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"],
                "1",
            )


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(self, tokens, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(token) for token in tokens)


class RulerAdapterTests(unittest.TestCase):
    def test_scores_exact_and_hallucinated_codes(self):
        task = RulerNIAHTask(model="unused")
        examples, context = task.load_fixture()
        exact = task.parse("NIAH-AB12-CD34", examples[0], context)
        wrong = task.parse("NIAH-FFFF-0000", examples[0], context)
        wrong_prefix = task.parse("NIAC-AB12-CD34", examples[0], context)
        extra_prose = task.parse(
            "The code is NIAH-AB12-CD34.",
            examples[0],
            context,
        )

        self.assertEqual(task.score(exact, examples[0])["correct"], 1.0)
        wrong_score = task.score(wrong, examples[0])
        self.assertEqual(wrong_score["correct"], 0.0)
        self.assertEqual(wrong_score["format_valid_wrong"], 1.0)
        self.assertEqual(wrong_score["hallucination"], 1.0)
        self.assertEqual(wrong_score["omission"], 1.0)
        wrong_prefix_score = task.score(wrong_prefix, examples[0])
        self.assertEqual(wrong_prefix_score["correct"], 0.0)
        self.assertEqual(wrong_prefix_score["format_valid_wrong"], 1.0)
        self.assertEqual(wrong_prefix_score["hallucination"], 1.0)
        self.assertEqual(wrong_prefix_score["omission"], 1.0)
        extra_score = task.score(extra_prose, examples[0])
        self.assertEqual(extra_score["correct"], 0.0)
        self.assertEqual(extra_score["retrieval_rate"], 1.0)
        self.assertEqual(extra_score["format_valid_wrong"], 0.0)

    def test_multi_code_order_is_part_of_exact_correctness(self):
        task = RulerNIAHTask(model="unused")
        example = Example(
            item_id=1,
            payload={},
            gold=["NIAH-AAAA-1111", "NIAH-BBBB-2222"],
        )
        parsed = task.parse(
            "NIAH-BBBB-2222 | NIAH-AAAA-1111",
            example,
            {},
        )

        score = task.score(parsed, example)

        self.assertEqual(score["correct"], 0.0)
        self.assertEqual(score["retrieval_rate"], 1.0)
        self.assertEqual(score["format_valid_wrong"], 1.0)

    def test_hard_multi_uses_desalient_targets_decoys_and_heterogeneous_filler(self):
        task = RulerNIAHTask(
            model="unused",
            context_lengths=(8192,),
            depths=(0.5,),
            modes=("multi",),
            hard=True,
            distractor_count=4,
            multi_key_count=6,
            context_reserve_tokens=256,
        )
        task._tokenizer = CharacterTokenizer()

        examples, _ = task.load(split="dev", limit=None, seed=13)
        example = examples[0]
        context = example.payload["context"]
        found_codes = re.findall(
            r"[A-Z]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}",
            context,
        )

        self.assertEqual(len(example.gold), 6)
        self.assertEqual(len(found_codes), 10)
        self.assertNotIn("target-", context)
        self.assertNotIn("decoy-", context)
        self.assertIn("Ledger item", context)
        self.assertIn("Daily note", context)
        for target in example.gold:
            self.assertEqual(context.count(target), 1)

    def test_hard_depth_changes_position_not_target_identity(self):
        task = RulerNIAHTask(
            model="unused",
            context_lengths=(8192,),
            depths=(0.1, 0.9),
            modes=("multi",),
            hard=True,
            distractor_count=4,
            multi_key_count=5,
            aggregate_match_count=3,
            context_reserve_tokens=256,
        )
        task._tokenizer = CharacterTokenizer()

        examples, _ = task.load(split="dev", limit=None, seed=13)

        self.assertEqual(examples[0].gold, examples[1].gold)
        self.assertEqual(
            examples[0].payload["cases"],
            examples[1].payload["cases"],
        )
        self.assertNotEqual(
            examples[0].payload["context"],
            examples[1].payload["context"],
        )

    def test_hard_difficulty_knobs_do_not_change_target_identity(self):
        easier = RulerNIAHTask(
            model="unused",
            context_lengths=(8192,),
            depths=(0.5,),
            modes=("multi",),
            hard=True,
            distractor_count=3,
            multi_key_count=5,
            aggregate_match_count=3,
            context_reserve_tokens=256,
        )
        harder = RulerNIAHTask(
            model="unused",
            context_lengths=(8192,),
            depths=(0.5,),
            modes=("multi",),
            hard=True,
            distractor_count=6,
            multi_key_count=5,
            aggregate_match_count=3,
            context_reserve_tokens=256,
        )
        easier._tokenizer = CharacterTokenizer()
        harder._tokenizer = CharacterTokenizer()

        easier_examples, _ = easier.load(split="dev", limit=None, seed=13)
        harder_examples, _ = harder.load(split="dev", limit=None, seed=13)

        self.assertEqual(easier_examples[0].gold, harder_examples[0].gold)
        self.assertEqual(
            easier_examples[0].payload["cases"],
            harder_examples[0].payload["cases"],
        )

    def test_hard_generation_cap_fits_multi_code_answer(self):
        task = RulerNIAHTask(model="unused", hard=True)

        self.assertEqual(task.max_new_tokens(), 128)

    def test_aggregate_mode_counts_exact_integer_only(self):
        task = RulerNIAHTask(
            model="unused",
            modes=("aggregate",),
            hard=True,
        )
        example = Example(
            item_id=1,
            payload={"mode": "aggregate"},
            gold=6,
        )

        exact = task.score(task.parse("6", example, {}), example)
        prose = task.score(task.parse("There are 6.", example, {}), example)
        wrong = task.score(task.parse("5", example, {}), example)

        self.assertEqual(exact["correct"], 1.0)
        self.assertEqual(prose["correct"], 0.0)
        self.assertEqual(prose["parse_success"], 1.0)
        self.assertEqual(wrong["format_valid_wrong"], 1.0)
        self.assertEqual(wrong["hallucination"], 1.0)


class Stage2RunGuardTests(unittest.TestCase):
    def test_rejects_task_configuration_drift_from_prompt_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "lock.json"
            lock_path.write_text(
                json.dumps(
                    {
                        "task": "ruler_niah",
                        "model": "model",
                        "selected_variants": {"neutral_empty": 0},
                        "task_configuration": {
                            "context_lengths": [32768],
                            "hard": True,
                        },
                    }
                )
            )

            with self.assertRaisesRegex(SystemExit, "does not match"):
                _load_prompt_lock(
                    lock_path,
                    task_name="ruler_niah",
                    model="model",
                    task_configuration={
                        "context_lengths": [65536],
                        "hard": True,
                    },
                )

    def test_old_prompt_lock_without_task_configuration_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "lock.json"
            lock_path.write_text(
                json.dumps(
                    {
                        "task": "ruler_niah",
                        "model": "model",
                        "selected_variants": {"neutral_empty": 0},
                    }
                )
            )

            selected, _ = _load_prompt_lock(
                lock_path,
                task_name="ruler_niah",
                model="model",
                task_configuration={"context_lengths": [131072]},
            )

        self.assertEqual(selected, {"neutral_empty": 0})

    def test_rejects_rendered_prompt_that_exceeds_model_window(self):
        with self.assertRaisesRegex(SystemExit, "exceeds"):
            _validate_rendered_prompt_lengths(
                [130000, 131010],
                max_new_tokens=64,
                max_model_len=131072,
            )

        self.assertEqual(
            _validate_rendered_prompt_lengths(
                [130000, 131000],
                max_new_tokens=64,
                max_model_len=131072,
            ),
            131000,
        )


class PromptSelectionTests(unittest.TestCase):
    def test_selects_best_dev_variant_and_breaks_ties_by_index(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            metadata = {
                "split": "dev",
                "all_variants": True,
                "task": "ifeval",
                "model": "model",
                "seed": 13,
                "conditions": ["neutral_empty", "positive_pride"],
                "primary_hypothesis_signals": ["strict_prompt"],
            }
            (path / "metadata.json").write_text(json.dumps(metadata))
            with (path / "summary.csv").open("w", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["condition", "strict_prompt_mean"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "condition": "neutral_empty::v0",
                            "strict_prompt_mean": 0.4,
                        },
                        {
                            "condition": "positive_pride::v0",
                            "strict_prompt_mean": 0.5,
                        },
                        {
                            "condition": "positive_pride::v1",
                            "strict_prompt_mean": 0.6,
                        },
                        {
                            "condition": "positive_pride::v2",
                            "strict_prompt_mean": 0.6,
                        },
                    ]
                )

            lock = select_variants(path)

        self.assertEqual(lock["selected_variants"]["neutral_empty"], 0)
        self.assertEqual(lock["selected_variants"]["positive_pride"], 1)
        self.assertEqual(lock["version"], 2)


class AggregateWriterTests(unittest.TestCase):
    def test_csv_header_unions_heterogeneous_task_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "combined.csv"

            _write_csv(
                path,
                [
                    {"task": "ifeval", "strict_prompt_mean": 0.8},
                    {"task": "ruler_niah", "retrieval_rate_mean": 1.0},
                ],
            )

            with path.open(newline="") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(
            set(rows[0]),
            {"task", "strict_prompt_mean", "retrieval_rate_mean"},
        )
        self.assertEqual(rows[1]["retrieval_rate_mean"], "1.0")


if __name__ == "__main__":
    unittest.main()
