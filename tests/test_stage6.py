import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from emotion_stage6.analysis import joint_three_control
from emotion_stage6.make_panel_lock import make_panel_lock
from emotion_stage6.make_steering_lock import make_steering_lock
from emotion_stage6.run_stage6 import DOSE_GRID, build_stage6_arms
from emotion_stage6.tasks import RefusalDriftTask, SycophancyTask


class Stage6TaskTests(unittest.TestCase):
    def test_fixed_test_sizes_and_conditional_power(self):
        sycophancy, _ = SycophancyTask().load("test", None, 13)
        refusal, _ = RefusalDriftTask().load("test", None, 13)
        self.assertEqual(len(sycophancy), 600)
        self.assertEqual(len(refusal), 960)
        self.assertEqual(
            sum(bool(example.payload["should_refuse"]) for example in refusal),
            480,
        )
        self.assertEqual(
            sum(not bool(example.payload["should_refuse"]) for example in refusal),
            480,
        )

    def test_refusal_surface_has_no_exact_duplicate_requests(self):
        requests = [
            example.payload["request"]
            for example in RefusalDriftTask()._all_examples()
        ]
        self.assertEqual(len(requests), len(set(requests)))

    def test_arm_contract(self):
        self.assertEqual(len(build_stage6_arms("sycophancy")), 8)
        refusal = build_stage6_arms("refusal_drift")
        self.assertEqual(len(refusal), 32)
        self.assertEqual(len(DOSE_GRID), 6)
        self.assertIn(
            "neutral_length_matched__negative_a1.0",
            {arm.name for arm in refusal},
        )
        self.assertEqual(
            len(build_stage6_arms("refusal_drift", prompt_only=True)),
            8,
        )


class Stage6LockTests(unittest.TestCase):
    def test_panel_lock_is_model_specific(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dirs = []
            for task, admitted in (
                ("sycophancy", ["sycophancy"]),
                ("refusal_drift", ["unsafe_compliance"]),
            ):
                path = root / task
                path.mkdir()
                (path / "admission.json").write_text(
                    json.dumps(
                        {
                            "version": 1,
                            "stage": 6,
                            "task": task,
                            "model": "Qwen/Qwen3-14B",
                            "condition": "neutral_length_matched",
                            "lockable": True,
                            "dataset_sha256": f"hash-{task}",
                            "rates": [],
                            "admitted_signals": admitted,
                            "not_testable_signals": [],
                        }
                    ),
                    encoding="utf-8",
                )
                dirs.append(path)
            lock = make_panel_lock(dirs)
        self.assertEqual(lock["model"], "Qwen/Qwen3-14B")
        self.assertEqual(len(lock["admitted_signals"]), 2)

    def test_fixed_steering_lock_uses_exact_row(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            probe = root / "probe"
            vectors = root / "vectors"
            probe.mkdir()
            vectors.mkdir()
            (probe / "probe_report.json").write_text(
                json.dumps(
                    {
                        "model": "Qwen/Qwen3-14B",
                        "best_by_pair_and_policy": [
                            {
                                "pair": "diligence",
                                "policy": "last",
                                "layer": 27,
                                "passes_gate": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            np.savez(vectors / "vectors.npz", value=np.asarray([1.0]))
            (vectors / "extraction_meta.json").write_text(
                json.dumps(
                    {
                        "model": "Qwen/Qwen3-14B",
                        "layers": [27],
                    }
                ),
                encoding="utf-8",
            )
            lock = make_steering_lock(
                probe,
                vectors,
                direction="diligence",
                policy="last",
                layer=27,
                alpha=0.5,
            )
        self.assertEqual(lock["layers"], [27])
        self.assertEqual(lock["policy"], "last")


class Stage6AnalysisTests(unittest.TestCase):
    def test_joint_test_uses_worst_component_and_direction(self):
        rows = []
        for item_id in range(12):
            values = {
                "treatment": 1.0,
                "none": 0.0,
                "random": 0.0,
                "shuffled": 0.0 if item_id < 7 else 1.0,
            }
            for condition, value in values.items():
                rows.append(
                    {
                        "condition": condition,
                        "item_id": item_id,
                        "signal": value,
                    }
                )
        joint, components = joint_three_control(
            rows,
            condition="treatment",
            baselines=("none", "random", "shuffled"),
            signal="signal",
            seed=13,
            reps=100,
        )
        self.assertEqual(
            joint["joint_p_value"],
            max(float(row["p_value"]) for row in components),
        )
        self.assertTrue(joint["direction_consistent"])


if __name__ == "__main__":
    unittest.main()
