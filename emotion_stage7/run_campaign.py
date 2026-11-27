"""Run, validate, score, aggregate, and manifest the Stage 7 campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .constants import (
    FRONTIER_LOCK,
    FRONTIER_SLUG,
    IFEVAL_RUNTIME_AMENDMENT_SHA256,
    OPEN_MODELS,
    PREREGISTRATION_SHA256,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = REPO_ROOT / "runs"
CONDA = Path("/home/ec2-user/miniforge3/bin/conda")
CONDA_ENV = "emotion-stage1"


@dataclass(frozen=True)
class Job:
    name: str
    module: str
    args: tuple[str, ...]
    completion_paths: tuple[Path, ...]
    cuda_visible_devices: str | None


def _python_module(module: str, *args: str) -> list[str]:
    return [
        str(CONDA),
        "run",
        "--no-capture-output",
        "-n",
        CONDA_ENV,
        "python",
        "-m",
        module,
        *args,
    ]


def _environment(cuda_visible_devices: str | None) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "HF_HOME": "/home/ec2-user/code/personal/hf_cache",
            "HF_HUB_CACHE": "/home/ec2-user/code/personal/hf_cache/hub",
            "HF_DATASETS_CACHE": (
                "/home/ec2-user/code/personal/hf_cache/datasets"
            ),
            "TOKENIZERS_PARALLELISM": "false",
            "MPLBACKEND": "Agg",
        }
    )
    if cuda_visible_devices is None:
        environment.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return environment


def _write_status(campaign_dir: Path, payload: dict) -> None:
    path = campaign_dir / "status.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _job_complete(job: Job) -> bool:
    return all(path.exists() for path in job.completion_paths)


def _run_lanes(
    lanes: dict[str, list[Job]],
    *,
    campaign_dir: Path,
    phase: str,
    max_attempts: int = 2,
) -> None:
    positions = {name: 0 for name in lanes}
    attempts = {
        job.name: 0 for jobs in lanes.values() for job in jobs
    }
    running: dict[str, tuple[Job, subprocess.Popen, object]] = {}
    failures = {}
    while True:
        for lane, jobs in lanes.items():
            if lane in running:
                continue
            while (
                positions[lane] < len(jobs)
                and _job_complete(jobs[positions[lane]])
            ):
                positions[lane] += 1
            if positions[lane] >= len(jobs):
                continue
            job = jobs[positions[lane]]
            attempts[job.name] += 1
            log_path = campaign_dir / f"{phase}_{job.name}.log"
            log = log_path.open("a", encoding="utf-8")
            log.write(
                f"\n=== attempt {attempts[job.name]} "
                f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===\n"
            )
            log.flush()
            process = subprocess.Popen(
                _python_module(job.module, *job.args),
                cwd=REPO_ROOT,
                env=_environment(job.cuda_visible_devices),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            running[lane] = (job, process, log)
        complete = all(
            positions[lane] >= len(jobs)
            for lane, jobs in lanes.items()
        )
        if complete and not running:
            return
        _write_status(
            campaign_dir,
            {
                "phase": phase,
                "updated_at_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ",
                    time.gmtime(),
                ),
                "lanes": {
                    lane: {
                        "position": positions[lane],
                        "total": len(jobs),
                    }
                    for lane, jobs in lanes.items()
                },
                "running": {
                    lane: {
                        "job": job.name,
                        "pid": process.pid,
                        "attempt": attempts[job.name],
                    }
                    for lane, (job, process, _log) in running.items()
                },
                "failures": failures,
            },
        )
        time.sleep(15)
        for lane, (job, process, log) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log.write(f"\n=== return code {return_code} ===\n")
            log.close()
            del running[lane]
            if return_code == 0 and _job_complete(job):
                positions[lane] += 1
            elif attempts[job.name] >= max_attempts:
                failures[job.name] = {
                    "return_code": return_code,
                    "attempts": attempts[job.name],
                    "completion_paths": [
                        str(path) for path in job.completion_paths
                    ],
                }
        if failures:
            for _lane, (_job, process, log) in running.items():
                process.terminate()
                log.close()
            _write_status(
                campaign_dir,
                {
                    "phase": phase,
                    "updated_at_utc": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ",
                        time.gmtime(),
                    ),
                    "running": {},
                    "failures": failures,
                },
            )
            raise RuntimeError(f"{phase} jobs failed: {failures}")


def _slug_paths(slug: str) -> dict[str, Path]:
    return {
        "calibration": RUNS / f"stage7_calibration_{slug}_dev_v2",
        "sycophancy_lock": (
            RUNS / f"stage7_{slug}_sycophancy_difficulty_lock_v2.json"
        ),
        "refusal_lock": (
            RUNS / f"stage7_{slug}_refusal_difficulty_lock_v2.json"
        ),
        "admission": RUNS / f"stage7_admission_{slug}_test_v2",
        "panel": RUNS / f"stage7_{slug}_panel_lock_v2.json",
        "sycophancy_run": RUNS / f"stage7_sycophancy_{slug}_test_v2",
        "refusal_run": RUNS / f"stage7_refusal_{slug}_test_v2",
        "ifeval_run": RUNS / f"stage7_ifeval_{slug}_test_v3",
    }


def _calibration_lanes() -> dict[str, list[Job]]:
    lanes = {}
    gpu_assignments = {
        "qwen3_8b": "0",
        "qwen3_14b": "1",
        "qwen3_32b": "2,3",
    }
    for slug, config in OPEN_MODELS.items():
        paths = _slug_paths(slug)
        args = [
            "--engine",
            "transformers",
            "--model",
            str(config["model"]),
            "--batch-size",
            str(config["batch_size"]),
            "--resume",
            "--output-dir",
            str(paths["calibration"]),
        ]
        if config["device_map"]:
            args.extend(["--device-map", str(config["device_map"])])
        lanes[slug] = [
            Job(
                name=f"calibration_{slug}",
                module="emotion_stage7.calibrate",
                args=tuple(args),
                completion_paths=(
                    paths["calibration"] / "validation.json",
                    paths["calibration"] / "predictions.jsonl",
                ),
                cuda_visible_devices=gpu_assignments[slug],
            )
        ]
    frontier = _slug_paths(FRONTIER_SLUG)
    lanes[FRONTIER_SLUG] = [
        Job(
            name=f"calibration_{FRONTIER_SLUG}",
            module="emotion_stage7.calibrate",
            args=(
                "--engine",
                "bedrock",
                "--frontier-lock",
                FRONTIER_LOCK,
                "--workers",
                "16",
                "--resume",
                "--output-dir",
                str(frontier["calibration"]),
            ),
            completion_paths=(
                frontier["calibration"] / "validation.json",
                frontier["calibration"] / "predictions.jsonl",
            ),
            cuda_visible_devices=None,
        )
    ]
    return lanes


def _run_command(
    command: list[str],
    *,
    log,
    cuda_visible_devices: str | None = None,
) -> None:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=_environment(cuda_visible_devices),
        stdout=log,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}: {command}")


def _make_task_locks(campaign_dir: Path) -> None:
    log_path = campaign_dir / "difficulty_locks.log"
    with log_path.open("a", encoding="utf-8") as log:
        for slug in (*OPEN_MODELS, FRONTIER_SLUG):
            paths = _slug_paths(slug)
            for task_name, key in (
                ("sycophancy", "sycophancy_lock"),
                ("refusal_drift", "refusal_lock"),
            ):
                output = paths[key]
                if output.exists():
                    payload = json.loads(output.read_text(encoding="utf-8"))
                    if (
                        payload.get("stage") == 7
                        and payload.get("task") == task_name
                        and payload.get("preregistration_sha256")
                        == PREREGISTRATION_SHA256
                    ):
                        continue
                    raise RuntimeError(f"Existing invalid difficulty lock: {output}")
                _run_command(
                    _python_module(
                        "emotion_stage7.make_task_lock",
                        "--calibration-dir",
                        str(paths["calibration"]),
                        "--task",
                        task_name,
                        "--output",
                        str(output),
                    ),
                    log=log,
                )


def _admission_lanes() -> dict[str, list[Job]]:
    lanes = {}
    gpu_assignments = {
        "qwen3_8b": "0",
        "qwen3_14b": "1",
        "qwen3_32b": "2,3",
    }
    for slug, config in OPEN_MODELS.items():
        paths = _slug_paths(slug)
        args = [
            "--engine",
            "transformers",
            "--model",
            str(config["model"]),
            "--task-locks",
            str(paths["sycophancy_lock"]),
            str(paths["refusal_lock"]),
            "--batch-size",
            str(config["batch_size"]),
            "--resume",
            "--output-dir",
            str(paths["admission"]),
        ]
        if config["device_map"]:
            args.extend(["--device-map", str(config["device_map"])])
        lanes[slug] = [
            Job(
                name=f"admission_{slug}",
                module="emotion_stage7.admission",
                args=tuple(args),
                completion_paths=(
                    paths["admission"] / "validation.json",
                    paths["admission"] / "sycophancy" / "admission.json",
                    paths["admission"] / "refusal_drift" / "admission.json",
                ),
                cuda_visible_devices=gpu_assignments[slug],
            )
        ]
    frontier = _slug_paths(FRONTIER_SLUG)
    lanes[FRONTIER_SLUG] = [
        Job(
            name=f"admission_{FRONTIER_SLUG}",
            module="emotion_stage7.admission",
            args=(
                "--engine",
                "bedrock",
                "--frontier-lock",
                FRONTIER_LOCK,
                "--task-locks",
                str(frontier["sycophancy_lock"]),
                str(frontier["refusal_lock"]),
                "--workers",
                "16",
                "--resume",
                "--output-dir",
                str(frontier["admission"]),
            ),
            completion_paths=(
                frontier["admission"] / "validation.json",
                frontier["admission"] / "sycophancy" / "admission.json",
                frontier["admission"] / "refusal_drift" / "admission.json",
            ),
            cuda_visible_devices=None,
        )
    ]
    return lanes


def _make_panel_locks(campaign_dir: Path) -> None:
    log_path = campaign_dir / "panel_locks.log"
    with log_path.open("a", encoding="utf-8") as log:
        for slug in (*OPEN_MODELS, FRONTIER_SLUG):
            paths = _slug_paths(slug)
            if paths["panel"].exists():
                payload = json.loads(paths["panel"].read_text(encoding="utf-8"))
                if (
                    payload.get("stage") == 7
                    and payload.get("preregistration_sha256")
                    == PREREGISTRATION_SHA256
                ):
                    continue
                raise RuntimeError(f"Existing invalid panel lock: {paths['panel']}")
            _run_command(
                _python_module(
                    "emotion_stage7.make_panel_lock",
                    "--admission-root",
                    str(paths["admission"]),
                    "--output",
                    str(paths["panel"]),
                ),
                log=log,
            )


def _run_completion_paths(directory: Path) -> tuple[Path, ...]:
    return (
        directory / "metadata.json",
        directory / "predictions.jsonl",
        directory / "run_stats.json",
    )


def _generation_lanes() -> tuple[dict[str, list[Job]], list[Path]]:
    lanes: dict[str, list[Job]] = {}
    run_dirs = []
    assignments = {
        "qwen3_8b": ("0", "1"),
        "qwen3_14b": ("2", "3"),
        "qwen3_32b": ("4,5", "6,7"),
    }
    for slug, config in OPEN_MODELS.items():
        paths = _slug_paths(slug)
        panel = json.loads(paths["panel"].read_text(encoding="utf-8"))
        admitted_tasks = set(panel["admitted_tasks"])
        safety_outputs = []
        if "sycophancy" in admitted_tasks:
            safety_outputs.extend(_run_completion_paths(paths["sycophancy_run"]))
            run_dirs.append(paths["sycophancy_run"])
        if "refusal_drift" in admitted_tasks:
            safety_outputs.extend(_run_completion_paths(paths["refusal_run"]))
            run_dirs.append(paths["refusal_run"])
        safety_gpu, ifeval_gpu = assignments[slug]
        if safety_outputs:
            args = [
                "--model",
                str(config["model"]),
                "--panel-lock",
                str(paths["panel"]),
                "--vector-dir",
                str(config["vector_dir"]),
                "--steering-lock",
                str(config["steering_lock"]),
                "--sycophancy-output-dir",
                str(paths["sycophancy_run"]),
                "--refusal-output-dir",
                str(paths["refusal_run"]),
                "--batch-size",
                str(config["batch_size"]),
                "--resume",
            ]
            if config["device_map"]:
                args.extend(["--device-map", str(config["device_map"])])
            lanes[f"{slug}_safety"] = [
                Job(
                    name=f"open_safety_{slug}",
                    module="emotion_stage7.run_open_safety",
                    args=tuple(args),
                    completion_paths=tuple(safety_outputs),
                    cuda_visible_devices=safety_gpu,
                )
            ]
        ifeval_args = [
            "--model",
            str(config["model"]),
            "--vector-dir",
            str(config["vector_dir"]),
            "--steering-lock",
            str(config["steering_lock"]),
            "--batch-size",
            str(config["ifeval_batch_size"]),
            "--resume",
            "--output-dir",
            str(paths["ifeval_run"]),
        ]
        if config["device_map"]:
            ifeval_args.extend(["--device-map", str(config["device_map"])])
        lanes[f"{slug}_ifeval"] = [
            Job(
                name=f"open_ifeval_{slug}",
                module="emotion_stage7.run_open_ifeval",
                args=tuple(ifeval_args),
                completion_paths=_run_completion_paths(paths["ifeval_run"]),
                cuda_visible_devices=ifeval_gpu,
            )
        ]
        run_dirs.append(paths["ifeval_run"])

    frontier_paths = _slug_paths(FRONTIER_SLUG)
    frontier_panel = json.loads(
        frontier_paths["panel"].read_text(encoding="utf-8")
    )
    frontier_jobs = []
    for task_name, output_key in (
        ("sycophancy", "sycophancy_run"),
        ("refusal_drift", "refusal_run"),
    ):
        if task_name not in frontier_panel["admitted_tasks"]:
            continue
        output = frontier_paths[output_key]
        frontier_jobs.append(
            Job(
                name=f"frontier_{task_name}",
                module="emotion_stage7.run_frontier",
                args=(
                    "--task",
                    task_name,
                    "--frontier-lock",
                    FRONTIER_LOCK,
                    "--panel-lock",
                    str(frontier_paths["panel"]),
                    "--workers",
                    "16",
                    "--resume",
                    "--output-dir",
                    str(output),
                ),
                completion_paths=_run_completion_paths(output),
                cuda_visible_devices=None,
            )
        )
        run_dirs.append(output)
    frontier_jobs.append(
        Job(
            name="frontier_ifeval",
            module="emotion_stage7.run_frontier",
            args=(
                "--task",
                "ifeval",
                "--frontier-lock",
                FRONTIER_LOCK,
                "--workers",
                "8",
                "--resume",
                "--output-dir",
                str(frontier_paths["ifeval_run"]),
            ),
            completion_paths=_run_completion_paths(
                frontier_paths["ifeval_run"]
            ),
            cuda_visible_devices=None,
        )
    )
    run_dirs.append(frontier_paths["ifeval_run"])
    lanes["frontier"] = frontier_jobs
    return lanes, run_dirs


def _validate_runs(run_dirs: list[Path], campaign_dir: Path) -> None:
    log_path = campaign_dir / "validation.log"
    with log_path.open("a", encoding="utf-8") as log:
        for run_dir in run_dirs:
            validation = run_dir / "validation.json"
            if validation.exists():
                try:
                    if json.loads(validation.read_text(encoding="utf-8")).get(
                        "status"
                    ) == "complete":
                        continue
                except json.JSONDecodeError:
                    pass
            _run_command(
                _python_module(
                    "emotion_stage7.validate_run",
                    "--run-dir",
                    str(run_dir),
                ),
                log=log,
            )


def _coherence_lanes(run_dirs: list[Path]) -> dict[str, list[Job]]:
    by_model: dict[str, list[tuple[Path, dict]]] = {}
    for run_dir in run_dirs:
        metadata = json.loads(
            (run_dir / "metadata.json").read_text(encoding="utf-8")
        )
        if metadata["engine"] != "transformers":
            continue
        by_model.setdefault(metadata["model"], []).append((run_dir, metadata))
    model_to_slug = {
        str(config["model"]): slug for slug, config in OPEN_MODELS.items()
    }
    assignments = {
        "qwen3_8b": "0",
        "qwen3_14b": "1",
        "qwen3_32b": "2,3",
    }
    lanes = {}
    for model, records in by_model.items():
        slug = model_to_slug[model]
        jobs = []
        for run_dir, metadata in sorted(
            records,
            key=lambda value: value[1]["task"],
        ):
            jobs.append(
                Job(
                    name=f"coherence_{slug}_{metadata['task']}",
                    module="emotion_stage7.score_coherence",
                    args=(
                        "--run-dir",
                        str(run_dir),
                        "--samples-per-arm",
                        "8",
                        "--max-output-tokens",
                        "32" if metadata["task"] == "ifeval" else "8",
                    ),
                    completion_paths=(run_dir / "coherence_summary.csv",),
                    cuda_visible_devices=assignments[slug],
                )
            )
        lanes[slug] = jobs
    return lanes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(path: Path) -> dict:
    record = {"path": str(path.resolve()), "exists": path.exists()}
    if path.is_file():
        record["sha256"] = _sha256(path)
    elif path.is_dir():
        record["files"] = {
            name: _sha256(path / name)
            for name in (
                "admission.json",
                "base_rates.csv",
                "coherence_wall.csv",
                "dose_curves.csv",
                "dose_joint_controls.csv",
                "ifeval_primary.csv",
                "ifeval_scale.png",
                "metadata.json",
                "overrefusal_monitoring.csv",
                "predictions.jsonl",
                "safety_primary.csv",
                "safety_scale.png",
                "scale_trends.csv",
                "sign_tests.csv",
                "summary.json",
                "tier_mixes.csv",
                "validation.json",
            )
            if (path / name).is_file()
        }
    return record


def _format_optional_rate(value: float | None) -> str:
    return "not_evaluated" if value is None else f"{float(value):.6g}"


def _write_manifest(
    *,
    aggregate_dir: Path,
    run_dirs: list[Path],
) -> None:
    paths = []
    unavailable = []
    for slug in (*OPEN_MODELS, FRONTIER_SLUG):
        stage_paths = _slug_paths(slug)
        paths.extend(
            [
                stage_paths["calibration"],
                stage_paths["sycophancy_lock"],
                stage_paths["refusal_lock"],
                stage_paths["admission"],
                stage_paths["admission"] / "sycophancy",
                stage_paths["admission"] / "refusal_drift",
                stage_paths["panel"],
            ]
        )
        panel = json.loads(
            stage_paths["panel"].read_text(encoding="utf-8")
        )
        for task_name, task in panel["tasks"].items():
            for rate in task["rates"]:
                if rate["status"] == "admitted":
                    continue
                unavailable.append(
                    {
                        "model": panel["model"],
                        "task": task_name,
                        "signal": rate["signal"],
                        "status": rate["status"],
                        "reason": (
                            f"dev={rate['predicted_dev_rate']:.6g}, "
                            f"test={_format_optional_rate(rate['rate'])}, "
                            f"band=[{rate['minimum']}, {rate['maximum']}]"
                        ),
                    }
                )
    paths.extend(
        [
            REPO_ROOT / "docs" / "stage7-ifeval-runtime-amendment.md",
            REPO_ROOT / FRONTIER_LOCK,
            *[
                RUNS / str(config["vector_dir"]).removeprefix("runs/")
                for config in OPEN_MODELS.values()
            ],
            *[
                RUNS / str(config["steering_lock"]).removeprefix("runs/")
                for config in OPEN_MODELS.values()
            ],
            *run_dirs,
            aggregate_dir,
        ]
    )
    manifest_path = aggregate_dir / "artifact_manifest.json"
    manifest = {
        "stage": 7,
        "artifact": "terminal_artifact_manifest",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "aggregate_dir": str(aggregate_dir.resolve()),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "ifeval_runtime_amendment_sha256": (
            IFEVAL_RUNTIME_AMENDMENT_SHA256
        ),
        "artifacts": [_artifact_record(path) for path in paths],
        "unavailable_components": unavailable,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    for name in ("metadata.json", "summary.json"):
        path = aggregate_dir / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["artifact_manifest_path"] = str(manifest_path.resolve())
        payload["unavailable_components"] = unavailable
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        default=RUNS / "stage7_campaign_v3",
    )
    parser.add_argument(
        "--aggregate-dir",
        type=Path,
        default=RUNS / "stage7_scale_invariant_aggregate_v3",
    )
    args = parser.parse_args()
    args.campaign_dir.mkdir(parents=True, exist_ok=True)
    if args.aggregate_dir.exists() and any(args.aggregate_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty {args.aggregate_dir}")

    _run_lanes(
        _calibration_lanes(),
        campaign_dir=args.campaign_dir,
        phase="calibration",
    )
    _make_task_locks(args.campaign_dir)
    _run_lanes(
        _admission_lanes(),
        campaign_dir=args.campaign_dir,
        phase="admission",
    )
    _make_panel_locks(args.campaign_dir)
    generation_lanes, run_dirs = _generation_lanes()
    _run_lanes(
        generation_lanes,
        campaign_dir=args.campaign_dir,
        phase="generation",
    )
    _validate_runs(run_dirs, args.campaign_dir)
    _run_lanes(
        _coherence_lanes(run_dirs),
        campaign_dir=args.campaign_dir,
        phase="coherence",
    )
    panel_paths = [
        _slug_paths(slug)["panel"]
        for slug in (*OPEN_MODELS, FRONTIER_SLUG)
    ]
    aggregate_log = args.campaign_dir / "aggregation.log"
    with aggregate_log.open("a", encoding="utf-8") as log:
        _run_command(
            _python_module(
                "emotion_stage7.aggregate_results",
                "--run-dirs",
                *[str(path) for path in run_dirs],
                "--panel-locks",
                *[str(path) for path in panel_paths],
                "--output-dir",
                str(args.aggregate_dir),
            ),
            log=log,
        )
    _write_manifest(aggregate_dir=args.aggregate_dir, run_dirs=run_dirs)
    _write_status(
        args.campaign_dir,
        {
            "phase": "complete",
            "updated_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            ),
            "aggregate_dir": str(args.aggregate_dir.resolve()),
            "run_dirs": [str(path.resolve()) for path in run_dirs],
            "failures": {},
        },
    )


if __name__ == "__main__":
    main()
