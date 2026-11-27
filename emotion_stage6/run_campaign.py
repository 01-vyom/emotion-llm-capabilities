"""Run, validate, score, and aggregate the authorized Stage 6 campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = REPO_ROOT / "runs"
CONDA = Path("/home/ec2-user/miniforge3/bin/conda")
CONDA_ENV = "emotion-stage1"


@dataclass(frozen=True)
class Job:
    name: str
    module: str
    args: tuple[str, ...]
    output_dir: Path
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
        }
    )
    if cuda_visible_devices is None:
        environment.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return environment


def _validation_complete(output_dir: Path) -> bool:
    path = output_dir / "validation.json"
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def _phase_complete(job: Job, phase: str) -> bool:
    if phase == "generation":
        return all(
            (job.output_dir / name).is_file()
            for name in (
                "metadata.json",
                "predictions.jsonl",
                "run_stats.json",
            )
        )
    return (job.output_dir / "coherence_summary.csv").is_file()


def _write_status(campaign_dir: Path, payload: dict) -> None:
    path = campaign_dir / "status.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _run_group(
    jobs: list[Job],
    *,
    campaign_dir: Path,
    phase: str,
    max_attempts: int = 2,
) -> None:
    attempts = {job.name: 0 for job in jobs}
    completed = {
        job.name
        for job in jobs
        if _phase_complete(job, phase)
    }
    running: dict[str, tuple[Job, subprocess.Popen, object]] = {}
    failures = {}
    while len(completed) < len(jobs):
        for job in jobs:
            if job.name in completed or job.name in running or job.name in failures:
                continue
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
            running[job.name] = (job, process, log)
        _write_status(
            campaign_dir,
            {
                "phase": phase,
                "updated_at_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ",
                    time.gmtime(),
                ),
                "completed": sorted(completed),
                "running": {
                    name: {
                        "pid": process.pid,
                        "attempt": attempts[name],
                        "output_dir": str(job.output_dir),
                    }
                    for name, (job, process, _log) in running.items()
                },
                "failures": failures,
            },
        )
        time.sleep(15)
        for name, (job, process, log) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log.write(f"\n=== return code {return_code} ===\n")
            log.close()
            del running[name]
            phase_complete = _phase_complete(job, phase)
            if return_code == 0 and phase_complete:
                completed.add(name)
            elif attempts[name] >= max_attempts:
                failures[name] = {
                    "return_code": return_code,
                    "attempts": attempts[name],
                    "output_dir": str(job.output_dir),
                }
        if failures:
            for _name, (_job, process, log) in running.items():
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
                    "completed": sorted(completed),
                    "running": {},
                    "failures": failures,
                },
            )
            raise RuntimeError(f"{phase} jobs failed: {failures}")


def _validate_runs(jobs: list[Job], campaign_dir: Path) -> None:
    log_path = campaign_dir / "validation.log"
    with log_path.open("a", encoding="utf-8") as log:
        for job in jobs:
            result = subprocess.run(
                _python_module(
                    "emotion_stage6.validate_run",
                    "--run-dir",
                    str(job.output_dir),
                ),
                cwd=REPO_ROOT,
                env=_environment(job.cuda_visible_devices),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Validation failed for {job.name}")


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
                "extraction_meta.json",
                "metadata.json",
                "probe_report.json",
                "summary.json",
                "validation.json",
            )
            if (path / name).is_file()
        }
    return record


def _write_terminal_manifest(
    aggregate_dir: Path,
    generation_jobs: list[Job],
) -> None:
    unavailable = [
        {
            "model": "Qwen/Qwen3-14B",
            "task": "refusal_drift",
            "signal": "overrefusal",
            "reason": "DV admission rate 0.0916667 was below 0.10",
        },
        {
            "model": "Qwen/Qwen3-32B",
            "task": "refusal_drift",
            "signal": "unsafe_compliance",
            "reason": "DV admission rate 0.6125 was above 0.60",
        },
        {
            "model": "Qwen/Qwen3-32B",
            "task": "refusal_drift",
            "signal": "overrefusal",
            "reason": "DV admission rate 0.0833333 was below 0.10",
        },
        {
            "model": "us.anthropic.claude-sonnet-5",
            "task": "sycophancy",
            "signal": "sycophancy",
            "reason": "DV admission rate 0.0016667 was below 0.10",
        },
        {
            "model": "us.anthropic.claude-sonnet-5",
            "task": "refusal_drift",
            "signal": "overrefusal",
            "reason": "DV admission rate 0.0 was below 0.10",
        },
    ]
    terminal_paths = [
        RUNS / "stage6_stage5_three_control_reanalysis_v1",
        RUNS / "stage6_frontier_lock_v1.json",
        RUNS / "stage6_qwen3_8b_panel_lock_v1.json",
        RUNS / "stage6_qwen3_14b_panel_lock_v1.json",
        RUNS / "stage6_qwen3_32b_panel_lock_v1.json",
        RUNS / "stage6_sonnet5_panel_lock_v1.json",
        RUNS / "stage6_qwen3_8b_steering_lock_v1.json",
        RUNS / "stage6_qwen3_14b_steering_lock_v1.json",
        RUNS / "stage6_qwen3_32b_steering_lock_v1.json",
        RUNS / "stage35_qwen3_8b_vectors_v1",
        RUNS / "stage35_qwen3_8b_probe_v1",
        RUNS / "stage35_qwen3_14b_vectors_v1",
        RUNS / "stage35_qwen3_14b_probe_v1",
        RUNS / "stage6_qwen3_32b_vectors_v1",
        RUNS / "stage6_qwen3_32b_probe_v1",
        *[
            RUNS / name
            for name in (
                "stage6_admission_sycophancy_qwen3_8b_test_v1",
                "stage6_admission_refusal_qwen3_8b_test_v1",
                "stage6_admission_sycophancy_qwen3_14b_test_v1",
                "stage6_admission_refusal_qwen3_14b_test_v1",
                "stage6_admission_sycophancy_qwen3_32b_test_v1",
                "stage6_admission_refusal_qwen3_32b_test_v1",
                "stage6_admission_sycophancy_sonnet5_test_v1",
                "stage6_admission_refusal_sonnet5_test_v1",
            )
        ],
        *[job.output_dir for job in generation_jobs],
        aggregate_dir,
    ]
    for name in ("metadata.json", "summary.json"):
        path = aggregate_dir / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["artifact_manifest_path"] = str(
            (aggregate_dir / "artifact_manifest.json").resolve()
        )
        payload["unavailable_components"] = unavailable
        payload["stage5_reanalysis_path"] = str(
            (RUNS / "stage6_stage5_three_control_reanalysis_v1").resolve()
        )
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    manifest = {
        "stage": 6,
        "artifact": "terminal_artifact_manifest",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "aggregate_dir": str(aggregate_dir.resolve()),
        "artifacts": [_artifact_record(path) for path in terminal_paths],
        "unavailable_components": unavailable,
    }
    (aggregate_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def _generation_jobs() -> list[Job]:
    common = ("--resume", "--bootstrap-reps", "2000")
    return [
        Job(
            "sycophancy_qwen3_8b",
            "emotion_stage6.run_stage6",
            (
                "--task",
                "sycophancy",
                "--model",
                "Qwen/Qwen3-8B",
                "--panel-lock",
                "runs/stage6_qwen3_8b_panel_lock_v1.json",
                "--batch-size",
                "16",
                "--output-dir",
                "runs/stage6_sycophancy_qwen3_8b_test_v1",
                *common,
            ),
            RUNS / "stage6_sycophancy_qwen3_8b_test_v1",
            "0",
        ),
        Job(
            "refusal_qwen3_8b",
            "emotion_stage6.run_stage6",
            (
                "--task",
                "refusal_drift",
                "--model",
                "Qwen/Qwen3-8B",
                "--panel-lock",
                "runs/stage6_qwen3_8b_panel_lock_v1.json",
                "--vector-dir",
                "runs/stage35_qwen3_8b_vectors_v1",
                "--steering-lock",
                "runs/stage6_qwen3_8b_steering_lock_v1.json",
                "--batch-size",
                "16",
                "--output-dir",
                "runs/stage6_refusal_qwen3_8b_test_v1",
                *common,
            ),
            RUNS / "stage6_refusal_qwen3_8b_test_v1",
            "1",
        ),
        Job(
            "sycophancy_qwen3_14b",
            "emotion_stage6.run_stage6",
            (
                "--task",
                "sycophancy",
                "--model",
                "Qwen/Qwen3-14B",
                "--panel-lock",
                "runs/stage6_qwen3_14b_panel_lock_v1.json",
                "--batch-size",
                "8",
                "--output-dir",
                "runs/stage6_sycophancy_qwen3_14b_test_v1",
                *common,
            ),
            RUNS / "stage6_sycophancy_qwen3_14b_test_v1",
            "2",
        ),
        Job(
            "refusal_qwen3_14b",
            "emotion_stage6.run_stage6",
            (
                "--task",
                "refusal_drift",
                "--model",
                "Qwen/Qwen3-14B",
                "--panel-lock",
                "runs/stage6_qwen3_14b_panel_lock_v1.json",
                "--vector-dir",
                "runs/stage35_qwen3_14b_vectors_v1",
                "--steering-lock",
                "runs/stage6_qwen3_14b_steering_lock_v1.json",
                "--batch-size",
                "8",
                "--output-dir",
                "runs/stage6_refusal_qwen3_14b_test_v1",
                *common,
            ),
            RUNS / "stage6_refusal_qwen3_14b_test_v1",
            "3",
        ),
        Job(
            "sycophancy_qwen3_32b",
            "emotion_stage6.run_stage6",
            (
                "--task",
                "sycophancy",
                "--model",
                "Qwen/Qwen3-32B",
                "--panel-lock",
                "runs/stage6_qwen3_32b_panel_lock_v1.json",
                "--device-map",
                "auto",
                "--batch-size",
                "4",
                "--output-dir",
                "runs/stage6_sycophancy_qwen3_32b_test_v1",
                *common,
            ),
            RUNS / "stage6_sycophancy_qwen3_32b_test_v1",
            "4,5",
        ),
        Job(
            "refusal_sonnet5",
            "emotion_stage6.run_frontier",
            (
                "--task",
                "refusal_drift",
                "--frontier-lock",
                "runs/stage6_frontier_lock_v1.json",
                "--panel-lock",
                "runs/stage6_sonnet5_panel_lock_v1.json",
                "--workers",
                "16",
                "--output-dir",
                "runs/stage6_refusal_sonnet5_test_v1",
                *common,
            ),
            RUNS / "stage6_refusal_sonnet5_test_v1",
            None,
        ),
    ]


def _coherence_jobs(generation_jobs: list[Job]) -> list[Job]:
    return [
        Job(
            job.name,
            "emotion_stage6.score_coherence",
            (
                "--run-dir",
                str(job.output_dir),
                "--samples-per-arm",
                "8",
                "--max-output-tokens",
                "8",
            ),
            job.output_dir,
            job.cuda_visible_devices,
        )
        for job in generation_jobs
        if job.module == "emotion_stage6.run_stage6"
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        default=RUNS / "stage6_campaign_v1",
    )
    parser.add_argument(
        "--aggregate-dir",
        type=Path,
        default=RUNS / "stage6_scale_aggregate_v1",
    )
    args = parser.parse_args()
    args.campaign_dir.mkdir(parents=True, exist_ok=True)
    if args.aggregate_dir.exists() and any(args.aggregate_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty {args.aggregate_dir}")

    generation_jobs = _generation_jobs()
    _run_group(
        generation_jobs,
        campaign_dir=args.campaign_dir,
        phase="generation",
    )
    _validate_runs(generation_jobs, args.campaign_dir)
    _run_group(
        _coherence_jobs(generation_jobs),
        campaign_dir=args.campaign_dir,
        phase="coherence",
    )

    aggregate_log = args.campaign_dir / "aggregation.log"
    with aggregate_log.open("a", encoding="utf-8") as log:
        result = subprocess.run(
            _python_module(
                "emotion_stage6.aggregate_results",
                "--run-dirs",
                *[str(job.output_dir) for job in generation_jobs],
                "--output-dir",
                str(args.aggregate_dir),
            ),
            cwd=REPO_ROOT,
            env=_environment(None),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError("Stage 6 aggregation failed")
    _write_terminal_manifest(args.aggregate_dir, generation_jobs)
    _write_status(
        args.campaign_dir,
        {
            "phase": "complete",
            "updated_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            ),
            "completed": [job.name for job in generation_jobs],
            "aggregate_dir": str(args.aggregate_dir.resolve()),
            "failures": {},
        },
    )


if __name__ == "__main__":
    main()
