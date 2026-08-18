"""Execution engine for the steps declared in `pipeline/steps.py`.

Responsibilities (and nothing else -- no analysis logic lives here):

- Run each Step as an unmodified `python3 <script>.py [args]` subprocess,
  with the working directory, environment variables, and arguments exactly
  as declared on the Step. Every script's own code path is untouched.
- Capture stdout/stderr for each step into
  `results/<run-timestamp>/logs/<step_name>.log`.
- "onetime" steps (stage 2/3): cached under `./harmonised/<step-name>/`.
    * auto  (default) - skip and restore from cache if the cache already has
      every declared output; otherwise run and populate the cache.
    * force - always (re-)run and overwrite the cache.
    * skip  - never run; restore from cache if present, otherwise leave the
      step un-run (only a warning, so `--only-steps` subsets still work).
  A onetime step's outputs are, after running or restoring, mirrored to
  their README-documented location (repo root and/or results/, per the
  Step's own `outputs` paths) so downstream steps that read them back (e.g.
  pooled_hr_stratified.py reading results/loco_risk_novel5.csv) see them
  exactly as the manual workflow produced them.
- "always" steps (stage 4/5/6): re-run every invocation of main.py, reading
  the current repo-root/results/ state and writing results/, media/, paper/
  as they always have. Their declared outputs are additionally snapshotted
  (copied) into `results/<run-timestamp>/` so each run's figures/tables are
  preserved alongside that run's logs.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from pipeline.steps import Step

REPO_ROOT = Path(__file__).resolve().parent.parent
harmonised_DIR = REPO_ROOT / "harmonised"
RESULTS_DIR = REPO_ROOT / "results"


class StepResult:
    def __init__(self, step: Step, status: str, detail: str = ""):
        self.step = step
        self.status = status  # "ran" | "skipped-cached" | "skipped-unavailable" | "failed"
        self.detail = detail


def _step_cache_dir(step: Step) -> Path:
    return harmonised_DIR / step.name


def _cache_complete(step: Step) -> bool:
    cache_dir = _step_cache_dir(step)
    if not step.outputs:
        return cache_dir.exists()
    return all((cache_dir / Path(out).name).exists() for out in step.outputs)


def _restore_from_cache(step: Step) -> None:
    cache_dir = _step_cache_dir(step)
    for out in step.outputs:
        src = cache_dir / Path(out).name
        dst = (REPO_ROOT / step.cwd / out).resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, dst)
        _mirror_to_results(step, out)
        _mirror_to_extra(step, out)


def _populate_cache(step: Step) -> None:
    cache_dir = _step_cache_dir(step)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for out in step.outputs:
        src = (REPO_ROOT / step.cwd / out).resolve()
        if src.exists():
            shutil.copy2(src, cache_dir / Path(out).name)


def _mirror_to_results(step: Step, out: str) -> None:
    """Ensure a onetime step's output is also visible at results/<basename>,
    since some downstream stage-3 scripts read it back from there."""
    src = (REPO_ROOT / step.cwd / out).resolve()
    if not src.exists():
        return
    dst = RESULTS_DIR / Path(out).name
    if src.resolve() == dst.resolve():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _mirror_to_extra(step: Step, out: str) -> None:
    src = (REPO_ROOT / step.cwd / out).resolve()
    if not src.exists():
        return
    for extra_dir in step.extra_copy:
        dst = (REPO_ROOT / extra_dir / Path(out).name).resolve()
        if src.resolve() == dst.resolve():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _snapshot_to_run_dir(step: Step, run_dir: Path) -> None:
    for out in step.outputs:
        src = (REPO_ROOT / step.cwd / out).resolve()
        if src.exists():
            dst = run_dir / "outputs" / step.name / Path(out).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


class Runner:
    def __init__(self, run_dir: Path, one_time_mode: str = "auto", dry_run: bool = False):
        self.run_dir = run_dir
        self.one_time_mode = one_time_mode  # "auto" | "skip" | "force"
        self.dry_run = dry_run
        self.log_dir = run_dir / "logs"
        if not dry_run:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def plan_line(self, step: Step) -> str:
        if step.script is None:
            return f"[stage {step.stage}] {step.name:<32} UNAVAILABLE  ({step.note})"
        if step.category == "onetime":
            cached = _cache_complete(step)
            if self.one_time_mode == "skip":
                action = "restore-cache" if cached else "SKIP (no cache, will be missing)"
            elif self.one_time_mode == "force":
                action = "run (force)"
            else:
                action = "restore-cache" if cached else "run (no cache yet)"
        else:
            action = "run (always)"
        cwd_note = f" cwd={step.cwd}" if step.cwd != "." else ""
        env_note = f" env={step.env}" if step.env else ""
        return f"[stage {step.stage}] {step.name:<32} {action}{cwd_note}{env_note}"

    def run_step(self, step: Step) -> StepResult:
        if step.script is None:
            return StepResult(step, "skipped-unavailable", step.note)

        if step.category == "onetime":
            cached = _cache_complete(step)
            if self.one_time_mode == "skip":
                if cached:
                    if not self.dry_run:
                        _restore_from_cache(step)
                    return StepResult(step, "skipped-cached", "restored from harmonised/ cache")
                return StepResult(step, "skipped-unavailable", "no cache present, --one-time skip")
            if self.one_time_mode == "auto" and cached:
                if not self.dry_run:
                    _restore_from_cache(step)
                return StepResult(step, "skipped-cached", "restored from harmonised/ cache")
            # one_time_mode == "force", or auto with no cache: fall through to run.

        if self.dry_run:
            return StepResult(step, "ran", "dry-run: not executed")

        if step.category == "onetime":
            # Some stage-2 scripts (e.g. run_permutation_search.py) append
            # to their CSV output rather than truncating it. Remove stale
            # outputs before a real (re-)run so re-execution doesn't
            # silently accumulate duplicate rows on top of a prior run or a
            # stale cache copy.
            for out in step.outputs:
                stale = (REPO_ROOT / step.cwd / out).resolve()
                if stale.exists():
                    stale.unlink()

        result = self._execute(step)
        if result.status == "ran" and step.category == "onetime":
            _populate_cache(step)
            for out in step.outputs:
                _mirror_to_results(step, out)
                _mirror_to_extra(step, out)
        if result.status == "ran":
            _snapshot_to_run_dir(step, self.run_dir)
        return result

    def _execute(self, step: Step) -> StepResult:
        cwd = (REPO_ROOT / step.cwd).resolve()
        cwd.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(step.env)
        script_path = (REPO_ROOT / step.script).resolve()
        cmd = [sys.executable, str(script_path), *step.args]
        log_path = self.log_dir / f"{step.name}.log"
        start = time.time()
        with open(log_path, "w") as logf:
            logf.write(f"$ cd {cwd}\n$ {' '.join(f'{k}={v}' for k, v in step.env.items())} "
                       f"{' '.join(cmd)}\n\n")
            logf.flush()
            proc = subprocess.run(
                cmd, cwd=str(cwd), env=env, stdout=logf, stderr=subprocess.STDOUT,
            )
        elapsed = time.time() - start
        if proc.returncode != 0:
            return StepResult(
                step, "failed",
                f"exit {proc.returncode} after {elapsed:.1f}s; see {log_path.relative_to(REPO_ROOT)}",
            )
        return StepResult(step, "ran", f"{elapsed:.1f}s; log at {log_path.relative_to(REPO_ROOT)}")
