#!/usr/bin/env python3
"""Single entrypoint that runs the Novel5 analysis pipeline in the order
documented in README.md (stages 1-6). See `pipeline/steps.py` for the exact
step registry and `pipeline/runner.py` for the execution engine.

This script only *orchestrates* the existing, unmodified analysis scripts
(same script, same args, same env vars, same working directory each script
already expects) -- it does not reimplement or alter any analysis logic.

Stage 0 (harmonise.py) is intentionally out of scope: it is a one-time raw
-> harmonised/*.parquet preparation step, run separately, outside this
orchestrator (see CLAUDE.md / README.md "Data dependency").

Usage:
    python3 main.py                       # run everything, auto onetime-caching
    python3 main.py --dry-run             # print the plan, run nothing
    python3 main.py --list                # list all known steps and exit
    python3 main.py --one-time force      # force-regenerate every onetime step
    python3 main.py --one-time skip       # never (re)run onetime steps
    python3 main.py --stage 4 5 6         # only run these stages
    python3 main.py --only-steps make_tables verify_manuscript_numbers
    python3 main.py --skip-steps benchmark_within
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.runner import REPO_ROOT as RUNNER_ROOT, RESULTS_DIR, Runner  # noqa: E402
from pipeline.steps import STEPS, stages  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true",
                    help="print the execution plan without running anything")
    p.add_argument("--list", action="store_true",
                    help="list all known steps (name, stage, category) and exit")
    p.add_argument("--one-time", choices=["auto", "skip", "force"], default="auto",
                    help="onetime (stage 2/3) step handling: auto=skip if cached in "
                         "./harmonised/, run and cache otherwise (default); "
                         "skip=never run, restore cache only if present; "
                         "force=always (re)run and refresh the cache")
    p.add_argument("--stage", type=int, nargs="+", default=None,
                    help="restrict to these stage numbers (e.g. --stage 4 5 6)")
    p.add_argument("--only-steps", nargs="+", default=None, metavar="STEP",
                    help="restrict to exactly these step names")
    p.add_argument("--skip-steps", nargs="+", default=None, metavar="STEP",
                    help="exclude these step names from the run")
    return p.parse_args(argv)


def select_steps(args: argparse.Namespace) -> list:
    selected = list(STEPS)
    if args.stage:
        selected = [s for s in selected if s.stage in args.stage]
    if args.only_steps:
        wanted = set(args.only_steps)
        unknown = wanted - {s.name for s in STEPS}
        if unknown:
            raise SystemExit(f"unknown step name(s): {sorted(unknown)}")
        selected = [s for s in selected if s.name in wanted]
    if args.skip_steps:
        selected = [s for s in selected if s.name not in set(args.skip_steps)]
    return selected


def ensure_prerequisites() -> None:
    """Stage-0 prerequisite check (see README.md Prerequisites). Only copies
    a file into place if it's missing -- never overwrites, never touches
    harmonised/ (the source of truth), never runs harmonise.py itself."""
    results_gene_sets = RESULTS_DIR / "gene_sets.json"
    harmonised_gene_sets = REPO_ROOT / "harmonised" / "gene_sets.json"
    if not results_gene_sets.exists() and harmonised_gene_sets.exists():
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(harmonised_gene_sets, results_gene_sets)
        print(f"prerequisite: copied {harmonised_gene_sets.relative_to(REPO_ROOT)} "
              f"-> {results_gene_sets.relative_to(REPO_ROOT)}")


def list_steps() -> None:
    for stage in stages():
        print(f"\n=== stage {stage} ===")
        for s in STEPS:
            if s.stage != stage:
                continue
            avail = "unavailable" if s.script is None else s.script
            print(f"  {s.name:<32} [{s.category:<7}] {avail}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list:
        list_steps()
        return 0

    selected = select_steps(args)
    if not selected:
        print("No steps selected.")
        return 0

    if not args.dry_run:
        ensure_prerequisites()

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RESULTS_DIR / timestamp
    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        RUNNER_ROOT.joinpath("harmonised").mkdir(parents=True, exist_ok=True)

    runner = Runner(run_dir=run_dir, one_time_mode=args.one_time, dry_run=args.dry_run)

    print(f"Run directory: {run_dir.relative_to(REPO_ROOT)}" if not args.dry_run else
          "(dry run: no run directory created)")
    print(f"one-time mode: {args.one_time}\n")

    failures = []
    for step in selected:
        line = runner.plan_line(step)
        if args.dry_run:
            print(line)
            continue
        print(line, "...", flush=True)
        result = runner.run_step(step)
        print(f"    -> {result.status}: {result.detail}")
        if result.status == "failed":
            failures.append(step.name)

    if args.dry_run:
        return 0

    summary_path = run_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"run: {timestamp}\n")
        f.write(f"one-time mode: {args.one_time}\n")
        f.write(f"steps selected: {len(selected)}\n")
        f.write(f"failures: {failures if failures else 'none'}\n")

    if failures:
        print(f"\n{len(failures)} step(s) failed: {failures}")
        print(f"See {run_dir.relative_to(REPO_ROOT)}/logs/<step>.log for details.")
        return 1

    print(f"\nAll steps completed. Logs and output snapshots in "
          f"{run_dir.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
