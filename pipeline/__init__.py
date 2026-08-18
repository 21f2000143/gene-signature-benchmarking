"""Orchestration package for the Novel5 analysis pipeline.

This package does not implement any analysis itself — it only sequences the
existing, unmodified `run_*.py` / `make_fig_*.py` / `make_tables.py` /
`verify_manuscript_numbers.py` scripts documented in README.md, in the order
documented there. See `pipeline/steps.py` for the step registry and
`pipeline/runner.py` for the execution/caching/snapshotting engine.
"""
