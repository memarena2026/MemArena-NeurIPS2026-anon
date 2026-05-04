"""Smoke test for `scripts/reproduce_figures.py`.

The paper-figure generators under `memarena/figures/` need the released
CSV artifacts from the Hugging Face dataset repo. When those aren't on
disk the test is skipped — CI still runs, while a local developer who has
symlinked or downloaded the artifacts exercises the pipeline.

We use `gen_tables` as the smoke module: it produces pure LaTeX output
(easy to check for non-trivial length) and exercises the full paper_data
loader end-to-end on the canonical MemArena-L run.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CANONICAL_RUN = REPO / "MASim" / "runs" / "l_20260408_111046"
LOADER = REPO / "memarena" / "figures" / "paper_data.py"
TABLES_OUT = REPO / "memarena" / "figures" / "tables" / "main_L.tex"


pytestmark = pytest.mark.skipif(
    not (CANONICAL_RUN.exists() and LOADER.exists()),
    reason=(
        "baseline run + paper_data loader not present; "
        "symlink MASim/runs/l_20260408_111046 and copy paper_data.py to exercise"
    ),
)


def test_gen_tables_smoke() -> None:
    if TABLES_OUT.exists():
        TABLES_OUT.unlink()

    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "reproduce_figures.py"),
         "--name", "gen_tables"],
        check=True, cwd=str(REPO),
    )

    assert TABLES_OUT.exists(), f"{TABLES_OUT} was not written"
    body = TABLES_OUT.read_text()
    assert len(body) > 500, f"{TABLES_OUT} suspiciously short ({len(body)} chars)"
    assert "\\begin{tabular}" in body, "expected a tabular environment in main_L.tex"
