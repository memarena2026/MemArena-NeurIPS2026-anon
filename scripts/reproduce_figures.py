#!/usr/bin/env python3
"""Re-run the paper's figure / table generators.

Thin dispatcher over `memarena/figures/`. Each figure module exposes a
`main()` callable; this script lists them, runs a named one, or runs
all of them in order. Input data expectations are documented per-script
and usually require locally generated evaluation result files. The hosted
dataset download contains benchmark data only, not baseline result files.

Usage
-----
  python scripts/reproduce_figures.py --list
  python scripts/reproduce_figures.py --name analyze_finding2
  python scripts/reproduce_figures.py --all
  python scripts/reproduce_figures.py --all --out-dir out/paper_accuracy_10a5d_input
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from memarena.runtime import configure_live_output
from memarena.figures.paths import artifact_root

# Modules shipped under `memarena/figures/`. Listed in the order they
# appear in the paper (roughly §5 → §7 → Appendix).
FIGURE_MODULES: list[tuple[str, str]] = [
    ("analyze_finding2",          "Figure: Finding 2 structure-vs-compute"),
    ("analyze_deployment_triptych", "Figure: deployment triptych (on-device)"),
    ("analyze_efficiency",        "Table: on-device efficiency"),
    ("analyze_efficiency_d13",    "Table: D13 efficiency slice"),
    ("analyze_memory_amplification", "Figure: memory amplification across models"),
    ("analyze_ingest_amortization",  "Figure: ingest amortization over days"),
    ("analyze_rec_rea_scatter",   "Figure: Recall vs Reasoning scatter"),
    ("analyze_cross_judge",       "Table: cross-judge κ"),
    ("gen_fig_d6_v2",             "Figure: D6 content/social/privacy-utility triptych"),
    ("analyze_d6_split",          "Figure: D6 three-category split"),
    ("analyze_extractor_tradeoff", "Table: extractor accuracy-vs-latency tradeoff"),
    ("analyze_refusal_provenance", "Table: refusal provenance audit"),
    ("gen_tables",                "Dispatch: all paper tables"),
    ("gen_pvalues",               "Table: bootstrap p-values"),
    ("gen_privacy_table",         "Table: privacy / utility"),
    ("gen_tokenf1",               "Table: token-level F1"),
    # generate_tables_figures is a "rerun everything" meta-dispatcher that
    # subprocess-runs analyze_efficiency (and others) internally. Listing it in
    # --all causes double execution and, worse, lets its older analyze_efficiency
    # output overwrite the richer analyze_efficiency_d13 output. Invoke it
    # directly via `python -m memarena.figures.generate_tables_figures` if you
    # need the bundled runner explicitly.
]


def _print_listing() -> None:
    w = max(len(n) for n, _ in FIGURE_MODULES)
    print(f"Available figure / table modules in memarena.figures:")
    for name, desc in FIGURE_MODULES:
        print(f"  {name:<{w}}   {desc}")
    print()
    print("Each module also runs standalone: `python -m memarena.figures.<name>`")


def _run(name: str, extra: list[str]) -> int:
    mod_name = f"memarena.figures.{name}"
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as ex:
        print(f"[reproduce-figures] {name}: import failed → {ex}", file=sys.stderr)
        return 1
    # Each figure module is expected to expose either a `main()` callable or
    # be runnable via `if __name__ == '__main__'`. We prefer `main()` if it
    # exists so we can feed extra CLI args through `sys.argv`.
    orig_argv = sys.argv
    try:
        sys.argv = [mod_name, *extra]
        if hasattr(mod, "main") and callable(mod.main):
            rc = mod.main() or 0
        else:
            # Fallback: re-execute the module's top-level code.
            importlib.reload(mod)
            rc = 0
    except SystemExit as ex:
        rc = int(ex.code or 0)
    except Exception as ex:
        print(f"[reproduce-figures] {name}: runtime error → {ex}", file=sys.stderr)
        rc = 2
    finally:
        sys.argv = orig_argv
    if rc == 0:
        print(f"[reproduce-figures] {name}: ok")
    else:
        print(f"[reproduce-figures] {name}: exit {rc}")
    return rc


def main() -> int:
    configure_live_output()
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--list", action="store_true",
                   help="list all available modules and exit")
    g.add_argument("--name", help="run a single module by name")
    g.add_argument("--all", action="store_true",
                   help="run every module in sequence (best-effort)")
    ap.add_argument(
        "--out-dir",
        type=Path,
        help=(
            "artifact output root; generated files go under figures/ and tables/ "
            "inside this directory. Defaults to MEMARENA_FIGURE_OUT_DIR, then "
            "MEMARENA_RUN_DIR, then the legacy source-tree path."
        ),
    )
    ap.add_argument("remaining", nargs=argparse.REMAINDER,
                    help="extra args forwarded to the underlying module")
    args = ap.parse_args()

    if args.out_dir:
        out_dir = args.out_dir.expanduser()
        if not out_dir.is_absolute():
            out_dir = (REPO_ROOT / out_dir).resolve()
        os.environ["MEMARENA_FIGURE_OUT_DIR"] = str(out_dir)
    print(f"[reproduce-figures] artifact root: {artifact_root()}")

    if args.list or (not args.name and not args.all):
        _print_listing()
        return 0

    if args.name:
        names = [args.name]
    else:
        names = [n for n, _ in FIGURE_MODULES]

    rc_all = 0
    for n in names:
        rc = _run(n, args.remaining or [])
        rc_all = rc_all or rc
    return rc_all


if __name__ == "__main__":
    sys.exit(main())
