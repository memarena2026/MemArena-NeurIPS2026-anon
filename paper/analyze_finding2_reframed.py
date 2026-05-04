#!/usr/bin/env python3
"""Quantify the reframed Finding 2 argument from existing MemArena artefacts.

Candidate headline
------------------
"Better query-time retrieval helps, but only broader structured ingest closes
the gap."

This script consolidates the evidence for that claim from the current repo:

1. Published main-table anchors (`paper/tables/main_SML.tex`).
2. Retrieval-side ablations (dense retrieval, HippoRAG, targeted strong-RAG).
3. Broader-ingest controls (Memobase vs Oracle, H5 Oracle-structured,
   Oracle+distractors).
4. Existing significance summaries (`sanity_reports/phase9_pvalues.csv`).

Outputs
-------
* stdout: compact Markdown report for paper-writing.
* `paper/history/finding2_reframed.json`: machine-readable summary.

The script is read-only on experiment artefacts.
"""

from __future__ import annotations

import csv
import json
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_data import (  # noqa: E402
    DEFAULT_RUN,
    DIM_KEYS_PAPER,
    MODEL_ORDER,
    MODEL_TEX,
    SEEDS,
    load_all_cells,
)

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
RUN = DEFAULT_RUN

MAIN_TABLE_TEX = ROOT / "tables" / "main_SML.tex"
OUT_JSON = ROOT / "history" / "finding2_reframed.json"
PVALUES_CSV = PROJECT_ROOT / "sanity_reports" / "phase9_pvalues.csv"

PAPER_DIM_FAMILIES = {
    "direct_lookup": ["d1_cloze", "d3_factual_qa"],
    "integrative": ["d2_metadata", "d4_cross_session"],
}

SUBDIM_TO_PAPER_DIM = {
    "d1_conflict": "d4_cross_session",
    "d2_anaphora": "d4_cross_session",
    "d3_confabulation": "d5_abstention",
    "d4_permission": "d6_permission",
    "d5_cloze": "d1_cloze",
    "d6_metadata": "d2_metadata",
    "d7_qa": "d3_factual_qa",
    "d8_temporal": "d3_factual_qa",
    "d10_counterfactual": "d3_factual_qa",
}

PAPER_DIM_TO_SUBDIMS = {
    "d1_cloze": ["d5_cloze"],
    "d2_metadata": ["d6_metadata"],
    "d3_factual_qa": ["d7_qa", "d8_temporal", "d10_counterfactual"],
    "d4_cross_session": ["d1_conflict", "d2_anaphora"],
    "d5_abstention": ["d3_confabulation"],
    "d6_permission": ["d4_permission"],
}

CAPACITY_SUFFICIENT_MODELS = [m for m in MODEL_ORDER if m != "7b"]

CELL_RE = re.compile(
    r"(?P<mean>-?\d+(?:\.\d+)?)\{\\scriptsize\$\\pm\$(?P<std>-?\d+(?:\.\d+)?)\}"
)
ROW_HEADER_RE = re.compile(r"\\multirow\{4\}\{\*\}\{(?P<reader>[^}]+)\}")
METRIC_RE = re.compile(r"&\s*(Rec\.|Rea\.|Abst\.|Avg)\s*&")

DENSE_BGE_PATHS = {
    "0_6b": RUN / "eval_results/ablations/p1c/dense_bge_m3/dense_bge_m3/runs/"
             "p1c_rag_bge_m3_0_6b_s1-20260418-232748/evaluation_results_p1c_rag_bge_m3_0_6b_s1.json",
    "llama3b": RUN / "eval_results/ablations/p1c/dense_bge_m3/dense_bge_m3/runs/"
                "p1c_rag_bge_m3_llama3b_s1-20260418-233905/evaluation_results_p1c_rag_bge_m3_llama3b_s1.json",
    "7b": RUN / "eval_results/ablations/p1c/dense_bge_m3/dense_bge_m3/runs/"
          "p1c_rag_bge_m3_7b_s1-20260419-011158/evaluation_results_p1c_rag_bge_m3_7b_s1.json",
    "8b": RUN / "eval_results/ablations/p1c/dense_bge_m3/dense_bge_m3/runs/"
          "p1c_rag_bge_m3_8b_s1-20260419-000353/evaluation_results_p1c_rag_bge_m3_8b_s1.json",
    "32b": RUN / "eval_results/ablations/p1c/dense_bge_m3/dense_bge_m3/runs/"
           "p1c_rag_bge_m3_32b_s1-20260419-003926/evaluation_results_p1c_rag_bge_m3_32b_s1.json",
}

BM25_K5_PATHS = {
    "0_6b": RUN / "eval_results/ablations/p1c/bm25_k5/inmem/evaluation_results_p1c_bm25_k5_0_6b_s1.json",
    "llama3b": RUN / "eval_results/ablations/p1c/bm25_k5/inmem/evaluation_results_p1c_bm25_k5_llama3b_s1.json",
    "7b": RUN / "eval_results/ablations/p1c/bm25_k5/inmem/evaluation_results_p1c_bm25_k5_7b_s1.json",
    "8b": RUN / "eval_results/ablations/p1b/inmem/evaluation_results_p1b_rag_k5_8b_s1.json",
    "32b": RUN / "eval_results/ablations/p1c/bm25_k5/inmem/evaluation_results_p1c_bm25_k5_32b_s1.json",
}

H5_PATHS = {
    model: RUN / f"eval_results/ablations/h5_oracle_structured/evaluation_results_h5_oracle_structured_{model}_s1.json"
    for model in MODEL_ORDER
}

ORACLE_S1_PATHS = {
    model: RUN / f"eval_results/oracle/evaluation_results_oracle_{model}_4omini.json"
    for model in MODEL_ORDER
}

ORACLE_DISTRACTOR_PATHS = {
    model: RUN / "eval_results/ablations/review4/oracle_distract/oracle_with_distractors"
           / f"evaluation_results_review4_oracle_distract_{model}_s1.json"
    for model in MODEL_ORDER
}

ORACLE_RANDOM_DISTRACTOR_PATHS = {
    model: RUN / "eval_results/ablations/review4/oracle_distract/oracle_with_random_distractors"
           / f"evaluation_results_review4_oracle_distract_random_{model}_s1.json"
    for model in MODEL_ORDER
}

HIPPORAG_PATHS = {
    seed: RUN / f"eval_results_{seed}/memory_cache/memory_cache/evaluation_results_memcache_hipporag_A_paired_qwen3_8b.json"
    for seed in ["s2", "s3", "s4"]
}

TARGETED_RETRIEVAL_PATHS = {
    "plain_rag": {
        seed: RUN / ("eval_results" if seed == "s1" else f"eval_results_{seed}") / "inmem"
        / "evaluation_results_rag_8b_4omini.json"
        for seed in ["s1", "s2", "s3"]
    },
    "entity_filter": {
        seed: RUN / "eval_results/ablations/b2_rag_strong"
        / f"evaluation_results_b2_rag_entity_filter_8b_{seed}_4omini.json"
        for seed in ["s1", "s2", "s3"]
    },
    "hybrid_rrf": {
        seed: RUN / "eval_results/ablations/b2_rag_strong"
        / f"evaluation_results_b2_rag_hybrid_rrf_8b_{seed}_4omini.json"
        for seed in ["s1", "s2", "s3"]
    },
    "temporal_prior_l05": {
        seed: RUN / "eval_results/ablations/b2_rag_strong"
        / f"evaluation_results_b2_rag_temporal_prior_8b_{seed}_lam0.5_4omini.json"
        for seed in ["s1", "s2", "s3"]
    },
    "memobase": {
        seed: RUN / f"eval_results_{seed}/memory_cache/memory_cache/evaluation_results_memcache_memobase_A_paired_qwen3_8b.json"
        for seed in ["s2", "s3", "s4"]
    },
}


def _round(x: float, ndigits: int = 2) -> float:
    return round(float(x), ndigits)


def parse_main_table_means(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    """Parse `paper/tables/main_SML.tex` into reader -> metric -> backend."""
    backends = ["Oracle", "Vanilla", "RAG", "Memobase", "MemOS"]
    out: dict[str, dict[str, dict[str, float]]] = {}
    current_reader: str | None = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        m_hdr = ROW_HEADER_RE.search(line)
        if m_hdr:
            current_reader = m_hdr.group("reader")
        if current_reader is None:
            continue
        m_met = METRIC_RE.search(line)
        if not m_met:
            continue
        metric = m_met.group(1)
        cells = CELL_RE.findall(line)
        if len(cells) != 5:
            continue
        out.setdefault(current_reader, {})
        out[current_reader][metric] = {
            backend: float(mean)
            for backend, (mean, _std) in zip(backends, cells)
        }
    return out


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _summary_dim_acc(summary: dict) -> dict[str, float]:
    fam = summary.get("accuracy_by_task_family")
    if fam:
        return {k: float(v) for k, v in fam.items()}
    by_dim = summary.get("by_dimension")
    if by_dim:
        out = {}
        for k, v in by_dim.items():
            if isinstance(v, dict) and "accuracy" in v:
                out[k] = float(v["accuracy"])
            elif isinstance(v, (int, float)):
                out[k] = float(v)
        return out
    return {}


def paper_dim_acc_from_json(path: Path) -> dict[str, float]:
    """Compute paper-dimension accuracies from a raw eval JSON."""
    data = _load_json(path)
    subdim_acc = _summary_dim_acc(data["summary"])
    if not subdim_acc:
        buckets: dict[str, list[int]] = {}
        for item in data.get("details", []):
            qid = item.get("question_id") or item.get("instance_id") or ""
            dim = qid.split("_")[0]
            correct = item.get("correct")
            if correct is None:
                correct = item.get("judge_correct")
            if correct is None:
                continue
            buckets.setdefault(dim, []).append(1 if correct else 0)
        subdim_acc = {k: sum(v) / len(v) for k, v in buckets.items() if v}

    out = {}
    for paper_dim, subdims in PAPER_DIM_TO_SUBDIMS.items():
        vals = [subdim_acc[s] for s in subdims if s in subdim_acc]
        if vals:
            out[paper_dim] = sum(vals) / len(vals)
    return out


def paper_avg_from_json(path: Path) -> float:
    per_dim = paper_dim_acc_from_json(path)
    vals = [per_dim[d] for d in DIM_KEYS_PAPER[:5] if d in per_dim]
    if len(vals) != 5:
        raise ValueError(f"Missing paper dims in {path}")
    return sum(vals) / len(vals) * 100


def _mean_std(vals: list[float]) -> tuple[float, float]:
    if len(vals) == 1:
        return vals[0], 0.0
    return st.mean(vals), st.stdev(vals)


def mean_paper_dim(grid, backend: str, model: str) -> dict[str, float]:
    accs: dict[str, list[float]] = {}
    for seed in SEEDS:
        cell = grid.get((seed, backend, model))
        if cell is None:
            continue
        for dim, acc in cell.accuracy_by_paper_dim().items():
            if acc is None:
                continue
            accs.setdefault(dim, []).append(acc)
    return {dim: sum(vals) / len(vals) for dim, vals in accs.items() if vals}


def subset_accuracy_from_json(path: Path, prefixes: tuple[str, ...]) -> float:
    data = _load_json(path)
    flags = []
    for item in data.get("details", []):
        qid = item.get("question_id") or item.get("instance_id") or ""
        if not qid.startswith(prefixes):
            continue
        correct = item.get("correct")
        if correct is None:
            correct = item.get("judge_correct")
        if correct is None:
            continue
        flags.append(1 if correct else 0)
    if not flags:
        raise ValueError(f"No matching rows in {path}")
    return sum(flags) / len(flags) * 100


def main_table_anchor(main_means: dict[str, dict[str, dict[str, float]]]) -> dict:
    out = {}
    for reader in ["Qwen3-8B", "Qwen3-32B"]:
        row = main_means[reader]["Avg"]
        gap_total = row["Oracle"] - row["Vanilla"]
        rag_closure = 100.0 * (row["RAG"] - row["Vanilla"]) / gap_total
        out[reader] = {
            "vanilla_avg": _round(row["Vanilla"], 1),
            "rag_avg": _round(row["RAG"], 1),
            "oracle_avg": _round(row["Oracle"], 1),
            "memobase_avg": _round(row["Memobase"], 1),
            "memos_avg": _round(row["MemOS"], 1),
            "rag_closes_vanilla_to_oracle_gap_pct": _round(rag_closure, 1),
            "rag_residual_to_oracle_pp": _round(row["Oracle"] - row["RAG"], 1),
            "rag_residual_to_memobase_pp": _round(row["Memobase"] - row["RAG"], 1),
        }
    return out


def dense_retrieval_summary(main_means: dict[str, dict[str, dict[str, float]]]) -> dict:
    per_model = {}
    deltas = []
    residuals_to_memobase = []
    for model in MODEL_ORDER:
        bm25 = paper_avg_from_json(BM25_K5_PATHS[model])
        dense = paper_avg_from_json(DENSE_BGE_PATHS[model])
        reader = MODEL_TEX[model]
        memobase_pub = main_means[reader]["Avg"]["Memobase"]
        oracle_pub = main_means[reader]["Avg"]["Oracle"]
        delta = dense - bm25
        deltas.append(delta)
        residuals_to_memobase.append(memobase_pub - dense)
        per_model[reader] = {
            "bm25_k5_avg": _round(bm25),
            "dense_bge_m3_avg": _round(dense),
            "delta_vs_bm25_k5_pp": _round(delta),
            "residual_to_memobase_pub_pp": _round(memobase_pub - dense),
            "residual_to_oracle_pub_pp": _round(oracle_pub - dense),
        }
    return {
        "per_reader": per_model,
        "mean_delta_vs_bm25_k5_pp": _round(st.mean(deltas)),
        "mean_residual_to_memobase_pub_pp": _round(st.mean(residuals_to_memobase)),
    }


def hipporag_summary(main_means: dict[str, dict[str, dict[str, float]]]) -> dict:
    vals = [paper_avg_from_json(path) for path in HIPPORAG_PATHS.values()]
    mean_avg, std_avg = _mean_std(vals)
    q8 = main_means["Qwen3-8B"]["Avg"]
    return {
        "per_seed_avg": {seed: _round(paper_avg_from_json(path)) for seed, path in HIPPORAG_PATHS.items()},
        "mean_avg": _round(mean_avg),
        "std_avg": _round(std_avg),
        "gap_vs_q8_rag_pub_pp": _round(mean_avg - q8["RAG"]),
        "gap_vs_q8_oracle_pub_pp": _round(mean_avg - q8["Oracle"]),
        "gap_vs_q8_memobase_pub_pp": _round(mean_avg - q8["Memobase"]),
        "gap_vs_q8_memos_pub_pp": _round(mean_avg - q8["MemOS"]),
    }


def targeted_retrieval_probe_summary() -> dict:
    prefixes = ("d1_", "d3_", "d4_")
    per_variant = {}
    means = {}
    for variant, paths in TARGETED_RETRIEVAL_PATHS.items():
        vals = [subset_accuracy_from_json(path, prefixes) for path in paths.values()]
        mean_avg, std_avg = _mean_std(vals)
        per_variant[variant] = {
            "per_seed_subset_acc": {seed: _round(subset_accuracy_from_json(path, prefixes)) for seed, path in paths.items()},
            "mean_subset_acc": _round(mean_avg),
            "std_subset_acc": _round(std_avg),
        }
        means[variant] = mean_avg
    return {
        "subset_note": "Targeted 540-item wave-1 probe on internal d1/d3/d4 instances only.",
        "variants": per_variant,
        "best_query_time_variant": {
            "name": "hybrid_rrf",
            "mean_subset_acc": _round(means["hybrid_rrf"]),
            "delta_vs_plain_rag_pp": _round(means["hybrid_rrf"] - means["plain_rag"]),
            "residual_to_memobase_pp": _round(means["memobase"] - means["hybrid_rrf"]),
        },
    }


def memobase_vs_oracle_summary(
    main_means: dict[str, dict[str, dict[str, float]]], grid
) -> dict:
    avg_rows = {}
    for reader in MODEL_TEX.values():
        row = main_means[reader]["Avg"]
        avg_rows[reader] = {
            "oracle_avg": _round(row["Oracle"], 1),
            "memobase_avg": _round(row["Memobase"], 1),
            "delta_pp": _round(row["Memobase"] - row["Oracle"], 1),
        }

    reasoning_rows = {}
    pvalues = list(csv.DictReader(PVALUES_CSV.open()))
    for model in CAPACITY_SUFFICIENT_MODELS:
        reader = MODEL_TEX[model]
        metric_row = next(
            r for r in pvalues
            if r["A"] == f"memobase_{model}" and r["B"] == f"oracle_{model}" and r["metric"] == "Rea"
        )
        reasoning_rows[reader] = {
            "delta_pp": _round(float(metric_row["delta_pp"])),
            "ci_lo_pp": _round(float(metric_row["ci_lo_pp"])),
            "ci_hi_pp": _round(float(metric_row["ci_hi_pp"])),
            "p_two_sided": float(metric_row["p_two_sided"]),
        }

    per_dim = {}
    for dim in DIM_KEYS_PAPER[:6]:
        memo_minus_oracle = []
        for model in CAPACITY_SUFFICIENT_MODELS:
            memo = mean_paper_dim(grid, "memobase", model)
            oracle = mean_paper_dim(grid, "oracle", model)
            memo_minus_oracle.append((memo[dim] - oracle[dim]) * 100)
        per_dim[dim] = _round(st.mean(memo_minus_oracle))

    family = {}
    for family_name, dims in PAPER_DIM_FAMILIES.items():
        family[family_name] = {
            "dims": dims,
            "mean_memobase_minus_oracle_pp": _round(st.mean(per_dim[d] for d in dims)),
        }

    return {
        "avg_pub": avg_rows,
        "reasoning_paired_bootstrap": reasoning_rows,
        "capacity_sufficient_dim_deltas_pp": per_dim,
        "family_aggregate_pp": family,
        "headline_contrast": {
            "d2_metadata_pp": per_dim["d2_metadata"],
            "d1_cloze_pp": per_dim["d1_cloze"],
            "integrative_family_pp": family["integrative"]["mean_memobase_minus_oracle_pp"],
            "direct_lookup_family_pp": family["direct_lookup"]["mean_memobase_minus_oracle_pp"],
        },
    }


def h5_oracle_structured_summary() -> dict:
    per_model = {}
    deltas = []
    for model in MODEL_ORDER:
        oracle = paper_avg_from_json(ORACLE_S1_PATHS[model])
        structured = paper_avg_from_json(H5_PATHS[model])
        delta = structured - oracle
        deltas.append(delta)
        per_model[MODEL_TEX[model]] = {
            "oracle_s1_avg": _round(oracle),
            "oracle_structured_avg": _round(structured),
            "delta_pp": _round(delta),
        }
    return {
        "per_reader": per_model,
        "mean_delta_pp": _round(st.mean(deltas)),
    }


def oracle_distractor_summary() -> dict:
    per_model = {}
    positive_four = []
    random_minus_recency = []
    for model in MODEL_ORDER:
        oracle = paper_avg_from_json(ORACLE_S1_PATHS[model])
        recency = paper_avg_from_json(ORACLE_DISTRACTOR_PATHS[model])
        random = paper_avg_from_json(ORACLE_RANDOM_DISTRACTOR_PATHS[model])
        delta = recency - oracle
        if model != "7b":
            positive_four.append(delta)
        random_minus_recency.append(random - recency)
        per_model[MODEL_TEX[model]] = {
            "oracle_s1_avg": _round(oracle),
            "oracle_plus_recency_distractors_avg": _round(recency),
            "delta_recency_minus_oracle_pp": _round(delta),
            "oracle_plus_random_distractors_avg": _round(random),
            "delta_random_minus_recency_pp": _round(random - recency),
        }
    return {
        "per_reader": per_model,
        "mean_delta_recency_minus_oracle_four_non_mistral_pp": _round(st.mean(positive_four)),
        "mean_delta_random_minus_recency_all_readers_pp": _round(st.mean(random_minus_recency)),
    }


def _print_report(payload: dict) -> None:
    print("# Finding 2 Reframed")
    print()
    print("Candidate claim: Better query-time retrieval helps, but only broader structured ingest closes the gap.")
    print()

    print("## 1. Main-table anchors")
    for reader, row in payload["main_table_anchor"].items():
        print(
            f"- {reader}: Vanilla {row['vanilla_avg']:.1f}, RAG {row['rag_avg']:.1f}, "
            f"Oracle {row['oracle_avg']:.1f}, Memobase {row['memobase_avg']:.1f}, "
            f"MemOS {row['memos_avg']:.1f}; RAG closes {row['rag_closes_vanilla_to_oracle_gap_pct']:.1f}% "
            f"of the Vanilla->Oracle gap."
        )
    print()

    dense = payload["retrieval_side"]["dense_bge_m3"]
    print("## 2. Retrieval-side improvements help, but plateau below the structured-ingest regime")
    print(
        f"- Dense BGE-M3 vs BM25@k=5: mean +{dense['mean_delta_vs_bm25_k5_pp']:.2f} pp across 5 readers; "
        f"mean residual to published Memobase = {dense['mean_residual_to_memobase_pub_pp']:.2f} pp."
    )
    q8_dense = dense["per_reader"]["Qwen3-8B"]
    print(
        f"- Qwen3-8B dense anchor: Avg {q8_dense['dense_bge_m3_avg']:.2f}, "
        f"still {q8_dense['residual_to_oracle_pub_pp']:.2f} pp below published Oracle "
        f"and {q8_dense['residual_to_memobase_pub_pp']:.2f} pp below published Memobase."
    )

    hippo = payload["retrieval_side"]["hipporag_q8"]
    print(
        f"- HippoRAG x Qwen3-8B: {hippo['mean_avg']:.2f} +- {hippo['std_avg']:.2f}; "
        f"{hippo['gap_vs_q8_rag_pub_pp']:+.2f} pp vs published RAG, "
        f"{hippo['gap_vs_q8_oracle_pub_pp']:+.2f} pp vs published Oracle."
    )

    probe = payload["retrieval_side"]["targeted_wave1_probe"]["best_query_time_variant"]
    print(
        f"- Targeted wave-1 probe ({payload['retrieval_side']['targeted_wave1_probe']['subset_note']}): "
        f"best hybrid-RRF reaches {probe['mean_subset_acc']:.2f}, "
        f"{probe['delta_vs_plain_rag_pp']:+.2f} pp vs plain RAG, "
        f"but still {probe['residual_to_memobase_pp']:.2f} pp below Memobase on the same subset."
    )
    print()

    memo = payload["broader_ingest"]["memobase_vs_oracle"]
    print("## 3. Fixed evidence is not the whole story; broader structured ingest changes the ceiling")
    for reader in ["Qwen3-0.6B", "Llama-3.2-3B", "Qwen3-8B", "Qwen3-32B"]:
        row = memo["avg_pub"][reader]
        print(
            f"- Memobase vs Oracle Avg on {reader}: {row['memobase_avg']:.1f} vs {row['oracle_avg']:.1f} "
            f"({row['delta_pp']:+.1f} pp)."
        )
    print(
        f"- Capacity-sufficient family aggregate: integrative dims {memo['headline_contrast']['integrative_family_pp']:+.2f} pp "
        f"vs direct-lookup dims {memo['headline_contrast']['direct_lookup_family_pp']:+.2f} pp "
        f"(Memobase - Oracle)."
    )

    h5 = payload["broader_ingest"]["h5_oracle_structured"]
    print(
        f"- H5 control (structure over Oracle evidence only): mean {h5['mean_delta_pp']:.2f} pp vs raw Oracle."
    )

    distract = payload["broader_ingest"]["oracle_distractors"]
    print(
        f"- Oracle + recency distractors: mean +{distract['mean_delta_recency_minus_oracle_four_non_mistral_pp']:.2f} pp "
        f"on the four non-Mistral readers."
    )
    print()

    print("## 4. Scope / exception")
    mistral = memo["avg_pub"]["Mistral-7B"]
    print(
        f"- Mistral-7B is the main exception: Memobase {mistral['memobase_avg']:.1f} vs Oracle {mistral['oracle_avg']:.1f} "
        f"({mistral['delta_pp']:+.1f} pp), consistent with the known ingest-side schema/capacity issue."
    )


def main() -> None:
    main_means = parse_main_table_means(MAIN_TABLE_TEX)
    grid = load_all_cells(DEFAULT_RUN)

    payload = {
        "candidate": "Better query-time retrieval helps, but only broader structured ingest closes the gap.",
        "data_sources": {
            "published_main_table": str(MAIN_TABLE_TEX),
            "run_root": str(RUN),
            "paired_bootstrap": str(PVALUES_CSV),
        },
        "main_table_anchor": main_table_anchor(main_means),
        "retrieval_side": {
            "dense_bge_m3": dense_retrieval_summary(main_means),
            "hipporag_q8": hipporag_summary(main_means),
            "targeted_wave1_probe": targeted_retrieval_probe_summary(),
        },
        "broader_ingest": {
            "memobase_vs_oracle": memobase_vs_oracle_summary(main_means, grid),
            "h5_oracle_structured": h5_oracle_structured_summary(),
            "oracle_distractors": oracle_distractor_summary(),
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    _print_report(payload)
    print()
    print(f"[finding2-reframed] wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
