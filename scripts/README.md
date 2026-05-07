# Scripts Directory

This directory contains MemArena's operational entry points: setup helpers,
dataset downloaders, service launchers, evaluation runners, D6 rejudging tools,
latency measurement scripts, ablation wrappers, and figure/table reproduction
utilities.

Run scripts from the repository root unless a script explicitly says otherwise.
Most experiment runners write artifacts under `out/`; the anonymous repository
does not include baseline result JSON/JSONL files.

## Common Entry Points

| Script | Purpose |
|---|---|
| `setup_venv.sh` | Create or refresh the repository-local Python virtual environment. |
| `activate_venv.sh` | Source this file to activate `.venv` for MemArena commands. |
| `verify_install.py` | Post-install health check for imports, dry-run tests, and local wiring. |
| `download_dataset.py` | Download the MemArena-L dataset from a configurable Hugging Face dataset repository. |
| `run_masim.py` | Launch MASim generation against an OpenAI-compatible SGLang endpoint. |
| `run_eval_matrix.sh` | Run a backend x reader x trial accuracy matrix. |
| `run_accuracy.py` | Python wrapper around `eval.cli` for one accuracy cell or dry-run evaluation. |
| `llmjudge.py` | Judge existing answer artifacts without rerunning reader inference. |
| `summarize_eval_progress.py` | Summarize matrix completion from manifests and logs. |
| `reproduce_figures.py` | Dispatch figure/table artifact generators from `memarena/figures/`. |

## Environment And Services

| Script | Purpose |
|---|---|
| `check_service_health.sh` | Check the five-reader service matrix: SGLang plus memory backends. |
| `setup_memory_backends.sh` | Set up official Memobase and MemOS service checkouts and containers. |
| `spin_per_trial_stacks.sh` | Start/stop per-cell isolated Memobase/MemOS docker-compose stacks. |
| `run_backend_sequential_trials_parallel.sh` | Run one reader with backends sequentially and trials in parallel. |
| `run_isolated_per_cell.sh` | Run a matrix where each backend/model/trial cell owns an isolated memory stack. |
| `git_add_results.sh` | Helper for force-adding whitelisted result artifacts on data-snapshot branches. Not needed for normal anonymous code release. |

## Dataset And Artifact Plumbing

| Script | Purpose |
|---|---|
| `build_experiments_index.py` | Build `experiments_index.csv` for the result registry. |
| `build_run_merged.py` | Stitch multiple output directories into one `MEMARENA_RUN_DIR` tree for figure scripts. |
| `promote_light_to_full.py` | Convert lightweight answer files into full answer-record JSON expected by `llmjudge.py`. |
| `stage_paper_accuracy_inputs.py` | Stage completed eval-matrix cells into the figure/table input layout. |
| `__init__.py` | Package marker for importing helper modules under `scripts`. |

## Main Evaluation Runners

| Script | Purpose |
|---|---|
| `run_l_baselines.sh` | Run baseline backends (`vanilla`, `oracle`, `inmem`) across MemArena-L readers and seeds. |
| `run_l_ablations.sh` | Run ablation backends across MemArena-L readers and seeds with the baseline-run lifecycle. |
| `run_l_all_models.sh` | Run structured memory backends such as Memobase/MemOS across readers and seeds. |
| `run_l_parallel_8b_32b.sh` | Specialized parallel MemOS build for 8B and 32B readers on one multi-GPU node. |
| `run_memsearch_phase2.sh` | Run MemSearch answer/judge cells against a shared prebuilt MemSearch cache. |
| `run_missing_memobase_memos.sh` | Fill missing Memobase/MemOS cells and upgrade D6 labels; outputs remain under `out/`. |

## D6 Permission And Rejudging Tools

| Script | Purpose |
|---|---|
| `rerun_d6_judge.py` | Rejudge main-grid D6 records with the current 5-label rubric. |
| `rerun_d6_judge_ablations.py` | Rejudge D6 records in standalone ablation output trees. |
| `rerun_d6_judge_g3a.py` | Rejudge G3a access-marker cells and harmonize D6 fields. |
| `rejudge_d6_armB.py` | Apply Arm-B D6 rejudge results to existing evaluation JSONs. |
| `backfill_d6_armB.py` | Backfill Arm-B D6 labels from sidecar JSONL files into evaluation JSONs. |
| `judge_d6_self_probe.py` | Judge D6 self-ego probe answer files with the 5-label rubric. |
| `judge_d6_ablation_3label.py` | Classify non-leak DENY responses into `NO_ACCESS`, `DONT_KNOW`, or `OTHER`. |
| `analyze_d6_ablation_buckets.py` | Summarize the 3-bucket labels and leak/disclose rates for D6 ablations. |
| `analyze_d6_paired_probe.py` | Compare third-party D6 probes with self-ego probes to separate evidence availability from disclosure policy. |
| `sanity_d6_judge.py` | Run hand-picked sanity cases through the current D6 judge. |

## Ablation And Result-Run Wrappers

| Script | Purpose |
|---|---|
| `run_b1_identity.py` | Requester-identity ablation for permission questions. |
| `run_b5_topk_sweep.sh` | Original BM25 top-k sweep wrapper; retained for provenance. |
| `run_b5_topk_sweep_v2.sh` | Corrected BM25/in-memory top-k sweep that actually changes retrieval top-k. |
| `run_d4_rules_baseline.py` | Deterministic rules baseline for permission/access-control instances. |
| `run_d6_self_probe.sh` | Run D6 self-ego probe inference across the 5x5x3 grid. |
| `run_d6_self_probe_8b_32b.sh` | Narrow self-probe wrapper for the 8B and 32B reader cells. |
| `run_d6_self_probe_complete.sh` | Completion wrapper for remaining D6 self-probe phases and cache builds. |
| `run_g3a_access_marker.sh` | G3a access-marker ablation using explicit allow/deny markers. |
| `run_g3a_norm_binding.sh` | Reader-side norm-binding ablation for D6 under Oracle evidence. |
| `run_omniscient_s3s4.sh` | Omniscient-backend ablation for additional seeds. |
| `run_step5_oracle_gated.sh` | A7 oracle-gated retrieval-time permission filtering sweep. |
| `run_step6_memobase_writer32.sh` | A4 Memobase writer-32B / smaller-reader sweep. |
| `run_step7_temporal_sweep.sh` | A8 temporal-window sweep over multiple recency windows. |
| `run_step9_full_panel.sh` | Full-panel rerun for temporal sweep plus Memobase writer32 8B cells. |

## Latency And Hardware Measurement

| Script | Purpose |
|---|---|
| `run_latency.py` | Single-cell latency harness for TTFT, throughput, total time, and optional ingest timing. |
| `run_latency_spark.sh` | Single-button Spark/edge-node latency workflow with low-resource defaults. |
| `run_latency_answer.sh` | Answer-only latency replay for one backend-reader cell using a fixed query subset. |
| `run_latency_answer_sweep.sh` | Multi-reader answer-phase latency sweep over vanilla, oracle, and in-memory backends. |
| `run_latency_memobase_ingest.sh` | Measure Memobase ingest cost for a fixed ego subset. |
| `run_latency_memsearch_ingest.sh` | Measure MemSearch ingest cost and embedding-side hardware telemetry. |
| `select_latency_queries.py` | Select a balanced fixed query subset for latency replay. |
| `fit_latency_model.py` | Fit per-reader latency models and summarize backend search overhead. |
| `gen_ttft_table.py` | Emit the main-text TTFT table from latency model artifacts. |
| `regen_ttft_main_table.py` | Regenerate the TTFT table artifact from fitted latency outputs. |
| `gen_latency_appendix_tables.py` | Emit appendix latency-fit and predicted-latency LaTeX tables. |
| `hw_probe.py` | Background hardware telemetry sampler for power, utilization, and memory. |
| `aggregate_hw_probe_simple.py` | Aggregate hardware telemetry over one cell when per-query markers are unavailable. |

## Figure, Table, And Analysis Helpers

| Script | Purpose |
|---|---|
| `reproduce_figures.py` | List or run figure/table generators under `memarena/figures/`. |
| `gen_ttft_table.py` | Build the main TTFT table artifact. |
| `gen_latency_appendix_tables.py` | Build latency-methodology appendix tables. |
| `analyze_d6_ablation_buckets.py` | Produce D6 ablation-bucket summaries for appendix tables. |
| `analyze_d6_paired_probe.py` | Produce paired third-party/self-ego D6 analysis artifacts. |

## Retrieval And External Backend Experiments

| Script | Purpose |
|---|---|
| `run_hipporag_index.py` | Build a HippoRAG index over MASim transcript sessions. |
| `run_hipporag_retrieve.py` | Retrieve from a HippoRAG index and emit memory-cache JSONL rows. |
| `run_memsearch_phase2.sh` | Run answer/judge cells using a reader-agnostic MemSearch cache. |

## `scripts/reproduce/` Utilities

These are lower-level reproduction and ablation utilities used for paper audits,
appendix experiments, and older reproduction paths.

| Script | Purpose |
|---|---|
| `reproduce/analyze_llm_timing.py` | Analyze LLM call timing from a MASim pipeline run, grouped by phase. |
| `reproduce/bootstrap_cis.py` | Compute bootstrap confidence intervals for evaluated cells. |
| `reproduce/build_memory_cache.py` | Build frozen retrieval caches for memory-system backends with day-by-day ingest. |
| `reproduce/compute_all_metrics.py` | Compute per-dimension metrics from evaluation result files. |
| `reproduce/cpu2_context_overlap.py` | Score whether D4 source sessions appeared in each backend's actual context. |
| `reproduce/cpu3_privacy_utility_auc.py` | Compute privacy/utility AUC-style summaries for permission instances. |
| `reproduce/eval_cleanup_watchdog.sh` | Keep evaluation result files synchronized with answer files during fresh reproduction runs. |
| `reproduce/generate_dataset.py` | Legacy compatibility wrapper for an older dataset-generation CLI. |
| `reproduce/generate_dataset.sh` | Shell compatibility wrapper for `reproduce/generate_dataset.py`. |
| `reproduce/hw_aggregate.py` | Aggregate hardware telemetry with per-query marker files. |
| `reproduce/hw_capability_probe.py` | Probe hardware telemetry capabilities and write a JSON report. |
| `reproduce/hw_probe.py` | Older hardware telemetry sampler used by Spark reproduction scripts. |
| `reproduce/retrieval_ablation.py` | Analyze top-k retrieval budget ablations from result directories. |
| `reproduce/run_b2_rag_strong.py` | Stronger RAG ablation with hybrid/dense retrieval variants. |
| `reproduce/run_b3_memobase_time_indexed.py` | Time-indexed Memobase ablation over existing caches. |
| `reproduce/run_c1_rag_provenance.py` | RAG-with-provenance chunk-header ablation. |
| `reproduce/run_h2_llm_sum_rag.py` | LLM summarization over retrieved RAG evidence before answering. |
| `reproduce/run_h3_policy_header.py` | D6 Oracle run with explicit policy header conditioning. |
| `reproduce/run_h4_reranker.py` | Cross-encoder reranker baseline on BM25 hits. |
| `reproduce/run_h5_oracle_structured.py` | Feed Oracle evidence through Memobase structured extraction and answer from profiles. |

## Result Files

Scripts that run experiments generally write under `out/`, including:

| Artifact Pattern | Meaning |
|---|---|
| `answer_results_*.json` | Reader outputs before judging. |
| `evaluation_results_*.json` | Judged accuracy records. |
| `memory_cache/*.jsonl` | Frozen memory-backend retrieval caches. |
| `matrix_logs/*.log` | Per-cell run logs. |
| `hw/*.csv` and `hw/*.json` | Hardware telemetry and aggregates. |

These artifacts are intentionally separate from the code release. Do not assume
they exist in a fresh checkout.
