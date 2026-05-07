# MemArena Hugging Face Release Playbook

This is the public-release playbook for **MemArena-L v1.0.0**. The repository
contains code, release metadata, and reproducibility scripts; the dataset
payload is uploaded to Hugging Face.

## Phase 0 - Inputs

You need the final MemArena-L dataset directory:

```text
MASim/datasets/memarena_l/
├── agents_personas.jsonl.gz
├── agent_schedules.jsonl.gz
├── corpus_sessions.jsonl.gz
├── events.jsonl
├── groups.json
├── locations.json
├── graph_edges.jsonl
├── ego_projections.json
├── ego_session_map.json
├── day_checkpoint.json
├── pipeline_report.json
└── eval_instances/{d1,d2,d3,d4,d5,d6,d7,d8,d10}_*.jsonl
```

Verify the payload:

```bash
test -f /path/to/MASim/datasets/memarena_l/corpus_sessions.jsonl.gz
test -f /path/to/MASim/datasets/memarena_l/eval_instances/d10_counterfactual.jsonl
```

## Phase 1 - Build a Tarball Release

Build the public release bundle:

```bash
python -m memarena.tools.bundle_memarena \
  --version 1.0.0 \
  --dataset-dir /path/to/MASim/datasets/memarena_l \
  --out out/release/
```

This produces:

```text
out/release/
├── memarena-l-v1.0.0.tar.gz
├── memarena-l-v1.0.0.SHA256SUMS
├── LICENSE
└── CITATION.cff
```

Verify checksums:

```bash
cd out/release
shasum -a 256 -c memarena-l-v1.0.0.SHA256SUMS
```

## Phase 2 - Prepare the Hugging Face Dataset Repo

Create a Hugging Face **Dataset** repository, then upload either:

1. the expanded dataset file tree, so users can browse `eval_instances/` and
   individual corpus files directly; or
2. `memarena-l-v1.0.0.tar.gz` plus `LICENSE`, `CITATION.cff`, and
   `SHA256SUMS`, if you want a compact archive-first release.

For the expanded layout, stage a folder manually:

```bash
mkdir -p out/hf_public
rsync -a /path/to/MASim/datasets/memarena_l/ out/hf_public/
cp docs/DATASET_CARD.md out/hf_public/README.md
cp docs/LICENSE out/hf_public/LICENSE
cp docs/CITATION.cff out/hf_public/CITATION.cff
cp docs/memarena-croissant.json out/hf_public/croissant.json
```

`docs/DATASET_CARD.md` contains the Hugging Face `configs` YAML that exposes
each `eval_instances/d*.jsonl` file as a separate viewer subset. Do not remove
that front matter: the task-specific `ground_truth` objects intentionally have
different schemas, and the raw corpus files contain heterogeneous nested JSON
metadata that HF's automatic Parquet converter cannot cast as one table. The
raw corpus files remain downloadable, but the HF viewer should be limited to
the eval-instance subsets.

Before upload, make sure `out/hf_public/croissant.json` uses the final
Hugging Face dataset URL:
`https://huggingface.co/datasets/anonymous/memarena-l`.

Upload:

```bash
source .venv/bin/activate
python -m pip install -U huggingface_hub
hf upload <owner>/<dataset> out/hf_public . --repo-type dataset
```

## Phase 3 - Post-upload Checks

Open the Hugging Face page and verify:

- `README.md` renders as the dataset card.
- `LICENSE` and `CITATION.cff` are visible.
- `corpus_sessions.jsonl.gz`, `pipeline_report.json`, and the nine released
  eval files (`d1`, `d2`, `d3`, `d4`, `d5`, `d6`, `d7`, `d8`, `d10`) are present.
- The dataset URL in `README.md`, `pyproject.toml`, and `croissant.json`
  matches the final Hugging Face repo.

Then test download from a clean checkout:

```bash
python scripts/download_dataset.py --out data/
```

Expected output:

```text
data/
├── hf_snapshot/
└── benchmark/
    ├── corpus_sessions.jsonl.gz
    ├── pipeline_report.json
    └── eval_instances/
```

## Phase 4 - Paper and Code Links

After the paper URL is final, update:

- `README.md`
- `docs/DATASET_CARD.md`
- `docs/CITATION.cff`
- `pyproject.toml`
- `docs/memarena-croissant.json`

Keep this repository focused on release code, documentation, reproducibility
scripts, and metadata. Do not commit paper drafts, review logs, local run
outputs, or copied MemArena workspace artifacts.
