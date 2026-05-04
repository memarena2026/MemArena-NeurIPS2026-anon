"""Emit a 4-reader × 5-backend TTFT table for the main-text efficiency finding.

Cells = T_search(backend) + T_TTFT_LLM(reader, backend), in ms (p50).
Italic † cells are extrapolated (memsearch/memobase × non-0_6b readers).
"""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC  = REPO / "out" / "latency_selection" / "llm_fit_models.json"
OUT  = REPO / "paper" / "tables" / "ttft_main.tex"

READERS = ["0_6b", "llama3b", "8b", "32b"]
READER_LABEL = {
    "0_6b":    r"Qwen3-0.6B",
    "llama3b": r"Llama-3.2-3B",
    "8b":      r"Qwen3-8B",
    "32b":     r"Qwen3-32B-AWQ",
}
BACKENDS = ["vanilla", "oracle", "inmem", "memobase", "memsearch"]
BACKEND_LABEL = {
    "vanilla":   r"Vanilla",
    "oracle":    r"Oracle",
    "inmem":     r"InMem",
    "memobase":  r"Memobase",
    "memsearch": r"MemSearch",
}


def fmt_total(p):
    if not p:
        return r"\texttt{n/a}"
    val = p["t_ttft_total_ms"]
    if p["extrapolated"]:
        return rf"\textit{{{val:.0f}}}$^\dagger$"
    return f"{val:.0f}"


def fmt_decomp(p):
    if not p:
        return r"\texttt{n/a}"
    s = p["t_search_meas_ms"]; l = p["t_ttft_llm_ms"]
    return f"{s:.0f}{{\\scriptsize\\,+\\,}}{l:.0f}"


def main():
    d = json.loads(SRC.read_text())
    pred = d["predicted_ttft_cells"]

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        r"\caption{End-to-end time-to-first-token (TTFT, ms, p50) on Spark GB10 at \texttt{concurrency=1}. Each cell is decomposed as $T_{\text{search}} + T_{\text{TTFT,LLM}}$ — search overhead from the backend's retrieval pipeline plus the reader's prefill cost on the resulting prompt. \emph{Roman}: directly measured. \emph{Italic with $\dagger$}: $T_{\text{TTFT,LLM}}$ is composed from this reader's TTFT fit (Eq.~\ref{eq:lat-fit-ttft}) on \texttt{0\_6b}'s prompt-length distribution, since memsearch / memobase were exercised only on \texttt{0\_6b} (\S\ref{app:lat:extrapolation}). Each backend's $T_{\text{search}}$ row reports the per-question retrieval median used to build every cell in that column. \texttt{n/a} marks Mistral-7B (\S\ref{app:lat:caveats}).}",
        r"\label{tab:ttft-main}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "r" * len(BACKENDS) + r"}",
        r"\toprule",
        r"Reader & " + " & ".join(BACKEND_LABEL[b] for b in BACKENDS) + r" \\",
        r"\midrule",
    ]

    for r in READERS:
        cells = [fmt_total(pred.get(f"{b}_{r}")) for b in BACKENDS]
        lines.append(f"{READER_LABEL[r]} & " + " & ".join(cells) + r" \\")

    lines += [r"\midrule"]
    sh = d["search_overhead_ms"]
    sh_row = ["$T_{\\text{search}}$"] + [
        f"{(sh.get(b) or {}).get('median', 0.0):.0f}" for b in BACKENDS
    ]
    lines.append(" & ".join(sh_row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
