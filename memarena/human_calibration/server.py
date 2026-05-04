#!/usr/bin/env python3
"""LAN-accessible human-calibration server with multi-reviewer progress sync.

Serves the static index.html + sample.json from this directory and adds a
tiny JSON API for persisting labels and computing live agreement metrics
across three annotators (IDs 0/1/2).

Endpoints:
  GET  /                     -> index.html
  GET  /sample.json          -> the sample data
  GET  /api/state            -> {reviewers, counts, total_items}
  GET  /api/labels/<name>    -> {reviewer, labels: {id: value}}
  POST /api/label            -> body: {reviewer, id, value}
  GET  /api/stats            -> live Fleiss/Cohen kappa across all reviewers

Label values:
  * Binary track (D2-D5): 0 | 1 | "skip" | null(erase)
  * D6 5-label track: one of {DISCLOSE_CORRECT, DISCLOSE_WRONG,
    DONT_KNOW, REFUSE, OTHER} | "skip" | null

Storage: ``labels.json`` keyed by reviewer name → no merge across reviewers.

Usage:
  python3 -m memarena.human_calibration.server --port 8080
  # then open http://<your-lan-ip>:8080/ on any device on the same network
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import threading
from collections import Counter
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
LABELS_FILE = ROOT / "labels.json"
SAMPLE_FILE = ROOT / "sample.json"
_lock = threading.Lock()

D6_LABELS = ("DISCLOSE_CORRECT", "DISCLOSE_WRONG", "DONT_KNOW", "REFUSE", "OTHER")
BINARY_VALUES = (0, 1)
SKIP = "skip"
DEFAULT_REVIEWERS = ("0", "1", "2")


def _load_labels() -> dict:
    if LABELS_FILE.exists():
        try:
            return json.loads(LABELS_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {"reviewers": {}}


def _save_labels(data: dict) -> None:
    tmp = LABELS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(LABELS_FILE)


def _load_sample() -> list[dict]:
    try:
        return json.loads(SAMPLE_FILE.read_text()).get("sample", [])
    except Exception:
        return []


def _is_valid_value(value, dim_kind: str) -> bool:
    if value is None or value == SKIP:
        return True
    if dim_kind == "d6":
        return value in D6_LABELS
    return value in BINARY_VALUES


# ----------------- agreement metrics -----------------

def _cohens_kappa(rater_a: list, rater_b: list, *, categories: list) -> tuple[float, int]:
    """Cohen's κ over a fixed category list. Returns (kappa, n)."""
    n = len(rater_a)
    if n < 2:
        return float("nan"), n
    cat_index = {c: i for i, c in enumerate(categories)}
    K = len(categories)
    matrix = [[0] * K for _ in range(K)]
    for a, b in zip(rater_a, rater_b):
        if a not in cat_index or b not in cat_index:
            continue
        matrix[cat_index[a]][cat_index[b]] += 1
    total = sum(sum(row) for row in matrix)
    if total == 0:
        return float("nan"), 0
    po = sum(matrix[i][i] for i in range(K)) / total
    a_marg = [sum(matrix[i]) / total for i in range(K)]
    b_marg = [sum(matrix[i][j] for i in range(K)) / total for j in range(K)]
    pe = sum(a_marg[i] * b_marg[i] for i in range(K))
    if abs(1 - pe) < 1e-12:
        return 1.0 if abs(po - 1) < 1e-12 else 0.0, total
    return (po - pe) / (1 - pe), total


def _fleiss_kappa(items_ratings: list[list], *, categories: list) -> tuple[float, int]:
    """Fleiss' κ. ``items_ratings[i]`` = list of categorical labels (one per rater)
    for item i. Items are dropped if any rater is missing/invalid."""
    cat_index = {c: i for i, c in enumerate(categories)}
    K = len(categories)
    rows = []
    for ratings in items_ratings:
        if any(r not in cat_index for r in ratings):
            continue
        n = len(ratings)
        if n < 2:
            continue
        counts = [0] * K
        for r in ratings:
            counts[cat_index[r]] += 1
        rows.append((counts, n))
    N = len(rows)
    if N < 2:
        return float("nan"), N
    n = rows[0][1]
    if any(r[1] != n for r in rows):
        return float("nan"), N
    p_j = [sum(r[0][j] for r in rows) / (N * n) for j in range(K)]
    P_e = sum(p ** 2 for p in p_j)
    P_bar = sum(
        (sum(c * c for c in counts) - n) / (n * (n - 1))
        for counts, _ in rows
    ) / N
    if abs(1 - P_e) < 1e-12:
        return 1.0 if abs(P_bar - 1) < 1e-12 else 0.0, N
    return (P_bar - P_e) / (1 - P_e), N


def _majority_vote(values: list, *, allow_ties: bool):
    """Return majority value; ``None`` if no clear winner under allow_ties=False."""
    if not values:
        return None
    counts = Counter(values)
    top, top_n = counts.most_common(1)[0]
    runners = [v for v, n in counts.items() if n == top_n and v != top]
    if runners and not allow_ties:
        return None
    return top


def _compute_stats(reviewers: list[str]) -> dict:
    sample = _load_sample()
    if not sample:
        return {"error": "no sample"}
    by_id = {r["id"]: r for r in sample}

    with _lock:
        data = _load_labels()
        per_reviewer = {
            name: data.get("reviewers", {}).get(name, {}) for name in reviewers
        }

    progress = {}
    per_dim_progress = {}
    for name, lab in per_reviewer.items():
        completed = [iid for iid, v in lab.items()
                     if iid in by_id and v != SKIP and v is not None]
        progress[name] = {
            "completed": len(completed),
            "skipped": sum(1 for v in lab.values() if v == SKIP),
            "total": len(sample),
        }
        per_dim = {}
        for iid in completed:
            dim = by_id[iid]["dim"]
            per_dim[dim] = per_dim.get(dim, 0) + 1
        per_dim_progress[name] = per_dim

    # Build aligned matrices for every item the reviewers have agreed-or-disagreed on.
    bin_items = []   # list of {ratings: [r0, r1, r2], judge: 0/1}
    d6_items = []    # list of {ratings: [r0, r1, r2], judge: label}
    for item in sample:
        iid = item["id"]
        votes = []
        for name in reviewers:
            v = per_reviewer[name].get(iid)
            if v == SKIP or v is None:
                votes.append(None)
            else:
                votes.append(v)
        if any(v is None for v in votes):
            continue
        if item["dim_kind"] == "d6":
            d6_items.append({
                "ratings": votes,
                "judge": item["judge_label"],
                "id": iid,
            })
        else:
            d6_items_dummy = None  # noqa
            bin_items.append({
                "ratings": votes,
                "judge": 1 if item["judge_correct"] else 0,
                "id": iid,
            })

    # Fleiss' κ across all 3 reviewers
    fleiss_bin = _fleiss_kappa([x["ratings"] for x in bin_items],
                               categories=list(BINARY_VALUES))
    fleiss_d6 = _fleiss_kappa([x["ratings"] for x in d6_items],
                              categories=list(D6_LABELS))

    # majority gold vs LLM judge
    bin_majority = []
    bin_judge = []
    for x in bin_items:
        m = _majority_vote(x["ratings"], allow_ties=False)  # 3 raters, binary → no ties possible
        if m is None:
            continue
        bin_majority.append(m)
        bin_judge.append(x["judge"])
    cohen_bin, cohen_bin_n = _cohens_kappa(bin_majority, bin_judge,
                                           categories=list(BINARY_VALUES))

    d6_majority = []
    d6_judge = []
    d6_no_consensus = 0
    for x in d6_items:
        m = _majority_vote(x["ratings"], allow_ties=False)
        if m is None:
            d6_no_consensus += 1
            continue
        d6_majority.append(m)
        d6_judge.append(x["judge"])
    cohen_d6, cohen_d6_n = _cohens_kappa(d6_majority, d6_judge,
                                         categories=list(D6_LABELS))

    def _f(x):
        return None if (x is None or (isinstance(x, float) and math.isnan(x))) else round(x, 4)

    return {
        "reviewers": reviewers,
        "progress": progress,
        "per_dim_progress": per_dim_progress,
        "fleiss": {
            "binary": {"kappa": _f(fleiss_bin[0]), "n": fleiss_bin[1]},
            "d6":     {"kappa": _f(fleiss_d6[0]),  "n": fleiss_d6[1]},
        },
        "cohen_majority_vs_judge": {
            "binary": {"kappa": _f(cohen_bin), "n": cohen_bin_n},
            "d6":     {"kappa": _f(cohen_d6),  "n": cohen_d6_n,
                       "no_consensus": d6_no_consensus},
        },
    }


# ----------------- HTTP handler -----------------

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        p = urlparse(self.path)
        if p.path == "/api/state":
            return self._api_state()
        if p.path == "/api/stats":
            return self._api_stats()
        if p.path.startswith("/api/labels/"):
            reviewer = p.path[len("/api/labels/"):]
            return self._api_labels(reviewer)
        return super().do_GET()

    def do_POST(self):
        p = urlparse(self.path)
        if p.path != "/api/label":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length).decode("utf-8", errors="replace")
        try:
            payload = json.loads(body) if body else {}
            reviewer = str(payload.get("reviewer", "")).strip()[:64]
            qid = str(payload.get("id", "")).strip()[:128]
            if not reviewer or not qid:
                raise ValueError("reviewer and id are required")
            value = payload.get("value", None)
            dim_kind = str(payload.get("dim_kind", "binary"))
            if dim_kind not in ("binary", "d6"):
                raise ValueError(f"invalid dim_kind: {dim_kind!r}")
            if not _is_valid_value(value, dim_kind):
                raise ValueError(f"invalid value for dim_kind={dim_kind}: {value!r}")
        except Exception as e:
            self._send_json({"ok": False, "error": str(e)}, status=400)
            return

        with _lock:
            data = _load_labels()
            data.setdefault("reviewers", {})
            data["reviewers"].setdefault(reviewer, {})
            if value is None:
                data["reviewers"][reviewer].pop(qid, None)
            else:
                data["reviewers"][reviewer][qid] = value
            _save_labels(data)
            count = len(data["reviewers"][reviewer])
        self._send_json({"ok": True, "count": count})

    def _api_state(self):
        with _lock:
            data = _load_labels()
            rs = data.get("reviewers", {})
        def _completed(v):
            return sum(1 for x in v.values() if x != SKIP and x is not None)
        sample_total = len(_load_sample())
        summary = {
            "reviewers": sorted(rs.keys()),
            "counts":   {name: _completed(v) for name, v in rs.items()},
            "counts_with_skips": {name: len(v) for name, v in rs.items()},
            "total_items": sample_total,
        }
        self._send_json(summary)

    def _api_labels(self, reviewer: str):
        reviewer = reviewer.strip()[:64]
        if not reviewer:
            self._send_json({"ok": False, "error": "reviewer required"}, status=400)
            return
        with _lock:
            data = _load_labels()
            labels = data.get("reviewers", {}).get(reviewer, {})
        self._send_json({"reviewer": reviewer, "labels": labels})

    def _api_stats(self):
        # Always evaluate against the canonical 0/1/2 trio so the panel is
        # symmetric even before all three have started labelling.
        stats = _compute_stats(list(DEFAULT_REVIEWERS))
        self._send_json(stats)

    def _send_json(self, obj: dict, status: int = 200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        line = (fmt % args) if args else str(fmt)
        if "/api/state" in line or "/api/stats" in line:
            return
        super().log_message(fmt, *args)


def _local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    os.chdir(ROOT)
    if not SAMPLE_FILE.exists():
        raise SystemExit(
            f"missing {SAMPLE_FILE}\n"
            "Run: python3 -m memarena.human_calibration.build_human_calibration --n 500"
        )
    if not LABELS_FILE.exists():
        _save_labels({"reviewers": {}})

    ip = _local_ip()
    print("Human-calibration server")
    print(f"  Root:   {ROOT}")
    print(f"  Labels: {LABELS_FILE}")
    print(f"  Local:  http://localhost:{args.port}/")
    print(f"  LAN:    http://{ip}:{args.port}/")
    print("Press Ctrl+C to stop.\n")

    httpd = HTTPServer((args.host, args.port), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        httpd.server_close()


if __name__ == "__main__":
    main()
