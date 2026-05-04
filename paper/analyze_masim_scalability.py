#!/usr/bin/env python3
"""Artifact-based reconstruction of MASim intrinsic agent-state growth.

This script does *not* treat the released corpus as the memory being measured.
Instead, it replays the state-update rules visible in the MASim code over the
current artifacts to estimate what each live EntityAgent carried internally:

- persona/profile text;
- KnowledgeState facts learned from heard turns;
- final current_mood snapshot when day_checkpoint.json is available;
- auxiliary memory containers: rolling turn buffer, facet buffers, shared bulletin;
- optional posthoc activity memory if activity_log(.jsonl) exists.

Important limitation: full KnownFeeling histories are not serialized in the
current artifacts. day_checkpoint.json stores only the final current_mood, so
day-by-day feeling-token growth cannot be recovered without a generation-time
snapshot hook.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import statistics
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, DefaultDict, Deque, Dict, Iterable, Iterator, List, Optional, Tuple


try:
    from MASim.utils.tokens import count_tokens as _count_tokens
except Exception:  # pragma: no cover
    _count_tokens = None


PROFILE_METRICS = (
    "persona_tokens",
    "knowledge_fact_tokens",
    "current_mood_tokens_available",
    "profile_fact_mood_tokens_observed",
)
EXTRA_METRICS = (
    "rolling_turn_buffer_tokens",
    "facet_memory_tokens",
    "shared_bulletin_tokens",
    "session_history_tokens",
    "posthoc_activity_memory_tokens",
    "intrinsic_extra_memory_tokens",
)
ALL_TOKEN_METRICS = PROFILE_METRICS + EXTRA_METRICS + ("intrinsic_total_tokens_observed",)


def token_count(text: str) -> int:
    text = text or ""
    if not text:
        return 0
    if _count_tokens is not None:
        try:
            return int(_count_tokens(text))
        except Exception:
            pass
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def find_file(data_dir: Path, stem: str) -> Optional[Path]:
    for suffix in ("", ".jsonl", ".jsonl.gz", ".json", ".json.gz"):
        path = data_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with open_text(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_json(path: Path) -> Any:
    with open_text(path) as fh:
        return json.load(fh)


def is_assistant(agent_id: str) -> bool:
    return agent_id.startswith("assistant_") or agent_id == "__assistant__"


def day_from_timestamp(timestamp: Any, max_day: int) -> Optional[int]:
    try:
        ts = float(timestamp)
    except (TypeError, ValueError):
        return None
    day = int(math.floor(ts)) + 1
    if 1 <= day <= max_day:
        return day
    return None


def cumulative(counter: Counter, day: int) -> int:
    return sum(int(counter.get(d, 0)) for d in range(1, day + 1))


def percentile(values: List[int], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(values[lo])
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def persona_to_prompt_like_text(persona: Dict[str, Any]) -> str:
    """Mirror PersonaCard.to_prompt closely without importing runtime objects."""
    traits = ", ".join(persona.get("personality_traits") or []) or "none specified"
    expertise = ", ".join(persona.get("expertise") or []) or "none specified"
    demographics = persona.get("demographics") or {}
    demo = "; ".join(f"{k}: {v}" for k, v in demographics.items()) if demographics else "N/A"
    hobbies = ", ".join(persona.get("hobbies") or []) or "not specified"
    concerns = "; ".join(persona.get("current_concerns") or []) or "nothing in particular"
    values = ", ".join(persona.get("values") or []) or "not specified"
    lines = [
        f"Name: {persona.get('name', '')}",
        f"Age: {persona.get('age', '')}",
        f"Occupation: {persona.get('occupation', '')}",
        f"Demographics: {demo}",
        f"Personality: {traits}",
        f"Expertise: {expertise}",
        f"Hobbies: {hobbies}",
        f"Values: {values}",
        f"Currently thinking about: {concerns}",
        f"Communication style: {persona.get('communication_style', '')}",
    ]
    if persona.get("education_level"):
        lines.append(f"Education: {persona['education_level']}")
    if persona.get("work_schedule"):
        lines.append(f"Work schedule: {persona['work_schedule']}")
    if persona.get("daily_routine_notes"):
        lines.append(f"Daily routine: {persona['daily_routine_notes']}")
    if persona.get("speaking_style"):
        lines.append(f"Speaking mannerisms: {persona['speaking_style']}")
    if persona.get("backstory"):
        lines.append(f"Background: {persona['backstory']}")
    relationships = persona.get("relationships") or {}
    if relationships:
        rels = "; ".join(f"{k} ({v})" for k, v in list(relationships.items())[:5])
        lines.append(f"Key relationships: {rels}")
    tech_affinity = float(persona.get("tech_affinity") or 0.0)
    if tech_affinity >= 0.7:
        lines.append("Technology comfort: very comfortable with technology")
    elif tech_affinity >= 0.4:
        lines.append("Technology comfort: somewhat comfortable with technology")
    else:
        lines.append("Technology comfort: not very tech-savvy")
    return "\n".join(lines)


def json_string_tokens(obj: Any) -> int:
    """Count only string payloads in a serialized memory object."""
    if isinstance(obj, str):
        return token_count(obj)
    if isinstance(obj, dict):
        return sum(json_string_tokens(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(json_string_tokens(v) for v in obj)
    return 0


def turn_memory_tokens(turn: Dict[str, Any]) -> int:
    """Approximate EntityAgent.memory_buffer payload size for one DialogueTurn."""
    return json_string_tokens(turn)


def turn_text(turn: Dict[str, Any]) -> str:
    speaker = turn.get("speaker_id", "")
    text = turn.get("text", "")
    return f"{speaker}: {text}" if speaker else str(text)


def fact_contents(turn: Dict[str, Any]) -> List[str]:
    facts = [str(x).strip() for x in (turn.get("extracted_facts") or []) if str(x).strip()]
    if facts:
        return facts
    text = str(turn.get("text") or "").strip()
    return [text] if text else []


def heard_by_agent(session: Dict[str, Any], turn: Dict[str, Any], agent_id: str) -> bool:
    if turn.get("speaker_id") == agent_id:
        return False
    if turn.get("listener_id") == agent_id:
        return True
    is_group = bool(
        turn.get("listener_id") == "group"
        or (turn.get("metadata") or {}).get("group_conversation")
        or (session.get("metadata") or {}).get("group_conversation")
    )
    return is_group and agent_id in (session.get("participants") or [])


def visible_turn_for_buffer(session: Dict[str, Any], turn: Dict[str, Any], agent_id: str) -> bool:
    """Reconstruct what would be appended by EntityAgent.receive_turn."""
    if turn.get("speaker_id") == agent_id:
        return True
    if turn.get("listener_id") == agent_id:
        return True
    is_group = bool(
        turn.get("listener_id") == "group"
        or (turn.get("metadata") or {}).get("group_conversation")
        or (session.get("metadata") or {}).get("group_conversation")
    )
    return is_group and agent_id in (session.get("participants") or [])


def slug_to_display(slug: str) -> str:
    if slug.startswith("assistant_"):
        slug = "assistant"
    return " ".join(part.capitalize() for part in slug.split("_"))


def load_agents(data_dir: Path) -> Tuple[List[str], Dict[str, str], Dict[str, int]]:
    path = find_file(data_dir, "agents_personas")
    if path is None:
        raise FileNotFoundError(f"agents_personas(.jsonl[.gz]) not found in {data_dir}")
    agent_ids: List[str] = []
    agent_names: Dict[str, str] = {}
    persona_tokens: Dict[str, int] = {}
    for row in iter_jsonl(path):
        agent_id = row.get("agent_id", "")
        if not agent_id or is_assistant(agent_id):
            continue
        persona = row.get("persona") or {}
        agent_ids.append(agent_id)
        agent_names[agent_id] = persona.get("name") or agent_id
        persona_tokens[agent_id] = token_count(persona_to_prompt_like_text(persona))
    agent_ids.sort()
    return agent_ids, agent_names, persona_tokens


def load_checkpoint(data_dir: Path) -> Dict[str, Any]:
    path = find_file(data_dir, "day_checkpoint")
    if path is None:
        return {}
    return load_json(path)


def load_config_flags(data_dir: Path) -> Dict[str, Any]:
    path = find_file(data_dir, "pipeline_report")
    if path is None:
        return {
            "memory_context_sessions": None,
            "facets_enabled": None,
        }
    raw = load_json(path).get("config", "")
    text = str(raw)
    flags: Dict[str, Any] = {}
    for name in ("memory_context_sessions", "facets_enabled"):
        match = re.search(rf"{name}=(True|False)", text)
        flags[name] = (match.group(1) == "True") if match else None
    return flags


def add_counter(table: DefaultDict[str, Counter], agent_id: str, day: Optional[int], value: int) -> None:
    if agent_id and day is not None and value:
        table[agent_id][day] += int(value)


def reconstruct(
    data_dir: Path,
    max_day: int,
    include_activity_log: bool,
    include_rolling_turn_buffer: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    agent_ids, agent_names, persona_tokens = load_agents(data_dir)
    agent_set = set(agent_ids)
    checkpoint = load_checkpoint(data_dir)
    flags = load_config_flags(data_dir)

    corpus_path = find_file(data_dir, "corpus_sessions")
    if corpus_path is None:
        raise FileNotFoundError(f"corpus_sessions(.jsonl[.gz]) not found in {data_dir}")

    sessions = sorted(iter_jsonl(corpus_path), key=lambda s: float(s.get("start_time", 0.0)))
    session_start = {s.get("session_id", ""): float(s.get("start_time", 0.0)) for s in sessions}

    fact_tokens_by_day: DefaultDict[str, Counter] = defaultdict(Counter)
    fact_count_by_day: DefaultDict[str, Counter] = defaultdict(Counter)
    rolling_buffers: Dict[str, Deque[Tuple[int, int]]] = {
        aid: deque(maxlen=50) for aid in agent_ids
    }
    rolling_tokens_by_day: DefaultDict[str, Counter] = defaultdict(Counter)

    facet_buffers: Dict[str, Dict[str, Deque[Dict[str, Any]]]] = {
        aid: defaultdict(lambda: deque(maxlen=30)) for aid in agent_ids
    }
    shared_bulletins: Dict[str, Deque[str]] = {aid: deque(maxlen=50) for aid in agent_ids}
    facet_tokens_snapshot: DefaultDict[str, Counter] = defaultdict(Counter)
    bulletin_tokens_snapshot: DefaultDict[str, Counter] = defaultdict(Counter)
    facet_entry_count_snapshot: DefaultDict[str, Counter] = defaultdict(Counter)
    bulletin_entry_count_snapshot: DefaultDict[str, Counter] = defaultdict(Counter)

    sessions_by_day: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for session in sessions:
        day = day_from_timestamp(session.get("start_time", 0.0), max_day)
        if day is not None:
            sessions_by_day[day].append(session)

    # Optional posthoc activity memory from Stage 5b. This is intrinsic after
    # ScheduleFactory.update_from_activity, but not part of the day-loop object
    # unless activity_log was generated.
    activity_tokens_by_day: DefaultDict[str, Counter] = defaultdict(Counter)
    activity_count_by_day: DefaultDict[str, Counter] = defaultdict(Counter)
    activity_path = find_file(data_dir, "activity_log") or find_file(data_dir, "activity_logs")
    if include_activity_log and activity_path is not None:
        for entry in iter_jsonl(activity_path):
            aid = entry.get("agent_id", "")
            if aid not in agent_set:
                continue
            linked = entry.get("linked_session_id") or ""
            if linked in session_start:
                day = day_from_timestamp(session_start[linked], max_day)
            else:
                day = day_from_timestamp(entry.get("start_time", 0.0), max_day)
            text = str(entry.get("memory_text") or "")
            add_counter(activity_tokens_by_day, aid, day, token_count(text))
            add_counter(activity_count_by_day, aid, day, 1)

    # Replay day by day to match the generation loop's state-growth view.
    for day in range(1, max_day + 1):
        for session in sessions_by_day.get(day, []):
            participants = [p for p in (session.get("participants") or []) if p in agent_set]
            turns = session.get("turns") or []

            for aid in participants:
                for turn in turns:
                    if heard_by_agent(session, turn, aid):
                        for fact in fact_contents(turn):
                            add_counter(fact_tokens_by_day, aid, day, token_count(fact))
                            add_counter(fact_count_by_day, aid, day, 1)
                    if include_rolling_turn_buffer and visible_turn_for_buffer(session, turn, aid):
                        rolling_buffers[aid].append((day, turn_memory_tokens(turn)))

            domain = str(session.get("interest_domain") or "")
            if domain:
                for aid in participants:
                    partner_ids = [p for p in (session.get("participants") or []) if p != aid]
                    partners = ", ".join(slug_to_display(p) for p in partner_ids) or "others"
                    facts: List[str] = []
                    for turn in turns:
                        if turn.get("speaker_id") != aid and turn.get("extracted_facts"):
                            facts.extend([str(x) for x in turn.get("extracted_facts", [])[:3]])
                    entry = {
                        "session_id": session.get("session_id", ""),
                        "domain": domain,
                        "partners": partners,
                        "timestamp": session.get("start_time", 0.0),
                        "facts": facts[:5],
                        "summary": f"Talked with {partners} about {domain.replace('_', ' ')}.",
                    }
                    facet_buffers[aid][domain].append(entry)

        # build_daily_bulletin() runs once at the end of each day when facets
        # are enabled. It appends one line per existing facet's latest entry.
        if flags.get("facets_enabled") is not False:
            for aid in agent_ids:
                for domain, buffer in facet_buffers[aid].items():
                    if not buffer:
                        continue
                    latest = buffer[-1]
                    facts = latest.get("facts", [])
                    partners = latest.get("partners", "someone")
                    if facts:
                        line = f"[{domain}] {partners}: {str(facts[0])[:80]}"
                    else:
                        line = f"[{domain}] {str(latest.get('summary', ''))[:80]}"
                    shared_bulletins[aid].append(line)

        for aid in agent_ids:
            facet_tokens = sum(json_string_tokens(entry) for buf in facet_buffers[aid].values() for entry in buf)
            facet_entries = sum(len(buf) for buf in facet_buffers[aid].values())
            bulletin_tokens = sum(token_count(x) for x in shared_bulletins[aid])
            rolling_tokens = sum(tok for _, tok in rolling_buffers[aid])
            facet_tokens_snapshot[aid][day] = facet_tokens
            facet_entry_count_snapshot[aid][day] = facet_entries
            bulletin_tokens_snapshot[aid][day] = bulletin_tokens
            bulletin_entry_count_snapshot[aid][day] = len(shared_bulletins[aid])
            rolling_tokens_by_day[aid][day] = rolling_tokens

    final_day = int(checkpoint.get("completed_day", max_day - 1)) + 1 if checkpoint else max_day
    final_moods = checkpoint.get("agent_moods", {}) if checkpoint else {}
    checkpoint_fact_counts = checkpoint.get("agent_knowledge_sizes", {}) if checkpoint else {}

    rows: List[Dict[str, Any]] = []
    for aid in agent_ids:
        for day in range(1, max_day + 1):
            persona_tok = persona_tokens.get(aid, 0)
            fact_tok = cumulative(fact_tokens_by_day[aid], day)
            fact_count = cumulative(fact_count_by_day[aid], day)
            mood_tok = token_count(final_moods.get(aid, "")) if day == final_day else 0
            rolling_tok = rolling_tokens_by_day[aid][day]
            facet_tok = facet_tokens_snapshot[aid][day]
            bulletin_tok = bulletin_tokens_snapshot[aid][day]
            session_history_tok = 0  # memory_context_sessions=False in current runs; contents not serialized.
            activity_tok = cumulative(activity_tokens_by_day[aid], day)
            extra_tok = rolling_tok + facet_tok + bulletin_tok + session_history_tok + activity_tok
            profile_tok = persona_tok + fact_tok + mood_tok
            rows.append({
                "agent_id": aid,
                "agent_name": agent_names.get(aid, aid),
                "day": day,
                "persona_tokens": persona_tok,
                "knowledge_fact_tokens": fact_tok,
                "current_mood_tokens_available": mood_tok,
                "profile_fact_mood_tokens_observed": profile_tok,
                "rolling_turn_buffer_tokens": rolling_tok,
                "facet_memory_tokens": facet_tok,
                "shared_bulletin_tokens": bulletin_tok,
                "session_history_tokens": session_history_tok,
                "posthoc_activity_memory_tokens": activity_tok,
                "intrinsic_extra_memory_tokens": extra_tok,
                "intrinsic_total_tokens_observed": profile_tok + extra_tok,
                "knowledge_fact_count_replayed": fact_count,
                "checkpoint_final_fact_count": checkpoint_fact_counts.get(aid, "") if day == final_day else "",
                "checkpoint_count_gap": (
                    fact_count - int(checkpoint_fact_counts[aid])
                    if day == final_day and aid in checkpoint_fact_counts
                    else ""
                ),
                "facet_memory_entries": facet_entry_count_snapshot[aid][day],
                "shared_bulletin_entries": bulletin_entry_count_snapshot[aid][day],
                "posthoc_activity_memory_entries": cumulative(activity_count_by_day[aid], day),
            })

    summary_rows: List[Dict[str, Any]] = []
    for day in range(1, max_day + 1):
        day_rows = [r for r in rows if r["day"] == day]
        row: Dict[str, Any] = {"day": day, "n_agents": len(day_rows)}
        for metric in ALL_TOKEN_METRICS:
            vals = [int(r[metric]) for r in day_rows]
            row[f"{metric}_mean"] = round(statistics.mean(vals), 1) if vals else 0.0
            row[f"{metric}_p50"] = round(percentile(vals, 0.50), 1)
            row[f"{metric}_p90"] = round(percentile(vals, 0.90), 1)
            row[f"{metric}_max"] = max(vals) if vals else 0
            row[f"{metric}_sum"] = sum(vals)
        summary_rows.append(row)

    gaps = [
        int(r["checkpoint_count_gap"])
        for r in rows
        if r["day"] == final_day and r["checkpoint_count_gap"] != ""
    ]
    meta = {
        "data_dir": str(data_dir),
        "max_day": max_day,
        "n_agents": len(agent_ids),
        "n_sessions": len(sessions),
        "facets_enabled_from_report": flags.get("facets_enabled"),
        "memory_context_sessions_from_report": flags.get("memory_context_sessions"),
        "include_rolling_turn_buffer": include_rolling_turn_buffer,
        "include_activity_log": include_activity_log,
        "activity_log_path": str(activity_path) if activity_path else "",
        "known_feeling_limitation": (
            "Full KnownFeeling histories are not serialized. Only final current_mood "
            "from day_checkpoint.json is available, so day-level feeling-token growth "
            "cannot be recovered from current artifacts."
        ),
        "attached_assistant_note": (
            "MASim person-agent sessions use assistant_* speaker IDs, but the assistant "
            "is not a separate EntityAgent with serialized intrinsic state; PA sessions "
            "update the human EntityAgent's KnowledgeState."
        ),
        "checkpoint_fact_count_gap_min": min(gaps) if gaps else None,
        "checkpoint_fact_count_gap_mean": round(statistics.mean(gaps), 2) if gaps else None,
        "checkpoint_fact_count_gap_max": max(gaps) if gaps else None,
    }
    return rows, summary_rows, meta


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        if not rows:
            return
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def maybe_plot(rows: List[Dict[str, Any]], summary_rows: List[Dict[str, Any]], out_dir: Path) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    paths: List[Path] = []
    days = [int(r["day"]) for r in summary_rows]

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for metric, label in (
        ("profile_fact_mood_tokens_observed", "persona + facts + final mood"),
        ("intrinsic_extra_memory_tokens", "extra intrinsic memory"),
        ("intrinsic_total_tokens_observed", "total observed intrinsic"),
    ):
        ax.plot(days, [r[f"{metric}_mean"] for r in summary_rows], marker="o", label=f"{label} mean")
        ax.plot(days, [r[f"{metric}_p90"] for r in summary_rows], linestyle="--", alpha=0.7, label=f"{label} p90")
    ax.set_xlabel("simulation day")
    ax.set_ylabel("tokens per simulated person")
    ax.set_xticks(days)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = out_dir / "intrinsic_state_growth_summary.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    final_day = max(days)
    final_rows = [r for r in rows if int(r["day"]) == final_day]
    order = [r["agent_id"] for r in sorted(final_rows, key=lambda r: int(r["intrinsic_total_tokens_observed"]), reverse=True)]
    by_agent_day = {(r["agent_id"], int(r["day"])): r for r in rows}
    heat_metrics = (
        ("profile_fact_mood_tokens_observed", "persona/facts/mood"),
        ("intrinsic_extra_memory_tokens", "extra intrinsic memory"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, max(5.0, len(order) * 0.14)), sharey=True)
    for ax, (metric, title) in zip(axes, heat_metrics):
        matrix = [[int(by_agent_day[(aid, day)][metric]) for day in days] for aid in order]
        im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("day")
        ax.set_xticks(range(len(days)))
        ax.set_xticklabels(days)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("agent")
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(order, fontsize=6)
    fig.tight_layout()
    path = out_dir / "intrinsic_state_growth_heatmaps.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)
    return paths


def print_summary(summary_rows: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    print("MASim intrinsic state growth reconstruction")
    print(f"data_dir: {meta['data_dir']}")
    print(f"agents={meta['n_agents']} sessions={meta['n_sessions']}")
    print(f"facets_enabled={meta['facets_enabled_from_report']} memory_context_sessions={meta['memory_context_sessions_from_report']}")
    print("KnownFeeling limitation: final current_mood only; full day-level feeling history unavailable.")
    if meta["checkpoint_fact_count_gap_mean"] is not None:
        print(
            "final fact-count replay gap vs checkpoint: "
            f"min={meta['checkpoint_fact_count_gap_min']} "
            f"mean={meta['checkpoint_fact_count_gap_mean']} "
            f"max={meta['checkpoint_fact_count_gap_max']}"
        )
    print()
    print("day | profile_mean profile_p90 | extra_mean extra_p90 | total_mean total_p90")
    by_day = {int(r["day"]): r for r in summary_rows}
    for day in sorted({1, 5, 10, max(by_day)}):
        if day not in by_day:
            continue
        r = by_day[day]
        print(
            f"{day:>3} | "
            f"{r['profile_fact_mood_tokens_observed_mean']:>12} {r['profile_fact_mood_tokens_observed_p90']:>11} | "
            f"{r['intrinsic_extra_memory_tokens_mean']:>10} {r['intrinsic_extra_memory_tokens_p90']:>9} | "
            f"{r['intrinsic_total_tokens_observed_mean']:>10} {r['intrinsic_total_tokens_observed_p90']:>9}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("MASim/datasets/memarena_l"))
    parser.add_argument("--out-dir", type=Path, default=Path("paper/analysis_outputs/masim_intrinsic_scalability"))
    parser.add_argument("--max-day", type=int, default=15)
    parser.add_argument(
        "--include-activity-log",
        action="store_true",
        help="Include posthoc activity_log.memory_text as intrinsic activity memory when present.",
    )
    parser.add_argument(
        "--no-rolling-turn-buffer",
        action="store_true",
        help="Exclude reconstructed EntityAgent.memory_buffer rolling turns.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, summary_rows, meta = reconstruct(
        data_dir=args.data_dir,
        max_day=args.max_day,
        include_activity_log=args.include_activity_log,
        include_rolling_turn_buffer=not args.no_rolling_turn_buffer,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "intrinsic_state_growth_by_agent_day.csv", rows)
    write_csv(args.out_dir / "intrinsic_state_growth_summary_by_day.csv", summary_rows)
    write_json(args.out_dir / "intrinsic_state_growth_meta.json", meta)
    plot_paths: List[Path] = []
    if not args.no_plots:
        plot_paths = maybe_plot(rows, summary_rows, args.out_dir)
    print_summary(summary_rows, meta)
    print()
    print(f"wrote: {args.out_dir / 'intrinsic_state_growth_by_agent_day.csv'}")
    print(f"wrote: {args.out_dir / 'intrinsic_state_growth_summary_by_day.csv'}")
    print(f"wrote: {args.out_dir / 'intrinsic_state_growth_meta.json'}")
    for path in plot_paths:
        print(f"wrote: {path}")


if __name__ == "__main__":
    main()
