#!/usr/bin/env python3
"""
SAG2-style educational demo (NOT a paper reproduction).

Mirrors the query-time idea:
  BM25/FTS retrieve → (demo) evidence judge → expand terms → retrieve again

Real SAG2 uses TypeSafe JEV for evidence scoring.
This demo uses a labeled heuristic scorer only — no API keys.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "data" / "docs"
DB_PATH = ROOT / "data" / "index.sqlite"

DEFAULT_QUESTION = "收购了B公司的企业，其CTO后来加入了哪个项目？"

# Patterns for DEMO entity / key-phrase extraction (not JEV).
ENTITY_PATTERNS = [
    r"盘古大模型项目",
    r"盘古大模型",
    r"[A-D]公司",
    r"张三",
    r"李四",
    r"王五",
    r"CTO",
    r"CEO",
    r"全资收购",
    r"收购",
]

# Bridge entities useful for the classic multi-hop story
BRIDGE_ENTITIES = {"A公司", "B公司", "张三", "盘古大模型项目", "盘古大模型", "CTO", "全资收购"}

STOP_CHARS = set("的了在与和及或其后来哪个什么是被为到中对由从等并也还又而就都要可以会能把让给")
QUERY_STOP = {
    "的", "了", "在", "与", "和", "及", "或", "其", "后来", "哪个", "什么",
    "是", "被", "为", "到", "中", "对", "由", "从", "等", "企业", "项目",
}

# Negation / "mentioned only as absence" cues — DEMO judge uses these to demote distractors
NEGATION_CUES = ("无关", "未提及", "没有", "并非", "不是")


@dataclass
class Passage:
    doc_id: int
    path: str
    title: str
    text: str
    fts_score: float = 0.0
    evidence_score: float = 0.0
    kept: bool = False
    round_found: int = 0


@dataclass
class RoundTrace:
    round_id: int
    query_terms: list[str]
    candidates: list[Passage] = field(default_factory=list)
    kept: list[Passage] = field(default_factory=list)
    expansion_terms: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tokenization: character bigrams for Chinese + keep Latin/digit tokens
# Plain FTS5 has no Chinese segmenter; bigrams need no third-party deps.
# ---------------------------------------------------------------------------

def tokenize_for_fts(text: str) -> str:
    """Turn text into space-separated FTS tokens (char unigrams/bigrams + ascii)."""
    text = text.strip()
    tokens: list[str] = []
    for m in re.finditer(r"[A-Za-z0-9_]+", text):
        tokens.append(m.group(0).lower())
    cjk = re.findall(r"[\u4e00-\u9fff]", text)
    for ch in cjk:
        if ch not in STOP_CHARS:
            tokens.append(ch)
    for a, b in zip(cjk, cjk[1:]):
        tokens.append(a + b)
    return " ".join(tokens)


def term_to_fts_atoms(term: str) -> list[str]:
    """Map a human term (e.g. A公司) to the atomic FTS tokens we actually indexed."""
    raw = tokenize_for_fts(term).split()
    # Prefer multi-char tokens; keep ascii words; drop lone stop-ish chars already filtered
    atoms = []
    seen = set()
    for t in raw:
        if t in seen:
            continue
        if len(t) == 1 and "\u4e00" <= t <= "\u9fff":
            # keep unigram only if term itself is length-1 useful char
            if len(term) == 1:
                seen.add(t)
                atoms.append(t)
            continue
        seen.add(t)
        atoms.append(t)
    # If nothing survived (odd edge case), fall back to all raw tokens
    return atoms or raw


def extract_query_terms(question: str) -> list[str]:
    """Prefer multi-char entities, then useful bigrams / words from the question."""
    terms: list[str] = []
    seen: set[str] = set()

    def add(t: str) -> None:
        t = t.strip()
        if not t or t in QUERY_STOP or t in seen:
            return
        if len(t) == 1 and "\u4e00" <= t <= "\u9fff":
            return
        seen.add(t)
        terms.append(t)

    for pat in ENTITY_PATTERNS:
        for m in re.finditer(pat, question):
            add(m.group(0))

    for m in re.finditer(r"[A-Za-z0-9_]+", question):
        add(m.group(0))

    cjk = re.findall(r"[\u4e00-\u9fff]", question)
    for a, b in zip(cjk, cjk[1:]):
        if a in STOP_CHARS or b in STOP_CHARS:
            continue
        add(a + b)

    return terms


def extract_entities(text: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for pat in ENTITY_PATTERNS:
        for m in re.finditer(pat, text):
            s = m.group(0)
            if s not in seen:
                seen.add(s)
                found.append(s)
    return found


def looks_like_negation_only(text: str) -> bool:
    """True when entities appear mainly inside '无关/未提及' style distractors."""
    return any(cue in text for cue in NEGATION_CUES)


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

def load_docs(docs_dir: Path) -> list[tuple[str, str, str]]:
    rows = []
    for path in sorted(docs_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8").strip()
        lines = raw.splitlines()
        title = lines[0].lstrip("# ").strip() if lines else path.stem
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else raw
        rows.append((str(path), title, body))
    return rows


def build_index(db_path: Path, docs_dir: Path) -> sqlite3.Connection:
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE docs (
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL,
            title TEXT NOT NULL,
            text TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE VIRTUAL TABLE docs_fts USING fts5(
            tokens,
            content='',
            tokenize='unicode61'
        )
        """
    )
    for path, title, text in load_docs(docs_dir):
        cur.execute(
            "INSERT INTO docs(path, title, text) VALUES (?, ?, ?)",
            (path, title, text),
        )
        doc_id = cur.lastrowid
        tokens = tokenize_for_fts(f"{title}\n{text}")
        cur.execute(
            "INSERT INTO docs_fts(rowid, tokens) VALUES (?, ?)",
            (doc_id, tokens),
        )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Retrieve (FTS5 BM25)
# ---------------------------------------------------------------------------

def fts_query_from_terms(terms: list[str]) -> str:
    """
    Build an OR query over atomic FTS tokens.
    Human terms like「A公司」are expanded to indexed atoms (a, 公司, …).
    """
    parts: list[str] = []
    seen: set[str] = set()
    for t in terms:
        for atom in term_to_fts_atoms(t):
            safe = atom.replace('"', "")
            if not safe or safe in seen:
                continue
            seen.add(safe)
            parts.append(f'"{safe}"')
    return " OR ".join(parts) if parts else '""'


def retrieve(conn: sqlite3.Connection, terms: list[str], limit: int = 10) -> list[Passage]:
    if not terms:
        return []
    q = fts_query_from_terms(terms)
    try:
        rows = conn.execute(
            """
            SELECT d.id, d.path, d.title, d.text,
                   bm25(docs_fts) AS score
            FROM docs_fts
            JOIN docs d ON d.id = docs_fts.rowid
            WHERE docs_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (q, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    out = []
    for r in rows:
        raw = float(r["score"])
        out.append(
            Passage(
                doc_id=int(r["id"]),
                path=r["path"],
                title=r["title"],
                text=r["text"],
                fts_score=-raw,
            )
        )
    return out


# ---------------------------------------------------------------------------
# DEMO evidence judge — NOT real JEV
# ---------------------------------------------------------------------------

def demo_evidence_score(
    question: str,
    passage: Passage,
    known_context: list[str],
) -> float:
    """
    DEMO HEURISTIC (NOT TypeSafe JEV / not paper-faithful).

    Rewards passages that assert useful facts for the question / known bridge
    entities; demotes negation-only distractors.
    """
    full = passage.title + "\n" + passage.text
    q_ents = set(extract_entities(question))
    p_ents = set(extract_entities(full))
    ctx = set(known_context)

    if looks_like_negation_only(full):
        # Still allow a tiny score so the trace shows them as candidates,
        # but they should not pass the keep threshold.
        return 0.2 + 0.02 * passage.fts_score

    score = 0.0
    overlap = q_ents & p_ents
    score += 1.8 * len(overlap)

    q_terms = extract_query_terms(question)
    hit_terms = [t for t in q_terms if t in full]
    score += 0.7 * len(hit_terms)

    # Affirmative chain keys present as real content
    chain_keys = [
        "全资收购",
        "收购",
        "B公司",
        "A公司",
        "CTO",
        "张三",
        "盘古大模型项目",
        "盘古大模型",
    ]
    local_hits = [k for k in chain_keys if k in full]
    score += 0.9 * len(local_hits)

    # Bridge with previously kept evidence
    if ctx:
        shared = p_ents & ctx
        if shared:
            score += 2.2
        new_ents = p_ents - ctx
        if shared and new_ents:
            score += 1.8  # connects old → new (classic multi-hop step)

    # Prefer docs that introduce a missing bridge entity
    missing_bridges = BRIDGE_ENTITIES - ctx - q_ents
    if p_ents & missing_bridges:
        score += 1.0

    score += 0.05 * passage.fts_score
    return score


def select_evidence(
    question: str,
    candidates: list[Passage],
    known_context: list[str],
    already_kept_ids: set[int],
    threshold: float = 2.5,
    max_keep: int = 2,
) -> list[Passage]:
    scored: list[Passage] = []
    for c in candidates:
        c.evidence_score = demo_evidence_score(question, c, known_context)
        scored.append(c)
    scored.sort(key=lambda p: p.evidence_score, reverse=True)

    kept: list[Passage] = []
    for p in scored:
        if p.doc_id in already_kept_ids:
            continue
        if p.evidence_score >= threshold and len(kept) < max_keep:
            p.kept = True
            kept.append(p)

    # Guarantee progress on first useful hit
    if not kept:
        for p in scored:
            if p.doc_id in already_kept_ids:
                continue
            if p.evidence_score >= 1.5:
                p.kept = True
                kept = [p]
                break
    return kept


def expansion_terms_from_passages(
    passages: list[Passage],
    already: set[str],
    question: str,
    limit: int = 6,
) -> list[str]:
    """
    DEMO term expansion from kept evidence only.
    Prefer company / person bridges not yet searched.
    """
    q_ents = set(extract_entities(question))
    counts: Counter[str] = Counter()
    for p in passages:
        full = p.title + "\n" + p.text
        if looks_like_negation_only(full):
            continue
        for e in extract_entities(full):
            if e in already:
                continue
            # Skip ultra-generic alone if already implied
            weight = 4 if e in {"A公司", "B公司", "张三", "盘古大模型项目", "盘古大模型"} else 2
            if e in {"CTO", "收购", "全资收购"}:
                weight = 1
            if e in q_ents and e not in {"A公司", "张三", "盘古大模型项目"}:
                weight = max(1, weight - 1)
            # Deprioritize pure distractor entities
            if e in {"C公司", "D公司", "李四", "王五", "CEO"}:
                weight = 0
            if weight:
                counts[e] += weight
    return [t for t, _ in counts.most_common(limit)]


# ---------------------------------------------------------------------------
# Multi-hop loop
# ---------------------------------------------------------------------------

def run_sag2_demo(question: str, max_rounds: int = 2) -> tuple[list[RoundTrace], list[Passage]]:
    if not DOCS_DIR.exists() or not any(DOCS_DIR.glob("*.md")):
        raise SystemExit(f"No docs found under {DOCS_DIR}")

    print("=" * 72)
    print("SAG2-style local demo  |  educational only")
    print("Evidence judge = DEMO HEURISTIC (NOT real JEV / TypeSafe)")
    print("=" * 72)
    print(f"Docs dir : {DOCS_DIR}")
    print(f"DB path  : {DB_PATH}")
    print(f"Question : {question}")
    print()

    conn = build_index(DB_PATH, DOCS_DIR)
    n = conn.execute("SELECT COUNT(*) AS n FROM docs").fetchone()["n"]
    print(f"[index] Built SQLite FTS5 over {n} passages (char bigrams, no embeddings).")
    print()

    traces: list[RoundTrace] = []
    evidence_by_id: dict[int, Passage] = {}
    known_entities: list[str] = []
    used_terms: set[str] = set()

    terms = extract_query_terms(question)
    for r in range(0, max_rounds + 1):
        trace = RoundTrace(round_id=r, query_terms=list(terms))
        print("-" * 72)
        print(f"Round {r}  |  search terms: {terms}")
        print("-" * 72)

        candidates = retrieve(conn, terms, limit=10)
        for c in candidates:
            c.round_found = r
        trace.candidates = candidates

        if not candidates:
            print("  (no FTS hits)")
            traces.append(trace)
            break

        print("  Candidates (FTS5 BM25, higher display score = better):")
        for i, c in enumerate(candidates, 1):
            snip = c.text.replace("\n", " ")[:60]
            print(f"    [{i}] id={c.doc_id}  fts={c.fts_score:.3f}  {Path(c.path).name}")
            print(f"        {c.title} — {snip}…")

        kept = select_evidence(
            question,
            candidates,
            known_entities,
            already_kept_ids=set(evidence_by_id.keys()),
        )
        # refresh scores on stored objects
        for p in kept:
            evidence_by_id[p.doc_id] = p
            for e in extract_entities(p.title + "\n" + p.text):
                if e not in known_entities:
                    known_entities.append(e)
        trace.kept = kept

        print()
        print("  DEMO evidence judge (heuristic, NOT JEV) kept:")
        if not kept:
            print("    (nothing new kept)")
        for p in kept:
            print(
                f"    ✓ id={p.doc_id}  evidence_score={p.evidence_score:.2f}  "
                f"{Path(p.path).name}"
            )
            print(f"      entities: {extract_entities(p.title + chr(10) + p.text)}")

        used_terms.update(terms)
        expand = expansion_terms_from_passages(
            list(evidence_by_id.values()), used_terms, question
        )
        # Avoid re-searching terms already used (including atoms' parents)
        expand = [t for t in expand if t not in used_terms]
        trace.expansion_terms = expand
        print()
        print(f"  Expansion terms for next round: {expand if expand else '(none)'}")
        print()
        traces.append(trace)

        if r >= max_rounds or not expand:
            break
        terms = expand

    final = sorted(evidence_by_id.values(), key=lambda p: p.evidence_score, reverse=True)
    print("=" * 72)
    print("FINAL EVIDENCE SET (accumulated across hops)")
    print("=" * 72)
    if not final:
        print("  (empty)")
    for i, p in enumerate(final, 1):
        print(f"  [{i}] round={p.round_found}  score={p.evidence_score:.2f}")
        print(f"      file: {p.path}")
        print(f"      title: {p.title}")
        print(f"      text : {p.text}")
        print()

    print("=" * 72)
    print("Naive answer sketch from evidence (no LLM generation):")
    joined = " ".join(p.text for p in final)
    names = {Path(p.path).name for p in final}
    if {"01_acquisition.md", "02_cto.md", "03_pangu.md"}.issubset(names):
        answer = (
            "收购了B公司的企业是A公司；其CTO张三后来加入了「盘古大模型项目」。"
            "（由多跳证据串起来，非模型生成）"
        )
    elif "盘古" in joined:
        answer = "证据中出现盘古相关表述，但链条可能不完整，请查看上方 FINAL EVIDENCE。"
    else:
        answer = "（未能从证据拼出完整链条）"
    print(f"  {answer}")
    print()
    print("Mapping to real SAG2: index=SQLite FTS5 · retrieve=BM25 · judge=JEV · expand=terms")
    print("This demo replaces JEV with a keyword/entity heuristic. See README.md.")
    conn.close()
    return traces, final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SAG2-style educational multi-hop demo")
    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help="Chinese question (default: classic A/B/CTO/盘古 multi-hop)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="max expansion rounds after the initial retrieve (default: 2)",
    )
    args = parser.parse_args(argv)
    _, final = run_sag2_demo(args.question, max_rounds=args.rounds)

    names = {Path(p.path).name for p in final}
    needed = {"01_acquisition.md", "02_cto.md", "03_pangu.md"}
    if args.question == DEFAULT_QUESTION and not needed.issubset(names):
        print("WARNING: expected multi-hop docs not all in final evidence:", needed - names)
        return 2
    if args.question == DEFAULT_QUESTION:
        print("CHECK: multi-hop path succeeded — acquisition + CTO + 盘古 all present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
