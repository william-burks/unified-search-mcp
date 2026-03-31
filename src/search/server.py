from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from research_core import ResearchStore, ResearchVector
from research_core.graph import ResearchGraph

mcp = FastMCP("search")

_SHARED_DIR = os.path.expanduser(
    os.environ.get("RESEARCH_DATA_DIR", "~/ClaudeProjects/QuantResearcher/core/data")
)

_store: ResearchStore | None = None
_vector: ResearchVector | None = None


def _get_store() -> ResearchStore:
    global _store
    if _store is None:
        Path(_SHARED_DIR).mkdir(parents=True, exist_ok=True)
        _store = ResearchStore(db_path=str(Path(_SHARED_DIR) / "research.db"))
    return _store


def _get_vector() -> ResearchVector:
    global _vector
    if _vector is None:
        _vector = ResearchVector(chroma_path=str(Path(_SHARED_DIR) / "chroma"))
    return _vector


def _entity_summary(entity_type: str, entity_id: str, score: float) -> dict[str, Any] | None:
    """Fetch entity from store and build a standardized summary dict."""
    store = _get_store()
    if entity_type == "paper":
        obj = store.get_paper(entity_id)
        if obj is None:
            return None
        return {
            "entity_type": "paper",
            "entity_id": entity_id,
            "title": obj.title or entity_id,
            "score": round(score, 2),
            "tags": obj.tags,
            "summary": (obj.abstract or "")[:200],
        }
    elif entity_type == "experiment":
        obj = store.get_experiment(entity_id)
        if obj is None:
            return None
        return {
            "entity_type": "experiment",
            "entity_id": entity_id,
            "title": obj.title,
            "score": round(score, 2),
            "tags": obj.tags,
            "summary": (obj.conclusion or "")[:200],
        }
    elif entity_type == "hypothesis":
        obj = store.get_hypothesis(entity_id)
        if obj is None:
            return None
        return {
            "entity_type": "hypothesis",
            "entity_id": entity_id,
            "title": obj.title,
            "score": round(score, 2),
            "tags": obj.tags,
            "summary": (obj.description or "")[:200],
        }
    elif entity_type == "note":
        obj = store.get_note(entity_id)
        if obj is None:
            return None
        return {
            "entity_type": "note",
            "entity_id": entity_id,
            "title": obj.title,
            "score": round(score, 2),
            "tags": obj.tags,
            "summary": (obj.content or "")[:200],
        }
    return None


@mcp.tool()
async def unified_search(
    query: str,
    entity_types: list[str] | None = None,
    n: int = 15,
) -> list[dict]:
    """
    Semantic search across ALL research artifacts — papers, experiments,
    hypotheses, and notes — in a single query.

    ALWAYS display results as a table with columns:
    Type, Title, Relevance, ID, Tags

    Args:
        query: Natural language search query
        entity_types: Filter to specific types — ["paper", "experiment",
                      "hypothesis", "note"]. None searches all.
        n: Number of results to return (default 15)

    Returns results ranked by semantic relevance across entity types.
    Use this when you want to find everything related to a topic
    regardless of whether it's a paper, experiment, or hypothesis.
    """
    try:
        results = _get_vector().search(query, n=n, entity_types=entity_types)
    except Exception:
        return []

    summaries = []
    for r in results:
        summary = _entity_summary(r.entity_type, r.entity_id, r.score)
        if summary is not None:
            summaries.append(summary)

    return sorted(summaries, key=lambda x: x["score"], reverse=True)


@mcp.tool()
async def search_all_notes(query: str) -> list[dict]:
    """
    Full-text keyword search across all research notes using SQLite FTS5.
    Complements unified_search (semantic) with exact keyword matching.
    Returns notes attached to papers, experiments, or freeform.

    Args:
        query: Keyword search query (SQLite FTS5 syntax supported)
    """
    notes = _get_store().search_notes(query)
    return [
        {
            "id": n.id,
            "title": n.title,
            "content_preview": (n.content or "")[:300],
            "entity_type": n.entity_type,
            "entity_id": n.entity_id,
            "tags": n.tags,
        }
        for n in notes
    ]


@mcp.tool()
async def research_summary() -> dict:
    """
    High-level summary of everything in the research database.
    Use at the start of a session to orient yourself.
    Returns counts, recent activity, open hypotheses, and top tags.
    """
    store = _get_store()
    vector = _get_vector()

    papers = store.list_papers()
    experiments = store.list_experiments()
    tag_counts: dict[str, int] = {}

    # Paper stats
    source_counts: dict[str, int] = {}
    papers_with_text = 0
    for p in papers:
        source_counts[p.source] = source_counts.get(p.source, 0) + 1
        if p.full_text or p.text_path:
            papers_with_text += 1
        for t in p.tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1

    # Experiment stats
    exp_by_status: dict[str, int] = {}
    best_sharpe: float | None = None
    recent_exps = sorted(
        [e for e in experiments if e.created_at],
        key=lambda e: e.created_at,
        reverse=True,
    )[:3]
    for e in experiments:
        exp_by_status[e.status] = exp_by_status.get(e.status, 0) + 1
        sharpe = e.results.get("sharpe") if e.results else None
        if isinstance(sharpe, (int, float)):
            if best_sharpe is None or sharpe > best_sharpe:
                best_sharpe = sharpe
        for t in e.tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1

    # Hypothesis stats
    hyp_counts = {
        "open": len(store.list_hypotheses(status="open")),
        "testing": len(store.list_hypotheses(status="testing")),
        "confirmed": len(store.list_hypotheses(status="confirmed")),
        "rejected": len(store.list_hypotheses(status="rejected")),
    }
    for hyp in store.list_hypotheses():
        for t in hyp.tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1

    top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    # Vector stats
    try:
        vector_stats = {
            "total_chunks": vector.count(),
            "by_type": {
                "paper": vector.count("paper"),
                "experiment": vector.count("experiment"),
                "hypothesis": vector.count("hypothesis"),
                "note": vector.count("note"),
            },
        }
    except Exception:
        vector_stats = {"total_chunks": 0, "by_type": {}}

    return {
        "papers": {
            "total": store.paper_count(),
            "with_full_text": papers_with_text,
            "sources": source_counts,
        },
        "experiments": {
            "total": len(experiments),
            "by_status": exp_by_status,
            "best_sharpe": best_sharpe,
            "recent": [
                {"title": e.title, "dataset": e.dataset}
                for e in recent_exps
            ],
        },
        "hypotheses": hyp_counts,
        "top_tags": [{"tag": t, "count": c} for t, c in top_tags],
        "vector_index": vector_stats,
    }


@mcp.tool()
async def find_connections(entity_id: str, n: int = 10) -> dict:
    """
    Find everything in the research database semantically related to a
    given entity — whether it's a paper, experiment, or hypothesis.
    Returns related items across all entity types.

    More powerful than find_similar (papers only) or search_experiments —
    this crosses entity type boundaries.

    Example: find_connections("arxiv:2203.13820") returns related papers,
    experiments that tested this paper's ideas, and hypotheses connected
    to it.

    Args:
        entity_id: Any entity ID — paper ID, experiment UUID, hypothesis UUID
        n: Number of related items to return
    """
    store = _get_store()
    vector = _get_vector()

    # Resolve entity and build query text
    entity_type = None
    title = entity_id
    query_text = ""

    paper = store.get_paper(entity_id)
    if paper is not None:
        entity_type = "paper"
        title = paper.title or entity_id
        query_text = f"{paper.title or ''} {paper.abstract or ''}"
    else:
        exp = store.get_experiment(entity_id)
        if exp is not None:
            entity_type = "experiment"
            title = exp.title
            query_text = f"{exp.hypothesis or ''} {exp.conclusion or ''}"
        else:
            hyp = store.get_hypothesis(entity_id)
            if hyp is not None:
                entity_type = "hypothesis"
                title = hyp.title
                query_text = f"{hyp.title} {hyp.description or ''}"

    if not query_text:
        return {"error": f"Entity not found: {entity_id}"}

    try:
        results = vector.search(query_text.strip(), n=n + 1, entity_types=None)
    except Exception:
        return {"entity_id": entity_id, "entity_type": entity_type, "title": title, "related": []}

    related = []
    for r in results:
        if r.entity_id == entity_id:
            continue
        summary = _entity_summary(r.entity_type, r.entity_id, r.score)
        if summary is not None:
            related.append(summary)
        if len(related) >= n:
            break

    return {
        "entity_id": entity_id,
        "entity_type": entity_type,
        "title": title,
        "related": related,
    }


@mcp.tool()
async def graph_summary() -> dict:
    """
    Overview of the research knowledge graph.
    Returns relationship counts by type, most connected entities,
    and isolated entity counts (no relationships).
    Use to understand how your research artifacts are connected.
    """
    store = _get_store()
    graph = ResearchGraph(store)
    stats = graph.relationship_stats()

    # Find most connected entities
    all_rels = store.get_relationships()
    entity_counts: dict[str, int] = {}
    for rel in all_rels:
        entity_counts[rel.from_id] = entity_counts.get(rel.from_id, 0) + 1
        entity_counts[rel.to_id] = entity_counts.get(rel.to_id, 0) + 1

    most_connected = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    # Count isolated entities
    connected_ids = set(entity_counts.keys())
    papers_isolated = sum(1 for p in store.list_papers() if p.id not in connected_ids)
    exps_isolated = sum(1 for e in store.list_experiments() if e.id not in connected_ids)
    hyps_isolated = sum(1 for h in store.list_hypotheses() if h.id not in connected_ids)

    return {
        "total_relationships": sum(stats.values()),
        "by_type": stats,
        "most_connected": [{"entity_id": eid, "relationship_count": cnt} for eid, cnt in most_connected],
        "isolated_entities": {
            "papers": papers_isolated,
            "experiments": exps_isolated,
            "hypotheses": hyps_isolated,
        },
    }


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
