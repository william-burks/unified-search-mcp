from __future__ import annotations

import pytest
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone

import search.server as srv
from research_core import ResearchStore, ResearchVector, Paper, Experiment, Hypothesis, ResearchNote


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@pytest.fixture(autouse=True)
def use_tmp_store(tmp_path: Path, monkeypatch):
    store = ResearchStore(db_path=str(tmp_path / "research.db"))
    vector = ResearchVector(chroma_path=str(tmp_path / "chroma"))
    monkeypatch.setattr(srv, "_store", store)
    monkeypatch.setattr(srv, "_vector", vector)
    yield


def _paper(pid: str = None, title: str = "Test paper", abstract: str = "volatility forecasting") -> Paper:
    return Paper(
        id=pid or f"arxiv:{uuid4().hex[:8]}",
        title=title,
        authors=["Author A"],
        abstract=abstract,
        source="arxiv",
        url="https://example.com",
    )


def _experiment(title: str = "Test exp", hypothesis: str = "H", conclusion: str = "C") -> Experiment:
    now = _now()
    return Experiment(
        id=str(uuid4()),
        title=title,
        hypothesis=hypothesis,
        conclusion=conclusion,
        code="# placeholder",
        dataset="DS",
        status="complete",
        created_at=now,
        updated_at=now,
    )


def _hypothesis(title: str = "Test hyp", description: str = "D") -> Hypothesis:
    now = _now()
    return Hypothesis(
        id=str(uuid4()),
        title=title,
        description=description,
        status="open",
        created_at=now,
        updated_at=now,
    )


def _note(title: str = "Test note", content: str = "note content") -> ResearchNote:
    now = _now()
    return ResearchNote(
        id=str(uuid4()),
        title=title,
        content=content,
        entity_type=None,
        entity_id=None,
        created_at=now,
        updated_at=now,
    )


async def test_unified_search_returns_papers():
    store = srv._get_store()
    vector = srv._get_vector()
    p = _paper(title="HAR volatility model", abstract="Realized variance prediction using HAR")
    store.save_paper(p)
    vector.upsert(p.id, "paper", f"{p.title} {p.abstract}", {"title": p.title})

    results = await srv.unified_search("realized variance HAR")
    assert len(results) >= 1
    assert results[0]["entity_type"] == "paper"
    assert results[0]["entity_id"] == p.id


async def test_unified_search_filters_by_entity_type():
    store = srv._get_store()
    vector = srv._get_vector()
    p = _paper(title="Vol paper", abstract="volatility model")
    e = _experiment(title="Vol experiment", hypothesis="volatility experiment hypothesis", conclusion="conclusion")
    store.save_paper(p)
    store.save_experiment(e)
    vector.upsert(p.id, "paper", f"{p.title} {p.abstract}", {"title": p.title})
    vector.upsert(e.id, "experiment", f"{e.hypothesis} {e.conclusion}", {"title": e.title})

    results = await srv.unified_search("volatility", entity_types=["experiment"])
    assert all(r["entity_type"] == "experiment" for r in results)
    ids = [r["entity_id"] for r in results]
    assert e.id in ids
    assert p.id not in ids


async def test_unified_search_crosses_entity_types():
    store = srv._get_store()
    vector = srv._get_vector()
    p = _paper(title="HAR model paper", abstract="heterogeneous autoregressive volatility")
    e = _experiment(title="HAR experiment", hypothesis="heterogeneous autoregressive model test", conclusion="HAR works")
    store.save_paper(p)
    store.save_experiment(e)
    vector.upsert(p.id, "paper", f"{p.title} {p.abstract}", {"title": p.title})
    vector.upsert(e.id, "experiment", f"{e.hypothesis} {e.conclusion}", {"title": e.title})

    results = await srv.unified_search("heterogeneous autoregressive")
    entity_types = {r["entity_type"] for r in results}
    assert "paper" in entity_types
    assert "experiment" in entity_types


async def test_research_summary_counts_correctly():
    store = srv._get_store()
    for i in range(3):
        store.save_paper(_paper(f"arxiv:p{i}", f"Paper {i}"))

    e1 = _experiment("Complete exp")
    e1.status = "complete"
    e2 = _experiment("Running exp")
    e2.status = "running"
    store.save_experiment(e1)
    store.save_experiment(e2)

    h = _hypothesis("Open hyp")
    store.save_hypothesis(h)

    summary = await srv.research_summary()
    assert summary["papers"]["total"] == 3
    assert summary["experiments"]["total"] == 2
    assert summary["experiments"]["by_status"]["complete"] == 1
    assert summary["experiments"]["by_status"]["running"] == 1
    assert summary["hypotheses"]["open"] == 1


async def test_find_connections_excludes_self():
    store = srv._get_store()
    vector = srv._get_vector()
    p1 = _paper(title="Main paper", abstract="volatility forecasting model")
    p2 = _paper(title="Related paper", abstract="realized volatility GARCH")
    store.save_paper(p1)
    store.save_paper(p2)
    vector.upsert(p1.id, "paper", f"{p1.title} {p1.abstract}", {"title": p1.title})
    vector.upsert(p2.id, "paper", f"{p2.title} {p2.abstract}", {"title": p2.title})

    result = await srv.find_connections(p1.id, n=10)
    related_ids = [r["entity_id"] for r in result["related"]]
    assert p1.id not in related_ids
    assert result["entity_id"] == p1.id
    assert result["entity_type"] == "paper"


async def test_search_all_notes_keyword_match():
    store = srv._get_store()
    vector = srv._get_vector()
    n = _note(title="HAR note", content="heterogeneous autoregressive model for realized variance")
    store.save_note(n)
    vector.upsert(n.id, "note", f"{n.title} {n.content}", {"title": n.title})

    results = await srv.search_all_notes("heterogeneous")
    assert len(results) >= 1
    assert any(r["id"] == n.id for r in results)
    match = next(r for r in results if r["id"] == n.id)
    assert "heterogeneous" in match["content_preview"]
