# search-mcp

MCP server for unified semantic search across all QuantResearcher artifacts.

## Tools

| Tool | Purpose |
|------|---------|
| `unified_search` | Semantic search across papers, experiments, hypotheses, and notes in one query |
| `search_all_notes` | FTS5 keyword search across all research notes |
| `research_summary` | High-level dashboard — counts, best Sharpe, open hypotheses, top tags |
| `find_connections` | Find semantically related items across all entity types for a given entity |
| `graph_summary` | Overview of knowledge graph — relationship counts, most connected entities, isolated entities |

## Usage

```
# Session start — orient yourself
research_summary()

# Find everything related to a topic, regardless of entity type
unified_search("HAR realized volatility forecasting")

# Find everything semantically related to a specific paper or experiment
find_connections("arxiv:2203.13820")

# Keyword search across notes
search_all_notes("heterogeneous autoregressive")
```

## Storage

Read-only access to the shared `research_core` store at `RESEARCH_DATA_DIR`.
Does not write to the database — purely a query layer over the shared SQLite + ChromaDB index.

## Note

This repo depends on `research-core`, an internal package in my personal
research workspace (not published). The server won't install as-is; it's
shared here as a reference implementation of the MCP pattern — tool shape,
FastMCP wiring, and the layering of SQLite FTS5 + ChromaDB for hybrid search.
