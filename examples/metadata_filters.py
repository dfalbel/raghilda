"""Example: metadata-aware insertion and retrieval filtering."""

from typing import Annotated

from raghilda.store import DuckDBStore, IndexType
from raghilda.document import MarkdownDocument
from raghilda.chunker import MarkdownChunker


class MetadataSpec:
    tenant: str
    topic: str
    priority: int


# All supported schema declaration styles:
#
# 1) Dict with scalar Python types
SCHEMA_DICT = {
    "tenant": str,
    "topic": str,
    "priority": int,
}
#
# 2) Class annotations
SCHEMA_CLASS = MetadataSpec
#
# 3) DuckDB-only fixed-size vectors
SCHEMA_DUCKDB_WITH_VECTOR = {
    "tenant": str,
    "topic": str,
    "priority": int,
    "embedding25": Annotated[list[float], 25],
}

store = DuckDBStore.create(
    location=":memory:",
    embed=None,
    metadata=SCHEMA_CLASS,
)

chunker = MarkdownChunker()
docs = [
    MarkdownDocument(
        origin="guide.md",
        content="alpha beta gamma",
        metadata={"tenant": "docs", "topic": "guide", "priority": 10},
    ),
    MarkdownDocument(
        origin="blog.md",
        content="beta appears in this public blog post",
        metadata={"tenant": "blog", "topic": "post", "priority": 1},
    ),
    MarkdownDocument(
        origin="notes.md",
        content="beta is also mentioned in internal notes",
        metadata={"tenant": "docs", "topic": "notes", "priority": 2},
    ),
]
for doc in docs:
    store.insert(chunker.chunk_document(doc))

store.build_index(IndexType.BM25)

print("No filter:")
for chunk in store.retrieve("beta", top_k=6):
    print("-", chunk.text.strip(), chunk.metadata)

print("\nFiltered with SQL-like string:")
chunks_string = store.retrieve(
    "beta",
    top_k=6,
    metadata_filter="tenant IN ('docs', 'blog') AND priority IN (5, 10)",
)
for chunk in chunks_string:
    print("-", chunk.text.strip(), chunk.metadata)

print("\nFiltered with dict AST:")
chunks_dict = store.retrieve(
    "beta",
    top_k=6,
    metadata_filter={
        "type": "and",
        "filters": [
            {"type": "eq", "key": "tenant", "value": "docs"},
            {"type": "in", "key": "priority", "value": [5, 10]},
        ],
    },
)
for chunk in chunks_dict:
    print("-", chunk.text.strip(), chunk.metadata)
