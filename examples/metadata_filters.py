"""Example: attributes-aware insertion and retrieval filtering."""

from typing import Annotated

from raghilda.store import DuckDBStore, IndexType
from raghilda.document import MarkdownDocument
from raghilda.chunker import MarkdownChunker


class AttributesSpec:
    tenant: str                 # required
    priority: int = 0           # optional, default 0
    is_public: bool = False     # optional, default False
    topic: str | None = None    # optional, default None


# All supported schema declaration styles:
#
# 1) Dict with scalar Python types
SCHEMA_DICT = {
    "tenant": str,
    "topic": str,
    "priority": int,
}
#
# 2) Dict with explicit defaults (optional values)
SCHEMA_DICT_WITH_DEFAULTS = {
    "tenant": str,
    "priority": (int, 0),
    "is_public": (bool, False),
    "topic": (str | None, None),
}
#
# 3) Class annotations
SCHEMA_CLASS = AttributesSpec
#
# 4) DuckDB-only fixed-size vectors
SCHEMA_DUCKDB_WITH_VECTOR = {
    "tenant": str,
    "topic": str,
    "priority": int,
    "embedding25": Annotated[list[float], 25],
}

store = DuckDBStore.create(
    location=":memory:",
    embed=None,
    attributes=SCHEMA_CLASS,
)

chunker = MarkdownChunker()
docs = [
    MarkdownDocument(
        origin="guide.md",
        content="alpha beta gamma",
        attributes={"tenant": "docs", "topic": "guide", "priority": 10},
    ),
    MarkdownDocument(
        origin="blog.md",
        content="beta appears in this public blog post",
        attributes={"tenant": "blog", "topic": "post", "priority": 1},
    ),
    MarkdownDocument(
        origin="notes.md",
        content="beta is also mentioned in internal notes",
        attributes={"tenant": "docs", "topic": "notes", "priority": 2},
    ),
]
for doc in docs:
    store.insert(chunker.chunk_document(doc))

store.build_index(IndexType.BM25)

print("No filter:")
for chunk in store.retrieve("beta", top_k=6):
    print("-", chunk.text.strip(), chunk.attributes)

print("\nFiltered with SQL-like string:")
chunks_string = store.retrieve(
    "beta",
    top_k=6,
    attributes_filter="tenant IN ('docs', 'blog') AND priority IN (5, 10)",
)
for chunk in chunks_string:
    print("-", chunk.text.strip(), chunk.attributes)

print("\nFiltered with dict AST:")
chunks_dict = store.retrieve(
    "beta",
    top_k=6,
    attributes_filter={
        "type": "and",
        "filters": [
            {"type": "eq", "key": "tenant", "value": "docs"},
            {"type": "in", "key": "priority", "value": [5, 10]},
        ],
    },
)
for chunk in chunks_dict:
    print("-", chunk.text.strip(), chunk.attributes)
