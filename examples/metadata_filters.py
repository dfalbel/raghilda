"""Example: attributes-aware insertion and retrieval filtering."""

from typing import Annotated

from raghilda.store import DuckDBIndexType, DuckDBStore
from raghilda.document import MarkdownDocument
from raghilda.chunker import MarkdownChunker


class ExampleAttributesSpec:
    tenant: str                 # required
    priority: int = 0           # optional, default 0
    is_public: bool = False     # optional, default False
    topic: str | None = None    # optional, default None


# All supported schema declaration styles:
#
# 1) Dict with scalar Python types (portable across DuckDB/Chroma/OpenAI today)
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
SCHEMA_CLASS = ExampleAttributesSpec
#
# 4) DuckDB-only fixed-size vectors + JSON-like nested object (DuckDB struct-style)
SCHEMA_DUCKDB_WITH_VECTOR = {
    "tenant": str,
    "topic": str,
    "priority": int,
    "embedding5": Annotated[list[float], 5],
    "details": {
        "source": str,
        "lang": str,
        "flags": {
            "is_public": bool,
            "is_internal": bool,
        },
    },
}

COMPLEX_ATTRIBUTES_BATCH_FOR_INSERT = [
    {
        "tenant": "docs",
        "topic": "guide",
        "priority": 10,
        "embedding5": [float(i) for i in range(5)],
        "details": {
            "source": "handbook",
            "lang": "en",
            "flags": {
                "is_public": True,
                "is_internal": False,
            },
        },
    },
    {
        "tenant": "docs",
        "topic": "notes",
        "priority": 3,
        "embedding5": [float(i + 1) for i in range(5)],
        "details": {
            "source": "internal-wiki",
            "lang": "en",
            "flags": {
                "is_public": False,
                "is_internal": True,
            },
        },
    },
    {
        "tenant": "blog",
        "topic": "post",
        "priority": 8,
        "embedding5": [float(i + 2) for i in range(5)],
        "details": {
            "source": "marketing-site",
            "lang": "en",
            "flags": {
                "is_public": True,
                "is_internal": False,
            },
        },
    },
]

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

store.build_index(DuckDBIndexType.BM25)

print("No filter:")
for chunk in store.retrieve("beta", top_k=6):
    print("-", chunk.text.strip(), chunk.attributes)

print("\nFiltered with SQL-like string:")
chunks_string = store.retrieve(
    "beta",
    top_k=6,
    attributes_filter="""
    tenant IN ('docs', 'blog')
    AND priority IN (5, 10)
    """,
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

print("\n--- Complex store with vector + nested attributes ---")
complex_store = DuckDBStore.create(
    location=":memory:",
    embed=None,
    attributes=SCHEMA_DUCKDB_WITH_VECTOR,
)

complex_docs = [
    MarkdownDocument(
        origin="complex-guide.md",
        content="advanced alpha beta with vector and nested attributes",
        attributes=COMPLEX_ATTRIBUTES_BATCH_FOR_INSERT[0],
    ),
    MarkdownDocument(
        origin="complex-notes.md",
        content="beta appears in lower-priority internal notes",
        attributes=COMPLEX_ATTRIBUTES_BATCH_FOR_INSERT[1],
    ),
    MarkdownDocument(
        origin="complex-blog.md",
        content="public beta write-up for external readers",
        attributes=COMPLEX_ATTRIBUTES_BATCH_FOR_INSERT[2],
    ),
]
for doc in complex_docs:
    complex_store.insert(chunker.chunk_document(doc))

complex_store.build_index(DuckDBIndexType.BM25)

print("\nComplex query (no filter):")
complex_results = complex_store.retrieve("beta", top_k=5, deoverlap=False)
for chunk in complex_results:
    print("-", chunk.text.strip(), chunk.attributes)

print("\nComplex query (SQL-like filter with dot-path object access):")
complex_sql_filter_results = complex_store.retrieve(
    "beta",
    top_k=5,
    deoverlap=False,
    attributes_filter="""
    tenant = 'docs'
    AND details.source = 'handbook'
    AND details.flags.is_public = TRUE
    """,
)
for chunk in complex_sql_filter_results:
    print("-", chunk.text.strip(), chunk.attributes)

print("\nComplex query (dict AST filter):")
complex_dict_filter_results = complex_store.retrieve(
    "beta",
    top_k=5,
    deoverlap=False,
    attributes_filter={
        "type": "and",
        "filters": [
            {"type": "eq", "key": "tenant", "value": "blog"},
            {"type": "gte", "key": "priority", "value": 5},
        ],
    },
)
for chunk in complex_dict_filter_results:
    print("-", chunk.text.strip(), chunk.attributes)
