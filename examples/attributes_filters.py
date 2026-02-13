"""Example: attributes-aware insertion and retrieval filtering."""

from typing import Annotated

from raghilda.store import DuckDBStore
from raghilda.document import MarkdownDocument
from raghilda.chunker import MarkdownChunker


def show_results(title, chunks):
    print(f"\n{title}")
    if not chunks:
        print("  (no matches)")
        return
    for idx, chunk in enumerate(chunks, start=1):
        print(f"  {idx}. {chunk.text.strip()!r}")
        print(f"     attributes={chunk.attributes}")


class ExampleAttributesSchemaClass:
    tenant: str                 # required
    priority: int = 0           # optional, default 0
    is_public: bool = False     # optional, default False
    topic: str | None = None    # optional, default None


# All supported schema declaration styles:
#
# 1) Dict with scalar Python types (portable across DuckDB/Chroma/OpenAI today)
EXAMPLE_ATTRIBUTES_SCHEMA_DICT = {
    "tenant": str,
    "topic": str,
    "priority": int,
}
#
# 2) Dict with explicit defaults (optional values)
EXAMPLE_ATTRIBUTES_SCHEMA_DICT_WITH_DEFAULTS = {
    "tenant": str,
    "priority": (int, 0),
    "is_public": (bool, False),
    "topic": (str | None, None),
}
#
# 3) Class annotations
EXAMPLE_ATTRIBUTES_SCHEMA_SIMPLE = ExampleAttributesSchemaClass
#
# 4) DuckDB-only fixed-size vectors + JSON-like nested object (DuckDB struct-style)
EXAMPLE_ATTRIBUTES_SCHEMA_COMPLEX = {
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

EXAMPLE_ATTRIBUTES_VALUES_COMPLEX_BATCH = [
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
    attributes=EXAMPLE_ATTRIBUTES_SCHEMA_SIMPLE,
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

store.build_index("bm25")

print("=== Simple Store ===")
show_results(
    "Simple query (no filter)",
    store.retrieve("beta", top_k=3, deoverlap=False),
)
show_results(
    "Simple query (SQL-like filter)",
    store.retrieve(
        "beta",
        top_k=3,
        deoverlap=False,
        attributes_filter="""
          tenant IN ('docs', 'blog')
          AND priority IN (5, 10)
        """,
    ),
)
show_results(
    "Simple query (dict AST filter)",
    store.retrieve(
        "beta",
        top_k=3,
        deoverlap=False,
        attributes_filter={
            "type": "and",
            "filters": [
                {"type": "eq", "key": "tenant", "value": "docs"},
                {"type": "in", "key": "priority", "value": [5, 10]},
            ],
        },
    ),
)

print("\n=== Complex Store (Vector + Nested Attributes) ===")
complex_store = DuckDBStore.create(
    location=":memory:",
    embed=None,
    attributes=EXAMPLE_ATTRIBUTES_SCHEMA_COMPLEX,
)

complex_docs = [
    MarkdownDocument(
        origin="complex-guide.md",
        content="advanced alpha beta with vector and nested attributes",
        attributes=EXAMPLE_ATTRIBUTES_VALUES_COMPLEX_BATCH[0],
    ),
    MarkdownDocument(
        origin="complex-notes.md",
        content="beta appears in lower-priority internal notes",
        attributes=EXAMPLE_ATTRIBUTES_VALUES_COMPLEX_BATCH[1],
    ),
    MarkdownDocument(
        origin="complex-blog.md",
        content="public beta write-up for external readers",
        attributes=EXAMPLE_ATTRIBUTES_VALUES_COMPLEX_BATCH[2],
    ),
]
for doc in complex_docs:
    complex_store.insert(chunker.chunk_document(doc))

complex_store.build_index("bm25")

show_results(
    "Complex query (no filter)",
    complex_store.retrieve("beta", top_k=3, deoverlap=False),
)
show_results(
    "Complex query (SQL-like filter with dot-path object access)",
    complex_store.retrieve(
        "beta",
        top_k=3,
        deoverlap=False,
        attributes_filter="""
        tenant = 'docs'
        AND details.source = 'handbook'
        AND details.flags.is_public = TRUE
        """,
    ),
)
show_results(
    "Complex query (dict AST filter)",
    complex_store.retrieve(
        "beta",
        top_k=3,
        deoverlap=False,
        attributes_filter={
            "type": "and",
            "filters": [
                {"type": "eq", "key": "tenant", "value": "blog"},
                {"type": "gte", "key": "priority", "value": 5},
            ],
        },
    ),
)
