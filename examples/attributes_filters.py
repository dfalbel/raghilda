"""
Example: attributes-aware insertion and retrieval filtering.

This example shows you how to build a store for RAG where chunks are
augmented with attributes that you can use to filter or narrow the scope of
retrieval. You can also include those attributes with retrieved chunks so a
model gets extra context about where each chunk came from.

That’s metadata-aware insertion and retrieval filtering:

1. Define metadata schemas.
2. Store simple documents and run attribute-filtered retrieval with two
   filter syntaxes.
3. Repeat the same retrieval style with nested attributes.

The setup blocks are verbose by design so the query examples can be copied
into a minimal script and adapted quickly.
"""

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


# fmt: off
class ExampleAttributesSchemaClass:
    tenant: str                 # required
    priority: int = 0           # optional, default 0
    is_public: bool = False     # optional, default False
    topic: str | None = None    # optional, default None
# fmt: on

# All supported schema declaration styles:
#
# 1) Dict with scalar Python types (portable across DuckDB/Chroma/OpenAI today).
#    Best for quick schemas with only basic keys.
EXAMPLE_ATTRIBUTES_SCHEMA_DICT = {
    "tenant": str,
    "topic": str,
    "priority": int,
}
#
# 2) Dict with explicit defaults (optional values).
#    Useful when you want missing fields to resolve to defaults.
EXAMPLE_ATTRIBUTES_SCHEMA_DICT_WITH_DEFAULTS = {
    "tenant": str,
    "priority": (int, 0),
    "is_public": (bool, False),
    "topic": (str | None, None),
}
#
# 3) Class annotations.
#    Equivalent expressiveness with concise, type-checked-looking syntax.
EXAMPLE_ATTRIBUTES_SCHEMA_SIMPLE = ExampleAttributesSchemaClass
#
# 4) DuckDB-only fixed-size vectors + JSON-like nested object (DuckDB struct-style).
#    Required for array-like attributes and nested metadata in one row.
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

# You can skip directly here if you only care about usage:
#    - define a store,
#    - insert docs with metadata,
#    - query by text with optional attribute constraints.

# Choose a simple schema for the first pass so the filtering rules are easy
# to read.

store = DuckDBStore.create(
    location=":memory:",
    embed=None,
    attributes=EXAMPLE_ATTRIBUTES_SCHEMA_SIMPLE,
)

chunker = MarkdownChunker()
# The documents below are intentionally tiny so you can see exactly why each
# filter does or doesn't match.
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

# Build a text index before retrieval.
store.build_index("bm25")

print("=== Simple Store ===")
# First, confirm baseline retrieval without constraints.
show_results(
    "Simple query (no filter)",
    store.retrieve("beta", top_k=3, deoverlap=False),
)
# Next, apply an SQL-like filter string.
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
# Use the same semantics via the structured dict AST format.
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
# For nested attributes, the syntax is the same at the callsite; only the
# schema and filter keys become richer (dot-path access into objects).
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

# Rebuild index after loading the second dataset.
complex_store.build_index("bm25")

show_results(
    "Complex query (no filter)",
    complex_store.retrieve("beta", top_k=3, deoverlap=False),
)
# Dot-path example (`details.flags.is_public`) drills into nested fields.
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
# Equivalent semantics with dict AST for nested/typed metadata.
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
