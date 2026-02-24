import os
import hashlib
import json
from types import SimpleNamespace
from typing import Annotated, Any, cast
import httpx
import openai
import pytest
from raghilda.store import DuckDBStore, OpenAIStore
from raghilda.scrape import find_links
from raghilda.document import MarkdownDocument
from raghilda.chunk import MarkdownChunk, RetrievedChunk
from raghilda._attributes import AttributeFloatVectorType
from raghilda._duckdb_store import (
    RetrievedDuckDBMarkdownChunk,
)  # internal implementation
from raghilda._openai_store import _normalize_openai_attributes
from raghilda.embedding import EmbeddingOpenAI
from raghilda._embedding import EmbeddingProvider, EmbedInputType


class CountingEmbedding(EmbeddingProvider):
    def __init__(self):
        self.calls = 0

    def embed(
        self,
        x,
        input_type: EmbedInputType = EmbedInputType.DOCUMENT,
    ):
        self.calls += 1
        return [[float(len(text))] for text in x]

    def get_config(self):
        return {"type": "CountingEmbedding"}

    @classmethod
    def from_config(cls, config):
        return cls()


def _skip_if_unset(env_var: str) -> None:
    if not os.getenv(env_var):
        pytest.skip(f"{env_var} not set in environment variables")


def test_skip_if_unset_skips_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(pytest.skip.Exception):
        _skip_if_unset("OPENAI_API_KEY")


class TestDuckDBStore:
    @pytest.fixture
    def embed(self, request):
        try:
            value = request.param
            if isinstance(value, EmbeddingOpenAI):
                _skip_if_unset("OPENAI_API_KEY")
            return value
        except AttributeError:
            return None

    @pytest.fixture
    def store(self, embed):
        store = DuckDBStore.create(
            location=":memory:",
            embed=embed,
            overwrite=True,
            name="test_db",
            title="Test DuckDB Store",
        )
        return store

    @pytest.fixture
    def store_with_docs(self, store):
        doc = MarkdownDocument(origin="test", content="This is a test document.")
        doc.chunks = [
            _get_markdown_chunk(doc, start=0, end=4),
            _get_markdown_chunk(doc, start=5, end=7),
            _get_markdown_chunk(doc, start=8, end=9),
            _get_markdown_chunk(doc, start=10, end=14),
            _get_markdown_chunk(doc, start=15, end=23),
        ]
        store.insert(doc)
        return store

    def test_create_store(self, store):
        assert isinstance(store, DuckDBStore)
        assert store.metadata.name == "test_db"
        assert store.metadata.title == "Test DuckDB Store"
        assert store.metadata.embed is None

    @pytest.mark.parametrize("embed", [None, EmbeddingOpenAI()], indirect=True)
    def test_insert(self, store_with_docs):
        assert store_with_docs.size() == 1

    def test_insert_same_origin_skips_unchanged_by_default(self):
        embed = CountingEmbedding()
        store = DuckDBStore.create(
            location=":memory:",
            embed=embed,
            overwrite=True,
            name="insert_skip_unchanged",
        )
        calls_after_create = embed.calls
        doc = MarkdownDocument(origin="doc-1", content="hello world")
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=len(doc.content),
                text=doc.content,
                token_count=len(doc.content),
            )
        ]

        first_write = store.insert(doc)
        assert first_write.document.origin == "doc-1"
        assert first_write.document.content == "hello world"
        assert embed.calls == calls_after_create + 1

        second_write = store.insert(doc)
        assert second_write.action == "skipped"
        assert second_write.document.origin == "doc-1"
        assert second_write.document.content == "hello world"
        assert embed.calls == calls_after_create + 1

        rows = store.con.execute(
            "SELECT COUNT(*) FROM documents WHERE origin = 'doc-1'"
        ).fetchone()
        assert rows is not None
        assert rows[0] == 1

    def test_insert_same_origin_rewrites_when_skip_disabled(self):
        embed = CountingEmbedding()
        store = DuckDBStore.create(
            location=":memory:",
            embed=embed,
            overwrite=True,
            name="insert_force_rewrite",
        )
        calls_after_create = embed.calls
        doc = MarkdownDocument(origin="doc-1", content="hello world")
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=len(doc.content),
                text=doc.content,
                token_count=len(doc.content),
            )
        ]

        store.insert(doc)
        assert embed.calls == calls_after_create + 1

        store.insert(doc, skip_if_unchanged=False)
        assert embed.calls == calls_after_create + 2

    def test_insert_same_content_but_different_chunking_updates(self):
        embed = CountingEmbedding()
        store = DuckDBStore.create(
            location=":memory:",
            embed=embed,
            overwrite=True,
            name="insert_same_content_new_chunks",
        )
        calls_after_create = embed.calls

        content = "hello world"
        doc1 = MarkdownDocument(origin="doc-1", content=content)
        doc1.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=len(content),
                text=content,
                token_count=len(content),
            )
        ]
        first = store.insert(doc1)
        assert first.action == "inserted"
        assert embed.calls == calls_after_create + 1

        doc2 = MarkdownDocument(origin="doc-1", content=content)
        doc2.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text=content[0:5],
                token_count=5,
            ),
            MarkdownChunk(
                start_index=6,
                end_index=len(content),
                text=content[6:],
                token_count=len(content[6:]),
            ),
        ]
        second = store.insert(doc2)
        assert second.action == "replaced"
        assert embed.calls == calls_after_create + 2

        chunk_count = store.con.execute(
            "SELECT COUNT(*) FROM embeddings e JOIN documents d USING (doc_id) WHERE d.origin = 'doc-1'"
        ).fetchone()
        assert chunk_count is not None
        assert chunk_count[0] == 2

    def test_insert_same_layout_but_different_chunk_text_updates(self):
        embed = CountingEmbedding()
        store = DuckDBStore.create(
            location=":memory:",
            embed=embed,
            overwrite=True,
            name="insert_same_layout_new_text",
        )
        calls_after_create = embed.calls

        content = "hello world"
        doc1 = MarkdownDocument(origin="doc-1", content=content)
        doc1.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=len(content),
                text=content,
                token_count=len(content),
            )
        ]
        first = store.insert(doc1)
        assert first.action == "inserted"
        assert embed.calls == calls_after_create + 1

        doc2 = MarkdownDocument(origin="doc-1", content=content)
        doc2.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=len(content),
                text=content.upper(),
                token_count=len(content),
            )
        ]
        second = store.insert(doc2)
        assert second.action == "replaced"
        assert embed.calls == calls_after_create + 2

    def test_insert_same_multi_chunk_layout_skips_when_unchanged(self):
        embed = CountingEmbedding()
        store = DuckDBStore.create(
            location=":memory:",
            embed=embed,
            overwrite=True,
            name="insert_same_multi_chunk_skip",
        )
        calls_after_create = embed.calls
        content = "hello world"
        doc = MarkdownDocument(origin="doc-1", content=content)
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text=content[:5],
                token_count=5,
            ),
            MarkdownChunk(
                start_index=6,
                end_index=len(content),
                text=content[6:],
                token_count=len(content[6:]),
            ),
        ]

        first = store.insert(doc)
        assert first.action == "inserted"
        assert embed.calls == calls_after_create + 1

        second = store.insert(doc)
        assert second.action == "skipped"
        assert embed.calls == calls_after_create + 1

    def test_insert_requires_origin(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            name="insert_missing_origin",
        )
        doc = MarkdownDocument(origin=None, content="hello world")
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=len(doc.content),
                text=doc.content,
                token_count=len(doc.content),
            )
        ]

        with pytest.raises(ValueError, match="document.origin is required"):
            store.insert(doc)

    def test_insert_returns_replaced_document_when_updated(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            name="insert_replace_snapshot",
        )
        first = MarkdownDocument(origin="doc-1", content="hello world")
        first.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=len(first.content),
                text=first.content,
                token_count=len(first.content),
            )
        ]

        inserted = store.insert(first)
        assert inserted.action == "inserted"
        assert inserted.document.origin == "doc-1"
        assert inserted.document.content == "hello world"
        assert inserted.replaced_document is None

        second = MarkdownDocument(origin="doc-1", content="goodbye world")
        second.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=len(second.content),
                text=second.content,
                token_count=len(second.content),
            )
        ]

        updated = store.insert(second)
        assert updated.action == "replaced"
        assert updated.document.origin == "doc-1"
        assert updated.document.content == "goodbye world"
        assert updated.replaced_document is not None
        assert updated.replaced_document.origin == "doc-1"
        assert updated.replaced_document.content == "hello world"
        assert updated.replaced_document.chunks is not None
        assert len(updated.replaced_document.chunks) == 1

        restored = store.insert(updated.replaced_document, skip_if_unchanged=False)
        assert restored.action == "replaced"
        current = store.con.execute(
            "SELECT text FROM documents WHERE origin = 'doc-1'"
        ).fetchone()
        assert current is not None
        assert current[0] == "hello world"

    def test_insert_replaced_document_preserves_multi_chunk_text(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            name="insert_replace_multi_chunk_snapshot",
        )
        first = MarkdownDocument(origin="doc-1", content="hello world")
        first.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text="hello",
                token_count=5,
            ),
            MarkdownChunk(
                start_index=6,
                end_index=11,
                text="world",
                token_count=5,
            ),
        ]
        store.insert(first)

        second = MarkdownDocument(origin="doc-1", content="hello mars")
        second.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text="hello",
                token_count=5,
            ),
            MarkdownChunk(
                start_index=6,
                end_index=10,
                text="mars",
                token_count=4,
            ),
        ]
        updated = store.insert(second)
        assert updated.action == "replaced"
        assert updated.replaced_document is not None
        assert updated.replaced_document.chunks is not None
        assert [chunk.text for chunk in updated.replaced_document.chunks] == [
            "hello",
            "world",
        ]

    @pytest.mark.parametrize("embed", [EmbeddingOpenAI()], indirect=True)
    def test_retrieve_vss(self, store_with_docs):
        results = store_with_docs.retrieve_vss("test", top_k=3)
        assert len(results) == 3
        for chunk in results:
            assert isinstance(chunk, RetrievedDuckDBMarkdownChunk)
            assert chunk.text is not None

        results = store_with_docs.retrieve_vss("test", top_k=5)
        assert len(results) == 5

    @pytest.mark.parametrize("embed", [None, EmbeddingOpenAI()], indirect=True)
    def test_retrieve_bm25(self, store_with_docs):
        store_with_docs.build_index("bm25")
        results = store_with_docs.retrieve_bm25("document", top_k=3)
        assert len(results) == 3
        for chunk in results:
            assert isinstance(chunk, RetrievedDuckDBMarkdownChunk)
            assert chunk.text is not None

    @pytest.mark.parametrize("embed", [EmbeddingOpenAI()], indirect=True)
    def test_retrieve(self, store_with_docs):
        store_with_docs.build_index()
        results = store_with_docs.retrieve("document", top_k=3, deoverlap=False)
        assert len(results) > 3
        for chunk in results:
            assert isinstance(chunk, RetrievedDuckDBMarkdownChunk)
            assert chunk.text is not None

    def test_retrieve_with_deoverlap(self, store):
        # Create a document with overlapping chunks
        # "hello world test document" = 24 chars
        doc = MarkdownDocument(
            origin="test_deoverlap", content="hello world test document"
        )
        doc.chunks = [
            _get_markdown_chunk(doc, start=0, end=11),  # "hello world"
            _get_markdown_chunk(doc, start=6, end=16),  # "world test"
            _get_markdown_chunk(doc, start=12, end=25),  # "test document"
        ]
        store.insert(doc)
        store.build_index("bm25")

        # Without deoverlap, we may get multiple overlapping chunks
        results_no_deoverlap = store.retrieve("test", top_k=5, deoverlap=False)

        # With deoverlap, overlapping chunks should be merged
        results_deoverlap = store.retrieve("test", top_k=5, deoverlap=True)

        # Deoverlapped results should have fewer or equal chunks
        assert len(results_deoverlap) <= len(results_no_deoverlap)

        # Check that deoverlapped chunks don't have overlapping ranges
        for i, chunk1 in enumerate(results_deoverlap):
            for chunk2 in results_deoverlap[i + 1 :]:
                if chunk1.doc_id == chunk2.doc_id:
                    # Same document - ranges should not overlap
                    assert (
                        chunk1.end_index <= chunk2.start_index
                        or chunk2.end_index <= chunk1.start_index
                    ), (
                        f"Chunks overlap: [{chunk1.start_index}, {chunk1.end_index}) and [{chunk2.start_index}, {chunk2.end_index})"
                    )

    def test_retrieve_with_deoverlap_aggregates_attributes(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            attributes={"topic": str},
        )
        doc = MarkdownDocument(
            origin="test_deoverlap_attributes",
            content="alpha beta gamma",
            attributes={"topic": "first"},
        )
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=10,
                text="alpha beta",
                token_count=10,
                context="h1",
            ),
            MarkdownChunk(
                start_index=6,
                end_index=16,
                text="beta gamma",
                token_count=10,
                context="h2",
                attributes={"topic": "second"},
            ),
        ]
        store.insert(doc)
        store.build_index("bm25")

        results = store.retrieve("beta", top_k=5, deoverlap=True)

        assert len(results) == 1
        assert results[0].context == "h1"
        assert results[0].attributes == {"topic": ["first", "second"]}

    def test_create_store_with_attributes_schema(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            name="attributes_schema_db",
            title="Attributes Schema Store",
            attributes={
                "tenant": str,
                "priority": int,
                "is_public": bool,
            },
        )

        assert store.metadata.attributes_schema == {
            "tenant": str,
            "priority": int,
            "is_public": bool,
        }

        columns = store.con.execute("DESCRIBE embeddings").fetchall()
        columns_by_name = {row[0]: row[1] for row in columns}
        names = set(columns_by_name)
        assert "tenant" in names
        assert "priority" in names
        assert "is_public" in names
        assert columns_by_name["tenant"] == "VARCHAR"
        assert columns_by_name["priority"] == "INTEGER"
        assert columns_by_name["is_public"] == "BOOLEAN"

    def test_create_store_with_attributes_schema_class_annotations(self):
        class AttributesSpec:
            tenant: str
            priority: int
            is_public: bool

        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            attributes=AttributesSpec,
        )

        assert store.metadata.attributes_schema == {
            "tenant": str,
            "priority": int,
            "is_public": bool,
        }

    def test_create_store_rejects_invalid_attribute_names(self):
        with pytest.raises(ValueError, match="must match"):
            DuckDBStore.create(
                location=":memory:",
                embed=None,
                overwrite=True,
                attributes={"tenant-id": str},
            )

    def test_create_store_with_vector_attributes_annotation(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            attributes={
                "tenant": str,
                "embedding25": Annotated[list[float], 25],
            },
        )

        vector_type = store.metadata.attributes_schema["embedding25"]
        assert isinstance(vector_type, AttributeFloatVectorType)
        assert vector_type.dimension == 25

        columns = store.con.execute("DESCRIBE embeddings").fetchall()
        columns_by_name = {row[0]: row[1] for row in columns}
        assert columns_by_name["embedding25"] == "FLOAT[25]"

        vector = [float(i) for i in range(25)]
        doc = MarkdownDocument(
            origin="vector-attributes",
            content="hello vector attributes",
            attributes={"tenant": "docs", "embedding25": vector},
        )
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=len(doc.content),
                text=doc.content,
                token_count=len(doc.content),
            )
        ]
        store.insert(doc)
        store.build_index("bm25")

        results = store.retrieve(
            "hello",
            top_k=1,
            deoverlap=False,
        )
        assert len(results) == 1
        assert results[0].attributes is not None
        assert results[0].attributes["embedding25"] == pytest.approx(vector)

        filterable_columns = store._filterable_columns()
        assert "tenant" in filterable_columns
        assert "start_index" in filterable_columns
        assert "end_index" in filterable_columns
        assert "start" not in filterable_columns
        assert "end" not in filterable_columns
        assert "embedding25" not in filterable_columns

        with pytest.raises(ValueError, match="Unknown attribute column 'embedding25'"):
            store.retrieve(
                "hello",
                top_k=1,
                deoverlap=False,
                attributes_filter="embedding25 = 1",
            )

    def test_insert_same_vector_attributes_skips_when_unchanged(self):
        embed = CountingEmbedding()
        store = DuckDBStore.create(
            location=":memory:",
            embed=embed,
            overwrite=True,
            attributes={
                "tenant": str,
                "embedding3": Annotated[list[float], 3],
            },
        )
        calls_after_create = embed.calls

        vector = [1.0, 2.0, 3.0]
        doc = MarkdownDocument(
            origin="vector-unchanged",
            content="hello vector unchanged",
            attributes={"tenant": "docs", "embedding3": vector},
        )
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=len(doc.content),
                text=doc.content,
                token_count=len(doc.content),
            )
        ]

        first = store.insert(doc)
        assert first.action == "inserted"
        assert embed.calls == calls_after_create + 1

        second = store.insert(doc)
        assert second.action == "skipped"
        assert embed.calls == calls_after_create + 1

    def test_insert_and_retrieve_with_attributes_filter(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            attributes={
                "tenant": str,
                "priority": int,
                "is_public": bool | None,
            },
        )

        doc = MarkdownDocument(
            origin="attributes-test",
            content="alpha beta gamma",
            attributes={"tenant": "docs", "priority": 1},
        )
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text="alpha",
                token_count=5,
                attributes={"priority": 5, "is_public": False},
            ),
            MarkdownChunk(
                start_index=6,
                end_index=10,
                text="beta",
                token_count=4,
            ),
            MarkdownChunk(
                start_index=11,
                end_index=16,
                text="gamma",
                token_count=5,
            ),
        ]

        store.insert(doc)
        store.build_index("bm25")

        private_results = store.retrieve(
            "alpha",
            top_k=10,
            deoverlap=False,
            attributes_filter="tenant = 'docs' AND priority >= 5",
        )
        assert len(private_results) == 1
        assert private_results[0].text.strip() == "alpha"
        assert private_results[0].attributes == {
            "tenant": "docs",
            "priority": 5,
            "is_public": False,
        }

        public_results = store.retrieve(
            "beta",
            top_k=10,
            deoverlap=False,
            attributes_filter="tenant = 'docs' AND is_public IS NULL AND priority = 1",
        )
        assert len(public_results) == 1
        assert public_results[0].text.strip() == "beta"
        assert public_results[0].attributes == {
            "tenant": "docs",
            "priority": 1,
            "is_public": None,
        }

        dict_results = store.retrieve(
            "alpha",
            top_k=10,
            deoverlap=False,
            attributes_filter={
                "type": "and",
                "filters": [
                    {"type": "eq", "key": "tenant", "value": "docs"},
                    {"type": "in", "key": "priority", "value": [5, 10]},
                ],
            },
        )
        assert len(dict_results) == 1
        assert dict_results[0].text.strip() == "alpha"

        positional_results = store.retrieve(
            "alpha",
            top_k=10,
            deoverlap=False,
            attributes_filter="start_index = 0 AND end_index = 5",
        )
        assert len(positional_results) == 1
        assert positional_results[0].text.strip() == "alpha"

        with pytest.raises(ValueError, match="Unknown attribute column 'start'"):
            store.retrieve(
                "alpha",
                top_k=10,
                deoverlap=False,
                attributes_filter="start = 0",
            )

    def test_insert_preserves_declared_id_attribute(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            attributes={
                "id": str,
                "tenant": str,
            },
        )

        doc = MarkdownDocument(
            origin="id-attribute-test",
            content="alpha beta",
            attributes={"id": "attr-id-1", "tenant": "docs"},
        )
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text="alpha",
                token_count=5,
            )
        ]

        store.insert(doc)
        store.build_index("bm25")

        results = store.retrieve(
            "alpha",
            top_k=5,
            deoverlap=False,
            attributes_filter="id = 'attr-id-1'",
        )
        assert len(results) == 1
        assert results[0].attributes == {"id": "attr-id-1", "tenant": "docs"}

    def test_retrieve_rejects_text_attribute_filter(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
        )

        doc = MarkdownDocument(
            origin="text-filter-test",
            content="alpha beta",
        )
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text="alpha",
                token_count=5,
            )
        ]

        store.insert(doc)
        store.build_index("bm25")

        with pytest.raises(ValueError, match="Unknown attribute column 'text'"):
            store.retrieve(
                "alpha",
                top_k=5,
                deoverlap=False,
                attributes_filter="text = 'alpha'",
            )

    def test_insert_and_retrieve_with_nested_object_attributes_filter(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            attributes={
                "tenant": str,
                "details": {
                    "source": str,
                    "flags": {"is_public": bool, "is_internal": bool},
                },
            },
        )

        doc = MarkdownDocument(
            origin="nested-attributes-test",
            content="alpha beta gamma",
            attributes={
                "tenant": "docs",
                "details": {
                    "source": "handbook",
                    "flags": {"is_public": True, "is_internal": False},
                },
            },
        )
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text="alpha",
                token_count=5,
            )
        ]

        store.insert(doc)
        store.build_index("bm25")

        results = store.retrieve(
            "alpha",
            top_k=5,
            deoverlap=False,
            attributes_filter=(
                "tenant = 'docs' "
                "AND details.source = 'handbook' "
                "AND details.flags.is_public = TRUE"
            ),
        )
        assert len(results) == 1
        assert results[0].text.strip() == "alpha"
        assert results[0].attributes == {
            "tenant": "docs",
            "details": {
                "source": "handbook",
                "flags": {"is_public": True, "is_internal": False},
            },
        }

    def test_nested_object_attribute_rejects_hyphenated_field_name(self):
        with pytest.raises(ValueError, match="must match"):
            DuckDBStore.create(
                location=":memory:",
                embed=None,
                overwrite=True,
                attributes={
                    "details": {
                        "source-type": str,
                    },
                },
            )

    def test_insert_applies_inline_attribute_defaults(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            attributes={
                "tenant": str,
                "priority": (int, 0),
                "is_public": (bool, False),
                "topic": (str | None, None),
            },
        )

        doc = MarkdownDocument(
            origin="defaults-test",
            content="alpha beta",
            attributes={"tenant": "docs"},
        )
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text="alpha",
                token_count=5,
            )
        ]

        store.insert(doc)
        store.build_index("bm25")

        results = store.retrieve("alpha", top_k=5, deoverlap=False)
        assert len(results) == 1
        assert results[0].attributes == {
            "tenant": "docs",
            "priority": 0,
            "is_public": False,
            "topic": None,
        }

    def test_insert_missing_required_attribute_fails(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            attributes={"tenant": str, "priority": (int, 0)},
        )

        doc = MarkdownDocument(
            origin="required-fail",
            content="hello",
        )
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text="hello",
                token_count=5,
            )
        ]

        with pytest.raises(ValueError, match="Missing required attribute 'tenant'"):
            store.insert(doc)

    def test_insert_attributes_without_declared_schema_fails(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
        )

        doc = MarkdownDocument(
            origin="attributes-fail",
            content="hello",
            attributes={"tenant": "docs"},
        )
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text="hello",
                token_count=5,
            )
        ]

        with pytest.raises(ValueError, match="Unknown attribute key 'tenant'"):
            store.insert(doc)

    def test_insert_unknown_chunk_attributes_key_fails(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            attributes={"tenant": str},
        )

        doc = MarkdownDocument(
            origin="unknown-key-fail",
            content="hello",
            attributes={"tenant": "docs"},
        )
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text="hello",
                token_count=5,
                attributes={"unknown": "x"},
            )
        ]

        with pytest.raises(ValueError, match="Unknown attribute key 'unknown'"):
            store.insert(doc)

    def test_insert_rejects_float_for_int_attribute(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            attributes={"priority": int},
        )

        doc = MarkdownDocument(
            origin="type-mismatch-float-for-int",
            content="hello",
            attributes={"priority": 1.5},
        )
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text="hello",
                token_count=5,
            )
        ]

        with pytest.raises(
            ValueError,
            match="Invalid value for attributes 'priority': expected int, got float",
        ):
            store.insert(doc)

    def test_insert_rejects_int_for_float_attribute(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            attributes={"score": float},
        )

        doc = MarkdownDocument(
            origin="type-mismatch-int-for-float",
            content="hello",
            attributes={"score": 1},
        )
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text="hello",
                token_count=5,
            )
        ]

        with pytest.raises(
            ValueError,
            match="Invalid value for attributes 'score': expected float, got int",
        ):
            store.insert(doc)

    def test_connect_restores_attributes_schema(self, tmp_path):
        db_path = tmp_path / "attributes-connect.db"
        store = DuckDBStore.create(
            location=str(db_path),
            embed=None,
            overwrite=True,
            attributes={"tenant": str, "priority": int},
        )
        store.con.close()

        store2 = DuckDBStore.connect(str(db_path))
        assert store2.metadata.attributes_schema == {
            "tenant": str,
            "priority": int,
        }


class TestOpenAIStore:
    @pytest.fixture(autouse=True)
    def setup(self):
        _skip_if_unset("OPENAI_API_KEY")

    @pytest.fixture
    def store(self):
        store = OpenAIStore.create()
        try:
            yield store
        finally:
            try:
                store.client.vector_stores.delete(vector_store_id=store.store_id)
            except openai.AuthenticationError:
                pass

    @pytest.fixture(scope="class")
    def store_with_attributes(self):
        _skip_if_unset("OPENAI_API_KEY")
        store = OpenAIStore.create(attributes={"tenant": str, "priority": int})
        store.insert(
            MarkdownDocument(
                origin="doc-attrs",
                content="alpha bronze owl",
                attributes={"tenant": "docs", "priority": 2},
            ),
        )
        store.insert(
            MarkdownDocument(
                origin="docs-priority-1",
                content="alpha beta",
                attributes={"tenant": "docs", "priority": 1},
            ),
        )
        store.insert(
            MarkdownDocument(
                origin="ops-priority-5",
                content="alpha gamma",
                attributes={"tenant": "ops", "priority": 5},
            ),
        )
        store.insert(
            MarkdownDocument(
                origin="docs-priority-3",
                content="alpha alpha delta",
                attributes={"tenant": "docs", "priority": 3},
            ),
        )
        try:
            yield store
        finally:
            try:
                store.client.vector_stores.delete(vector_store_id=store.store_id)
            except openai.AuthenticationError:
                pass

    @pytest.fixture
    def store_with_class_attributes(self):
        class AttributesSpec:
            tenant: str
            priority: int

        _skip_if_unset("OPENAI_API_KEY")
        store = OpenAIStore.create(attributes=AttributesSpec)
        try:
            yield store
        finally:
            try:
                store.client.vector_stores.delete(vector_store_id=store.store_id)
            except openai.AuthenticationError:
                pass

    @pytest.fixture
    def store_with_docs(self, store):
        doc = MarkdownDocument(
            origin="test", content="hello world this is a document world world world"
        )
        store.insert(doc)
        return store

    def test_create_store(self, store):
        assert isinstance(store, OpenAIStore)
        assert isinstance(store.store_id, str)

    def test_insert(self, store_with_docs):
        assert store_with_docs.size() == 1

    def test_retrieve(self, store_with_docs):
        results = store_with_docs.retrieve("world", top_k=3)
        assert len(results) > 0
        for chunk in results:
            assert isinstance(chunk, RetrievedChunk)
            assert chunk.text is not None

    def test_connect_restores_attributes_schema(self, store_with_attributes):
        connected = OpenAIStore.connect(store_id=store_with_attributes.store_id)
        assert connected.attributes_schema == {"tenant": str, "priority": int}

    def test_create_accepts_class_attributes_schema(self, store_with_class_attributes):
        connected = OpenAIStore.connect(store_id=store_with_class_attributes.store_id)
        assert connected.attributes_schema == {"tenant": str, "priority": int}

    def test_insert_uses_document_attributes(self, store_with_attributes):
        results = store_with_attributes.retrieve(
            "bronze owl",
            top_k=5,
            attributes_filter="tenant = 'docs' AND priority = 2",
        )
        assert len(results) > 0
        assert all(chunk.attributes["tenant"] == "docs" for chunk in results)
        assert all(float(chunk.attributes["priority"]) == 2.0 for chunk in results)

    def test_retrieve_supports_attributes_filter(self, store_with_attributes):
        results = store_with_attributes.retrieve(
            "alpha",
            top_k=5,
            attributes_filter="tenant = 'docs' AND priority >= 2",
        )

        assert len(results) > 0
        assert all(chunk.attributes["tenant"] == "docs" for chunk in results)
        assert all(float(chunk.attributes["priority"]) >= 2.0 for chunk in results)

    def test_retrieve_supports_attributes_filter_ast(self, store_with_attributes):
        results = store_with_attributes.retrieve(
            "alpha",
            top_k=5,
            attributes_filter={
                "type": "and",
                "filters": [
                    {"type": "eq", "key": "tenant", "value": "docs"},
                    {"type": "in", "key": "priority", "value": [2, 3]},
                ],
            },
        )
        assert len(results) > 0
        assert all(chunk.attributes["tenant"] == "docs" for chunk in results)
        assert all(
            float(chunk.attributes["priority"]) in (2.0, 3.0) for chunk in results
        )

    def test_rejects_chunk_attributes(self, store_with_attributes):
        doc = MarkdownDocument(
            origin="chunk-attrs",
            content="hello",
            attributes={"tenant": "docs", "priority": 1},
        )
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text="hello",
                token_count=5,
                attributes={"tenant": "docs"},
            )
        ]

        with pytest.raises(
            ValueError, match="OpenAIStore does not support per-chunk attributes"
        ):
            store_with_attributes.insert(doc)


def test_openai_store_create_rejects_vector_attributes_schema():
    with pytest.raises(ValueError, match="Vector attribute types are not supported"):
        OpenAIStore.create(attributes={"embedding25": Annotated[list[float], 25]})


def test_openai_store_create_accepts_mapping_attributes_with_mocked_client(monkeypatch):
    class FakeVectorStores:
        def create(self, **kwargs):
            return SimpleNamespace(id="vs_test")

    fake_client = SimpleNamespace(vector_stores=FakeVectorStores())

    def fake_openai_client(*, api_key=None, base_url=None):
        return fake_client

    monkeypatch.setattr("raghilda._openai_store.openai.Client", fake_openai_client)

    store = OpenAIStore.create(attributes={"tenant": str, "priority": int})
    assert store.attributes_schema == {"tenant": str, "priority": int}
    assert set(store.attributes_spec.keys()) == {"tenant", "priority"}


def test_openai_store_normalize_attributes_preserves_large_ints():
    normalized = _normalize_openai_attributes(
        {
            "tenant_id": 9007199254740992,
            "doc_id": 9007199254740993,
            "score": 0.75,
            "active": True,
            "label": "docs",
        }
    )
    assert normalized == {
        "tenant_id": 9007199254740992,
        "doc_id": 9007199254740993,
        "score": 0.75,
        "active": True,
        "label": "docs",
    }
    assert isinstance(normalized["tenant_id"], int)
    assert isinstance(normalized["doc_id"], int)
    assert isinstance(normalized["score"], float)


def test_openai_store_create_rejects_object_attributes_schema():
    with pytest.raises(ValueError, match="Object attribute types are not supported"):
        OpenAIStore.create(attributes={"details": {"source": str}})


def test_openai_store_create_rejects_optional_attributes_schema():
    with pytest.raises(
        ValueError, match="Optional attribute values are not supported for 'topic'"
    ):
        OpenAIStore.create(attributes={"topic": str | None})


def test_openai_store_create_rejects_defaulted_attributes_schema():
    with pytest.raises(
        ValueError, match="Optional attribute values are not supported for 'priority'"
    ):
        OpenAIStore.create(attributes={"tenant": str, "priority": (int, 0)})


def test_openai_store_create_rejects_invalid_attribute_names():
    with pytest.raises(ValueError, match="must match"):
        OpenAIStore.create(attributes={"tenant-id": str})


@pytest.mark.parametrize(
    "attribute_name",
    ["_raghilda_origin", "_raghilda_content_hash"],
)
def test_openai_store_rejects_internal_attribute_names(attribute_name):
    with pytest.raises(ValueError, match=attribute_name):
        OpenAIStore(
            client=SimpleNamespace(),
            store_id="vs_test",
            attributes={attribute_name: str},
        )


def test_openai_store_insert_updates_when_attributes_change_for_same_content():
    content = "hello world"
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    class FakePage:
        def __init__(self, data):
            self.data = data

        def has_next_page(self):
            return False

        def get_next_page(self):
            raise AssertionError("No next page expected")

    class FakeVectorStoreFiles:
        def __init__(self):
            self.deleted_ids = []
            self.upload_calls = []
            self.page = FakePage(
                [
                    SimpleNamespace(
                        id="file_old",
                        created_at=1,
                        filename="doc.md",
                        attributes={
                            "_raghilda_origin": "doc",
                            "_raghilda_content_hash": content_hash,
                            "tenant": "old",
                        },
                    )
                ]
            )

        def list(self, **kwargs):
            return self.page

        def delete(self, file_id, **kwargs):
            self.deleted_ids.append(file_id)
            return SimpleNamespace(id=file_id, deleted=True)

        def upload_and_poll(self, **kwargs):
            self.upload_calls.append(kwargs)
            return SimpleNamespace(id="file_new")

    class FakeFiles:
        def content(self, file_id):
            assert file_id == "file_old"
            return SimpleNamespace(content=content.encode("utf-8"))

    fake_vector_store_files = FakeVectorStoreFiles()
    fake_client = SimpleNamespace(
        vector_stores=SimpleNamespace(files=fake_vector_store_files),
        files=FakeFiles(),
    )
    store = OpenAIStore(
        client=fake_client,
        store_id="vs_test",
        attributes={"tenant": str},
    )

    result = store.insert(
        MarkdownDocument(
            origin="doc",
            content=content,
            attributes={"tenant": "new"},
        )
    )
    assert result.action == "replaced"
    assert result.replaced_document is not None
    assert result.replaced_document.content == content
    assert fake_vector_store_files.deleted_ids == ["file_old"]
    assert len(fake_vector_store_files.upload_calls) == 1
    assert fake_vector_store_files.upload_calls[0]["attributes"]["tenant"] == "new"


def test_openai_store_insert_rejects_too_many_user_attributes():
    class FakePage:
        def __init__(self, data):
            self.data = data

        def has_next_page(self):
            return False

        def get_next_page(self):
            raise AssertionError("No next page expected")

    class FakeVectorStoreFiles:
        def __init__(self):
            self.upload_calls = []
            self.page = FakePage([])

        def list(self, **kwargs):
            return self.page

        def delete(self, file_id, **kwargs):
            raise AssertionError("delete should not be called")

        def upload_and_poll(self, **kwargs):
            self.upload_calls.append(kwargs)
            return SimpleNamespace(id="file_new")

    fake_vector_store_files = FakeVectorStoreFiles()
    fake_client = SimpleNamespace(
        vector_stores=SimpleNamespace(files=fake_vector_store_files),
        files=SimpleNamespace(content=lambda file_id: None),
    )
    attributes_schema = {f"k{i}": str for i in range(15)}
    store = OpenAIStore(
        client=fake_client,
        store_id="vs_test",
        attributes=attributes_schema,
    )
    document_attributes = {f"k{i}": f"v{i}" for i in range(15)}

    with pytest.raises(ValueError, match="at most 14 user attributes"):
        store.insert(
            MarkdownDocument(
                origin="doc",
                content="hello world",
                attributes=document_attributes,
            )
        )
    assert fake_vector_store_files.upload_calls == []


def test_openai_store_insert_ignores_matching_filename_without_internal_origin():
    content = "hello world"

    class FakePage:
        def __init__(self, data):
            self.data = data

        def has_next_page(self):
            return False

        def get_next_page(self):
            raise AssertionError("No next page expected")

    class FakeVectorStoreFiles:
        def __init__(self):
            self.deleted_ids = []
            self.upload_calls = []
            self.page = FakePage(
                [
                    SimpleNamespace(
                        id="file_existing",
                        created_at=1,
                        filename="doc.md",
                        attributes={
                            "tenant": "old",
                        },
                    )
                ]
            )

        def list(self, **kwargs):
            return self.page

        def delete(self, file_id, **kwargs):
            self.deleted_ids.append(file_id)
            return SimpleNamespace(id=file_id, deleted=True)

        def upload_and_poll(self, **kwargs):
            self.upload_calls.append(kwargs)
            return SimpleNamespace(id="file_new")

    fake_vector_store_files = FakeVectorStoreFiles()
    fake_client = SimpleNamespace(
        vector_stores=SimpleNamespace(files=fake_vector_store_files),
        files=SimpleNamespace(content=lambda file_id: SimpleNamespace(content=b"")),
    )
    store = OpenAIStore(
        client=fake_client,
        store_id="vs_test",
        attributes={"tenant": str},
    )

    result = store.insert(
        MarkdownDocument(
            origin="doc",
            content=content,
            attributes={"tenant": "new"},
        )
    )
    assert result.action == "inserted"
    assert result.replaced_document is None
    assert fake_vector_store_files.deleted_ids == []
    assert len(fake_vector_store_files.upload_calls) == 1
    assert fake_vector_store_files.upload_calls[0]["attributes"]["tenant"] == "new"


def test_openai_store_insert_ignores_unmanaged_matching_filename():
    content = "hello world"

    class FakePage:
        def __init__(self, data):
            self.data = data

        def has_next_page(self):
            return False

        def get_next_page(self):
            raise AssertionError("No next page expected")

    class FakeVectorStoreFiles:
        def __init__(self):
            self.deleted_ids = []
            self.upload_calls = []
            self.page = FakePage(
                [
                    SimpleNamespace(
                        id="file_unmanaged",
                        created_at=1,
                        filename="doc.md",
                        attributes={"source": "external"},
                    )
                ]
            )

        def list(self, **kwargs):
            return self.page

        def delete(self, file_id, **kwargs):
            self.deleted_ids.append(file_id)
            return SimpleNamespace(id=file_id, deleted=True)

        def upload_and_poll(self, **kwargs):
            self.upload_calls.append(kwargs)
            return SimpleNamespace(id="file_new")

    fake_vector_store_files = FakeVectorStoreFiles()
    fake_client = SimpleNamespace(
        vector_stores=SimpleNamespace(files=fake_vector_store_files),
        files=SimpleNamespace(content=lambda file_id: SimpleNamespace(content=b"")),
    )
    store = OpenAIStore(
        client=fake_client,
        store_id="vs_test",
        attributes={"tenant": str},
    )

    result = store.insert(
        MarkdownDocument(
            origin="doc",
            content=content,
            attributes={"tenant": "new"},
        )
    )

    assert result.action == "inserted"
    assert result.replaced_document is None
    assert fake_vector_store_files.deleted_ids == []
    assert len(fake_vector_store_files.upload_calls) == 1


def test_openai_store_insert_keeps_existing_file_when_upload_fails():
    content = "hello world"
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    class FakePage:
        def __init__(self, data):
            self.data = data

        def has_next_page(self):
            return False

        def get_next_page(self):
            raise AssertionError("No next page expected")

    class FakeVectorStoreFiles:
        def __init__(self):
            self.deleted_ids = []
            self.page = FakePage(
                [
                    SimpleNamespace(
                        id="file_old",
                        created_at=1,
                        filename="doc.md",
                        attributes={
                            "_raghilda_origin": "doc",
                            "_raghilda_content_hash": content_hash,
                            "tenant": "old",
                        },
                    )
                ]
            )

        def list(self, **kwargs):
            return self.page

        def delete(self, file_id, **kwargs):
            self.deleted_ids.append(file_id)
            return SimpleNamespace(id=file_id, deleted=True)

        def upload_and_poll(self, **kwargs):
            raise RuntimeError("upload failed")

    class FakeFiles:
        def content(self, file_id):
            assert file_id == "file_old"
            return SimpleNamespace(content=content.encode("utf-8"))

    fake_vector_store_files = FakeVectorStoreFiles()
    fake_client = SimpleNamespace(
        vector_stores=SimpleNamespace(files=fake_vector_store_files),
        files=FakeFiles(),
    )
    store = OpenAIStore(
        client=fake_client,
        store_id="vs_test",
        attributes={"tenant": str},
    )

    with pytest.raises(RuntimeError, match="upload failed"):
        store.insert(
            MarkdownDocument(
                origin="doc",
                content="new content",
                attributes={"tenant": "new"},
            )
        )
    assert fake_vector_store_files.deleted_ids == []


def test_openai_store_connect_restores_metadata_schema_with_mocked_client(
    monkeypatch,
):
    schema_json = json.dumps(
        {
            "tenant": {"type": "str", "nullable": False, "required": True},
            "priority": {"type": "int", "nullable": False, "required": True},
        }
    )

    class FakeVectorStores:
        def retrieve(self, *, vector_store_id):
            return SimpleNamespace(
                id=vector_store_id,
                metadata={"raghilda_attributes_schema_json": schema_json},
            )

    fake_client = SimpleNamespace(vector_stores=FakeVectorStores())

    def fake_openai_client(*, api_key=None, base_url=None):
        return fake_client

    monkeypatch.setattr("raghilda._openai_store.openai.Client", fake_openai_client)

    store = OpenAIStore.connect(store_id="vs_test")
    assert store.attributes_schema == {"tenant": str, "priority": int}


def test_openai_store_connect_rejects_internal_attribute_names_from_metadata(
    monkeypatch,
):
    schema_json = json.dumps(
        {
            "_raghilda_origin": {"type": "str", "nullable": False, "required": True},
        }
    )

    class FakeVectorStores:
        def retrieve(self, *, vector_store_id):
            return SimpleNamespace(
                id=vector_store_id,
                metadata={"raghilda_attributes_schema_json": schema_json},
            )

    fake_client = SimpleNamespace(vector_stores=FakeVectorStores())

    def fake_openai_client(*, api_key=None, base_url=None):
        return fake_client

    monkeypatch.setattr("raghilda._openai_store.openai.Client", fake_openai_client)

    with pytest.raises(ValueError, match="_raghilda_origin"):
        OpenAIStore.connect(store_id="vs_test")


def test_openai_store_connect_rejects_attributes_argument(monkeypatch):
    class FakeVectorStores:
        def retrieve(self, *, vector_store_id):
            return SimpleNamespace(id=vector_store_id, metadata={})

    fake_client = SimpleNamespace(vector_stores=FakeVectorStores())

    def fake_openai_client(*, api_key=None, base_url=None):
        return fake_client

    monkeypatch.setattr("raghilda._openai_store.openai.Client", fake_openai_client)

    with pytest.raises(TypeError, match="unexpected keyword argument 'attributes'"):
        cast(Any, OpenAIStore.connect)(
            store_id="vs_test",
            attributes={"tenant": str},
        )


def test_openai_store_insert_updates_when_snapshot_download_forbidden():
    old_content = "hello world"
    new_content = "hello world updated"
    old_hash = hashlib.sha256(old_content.encode("utf-8")).hexdigest()

    class FakePage:
        def __init__(self, data):
            self.data = data

        def has_next_page(self):
            return False

        def get_next_page(self):
            raise AssertionError("No next page expected")

    class FakeVectorStoreFiles:
        def __init__(self):
            self.deleted_ids = []
            self.upload_calls = []
            self.page = FakePage(
                [
                    SimpleNamespace(
                        id="file_old",
                        created_at=1,
                        filename="doc.md",
                        attributes={
                            "_raghilda_origin": "doc",
                            "_raghilda_content_hash": old_hash,
                            "tenant": "old",
                        },
                    )
                ]
            )

        def list(self, **kwargs):
            return self.page

        def delete(self, file_id, **kwargs):
            self.deleted_ids.append(file_id)
            return SimpleNamespace(id=file_id, deleted=True)

        def upload_and_poll(self, **kwargs):
            self.upload_calls.append(kwargs)
            return SimpleNamespace(id="file_new")

    class FakeFiles:
        def content(self, file_id):
            request = httpx.Request(
                "GET", f"https://api.openai.com/v1/files/{file_id}/content"
            )
            response = httpx.Response(400, request=request)
            raise openai.BadRequestError(
                "Not allowed to download files of purpose: assistants",
                response=response,
                body=None,
            )

    fake_vector_store_files = FakeVectorStoreFiles()
    fake_client = SimpleNamespace(
        vector_stores=SimpleNamespace(files=fake_vector_store_files),
        files=FakeFiles(),
    )
    store = OpenAIStore(
        client=fake_client,
        store_id="vs_test",
        attributes={"tenant": str},
    )

    result = store.insert(
        MarkdownDocument(
            origin="doc",
            content=new_content,
            attributes={"tenant": "new"},
        )
    )

    assert result.action == "replaced"
    assert result.replaced_document is None
    assert fake_vector_store_files.deleted_ids == ["file_old"]
    assert len(fake_vector_store_files.upload_calls) == 1


def _get_markdown_chunk(doc, start, end):
    return MarkdownChunk(
        start_index=start,
        end_index=end,
        text=doc.content[start:end],
        token_count=len(doc.content[start:end]),
    )


def test_ingest():
    _skip_if_unset("OPENAI_API_KEY")
    from raghilda.chunker import MarkdownChunker
    from raghilda.read import read_as_markdown

    links = find_links("https://r4ds.hadley.nz/base-R.html", validate=True)
    links = links[:3]

    store = DuckDBStore.create(
        location=":memory:",
        embed=EmbeddingOpenAI(),
        overwrite=True,
        name="ingest_db",
        title="Ingest Test DuckDB Store",
    )

    # Use smaller chunks to avoid exceeding embedding token limits on code-heavy pages
    chunker = MarkdownChunker(chunk_size=800)

    def prepare(uri: str):
        return chunker.chunk_document(read_as_markdown(uri))

    store.ingest(links, prepare=prepare)


def test_ingest_with_generator():
    """Test that ingest works with a generator (iterable without __len__)."""
    from raghilda.chunker import MarkdownChunker

    store = DuckDBStore.create(
        location=":memory:",
        embed=None,
        overwrite=True,
        name="ingest_generator_db",
    )

    chunker = MarkdownChunker()

    def make_docs():
        for i in range(3):
            doc = MarkdownDocument(origin=f"test_{i}", content=f"Document number {i}")
            yield doc

    # Use identity prepare since we're yielding MarkdownDocuments that need chunking
    store.ingest(
        make_docs(), prepare=lambda doc: chunker.chunk_document(doc), progress=False
    )
    assert store.size() == 3


def test_ingest_with_custom_prepare():
    """Test that ingest works with a custom prepare function."""
    store = DuckDBStore.create(
        location=":memory:",
        embed=None,
        overwrite=True,
        name="ingest_prepare_db",
    )

    def custom_prepare(item: dict) -> MarkdownDocument:
        doc = MarkdownDocument(origin=item["id"], content=item["text"])
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=len(item["text"]),
                text=item["text"],
                token_count=len(item["text"]),
            )
        ]
        return doc

    items = [
        {"id": "doc1", "text": "First document content"},
        {"id": "doc2", "text": "Second document content"},
    ]

    store.ingest(items, prepare=custom_prepare, progress=False)
    assert store.size() == 2


def test_ingest_lazy_evaluation():
    """Test that ingest consumes the iterator lazily, not all at once."""
    import threading
    import time

    store = DuckDBStore.create(
        location=":memory:",
        embed=None,
        overwrite=True,
        name="ingest_lazy_db",
    )

    consumed_count = 0
    max_concurrent_consumed = 0
    inserted_count = 0
    lock = threading.Lock()

    def tracking_generator():
        nonlocal consumed_count, max_concurrent_consumed
        for i in range(20):
            with lock:
                consumed_count += 1
                # Track max items consumed while inserts are still pending
                pending = consumed_count - inserted_count
                if pending > max_concurrent_consumed:
                    max_concurrent_consumed = pending
            yield {"id": f"doc_{i}", "text": f"Document content {i}"}

    def slow_prepare(item: dict) -> MarkdownDocument:
        nonlocal inserted_count
        # Simulate slow processing
        time.sleep(0.05)
        doc = MarkdownDocument(origin=item["id"], content=item["text"])
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=len(item["text"]),
                text=item["text"],
                token_count=len(item["text"]),
            )
        ]
        with lock:
            inserted_count += 1
        return doc

    # Use only 2 workers to make the test more deterministic
    store.ingest(
        tracking_generator(), prepare=slow_prepare, num_workers=2, progress=False
    )

    assert store.size() == 20
    # With lazy evaluation, max concurrent should be bounded by num_workers (plus some buffer)
    # If it consumed eagerly, all 20 would be consumed before any inserts complete
    assert max_concurrent_consumed <= 10, (
        f"Expected lazy consumption but {max_concurrent_consumed} items were consumed "
        f"concurrently, suggesting eager evaluation"
    )


def test_connect(tmp_path):
    _skip_if_unset("OPENAI_API_KEY")
    db_path = tmp_path / "test.db"

    # Create a store with embeddings
    store = DuckDBStore.create(
        location=str(db_path),
        embed=EmbeddingOpenAI(model="text-embedding-3-small"),
        name="connect_test",
        title="Connect Test Store",
    )
    doc = MarkdownDocument(origin="test", content="hello world")
    doc.chunks = [_get_markdown_chunk(doc, start=0, end=5)]
    store.insert(doc)
    store.build_index()
    store.con.close()

    # Connect to existing store
    store2 = DuckDBStore.connect(str(db_path))
    assert store2.metadata.name == "connect_test"
    assert store2.metadata.title == "Connect Test Store"
    assert store2.size() == 1

    # Embedding provider should be restored
    assert store2.metadata.embed is not None
    assert isinstance(store2.metadata.embed, EmbeddingOpenAI)
    assert store2.metadata.embed.model == "text-embedding-3-small"

    # Retrieve should work (uses both BM25 and VSS)
    results = store2.retrieve("hello", top_k=1)
    assert len(results) >= 1
    assert results[0].text == "hello"
