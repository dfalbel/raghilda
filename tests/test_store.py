import os
import socket
import pytest
from raghilda.store import DuckDBStore, OpenAIStore
from raghilda.scrape import find_links
from raghilda.document import MarkdownDocument
from raghilda.chunk import MarkdownChunk, RetrievedChunk
from raghilda._store import (
    RetrievedDuckDBMarkdownChunk,
    IndexType,
)  # internal implementation
from raghilda.embedding import EmbeddingOpenAI


def _can_reach_openai(timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection(("api.openai.com", 443), timeout=timeout):
            return True
    except OSError:
        return False


def _require_openai_integration() -> None:
    if "OPENAI_API_KEY" not in os.environ:
        pytest.skip("OPENAI_API_KEY not set in environment variables")
    if not _can_reach_openai():
        pytest.skip("OpenAI API is not reachable from this environment")


class TestDuckDBStore:
    @pytest.fixture
    def embed(self, request):
        try:
            value = request.param
            if isinstance(value, EmbeddingOpenAI):
                _require_openai_integration()
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
        store_with_docs.build_index(IndexType.BM25)
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
        store.build_index(IndexType.BM25)

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

    def test_create_store_with_metadata_schema(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            name="metadata_schema_db",
            title="Metadata Schema Store",
            metadata={
                "tenant": str,
                "priority": int,
                "is_public": bool,
            },
        )

        assert store.metadata.metadata_schema == {
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

    def test_insert_and_retrieve_with_metadata_filter(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            metadata={
                "tenant": str,
                "priority": int,
                "is_public": bool,
            },
        )

        doc = MarkdownDocument(
            origin="metadata-test",
            content="alpha beta gamma",
            metadata={"tenant": "docs", "priority": 1},
        )
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text="alpha",
                token_count=5,
                metadata={"priority": 5, "is_public": False},
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
        store.build_index(IndexType.BM25)

        private_results = store.retrieve(
            "alpha",
            top_k=10,
            deoverlap=False,
            metadata_filter="tenant = 'docs' AND priority >= 5",
        )
        assert len(private_results) == 1
        assert private_results[0].text.strip() == "alpha"
        assert private_results[0].metadata == {
            "tenant": "docs",
            "priority": 5,
            "is_public": False,
        }

        public_results = store.retrieve(
            "beta",
            top_k=10,
            deoverlap=False,
            metadata_filter="tenant = 'docs' AND is_public = NULL AND priority = 1",
        )
        assert len(public_results) == 1
        assert public_results[0].text.strip() == "beta"
        assert public_results[0].metadata == {
            "tenant": "docs",
            "priority": 1,
            "is_public": None,
        }

        dict_results = store.retrieve(
            "alpha",
            top_k=10,
            deoverlap=False,
            metadata_filter={
                "type": "and",
                "filters": [
                    {"type": "eq", "key": "tenant", "value": "docs"},
                    {"type": "in", "key": "priority", "value": [5, 10]},
                ],
            },
        )
        assert len(dict_results) == 1
        assert dict_results[0].text.strip() == "alpha"

    def test_insert_metadata_without_declared_schema_fails(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
        )

        doc = MarkdownDocument(
            origin="metadata-fail",
            content="hello",
            metadata={"tenant": "docs"},
        )
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text="hello",
                token_count=5,
            )
        ]

        with pytest.raises(ValueError, match="Unknown metadata key 'tenant'"):
            store.insert(doc)

    def test_insert_unknown_chunk_metadata_key_fails(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            metadata={"tenant": str},
        )

        doc = MarkdownDocument(
            origin="unknown-key-fail",
            content="hello",
            metadata={"tenant": "docs"},
        )
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text="hello",
                token_count=5,
                metadata={"unknown": "x"},
            )
        ]

        with pytest.raises(ValueError, match="Unknown metadata key 'unknown'"):
            store.insert(doc)

    def test_connect_restores_metadata_schema(self, tmp_path):
        db_path = tmp_path / "metadata-connect.db"
        store = DuckDBStore.create(
            location=str(db_path),
            embed=None,
            overwrite=True,
            metadata={"tenant": str, "priority": int},
        )
        store.con.close()

        store2 = DuckDBStore.connect(str(db_path))
        assert store2.metadata.metadata_schema == {
            "tenant": str,
            "priority": int,
        }


class TestOpenAIStore:
    @pytest.fixture(autouse=True)
    def setup(self):
        _require_openai_integration()

    @pytest.fixture
    def store(self):
        store = OpenAIStore.create()
        return store

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
        for _ in range(20):
            store_with_docs.insert(
                MarkdownDocument(
                    origin="test", content="hello world world world world world"
                )
            )
        results = store_with_docs.retrieve("world", top_k=3)
        assert len(results) > 0
        for chunk in results:
            assert isinstance(chunk, RetrievedChunk)
            assert chunk.text is not None


def _get_markdown_chunk(doc, start, end):
    return MarkdownChunk(
        start_index=start,
        end_index=end,
        text=doc.content[start:end],
        token_count=len(doc.content[start:end]),
    )


def test_ingest():
    _require_openai_integration()
    links = find_links("https://r4ds.hadley.nz/base-R.html", validate=True)

    store = DuckDBStore.create(
        location=":memory:",
        embed=EmbeddingOpenAI(),
        overwrite=True,
        name="ingest_db",
        title="Ingest Test DuckDB Store",
    )

    store.ingest(links)


def test_connect(tmp_path):
    _require_openai_integration()
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
