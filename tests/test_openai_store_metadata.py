from types import SimpleNamespace

import pytest

from raghilda._openai_store import OpenAIStore
from raghilda.document import MarkdownDocument
from raghilda.chunk import MarkdownChunk


class _FakeVectorStoreFilesAPI:
    def __init__(self):
        self.upload_calls = []

    def upload_and_poll(self, **kwargs):
        self.upload_calls.append(kwargs)
        return SimpleNamespace(id="vsf_123")


class _FakeVectorStoresAPI:
    def __init__(self):
        self.files = _FakeVectorStoreFilesAPI()
        self.search_calls = []
        self.retrieve_calls = []

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    content=[SimpleNamespace(text="hello from openai store")],
                    score=0.99,
                    attributes={"tenant": "docs", "priority": 2.0},
                )
            ]
        )

    def retrieve(self, **kwargs):
        self.retrieve_calls.append(kwargs)
        return SimpleNamespace(
            file_counts=SimpleNamespace(total=1),
            metadata={},
        )


class _FakeOpenAIClient:
    def __init__(self):
        self.vector_stores = _FakeVectorStoresAPI()


def test_openai_store_insert_uses_document_metadata_as_attributes():
    client = _FakeOpenAIClient()
    store = OpenAIStore(
        client=client,
        store_id="vs_123",
        metadata={"tenant": str, "priority": int},
    )

    doc = MarkdownDocument(
        origin="openai.md",
        content="hello world",
        metadata={"tenant": "docs"},
    )
    store.insert(doc)

    assert len(client.vector_stores.files.upload_calls) == 1
    call = client.vector_stores.files.upload_calls[0]
    assert call["vector_store_id"] == "vs_123"
    assert call["attributes"] == {"tenant": "docs"}


def test_openai_store_retrieve_supports_metadata_filter():
    client = _FakeOpenAIClient()
    store = OpenAIStore(
        client=client,
        store_id="vs_123",
        metadata={"tenant": str, "priority": int},
    )

    chunks = store.retrieve(
        "hello",
        top_k=3,
        metadata_filter="tenant = 'docs' AND priority >= 2",
    )
    assert len(chunks) == 1
    assert chunks[0].metadata == {"tenant": "docs", "priority": 2.0}

    assert len(client.vector_stores.search_calls) == 1
    filters = client.vector_stores.search_calls[0]["filters"]
    assert filters == {
        "type": "and",
        "filters": [
            {"type": "eq", "key": "tenant", "value": "docs"},
            {"type": "gte", "key": "priority", "value": 2.0},
        ],
    }

    chunks = store.retrieve(
        "hello",
        top_k=3,
        metadata_filter={
            "type": "and",
            "filters": [
                {"type": "eq", "key": "tenant", "value": "docs"},
                {"type": "in", "key": "priority", "value": [1, 2, 3]},
            ],
        },
    )
    assert len(chunks) == 1

    filters = client.vector_stores.search_calls[-1]["filters"]
    assert filters == {
        "type": "and",
        "filters": [
            {"type": "eq", "key": "tenant", "value": "docs"},
            {"type": "in", "key": "priority", "value": [1.0, 2.0, 3.0]},
        ],
    }


def test_openai_store_rejects_chunk_metadata():
    client = _FakeOpenAIClient()
    store = OpenAIStore(
        client=client,
        store_id="vs_123",
        metadata={"tenant": str},
    )

    doc = MarkdownDocument(
        origin="openai.md",
        content="hello",
        metadata={"tenant": "docs"},
    )
    doc.chunks = [
        MarkdownChunk(
            start_index=0,
            end_index=5,
            text="hello",
            token_count=5,
            metadata={"tenant": "docs"},
        )
    ]

    with pytest.raises(
        ValueError, match="OpenAIStore does not support per-chunk metadata"
    ):
        store.insert(doc)
