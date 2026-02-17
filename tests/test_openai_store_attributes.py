import json
from types import SimpleNamespace
from typing import Annotated

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
    def __init__(self, *, retrieve_metadata=None):
        self.files = _FakeVectorStoreFilesAPI()
        self.create_calls = []
        self.search_calls = []
        self.retrieve_calls = []
        self.retrieve_metadata = dict(retrieve_metadata or {})

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return SimpleNamespace(id="vs_123")

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
            metadata=self.retrieve_metadata,
        )


class _FakeOpenAIClient:
    def __init__(self, *, retrieve_metadata=None):
        self.vector_stores = _FakeVectorStoresAPI(retrieve_metadata=retrieve_metadata)


def test_openai_store_create_accepts_class_attributes_schema(monkeypatch):
    class AttributesSpec:
        tenant: str
        priority: int

    client = _FakeOpenAIClient()
    monkeypatch.setattr("raghilda._openai_store.openai.Client", lambda **kwargs: client)

    store = OpenAIStore.create(attributes=AttributesSpec, name="my-store")
    assert store.attributes_schema == {"tenant": str, "priority": int}

    assert len(client.vector_stores.create_calls) == 1
    schema_json = client.vector_stores.create_calls[0]["metadata"][
        "raghilda_attributes_schema_json"
    ]
    assert json.loads(schema_json) == {
        "tenant": {"type": "str", "nullable": False, "required": True},
        "priority": {"type": "int", "nullable": False, "required": True},
    }


def test_openai_store_create_rejects_vector_attributes_schema():
    with pytest.raises(ValueError, match="Vector attribute types are not supported"):
        OpenAIStore.create(attributes={"embedding25": Annotated[list[float], 25]})


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


def test_openai_store_insert_uses_document_attributes():
    client = _FakeOpenAIClient()
    store = OpenAIStore(
        client=client,
        store_id="vs_123",
        attributes={"tenant": str, "priority": int},
    )

    doc = MarkdownDocument(
        origin="openai.md",
        content="hello world",
        attributes={"tenant": "docs", "priority": 1},
    )
    store.insert(doc)

    assert len(client.vector_stores.files.upload_calls) == 1
    call = client.vector_stores.files.upload_calls[0]
    assert call["vector_store_id"] == "vs_123"
    assert call["attributes"] == {"tenant": "docs", "priority": 1.0}


def test_openai_store_retrieve_supports_attributes_filter():
    client = _FakeOpenAIClient()
    store = OpenAIStore(
        client=client,
        store_id="vs_123",
        attributes={"tenant": str, "priority": int},
    )

    chunks = store.retrieve(
        "hello",
        top_k=3,
        attributes_filter="tenant = 'docs' AND priority >= 2",
    )
    assert len(chunks) == 1
    assert chunks[0].attributes == {"tenant": "docs", "priority": 2.0}

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
        attributes_filter={
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


def test_openai_store_rejects_chunk_attributes():
    client = _FakeOpenAIClient()
    store = OpenAIStore(
        client=client,
        store_id="vs_123",
        attributes={"tenant": str},
    )

    doc = MarkdownDocument(
        origin="openai.md",
        content="hello",
        attributes={"tenant": "docs"},
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
        store.insert(doc)
