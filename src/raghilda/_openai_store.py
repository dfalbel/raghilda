import openai
import json
from ._store import BaseStore
from .chunk import MarkdownChunk, RetrievedChunk, Metric
from .document import Document, MarkdownDocument
from typing import Any, Mapping, Optional, Sequence
from dataclasses import dataclass
from ._metadata import (
    MetadataFilter,
    MetadataType,
    MetadataValue,
    compile_filter_to_openai_filters,
    metadata_schema_from_json_dict,
    metadata_schema_from_sql_types,
    metadata_schema_to_json_dict,
    merge_metadata_values,
    normalize_metadata_schema,
)

_METADATA_SCHEMA_KEY = "raghilda_metadata_schema_json"
_LEGACY_METADATA_COLUMNS_KEY = "raghilda_metadata_columns_json"


@dataclass
class OpenAIMarkdownChunk(MarkdownChunk):
    """MarkdownChunk for OpenAI store - uses character count as token count"""

    def __init__(
        self,
        text: str,
        start_index: int = 0,
        end_index: Optional[int] = None,
        context=None,
        token_count=None,
        metadata=None,
    ):
        # Compute token_count if not provided (use character count)
        if token_count is None:
            token_count = len(text)

        # Compute end_index if not provided
        if end_index is None:
            end_index = len(text)

        # Initialize parent class
        super().__init__(
            text=text,
            start_index=start_index,
            end_index=end_index,
            token_count=token_count,
            context=context,
            metadata=metadata,
        )


@dataclass
class RetrievedOpenAIMarkdownChunk(OpenAIMarkdownChunk, RetrievedChunk):
    """OpenAIMarkdownChunk with retrieval metrics"""

    def __init__(
        self,
        text: str,
        start_index: int = 0,
        end_index: Optional[int] = None,
        context=None,
        token_count=None,
        metrics=None,
        metadata=None,
    ):
        # Initialize OpenAIMarkdownChunk
        super().__init__(
            text=text,
            start_index=start_index,
            end_index=end_index,
            context=context,
            token_count=token_count,
            metadata=metadata,
        )

        # Initialize metrics
        if metrics is None:
            metrics = []
        self.metrics = metrics


class OpenAIStore(BaseStore):
    """A vector store backed by OpenAI's Vector Store API.

    OpenAIStore uses OpenAI's hosted vector storage service for document
    storage and retrieval. Documents are uploaded as files and automatically
    chunked and embedded by OpenAI.

    Examples
    --------
    ```{python}
    #| eval: false
    from raghilda.store import OpenAIStore

    # Create a new store
    store = OpenAIStore.create(name="my-store")

    # Or connect to an existing store
    store = OpenAIStore.connect(store_id="vs_abc123")

    # Insert documents
    from raghilda.document import MarkdownDocument
    doc = MarkdownDocument(content="# Hello\\nWorld", origin="example.md")
    store.insert(doc)

    # Retrieve similar chunks
    chunks = store.retrieve("greeting", top_k=5)
    ```
    """

    @staticmethod
    def create(
        base_url: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
        *,
        metadata: Optional[Mapping[str, type[Any]]] = None,
        vector_store_metadata: Optional[Mapping[str, str]] = None,
        **kwargs,
    ):
        """Create a new OpenAI vector store.

        Parameters
        ----------
        base_url
            Base URL for the OpenAI API.
        api_key
            OpenAI API key. If None, uses the OPENAI_API_KEY environment variable.
        metadata
            Optional schema for user-defined metadata columns.
        vector_store_metadata
            Additional metadata to attach to the OpenAI vector store resource.
        **kwargs
            Additional arguments passed to the vector store creation
            (e.g., name, expires_after).

        Returns
        -------
        OpenAIStore
            A newly created store instance.
        """
        metadata_schema = normalize_metadata_schema(
            metadata=metadata,
            reserved_columns=set(),
        )

        client = openai.Client(api_key=api_key, base_url=base_url)
        api_vector_store_metadata = dict(vector_store_metadata or {})
        api_vector_store_metadata[_METADATA_SCHEMA_KEY] = json.dumps(
            metadata_schema_to_json_dict(metadata_schema)
        )
        kwargs["metadata"] = api_vector_store_metadata

        vector_store = client.vector_stores.create(**kwargs)
        return OpenAIStore(
            client,
            vector_store.id,
            metadata=metadata_schema,
        )

    @staticmethod
    def connect(
        store_id: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
        *,
        metadata: Optional[Mapping[str, type[Any]]] = None,
    ):
        """Connect to an existing OpenAI vector store.

        Parameters
        ----------
        store_id
            The ID of the vector store to connect to (e.g., "vs_abc123").
        base_url
            Base URL for the OpenAI API.
        api_key
            OpenAI API key. If None, uses the OPENAI_API_KEY environment variable.
        metadata
            Optional schema for user-defined metadata columns. If omitted,
            schema is loaded from the vector store metadata when available.

        Returns
        -------
        OpenAIStore
            A connected store instance.
        """
        client = openai.Client(api_key=api_key, base_url=base_url)
        vector_store = client.vector_stores.retrieve(vector_store_id=store_id)
        store_metadata = getattr(vector_store, "metadata", None) or {}

        resolved_metadata = normalize_metadata_schema(
            metadata=metadata,
            reserved_columns=set(),
        )
        if not resolved_metadata and store_metadata.get(_METADATA_SCHEMA_KEY):
            resolved_metadata = metadata_schema_from_json_dict(
                json.loads(store_metadata[_METADATA_SCHEMA_KEY])
            )
        elif not resolved_metadata and store_metadata.get(_LEGACY_METADATA_COLUMNS_KEY):
            resolved_metadata = metadata_schema_from_sql_types(
                json.loads(store_metadata[_LEGACY_METADATA_COLUMNS_KEY])
            )

        return OpenAIStore(
            client,
            store_id,
            metadata=resolved_metadata,
        )

    def __init__(
        self,
        client: Any,
        store_id: str,
        *,
        metadata: Optional[Mapping[str, MetadataType]] = None,
    ):
        self.client = client
        self.store_id = store_id
        self.metadata_schema = dict(metadata or {})

    def insert(
        self,
        document: Document,
        *,
        metadata: Optional[Mapping[str, MetadataValue]] = None,
    ) -> None:
        # Upload the document content as a file to the vector store
        # create a temporary file, write the content to it, and upload it
        if not isinstance(document, MarkdownDocument):
            raise ValueError("Only MarkdownDocument is supported for OpenAIStore")

        if document.chunks is not None:
            for chunk in document.chunks:
                if chunk.metadata:
                    raise ValueError(
                        "OpenAIStore does not support per-chunk metadata; use document-level metadata."
                    )

        resolved_metadata = merge_metadata_values(
            metadata_schema=self.metadata_schema,
            sources=[document.metadata, metadata],
        )
        attributes = _normalize_openai_attributes(resolved_metadata)

        file = ((document.origin or "") + ".md", document.content.encode("utf-8"))
        if attributes:
            self.client.vector_stores.files.upload_and_poll(
                file=file,
                vector_store_id=self.store_id,
                attributes=attributes,
            )
        else:
            self.client.vector_stores.files.upload_and_poll(
                file=file,
                vector_store_id=self.store_id,
            )

    def retrieve(
        self,
        text: str,
        top_k: int,
        *,
        metadata_filter: Optional[MetadataFilter] = None,
        **kwargs,
    ) -> Sequence[RetrievedOpenAIMarkdownChunk]:
        if metadata_filter is not None:
            if "filters" in kwargs:
                raise ValueError("Use either metadata_filter or filters, not both.")
            kwargs["filters"] = compile_filter_to_openai_filters(
                metadata_filter,
                allowed_columns=set(self.metadata_schema),
            )

        results = self.client.vector_stores.search(
            vector_store_id=self.store_id,
            query=text,
            max_num_results=top_k,
            **kwargs,
        )

        chunks = []
        for item in results.data:
            chunk_text = "\n\n".join([x.text for x in item.content])
            metadata_values = {
                key: (item.attributes or {}).get(key) for key in self.metadata_schema
            }
            chunk = RetrievedOpenAIMarkdownChunk(
                text=chunk_text,
                metrics=[Metric(name="similarity", value=item.score)],
                metadata=metadata_values,
            )
            chunks.append(chunk)

        return chunks

    def size(self):
        return self.client.vector_stores.retrieve(
            vector_store_id=self.store_id
        ).file_counts.total


def _normalize_openai_attributes(
    metadata: Mapping[str, MetadataValue],
) -> dict[str, str | float | bool]:
    out: dict[str, str | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, bool):
            out[key] = value
        elif isinstance(value, str):
            out[key] = value
        elif isinstance(value, (int, float)):
            out[key] = float(value)
        else:
            raise ValueError(
                f"Unsupported OpenAI metadata type for '{key}': {type(value).__name__}"
            )
    return out
