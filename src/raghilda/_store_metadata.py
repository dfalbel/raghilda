from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from ._attributes import AttributeSpec, AttributeType

if TYPE_CHECKING:
    from .embedding import EmbeddingProvider


class AttributesStoreMetadata(Protocol):
    """Shared metadata shape for stores that expose typed attributes."""

    name: str
    title: str
    attributes: dict[str, AttributeSpec]

    @property
    def attributes_spec(self) -> dict[str, AttributeSpec]: ...

    @property
    def attributes_schema(self) -> dict[str, AttributeType]: ...


class EmbeddedAttributesStoreMetadata(AttributesStoreMetadata, Protocol):
    """Metadata shape for stores with an embedding provider in store metadata."""

    embed: EmbeddingProvider | None


def attributes_schema_from_spec(
    attributes: dict[str, AttributeSpec],
) -> dict[str, AttributeType]:
    return {key: spec.attribute_type for key, spec in attributes.items()}
