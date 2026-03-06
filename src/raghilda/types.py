"""Protocol types for raghilda.

This module exports the protocol types for type checking compatibility
with chunks, documents, and chunkers.
"""

from ._types import (
    ChunkLike,
    DocumentLike,
    ChunkerLike,
    IntoChunk,
    IntoDocument,
    TokenCountChunkLike,
)

__all__ = [
    "ChunkLike",
    "TokenCountChunkLike",
    "DocumentLike",
    "ChunkerLike",
    "IntoChunk",
    "IntoDocument",
]
