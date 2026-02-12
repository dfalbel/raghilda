from typing import Annotated

import pytest

from raghilda._metadata import (
    MetadataFloatVectorType,
    metadata_schema_from_json_dict,
    metadata_schema_to_json_dict,
    normalize_metadata_schema,
)


def test_normalize_metadata_schema_accepts_class_annotations():
    class MetadataSpec:
        tenant: str
        priority: int
        is_public: bool

    schema = normalize_metadata_schema(
        MetadataSpec,
        reserved_columns=set(),
    )
    assert schema == {
        "tenant": str,
        "priority": int,
        "is_public": bool,
    }


def test_metadata_schema_roundtrip_with_vector_annotation():
    schema = normalize_metadata_schema(
        {"embedding25": Annotated[list[float], 25]},
        reserved_columns=set(),
        allow_vector_types=True,
    )
    assert schema["embedding25"] == MetadataFloatVectorType(dimension=25)

    encoded = metadata_schema_to_json_dict(schema)
    assert encoded == {"embedding25": "float_vector[25]"}

    decoded = metadata_schema_from_json_dict(encoded, allow_vector_types=True)
    assert decoded == schema


def test_normalize_metadata_schema_rejects_vector_for_backends_without_support():
    with pytest.raises(ValueError, match="Vector metadata types are not supported"):
        normalize_metadata_schema(
            {"embedding25": Annotated[list[float], 25]},
            reserved_columns=set(),
            allow_vector_types=False,
        )
