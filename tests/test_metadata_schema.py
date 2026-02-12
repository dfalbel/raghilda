from typing import Annotated

import pytest

from raghilda._metadata import (
    MetadataFloatVectorType,
    attributes_schema_from_json_dict,
    attributes_schema_to_json_dict,
    normalize_attributes_schema,
)


def test_normalize_attributes_schema_accepts_class_annotations():
    class AttributesSpec:
        tenant: str
        priority: int
        is_public: bool

    schema = normalize_attributes_schema(
        AttributesSpec,
        reserved_columns=set(),
    )
    assert schema == {
        "tenant": str,
        "priority": int,
        "is_public": bool,
    }


def test_attributes_schema_roundtrip_with_vector_annotation():
    schema = normalize_attributes_schema(
        {"embedding25": Annotated[list[float], 25]},
        reserved_columns=set(),
        allow_vector_types=True,
    )
    assert schema["embedding25"] == MetadataFloatVectorType(dimension=25)

    encoded = attributes_schema_to_json_dict(schema)
    assert encoded == {"embedding25": "float_vector[25]"}

    decoded = attributes_schema_from_json_dict(encoded, allow_vector_types=True)
    assert decoded == schema


def test_normalize_attributes_schema_rejects_vector_for_backends_without_support():
    with pytest.raises(ValueError, match="Vector attribute types are not supported"):
        normalize_attributes_schema(
            {"embedding25": Annotated[list[float], 25]},
            reserved_columns=set(),
            allow_vector_types=False,
        )
