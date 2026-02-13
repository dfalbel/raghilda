from __future__ import annotations

from dataclasses import dataclass
import re
import types
from typing import (
    Annotated,
    Any,
    Iterable,
    Mapping,
    Optional,
    Union,
    TypeAlias,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

AttributeScalar = str | int | float | bool
AttributeFilterValue = AttributeScalar | None
AttributeScalarType: TypeAlias = type[str] | type[int] | type[float] | type[bool]


@dataclass(frozen=True)
class AttributeFloatVectorType:
    dimension: int


@dataclass(frozen=True)
class AttributeStructType:
    fields: dict[str, "AttributeType"]


AttributeObjectValue: TypeAlias = dict[str, "AttributeValue"]
AttributeValue = AttributeScalar | list[float] | AttributeObjectValue | None
AttributeType: TypeAlias = (
    AttributeScalarType | AttributeFloatVectorType | AttributeStructType
)
AttributesSchemaSpec: TypeAlias = Mapping[str, Any] | type[Any]
AttributeFilter: TypeAlias = str | Mapping[str, Any]
_MISSING = object()


@dataclass(frozen=True)
class AttributeSpec:
    attribute_type: AttributeType
    nullable: bool
    required: bool
    default: AttributeValue = None


_ATTRIBUTE_SCALAR_TYPE_TO_NAME: dict[AttributeScalarType, str] = {
    str: "str",
    int: "int",
    float: "float",
    bool: "bool",
}
_ATTRIBUTE_NAME_TO_SCALAR_TYPE: dict[str, AttributeScalarType] = {
    value: key for key, value in _ATTRIBUTE_SCALAR_TYPE_TO_NAME.items()
}
_FLOAT_VECTOR_TYPE_PATTERN = re.compile(r"^float_vector\[(\d+)\]$")


def normalize_attributes_schema(
    attributes: Optional[AttributesSchemaSpec],
    *,
    reserved_columns: Iterable[str],
    allow_vector_types: bool = False,
    allow_struct_types: bool = True,
    allow_optional_values: bool = True,
) -> dict[str, AttributeType]:
    attributes_spec = normalize_attributes_spec(
        attributes=attributes,
        reserved_columns=reserved_columns,
        allow_vector_types=allow_vector_types,
        allow_struct_types=allow_struct_types,
        allow_optional_values=allow_optional_values,
    )
    return {key: spec.attribute_type for key, spec in attributes_spec.items()}


def normalize_attributes_spec(
    attributes: Optional[AttributesSchemaSpec],
    *,
    reserved_columns: Iterable[str],
    allow_vector_types: bool = False,
    allow_struct_types: bool = True,
    allow_optional_values: bool = True,
) -> dict[str, AttributeSpec]:
    schema_items = _attributes_schema_items(attributes)
    reserved = set(reserved_columns)
    spec: dict[str, AttributeSpec] = {}

    for key, item in schema_items.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Attribute column names must be non-empty strings")
        if key in reserved:
            raise ValueError(f"Attribute column '{key}' is reserved")

        attribute_type, nullable = _parse_attribute_type(
            key=key,
            annotation=item["annotation"],
            allow_vector_types=allow_vector_types,
            allow_struct_types=allow_struct_types,
        )
        has_default = item["has_default"]
        default_value = item["default"]

        if has_default:
            required = False
        elif nullable:
            required = False
            default_value = None
        else:
            required = True

        if not allow_optional_values and not required:
            raise ValueError(
                f"Optional attribute values are not supported for '{key}' in this backend"
            )

        if required:
            spec[key] = AttributeSpec(
                attribute_type=attribute_type,
                nullable=nullable,
                required=True,
            )
            continue

        if default_value is None and not nullable:
            raise ValueError(
                f"Default None for attribute '{key}' requires an optional type annotation"
            )

        normalized_default = _normalize_attribute_value(
            key,
            default_value,
            attribute_type,
            context="default attribute",
            allow_none=nullable,
        )
        spec[key] = AttributeSpec(
            attribute_type=attribute_type,
            nullable=nullable,
            required=False,
            default=normalized_default,
        )

    return spec


def _attributes_schema_items(
    attributes: Optional[AttributesSchemaSpec],
) -> dict[str, dict[str, Any]]:
    if attributes is None:
        return {}
    if isinstance(attributes, Mapping):
        out: dict[str, dict[str, Any]] = {}
        for key, value in attributes.items():
            if isinstance(value, tuple):
                if len(value) != 2:
                    raise ValueError(
                        f"Attribute schema tuple for '{key}' must be (type, default)"
                    )
                out[key] = {
                    "annotation": value[0],
                    "has_default": True,
                    "default": value[1],
                }
            else:
                out[key] = {
                    "annotation": value,
                    "has_default": False,
                    "default": None,
                }
        return out
    if isinstance(attributes, type):
        try:
            annotations = get_type_hints(attributes, include_extras=True)
        except Exception as e:
            raise ValueError(
                f"Failed to parse attribute annotations from '{attributes.__name__}': {e}"
            )
        class_vars = vars(attributes)
        out: dict[str, dict[str, Any]] = {}
        for key, annotation in annotations.items():
            if key in class_vars:
                out[key] = {
                    "annotation": annotation,
                    "has_default": True,
                    "default": class_vars[key],
                }
            else:
                out[key] = {
                    "annotation": annotation,
                    "has_default": False,
                    "default": None,
                }
        return out
    raise ValueError("attributes must be a mapping or a class with type annotations")


def _parse_attribute_type(
    *,
    key: str,
    annotation: Any,
    allow_vector_types: bool,
    allow_struct_types: bool,
) -> tuple[AttributeType, bool]:
    annotation, nullable = _unwrap_optional_annotation(annotation)

    if isinstance(annotation, type) and annotation in _ATTRIBUTE_SCALAR_TYPE_TO_NAME:
        return cast(AttributeScalarType, annotation), nullable

    if isinstance(annotation, AttributeFloatVectorType):
        if not allow_vector_types:
            raise ValueError(
                f"Vector attribute types are not supported for '{key}' in this backend"
            )
        return annotation, nullable

    if isinstance(annotation, Mapping):
        if not allow_struct_types:
            raise ValueError(
                f"Object attribute types are not supported for '{key}' in this backend"
            )
        return _parse_struct_annotation(
            key=key,
            annotation=annotation,
            allow_vector_types=allow_vector_types,
            allow_struct_types=allow_struct_types,
        ), nullable

    origin = get_origin(annotation)
    if origin is Annotated:
        args = get_args(annotation)
        base, base_nullable = _unwrap_optional_annotation(args[0])
        extras = args[1:]
        nullable = nullable or base_nullable

        if isinstance(base, type) and base in _ATTRIBUTE_SCALAR_TYPE_TO_NAME:
            return cast(AttributeScalarType, base), nullable

        if isinstance(base, Mapping):
            if extras:
                raise ValueError(
                    f"Unsupported attribute annotation for '{key}': {annotation}"
                )
            if not allow_struct_types:
                raise ValueError(
                    f"Object attribute types are not supported for '{key}' in this backend"
                )
            return _parse_struct_annotation(
                key=key,
                annotation=base,
                allow_vector_types=allow_vector_types,
                allow_struct_types=allow_struct_types,
            ), nullable

        vector_type = _parse_vector_annotation(base, extras)
        if vector_type is None:
            raise ValueError(
                f"Unsupported attribute annotation for '{key}': {annotation}"
            )
        if not allow_vector_types:
            raise ValueError(
                f"Vector attribute types are not supported for '{key}' in this backend"
            )
        return vector_type, nullable

    raise ValueError(
        f"Attribute type for '{key}' must be one of: str, int, float, bool, optional scalar (T | None), object mapping, or Annotated[list[float], N]"
    )


def _parse_struct_annotation(
    *,
    key: str,
    annotation: Mapping[str, Any],
    allow_vector_types: bool,
    allow_struct_types: bool,
) -> AttributeStructType:
    fields: dict[str, AttributeType] = {}
    for field_name, field_annotation in annotation.items():
        if not isinstance(field_name, str) or not field_name:
            raise ValueError(
                f"Object attribute field names for '{key}' must be non-empty strings"
            )
        field_type, field_nullable = _parse_attribute_type(
            key=f"{key}.{field_name}",
            annotation=field_annotation,
            allow_vector_types=allow_vector_types,
            allow_struct_types=allow_struct_types,
        )
        if field_nullable:
            raise ValueError(
                f"Optional object attribute field '{key}.{field_name}' is not supported"
            )
        fields[field_name] = field_type
    return AttributeStructType(fields=fields)


def _parse_vector_annotation(
    base: Any, extras: tuple[Any, ...]
) -> Optional[AttributeType]:
    base_origin = get_origin(base)
    base_args = get_args(base)
    if base_origin is not list or len(base_args) != 1 or base_args[0] is not float:
        return None

    dimensions = [
        x for x in extras if isinstance(x, int) and not isinstance(x, bool) and x > 0
    ]
    if len(dimensions) != 1:
        raise ValueError(
            "Vector attribute annotations must include exactly one positive integer dimension"
        )

    return AttributeFloatVectorType(dimension=dimensions[0])


def _unwrap_optional_annotation(annotation: Any) -> tuple[Any, bool]:
    origin = get_origin(annotation)
    if origin not in {Union, types.UnionType}:
        return annotation, False

    args = get_args(annotation)
    non_none_args = [arg for arg in args if arg is not type(None)]
    has_none = len(non_none_args) != len(args)
    if not has_none:
        return annotation, False
    if len(non_none_args) != 1:
        raise ValueError(
            f"Unsupported union annotation '{annotation}'. Only optional unions like T | None are supported."
        )
    return non_none_args[0], True


def attributes_schema_to_json_dict(
    attributes_schema: Mapping[str, AttributeType],
) -> dict[str, Any]:
    return {
        key: _attribute_type_to_json_value(value)
        for key, value in attributes_schema.items()
    }


def _attribute_type_to_name(attribute_type: AttributeType) -> str:
    if isinstance(attribute_type, AttributeFloatVectorType):
        return f"float_vector[{attribute_type.dimension}]"
    if isinstance(attribute_type, AttributeStructType):
        raise ValueError("Structured object attributes are serialized as mappings")
    return _ATTRIBUTE_SCALAR_TYPE_TO_NAME[attribute_type]


def _attribute_type_to_json_value(attribute_type: AttributeType) -> Any:
    if isinstance(attribute_type, AttributeStructType):
        return {
            "type": "struct",
            "fields": {
                key: _attribute_type_to_json_value(value)
                for key, value in attribute_type.fields.items()
            },
        }
    return _attribute_type_to_name(attribute_type)


def attributes_schema_from_json_dict(
    attributes_schema_json: Mapping[str, Any],
    *,
    allow_vector_types: bool = True,
    allow_struct_types: bool = True,
) -> dict[str, AttributeType]:
    schema: dict[str, AttributeType] = {}
    for key, value in attributes_schema_json.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Attribute column names must be non-empty strings")
        schema[key] = _attribute_type_from_json_value(
            key=key,
            value=value,
            allow_vector_types=allow_vector_types,
            allow_struct_types=allow_struct_types,
        )

    return schema


def attributes_spec_to_json_dict(
    attributes_spec: Mapping[str, AttributeSpec],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, spec in attributes_spec.items():
        payload: dict[str, Any] = {
            "type": _attribute_type_to_json_value(spec.attribute_type),
            "nullable": spec.nullable,
            "required": spec.required,
        }
        if not spec.required:
            payload["default"] = spec.default
        out[key] = payload
    return out


def attributes_spec_from_json_dict(
    attributes_spec_json: Mapping[str, Any],
    *,
    allow_vector_types: bool = True,
    allow_struct_types: bool = True,
    allow_optional_values: bool = True,
) -> dict[str, AttributeSpec]:
    out: dict[str, AttributeSpec] = {}
    for key, payload in attributes_spec_json.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Attribute column names must be non-empty strings")
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"Attribute spec for '{key}' must be a mapping with keys: type, nullable, required, default"
            )
        type_value = payload.get("type")
        if not isinstance(type_value, (str, Mapping)):
            raise ValueError(
                f"Attribute spec for '{key}' must include 'type' as a string or mapping"
            )
        nullable = payload.get("nullable")
        if not isinstance(nullable, bool):
            raise ValueError(
                f"Attribute spec for '{key}' must include boolean 'nullable'"
            )
        required = payload.get("required")
        if not isinstance(required, bool):
            raise ValueError(
                f"Attribute spec for '{key}' must include boolean 'required'"
            )
        if not allow_optional_values and not required:
            raise ValueError(
                f"Optional attribute values are not supported for '{key}' in this backend"
            )

        attribute_type = _attribute_type_from_json_value(
            key=key,
            value=type_value,
            allow_vector_types=allow_vector_types,
            allow_struct_types=allow_struct_types,
        )

        if required:
            out[key] = AttributeSpec(
                attribute_type=attribute_type,
                nullable=nullable,
                required=True,
            )
            continue

        default_raw = payload.get("default")
        if default_raw is None and not nullable:
            raise ValueError(
                f"Default None for attribute '{key}' requires nullable=true in serialized spec"
            )
        default_value = _normalize_attribute_value(
            key,
            default_raw,
            attribute_type,
            context="default attribute",
            allow_none=nullable,
        )
        out[key] = AttributeSpec(
            attribute_type=attribute_type,
            nullable=nullable,
            required=False,
            default=default_value,
        )
    return out


def _parse_attribute_type_name(type_name: str) -> Optional[AttributeType]:
    match = _FLOAT_VECTOR_TYPE_PATTERN.fullmatch(type_name)
    if match is not None:
        return AttributeFloatVectorType(dimension=int(match.group(1)))
    return None


def _attribute_type_from_json_value(
    *,
    key: str,
    value: Any,
    allow_vector_types: bool,
    allow_struct_types: bool,
) -> AttributeType:
    if isinstance(value, str):
        if value in _ATTRIBUTE_NAME_TO_SCALAR_TYPE:
            return _ATTRIBUTE_NAME_TO_SCALAR_TYPE[value]

        vector_type = _parse_attribute_type_name(value)
        if vector_type is None:
            raise ValueError(
                f"Attribute type for '{key}' must be one of: str, int, float, bool, float_vector[N], or struct"
            )
        if isinstance(vector_type, AttributeFloatVectorType) and not allow_vector_types:
            raise ValueError(
                f"Vector attribute types are not supported for '{key}' in this backend"
            )
        return vector_type

    if isinstance(value, Mapping):
        type_name = value.get("type")
        if type_name != "struct":
            raise ValueError(
                f"Attribute type mapping for '{key}' must include type='struct'"
            )
        if not allow_struct_types:
            raise ValueError(
                f"Object attribute types are not supported for '{key}' in this backend"
            )
        fields_value = value.get("fields")
        if not isinstance(fields_value, Mapping):
            raise ValueError(
                f"Attribute struct type for '{key}' must include mapping 'fields'"
            )

        fields: dict[str, AttributeType] = {}
        for field_name, field_type_json in fields_value.items():
            if not isinstance(field_name, str) or not field_name:
                raise ValueError(
                    f"Attribute struct field names for '{key}' must be non-empty strings"
                )
            fields[field_name] = _attribute_type_from_json_value(
                key=f"{key}.{field_name}",
                value=field_type_json,
                allow_vector_types=allow_vector_types,
                allow_struct_types=allow_struct_types,
            )
        return AttributeStructType(fields=fields)

    raise ValueError(
        f"Attribute type for '{key}' must be one of: str, int, float, bool, float_vector[N], or struct"
    )


def duckdb_sql_type_for_attribute_type(attribute_type: AttributeType) -> str:
    if attribute_type is str:
        return "VARCHAR"
    if attribute_type is int:
        return "INTEGER"
    if attribute_type is float:
        return "DOUBLE"
    if attribute_type is bool:
        return "BOOLEAN"
    if isinstance(attribute_type, AttributeFloatVectorType):
        return f"FLOAT[{attribute_type.dimension}]"
    if isinstance(attribute_type, AttributeStructType):
        fields_sql = ", ".join(
            f"{field_name} {duckdb_sql_type_for_attribute_type(field_type)}"
            for field_name, field_type in attribute_type.fields.items()
        )
        return f"STRUCT({fields_sql})"
    raise ValueError(f"Unsupported attribute type: {attribute_type}")


def merge_attribute_values(
    *,
    attributes_spec: Mapping[str, AttributeSpec],
    sources: Iterable[Optional[Mapping[str, Any]]],
) -> dict[str, AttributeValue]:
    merged: dict[str, AttributeValue | object] = {
        key: _MISSING for key in attributes_spec
    }

    for source in sources:
        if source is None:
            continue
        for key, value in source.items():
            if key not in attributes_spec:
                raise ValueError(
                    f"Unknown attribute key '{key}'. Declare it in attributes when creating the store."
                )
            spec = attributes_spec[key]
            merged[key] = _normalize_attribute_value(
                key,
                value,
                spec.attribute_type,
                allow_none=spec.nullable,
            )

    result: dict[str, AttributeValue] = {}
    for key, spec in attributes_spec.items():
        value = merged[key]
        if value is _MISSING:
            if spec.required:
                raise ValueError(
                    f"Missing required attribute '{key}'. Provide a value in document, insert call, or chunk attributes."
                )
            value = _copy_attribute_default(spec.default)
        result[key] = cast(AttributeValue, value)
    return result


def _copy_attribute_default(value: AttributeValue) -> AttributeValue:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return {key: _copy_attribute_default(item) for key, item in value.items()}
    return value


def _normalize_attribute_value(
    key: str,
    value: Any,
    attribute_type: AttributeType,
    *,
    context: str = "attributes",
    allow_none: bool = True,
) -> AttributeValue:
    if value is None:
        if allow_none:
            return None
        raise ValueError(
            f"Invalid value for {context} '{key}': expected non-null value, got NoneType"
        )

    if isinstance(attribute_type, AttributeFloatVectorType):
        if not isinstance(value, (list, tuple)):
            raise ValueError(
                f"Invalid value for {context} '{key}': expected list[float] with length {attribute_type.dimension}, got {type(value).__name__}"
            )
        if len(value) != attribute_type.dimension:
            raise ValueError(
                f"Invalid value for {context} '{key}': expected list[float] with length {attribute_type.dimension}, got length {len(value)}"
            )
        normalized: list[float] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError(
                    f"Invalid value for {context} '{key}': expected list[float], got element of type {type(item).__name__}"
                )
            normalized.append(float(item))
        return normalized

    if isinstance(attribute_type, AttributeStructType):
        if not isinstance(value, Mapping):
            raise ValueError(
                f"Invalid value for {context} '{key}': expected object mapping, got {type(value).__name__}"
            )
        extra_fields = set(value) - set(attribute_type.fields)
        if extra_fields:
            extra_display = ", ".join(sorted(extra_fields))
            raise ValueError(
                f"Invalid value for {context} '{key}': unknown object field(s): {extra_display}"
            )
        normalized_object: dict[str, AttributeValue] = {}
        for field_name, field_type in attribute_type.fields.items():
            if field_name not in value:
                raise ValueError(
                    f"Invalid value for {context} '{key}': missing object field '{field_name}'"
                )
            normalized_object[field_name] = _normalize_attribute_value(
                f"{key}.{field_name}",
                value[field_name],
                field_type,
                context=context,
                allow_none=False,
            )
        return normalized_object

    if attribute_type is str:
        ok = isinstance(value, str)
    elif attribute_type is bool:
        ok = isinstance(value, bool)
    elif attribute_type is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif attribute_type is float:
        ok = (isinstance(value, int) and not isinstance(value, bool)) or isinstance(
            value, float
        )
    else:
        ok = False

    if not ok:
        raise ValueError(
            f"Invalid value for {context} '{key}': expected {_attribute_type_label(attribute_type)}, got {type(value).__name__}"
        )
    return cast(AttributeValue, value)


def coerce_attribute_value_for_output(
    key: str,
    value: Any,
    attribute_type: AttributeType,
) -> AttributeValue:
    return _normalize_attribute_value(
        key,
        value,
        attribute_type,
        context="retrieved attributes",
        allow_none=True,
    )


def attribute_type_supports_filters(attribute_type: AttributeType) -> bool:
    return not isinstance(attribute_type, AttributeFloatVectorType)


def filterable_attribute_paths(
    attributes_schema: Mapping[str, AttributeType],
) -> set[str]:
    out: set[str] = set()
    for key, attribute_type in attributes_schema.items():
        out |= _filterable_attribute_paths_for_type(key, attribute_type)
    return out


def _filterable_attribute_paths_for_type(
    prefix: str,
    attribute_type: AttributeType,
) -> set[str]:
    if isinstance(attribute_type, AttributeFloatVectorType):
        return set()
    if isinstance(attribute_type, AttributeStructType):
        out: set[str] = set()
        for key, nested in attribute_type.fields.items():
            out |= _filterable_attribute_paths_for_type(f"{prefix}.{key}", nested)
        return out
    return {prefix}


def _attribute_type_label(attribute_type: AttributeType) -> str:
    if isinstance(attribute_type, AttributeFloatVectorType):
        return f"list[float] (length {attribute_type.dimension})"
    if isinstance(attribute_type, AttributeStructType):
        return "object"
    return attribute_type.__name__


@dataclass(frozen=True)
class FilterComparison:
    column: str
    operator: str
    value: AttributeFilterValue | list[AttributeScalar]


@dataclass(frozen=True)
class FilterLogical:
    operator: str
    children: list["FilterNode"]


FilterNode = FilterComparison | FilterLogical


def compile_filter_to_sql(
    attributes_filter: Optional[AttributeFilter],
    *,
    allowed_columns: Optional[Iterable[str]] = None,
) -> Optional[str]:
    node = _parse_filter_or_none(attributes_filter, allowed_columns=allowed_columns)
    if node is None:
        return None
    return _emit_sql(node)


def compile_filter_to_chroma_where(
    attributes_filter: Optional[AttributeFilter],
    *,
    allowed_columns: Optional[Iterable[str]] = None,
) -> Optional[dict[str, Any]]:
    node = _parse_filter_or_none(attributes_filter, allowed_columns=allowed_columns)
    if node is None:
        return None
    return _emit_chroma_where(node)


def compile_filter_to_openai_filters(
    attributes_filter: Optional[AttributeFilter],
    *,
    allowed_columns: Optional[Iterable[str]] = None,
) -> Optional[dict[str, Any]]:
    node = _parse_filter_or_none(attributes_filter, allowed_columns=allowed_columns)
    if node is None:
        return None
    return _emit_openai_filters(node)


def _parse_filter_or_none(
    attributes_filter: Optional[AttributeFilter],
    *,
    allowed_columns: Optional[Iterable[str]],
) -> Optional[FilterNode]:
    if attributes_filter is None:
        return None
    if isinstance(attributes_filter, str):
        text = attributes_filter.strip()
        if not text:
            return None
        parser = _FilterParser(text, allowed_columns=allowed_columns)
        return parser.parse()
    if isinstance(attributes_filter, Mapping):
        return _parse_filter_mapping_node(
            attributes_filter,
            allowed_columns=set(allowed_columns)
            if allowed_columns is not None
            else None,
        )
    raise TypeError(
        f"attributes_filter must be a string or mapping, got {type(attributes_filter).__name__}"
    )


def _parse_filter_mapping_node(
    node: Mapping[str, Any], *, allowed_columns: Optional[set[str]]
) -> FilterNode:
    node_type = node.get("type")
    if not isinstance(node_type, str) or not node_type:
        raise ValueError("Filter mapping nodes must include a non-empty string 'type'")
    node_type = node_type.lower()

    if node_type in {"and", "or"}:
        filters = node.get("filters")
        if not isinstance(filters, list) or len(filters) == 0:
            raise ValueError(
                f"Logical filter '{node_type}' must include a non-empty list in 'filters'"
            )
        children: list[FilterNode] = []
        for child in filters:
            if not isinstance(child, Mapping):
                raise ValueError("Each item in 'filters' must be a mapping")
            children.append(
                _parse_filter_mapping_node(child, allowed_columns=allowed_columns)
            )
        return FilterLogical(operator=node_type, children=children)

    if node_type in {"eq", "ne", "gt", "gte", "lt", "lte", "in", "nin"}:
        column = node.get("key")
        if not isinstance(column, str) or not column:
            raise ValueError(
                f"Comparison filter '{node_type}' must include string 'key'"
            )
        _validate_allowed_filter_column(column, allowed_columns)

        if "value" not in node:
            raise ValueError(f"Comparison filter '{node_type}' must include 'value'")
        raw_value = node["value"]

        if node_type in {"in", "nin"}:
            value_list = _parse_filter_list_value(raw_value)
            return FilterComparison(column=column, operator=node_type, value=value_list)

        value = _parse_filter_scalar_value(raw_value, allow_null=True)
        _validate_null_comparison_operator(operator=node_type, value=value)
        return FilterComparison(column=column, operator=node_type, value=value)

    raise ValueError(f"Unknown filter node type '{node_type}'")


def _parse_filter_scalar_value(value: Any, *, allow_null: bool) -> AttributeFilterValue:
    if value is None:
        if allow_null:
            return None
        raise ValueError("NULL is not allowed in this filter value")
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return value
    raise ValueError(f"Unsupported filter value type: {type(value).__name__}")


def _parse_filter_list_value(value: Any) -> list[AttributeScalar]:
    if not isinstance(value, list) or len(value) == 0:
        raise ValueError("IN/NIN filter values must be a non-empty list")
    parsed: list[AttributeScalar] = []
    for item in value:
        item_value = _parse_filter_scalar_value(item, allow_null=False)
        assert item_value is not None
        parsed.append(item_value)
    return parsed


def _validate_null_comparison_operator(
    *, operator: str, value: AttributeFilterValue
) -> None:
    if value is None and operator not in {"eq", "ne"}:
        raise ValueError("NULL is only allowed with = and != operators")


def _validate_allowed_filter_column(
    column: str, allowed_columns: Optional[set[str]]
) -> None:
    if allowed_columns is None:
        return
    if column not in allowed_columns:
        allowed = ", ".join(sorted(allowed_columns))
        raise ValueError(
            f"Unknown attribute column '{column}' in filter. Allowed columns: {allowed}"
        )


class _FilterParser:
    def __init__(self, text: str, *, allowed_columns: Optional[Iterable[str]]):
        self._tokens = _tokenize_filter(text)
        self._idx = 0
        self._allowed_columns = (
            set(allowed_columns) if allowed_columns is not None else None
        )

    def parse(self) -> FilterNode:
        node = self._parse_or()
        self._expect("EOF")
        return node

    def _parse_or(self) -> FilterNode:
        left = self._parse_and()
        children = [left]
        while self._match("OR"):
            children.append(self._parse_and())
        if len(children) == 1:
            return children[0]
        return FilterLogical(operator="or", children=children)

    def _parse_and(self) -> FilterNode:
        left = self._parse_primary()
        children = [left]
        while self._match("AND"):
            children.append(self._parse_primary())
        if len(children) == 1:
            return children[0]
        return FilterLogical(operator="and", children=children)

    def _parse_primary(self) -> FilterNode:
        if self._match("LPAREN"):
            node = self._parse_or()
            self._expect("RPAREN")
            return node
        return self._parse_comparison()

    def _parse_comparison(self) -> FilterNode:
        column = self._parse_identifier_path()
        _validate_allowed_filter_column(column, self._allowed_columns)

        if self._match("IS"):
            if self._match("NOT"):
                self._expect("NULL")
                return FilterComparison(column=column, operator="ne", value=None)
            self._expect("NULL")
            return FilterComparison(column=column, operator="eq", value=None)

        if self._match("NOT"):
            self._expect("IN")
            operator = "nin"
            value = self._parse_literal_list()
            return FilterComparison(column=column, operator=operator, value=value)

        if self._match("IN"):
            operator = "in"
            value = self._parse_literal_list()
            return FilterComparison(column=column, operator=operator, value=value)

        token = self._expect("OP")
        op_map = {
            "=": "eq",
            "!=": "ne",
            ">": "gt",
            ">=": "gte",
            "<": "lt",
            "<=": "lte",
        }
        operator = op_map[token.value]
        value = self._parse_literal_value()
        _validate_null_comparison_operator(operator=operator, value=value)
        if value is None and operator in {"eq", "ne"}:
            raise ValueError("Use IS NULL or IS NOT NULL for NULL comparisons")
        return FilterComparison(column=column, operator=operator, value=value)

    def _parse_identifier_path(self) -> str:
        token = self._expect("IDENT")
        assert isinstance(token.value, str)
        parts = [token.value]
        while self._match("DOT"):
            next_token = self._expect("IDENT")
            assert isinstance(next_token.value, str)
            parts.append(next_token.value)
        return ".".join(parts)

    def _parse_literal_list(self) -> list[AttributeScalar]:
        self._expect("LPAREN")
        values: list[AttributeScalar] = []
        values.append(self._parse_list_literal_value())
        while self._match("COMMA"):
            values.append(self._parse_list_literal_value())
        self._expect("RPAREN")
        return values

    def _parse_list_literal_value(self) -> AttributeScalar:
        value = self._parse_literal_value()
        if value is None:
            raise ValueError("NULL is not allowed inside IN (...) lists")
        return value

    def _parse_literal_value(self) -> AttributeFilterValue:
        token = self._peek()
        if token.kind == "STRING":
            self._idx += 1
            assert isinstance(token.value, str)
            return token.value
        if token.kind == "NUMBER":
            self._idx += 1
            assert isinstance(token.value, (int, float))
            return token.value
        if token.kind == "TRUE":
            self._idx += 1
            return True
        if token.kind == "FALSE":
            self._idx += 1
            return False
        if token.kind == "NULL":
            self._idx += 1
            return None
        raise ValueError(f"Expected a literal value, got '{token.raw}'")

    def _match(self, kind: str) -> bool:
        if self._peek().kind == kind:
            self._idx += 1
            return True
        return False

    def _expect(self, kind: str) -> "_Token":
        token = self._peek()
        if token.kind != kind:
            raise ValueError(f"Expected {kind}, got '{token.raw}'")
        self._idx += 1
        return token

    def _peek(self) -> "_Token":
        return self._tokens[self._idx]


@dataclass(frozen=True)
class _Token:
    kind: str
    raw: str
    value: Any = None


_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _tokenize_filter(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    i = 0
    n = len(text)
    while i < n:
        char = text[i]

        if char.isspace():
            i += 1
            continue

        if (
            text.startswith(">=", i)
            or text.startswith("<=", i)
            or text.startswith("!=", i)
        ):
            op = text[i : i + 2]
            tokens.append(_Token(kind="OP", raw=op, value=op))
            i += 2
            continue

        if char in ("=", ">", "<"):
            tokens.append(_Token(kind="OP", raw=char, value=char))
            i += 1
            continue

        if char == "(":
            tokens.append(_Token(kind="LPAREN", raw=char))
            i += 1
            continue
        if char == ")":
            tokens.append(_Token(kind="RPAREN", raw=char))
            i += 1
            continue
        if char == ",":
            tokens.append(_Token(kind="COMMA", raw=char))
            i += 1
            continue
        if char == ".":
            tokens.append(_Token(kind="DOT", raw=char))
            i += 1
            continue

        if char == "'":
            j = i + 1
            while j < n:
                if text[j] == "'":
                    if j + 1 < n and text[j + 1] == "'":
                        j += 2
                        continue
                    break
                j += 1
            if j >= n or text[j] != "'":
                raise ValueError("Unterminated string literal in attributes filter")
            raw = text[i : j + 1]
            value = raw[1:-1].replace("''", "'")
            tokens.append(_Token(kind="STRING", raw=raw, value=value))
            i = j + 1
            continue

        number_match = _NUMBER_RE.match(text, i)
        if number_match is not None:
            raw = number_match.group(0)
            value: int | float
            if "." in raw:
                value = float(raw)
            else:
                value = int(raw)
            tokens.append(_Token(kind="NUMBER", raw=raw, value=value))
            i = number_match.end()
            continue

        ident_match = _IDENT_RE.match(text, i)
        if ident_match is not None:
            raw = ident_match.group(0)
            keyword = raw.upper()
            if keyword in {"AND", "OR", "IN", "IS", "NOT", "TRUE", "FALSE", "NULL"}:
                tokens.append(_Token(kind=keyword, raw=raw))
            else:
                tokens.append(_Token(kind="IDENT", raw=raw, value=raw))
            i = ident_match.end()
            continue

        raise ValueError(f"Unexpected character '{char}' in attributes filter")

    tokens.append(_Token(kind="EOF", raw=""))
    return tokens


def _emit_sql(node: FilterNode) -> str:
    if isinstance(node, FilterLogical):
        joiner = " AND " if node.operator == "and" else " OR "
        return "(" + joiner.join(_emit_sql(child) for child in node.children) + ")"

    column = _sql_column_expression(node.column)
    if node.operator == "in":
        values = cast(list[AttributeScalar], node.value)
        values_sql = ", ".join(_sql_literal_scalar(value) for value in values)
        return f"{column} IN ({values_sql})"
    if node.operator == "nin":
        values = cast(list[AttributeScalar], node.value)
        values_sql = ", ".join(_sql_literal_scalar(value) for value in values)
        return f"{column} NOT IN ({values_sql})"

    value = cast(AttributeFilterValue, node.value)
    if value is None and node.operator == "eq":
        return f"{column} IS NULL"
    if value is None and node.operator == "ne":
        return f"{column} IS NOT NULL"

    op_map = {
        "eq": "=",
        "ne": "!=",
        "gt": ">",
        "gte": ">=",
        "lt": "<",
        "lte": "<=",
    }
    return f"{column} {op_map[node.operator]} {_sql_literal(value)}"


def _emit_chroma_where(node: FilterNode) -> dict[str, Any]:
    if isinstance(node, FilterLogical):
        key = "$and" if node.operator == "and" else "$or"
        return {key: [_emit_chroma_where(child) for child in node.children]}

    if node.value is None:
        raise ValueError("NULL is not supported in Chroma attributes filters")

    op_map = {
        "eq": "$eq",
        "ne": "$ne",
        "gt": "$gt",
        "gte": "$gte",
        "lt": "$lt",
        "lte": "$lte",
        "in": "$in",
        "nin": "$nin",
    }
    return {node.column: {op_map[node.operator]: node.value}}


def _emit_openai_filters(node: FilterNode) -> dict[str, Any]:
    if isinstance(node, FilterLogical):
        return {
            "type": node.operator,
            "filters": [_emit_openai_filters(child) for child in node.children],
        }

    if node.value is None:
        raise ValueError("NULL is not supported in OpenAI attributes filters")

    value = node.value
    if isinstance(value, list):
        coerced_list: list[str | float] = []
        for item in value:
            if isinstance(item, str):
                coerced_list.append(item)
            elif isinstance(item, bool):
                raise ValueError(
                    "Boolean values are not supported in IN/NIN for OpenAI filters"
                )
            elif isinstance(item, (int, float)):
                coerced_list.append(float(item))
            else:
                raise ValueError(
                    f"Unsupported filter value type: {type(item).__name__}"
                )
        value = coerced_list
    elif isinstance(value, bool):
        pass
    elif isinstance(value, str):
        pass
    elif isinstance(value, (int, float)):
        value = float(value)
    else:
        raise ValueError(f"Unsupported filter value type: {type(value).__name__}")

    return {
        "type": node.operator,
        "key": node.column,
        "value": value,
    }


def _sql_literal_scalar(value: AttributeScalar) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _sql_literal(value: AttributeFilterValue) -> str:
    if value is None:
        return "NULL"
    return _sql_literal_scalar(value)


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _sql_column_expression(column: str) -> str:
    parts = column.split(".")
    if len(parts) == 1:
        return _quote_identifier(parts[0])
    expr = _quote_identifier(parts[0])
    for field in parts[1:]:
        escaped = field.replace("'", "''")
        expr = f"struct_extract({expr}, '{escaped}')"
    return expr
