from __future__ import annotations

from dataclasses import dataclass
import re
from typing import (
    Annotated,
    Any,
    Iterable,
    Mapping,
    Optional,
    TypeAlias,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

MetadataScalar = str | int | float | bool
MetadataFilterValue = MetadataScalar | None
MetadataScalarType: TypeAlias = type[str] | type[int] | type[float] | type[bool]


@dataclass(frozen=True)
class MetadataFloatVectorType:
    dimension: int


MetadataValue = MetadataScalar | list[float] | None
MetadataType: TypeAlias = MetadataScalarType | MetadataFloatVectorType
AttributesSchemaSpec: TypeAlias = Mapping[str, Any] | type[Any]
MetadataFilter: TypeAlias = str | Mapping[str, Any]

_METADATA_SCALAR_TYPE_TO_NAME: dict[MetadataScalarType, str] = {
    str: "str",
    int: "int",
    float: "float",
    bool: "bool",
}
_METADATA_NAME_TO_SCALAR_TYPE: dict[str, MetadataScalarType] = {
    value: key for key, value in _METADATA_SCALAR_TYPE_TO_NAME.items()
}
_FLOAT_VECTOR_TYPE_PATTERN = re.compile(r"^float_vector\[(\d+)\]$")


def normalize_attributes_schema(
    attributes: Optional[AttributesSchemaSpec],
    *,
    reserved_columns: Iterable[str],
    allow_vector_types: bool = False,
) -> dict[str, MetadataType]:
    schema_items = dict(_attributes_schema_items(attributes))
    reserved = set(reserved_columns)
    schema: dict[str, MetadataType] = {}

    for key, annotation in schema_items.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Attribute column names must be non-empty strings")
        if key in reserved:
            raise ValueError(f"Attribute column '{key}' is reserved")

        schema[key] = _parse_metadata_type(
            key=key,
            annotation=annotation,
            allow_vector_types=allow_vector_types,
        )

    return schema


def _attributes_schema_items(
    attributes: Optional[AttributesSchemaSpec],
) -> Mapping[str, Any]:
    if attributes is None:
        return {}
    if isinstance(attributes, Mapping):
        return attributes
    if isinstance(attributes, type):
        try:
            return get_type_hints(attributes, include_extras=True)
        except Exception as e:
            raise ValueError(
                f"Failed to parse attribute annotations from '{attributes.__name__}': {e}"
            )
    raise ValueError("attributes must be a mapping or a class with type annotations")


def _parse_metadata_type(
    *,
    key: str,
    annotation: Any,
    allow_vector_types: bool,
) -> MetadataType:
    if isinstance(annotation, type) and annotation in _METADATA_SCALAR_TYPE_TO_NAME:
        return cast(MetadataScalarType, annotation)

    if isinstance(annotation, MetadataFloatVectorType):
        if not allow_vector_types:
            raise ValueError(
                f"Vector attribute types are not supported for '{key}' in this backend"
            )
        return annotation

    origin = get_origin(annotation)
    if origin is Annotated:
        args = get_args(annotation)
        base = args[0]
        extras = args[1:]

        if isinstance(base, type) and base in _METADATA_SCALAR_TYPE_TO_NAME:
            return cast(MetadataScalarType, base)

        vector_type = _parse_vector_annotation(base, extras)
        if vector_type is None:
            raise ValueError(
                f"Unsupported attribute annotation for '{key}': {annotation}"
            )
        if not allow_vector_types:
            raise ValueError(
                f"Vector attribute types are not supported for '{key}' in this backend"
            )
        return vector_type

    raise ValueError(
        f"Attribute type for '{key}' must be one of: str, int, float, bool, or Annotated[list[float], N]"
    )


def _parse_vector_annotation(
    base: Any, extras: tuple[Any, ...]
) -> Optional[MetadataType]:
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

    return MetadataFloatVectorType(dimension=dimensions[0])


def attributes_schema_to_json_dict(
    attributes_schema: Mapping[str, MetadataType],
) -> dict[str, str]:
    return {
        key: _metadata_type_to_name(value) for key, value in attributes_schema.items()
    }


def _metadata_type_to_name(metadata_type: MetadataType) -> str:
    if isinstance(metadata_type, MetadataFloatVectorType):
        return f"float_vector[{metadata_type.dimension}]"
    return _METADATA_SCALAR_TYPE_TO_NAME[metadata_type]


def attributes_schema_from_json_dict(
    attributes_schema_json: Mapping[str, Any],
    *,
    allow_vector_types: bool = True,
) -> dict[str, MetadataType]:
    schema: dict[str, MetadataType] = {}
    for key, value in attributes_schema_json.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Attribute column names must be non-empty strings")
        if not isinstance(value, str):
            raise ValueError(
                f"Attribute type for '{key}' must be one of: str, int, float, bool, or float_vector[N]"
            )

        if value in _METADATA_NAME_TO_SCALAR_TYPE:
            schema[key] = _METADATA_NAME_TO_SCALAR_TYPE[value]
            continue

        vector_type = _parse_metadata_type_name(value)
        if vector_type is None:
            raise ValueError(
                f"Attribute type for '{key}' must be one of: str, int, float, bool, or float_vector[N]"
            )
        if isinstance(vector_type, MetadataFloatVectorType) and not allow_vector_types:
            raise ValueError(
                f"Vector attribute types are not supported for '{key}' in this backend"
            )
        schema[key] = vector_type

    return schema


def _parse_metadata_type_name(type_name: str) -> Optional[MetadataType]:
    match = _FLOAT_VECTOR_TYPE_PATTERN.fullmatch(type_name)
    if match is not None:
        return MetadataFloatVectorType(dimension=int(match.group(1)))
    return None


def duckdb_sql_type_for_metadata_type(metadata_type: MetadataType) -> str:
    if metadata_type is str:
        return "VARCHAR"
    if metadata_type is int:
        return "INTEGER"
    if metadata_type is float:
        return "DOUBLE"
    if metadata_type is bool:
        return "BOOLEAN"
    if isinstance(metadata_type, MetadataFloatVectorType):
        return f"FLOAT[{metadata_type.dimension}]"
    raise ValueError(f"Unsupported attribute type: {metadata_type}")


def merge_metadata_values(
    *,
    attributes_schema: Optional[Mapping[str, MetadataType]] = None,
    metadata_schema: Optional[Mapping[str, MetadataType]] = None,
    sources: Iterable[Optional[Mapping[str, Any]]],
) -> dict[str, MetadataValue]:
    if attributes_schema is None:
        attributes_schema = metadata_schema
    elif metadata_schema is not None:
        raise ValueError("Use either attributes_schema or metadata_schema, not both.")
    if attributes_schema is None:
        raise ValueError("attributes_schema is required.")

    merged: dict[str, MetadataValue] = {key: None for key in attributes_schema}

    for source in sources:
        if source is None:
            continue
        for key, value in source.items():
            if key not in attributes_schema:
                raise ValueError(
                    f"Unknown attribute key '{key}'. Declare it in attributes when creating the store."
                )
            merged[key] = _normalize_metadata_value(
                key,
                value,
                attributes_schema[key],
            )

    return merged


def _normalize_metadata_value(
    key: str,
    value: Any,
    metadata_type: MetadataType,
    *,
    context: str = "attributes",
) -> MetadataValue:
    if value is None:
        return None

    if isinstance(metadata_type, MetadataFloatVectorType):
        if not isinstance(value, (list, tuple)):
            raise ValueError(
                f"Invalid value for {context} '{key}': expected list[float] with length {metadata_type.dimension}, got {type(value).__name__}"
            )
        if len(value) != metadata_type.dimension:
            raise ValueError(
                f"Invalid value for {context} '{key}': expected list[float] with length {metadata_type.dimension}, got length {len(value)}"
            )
        normalized: list[float] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError(
                    f"Invalid value for {context} '{key}': expected list[float], got element of type {type(item).__name__}"
                )
            normalized.append(float(item))
        return normalized

    if metadata_type is str:
        ok = isinstance(value, str)
    elif metadata_type is bool:
        ok = isinstance(value, bool)
    elif metadata_type is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif metadata_type is float:
        ok = (isinstance(value, int) and not isinstance(value, bool)) or isinstance(
            value, float
        )
    else:
        ok = False

    if not ok:
        raise ValueError(
            f"Invalid value for {context} '{key}': expected {metadata_type.__name__}, got {type(value).__name__}"
        )
    return cast(MetadataValue, value)


def coerce_metadata_value_for_output(
    key: str,
    value: Any,
    metadata_type: MetadataType,
) -> MetadataValue:
    return _normalize_metadata_value(
        key,
        value,
        metadata_type,
        context="retrieved attributes",
    )


def metadata_type_supports_filters(metadata_type: MetadataType) -> bool:
    return not isinstance(metadata_type, MetadataFloatVectorType)


@dataclass(frozen=True)
class FilterComparison:
    column: str
    operator: str
    value: MetadataFilterValue | list[MetadataScalar]


@dataclass(frozen=True)
class FilterLogical:
    operator: str
    children: list["FilterNode"]


FilterNode = FilterComparison | FilterLogical


def compile_filter_to_sql(
    attributes_filter: Optional[MetadataFilter],
    *,
    allowed_columns: Optional[Iterable[str]] = None,
) -> Optional[str]:
    node = _parse_filter_or_none(attributes_filter, allowed_columns=allowed_columns)
    if node is None:
        return None
    return _emit_sql(node)


def compile_filter_to_chroma_where(
    attributes_filter: Optional[MetadataFilter],
    *,
    allowed_columns: Optional[Iterable[str]] = None,
) -> Optional[dict[str, Any]]:
    node = _parse_filter_or_none(attributes_filter, allowed_columns=allowed_columns)
    if node is None:
        return None
    return _emit_chroma_where(node)


def compile_filter_to_openai_filters(
    attributes_filter: Optional[MetadataFilter],
    *,
    allowed_columns: Optional[Iterable[str]] = None,
) -> Optional[dict[str, Any]]:
    node = _parse_filter_or_none(attributes_filter, allowed_columns=allowed_columns)
    if node is None:
        return None
    return _emit_openai_filters(node)


def _parse_filter_or_none(
    attributes_filter: Optional[MetadataFilter],
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
        if column is None:
            column = node.get("column")
        if not isinstance(column, str) or not column:
            raise ValueError(
                f"Comparison filter '{node_type}' must include string 'key' (or 'column')"
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


def _parse_filter_scalar_value(value: Any, *, allow_null: bool) -> MetadataFilterValue:
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


def _parse_filter_list_value(value: Any) -> list[MetadataScalar]:
    if not isinstance(value, list) or len(value) == 0:
        raise ValueError("IN/NIN filter values must be a non-empty list")
    parsed: list[MetadataScalar] = []
    for item in value:
        item_value = _parse_filter_scalar_value(item, allow_null=False)
        assert item_value is not None
        parsed.append(item_value)
    return parsed


def _validate_null_comparison_operator(
    *, operator: str, value: MetadataFilterValue
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
        column = self._expect("IDENT").value
        assert isinstance(column, str)
        _validate_allowed_filter_column(column, self._allowed_columns)

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
        return FilterComparison(column=column, operator=operator, value=value)

    def _parse_literal_list(self) -> list[MetadataScalar]:
        self._expect("LPAREN")
        values: list[MetadataScalar] = []
        values.append(self._parse_list_literal_value())
        while self._match("COMMA"):
            values.append(self._parse_list_literal_value())
        self._expect("RPAREN")
        return values

    def _parse_list_literal_value(self) -> MetadataScalar:
        value = self._parse_literal_value()
        if value is None:
            raise ValueError("NULL is not allowed inside IN (...) lists")
        return value

    def _parse_literal_value(self) -> MetadataFilterValue:
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
            if keyword in {"AND", "OR", "IN", "NOT", "TRUE", "FALSE", "NULL"}:
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

    column = _quote_identifier(node.column)
    if node.operator == "in":
        values = cast(list[MetadataScalar], node.value)
        values_sql = ", ".join(_sql_literal_scalar(value) for value in values)
        return f"{column} IN ({values_sql})"
    if node.operator == "nin":
        values = cast(list[MetadataScalar], node.value)
        values_sql = ", ".join(_sql_literal_scalar(value) for value in values)
        return f"{column} NOT IN ({values_sql})"

    value = cast(MetadataFilterValue, node.value)
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


def _sql_literal_scalar(value: MetadataScalar) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _sql_literal(value: MetadataFilterValue) -> str:
    if value is None:
        return "NULL"
    return _sql_literal_scalar(value)


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'
