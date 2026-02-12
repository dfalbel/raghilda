from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping, Optional, TypeAlias, cast

MetadataScalar = str | int | float | bool
MetadataValue = MetadataScalar | None
MetadataType: TypeAlias = type[str] | type[int] | type[float] | type[bool]
MetadataFilter: TypeAlias = str | Mapping[str, Any]

_METADATA_TYPE_TO_NAME: dict[MetadataType, str] = {
    str: "str",
    int: "int",
    float: "float",
    bool: "bool",
}
_METADATA_NAME_TO_TYPE: dict[str, MetadataType] = {
    value: key for key, value in _METADATA_TYPE_TO_NAME.items()
}


def normalize_metadata_schema(
    metadata: Optional[Mapping[str, type[Any]]],
    *,
    reserved_columns: Iterable[str],
) -> dict[str, MetadataType]:
    schema = dict(metadata or {})
    reserved = set(reserved_columns)

    for key, value in schema.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Metadata column names must be non-empty strings")
        if key in reserved:
            raise ValueError(f"Metadata column '{key}' is reserved")
        if not isinstance(value, type) or value not in _METADATA_TYPE_TO_NAME:
            raise ValueError(
                f"Metadata type for '{key}' must be one of: str, int, float, bool"
            )
        schema[key] = cast(MetadataType, value)

    return schema


def metadata_schema_to_json_dict(
    metadata_schema: Mapping[str, MetadataType],
) -> dict[str, str]:
    return {
        key: _METADATA_TYPE_TO_NAME[value] for key, value in metadata_schema.items()
    }


def metadata_schema_from_json_dict(
    metadata_schema_json: Mapping[str, Any],
) -> dict[str, MetadataType]:
    schema: dict[str, MetadataType] = {}
    for key, value in metadata_schema_json.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Metadata column names must be non-empty strings")
        if not isinstance(value, str) or value not in _METADATA_NAME_TO_TYPE:
            raise ValueError(
                f"Metadata type for '{key}' must be one of: str, int, float, bool"
            )
        schema[key] = _METADATA_NAME_TO_TYPE[value]
    return schema


def duckdb_sql_type_for_metadata_type(metadata_type: MetadataType) -> str:
    if metadata_type is str:
        return "VARCHAR"
    if metadata_type is int:
        return "INTEGER"
    if metadata_type is float:
        return "DOUBLE"
    if metadata_type is bool:
        return "BOOLEAN"
    raise ValueError(f"Unsupported metadata type: {metadata_type}")


def merge_metadata_values(
    *,
    metadata_schema: Mapping[str, MetadataType],
    sources: Iterable[Optional[Mapping[str, Any]]],
) -> dict[str, MetadataValue]:
    merged: dict[str, MetadataValue] = {key: None for key in metadata_schema}

    for source in sources:
        if source is None:
            continue
        for key, value in source.items():
            if key not in metadata_schema:
                raise ValueError(
                    f"Unknown metadata key '{key}'. Declare it in metadata when creating the store."
                )
            _validate_metadata_value_type(key, value, metadata_schema[key])
            merged[key] = value

    return merged


def _validate_metadata_value_type(
    key: str, value: Any, metadata_type: MetadataType, *, context: str = "metadata"
) -> None:
    if value is None:
        return

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


@dataclass(frozen=True)
class FilterComparison:
    column: str
    operator: str
    value: MetadataValue | list[MetadataScalar]


@dataclass(frozen=True)
class FilterLogical:
    operator: str
    children: list["FilterNode"]


FilterNode = FilterComparison | FilterLogical


def compile_filter_to_sql(
    metadata_filter: Optional[MetadataFilter],
    *,
    allowed_columns: Optional[Iterable[str]] = None,
) -> Optional[str]:
    node = _parse_filter_or_none(metadata_filter, allowed_columns=allowed_columns)
    if node is None:
        return None
    return _emit_sql(node)


def compile_filter_to_chroma_where(
    metadata_filter: Optional[MetadataFilter],
    *,
    allowed_columns: Optional[Iterable[str]] = None,
) -> Optional[dict[str, Any]]:
    node = _parse_filter_or_none(metadata_filter, allowed_columns=allowed_columns)
    if node is None:
        return None
    return _emit_chroma_where(node)


def compile_filter_to_openai_filters(
    metadata_filter: Optional[MetadataFilter],
    *,
    allowed_columns: Optional[Iterable[str]] = None,
) -> Optional[dict[str, Any]]:
    node = _parse_filter_or_none(metadata_filter, allowed_columns=allowed_columns)
    if node is None:
        return None
    return _emit_openai_filters(node)


def _parse_filter_or_none(
    metadata_filter: Optional[MetadataFilter],
    *,
    allowed_columns: Optional[Iterable[str]],
) -> Optional[FilterNode]:
    if metadata_filter is None:
        return None
    if isinstance(metadata_filter, str):
        text = metadata_filter.strip()
        if not text:
            return None
        parser = _FilterParser(text, allowed_columns=allowed_columns)
        return parser.parse()
    if isinstance(metadata_filter, Mapping):
        return _parse_filter_mapping_node(
            metadata_filter,
            allowed_columns=set(allowed_columns)
            if allowed_columns is not None
            else None,
        )
    raise TypeError(
        f"metadata_filter must be a string or mapping, got {type(metadata_filter).__name__}"
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
        return FilterComparison(column=column, operator=node_type, value=value)

    raise ValueError(f"Unknown filter node type '{node_type}'")


def _parse_filter_scalar_value(value: Any, *, allow_null: bool) -> MetadataValue:
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


def _validate_allowed_filter_column(
    column: str, allowed_columns: Optional[set[str]]
) -> None:
    if allowed_columns is None:
        return
    if column not in allowed_columns:
        allowed = ", ".join(sorted(allowed_columns))
        raise ValueError(
            f"Unknown metadata column '{column}' in filter. Allowed columns: {allowed}"
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

    def _parse_literal_value(self) -> MetadataValue:
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
                raise ValueError("Unterminated string literal in metadata filter")
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

        raise ValueError(f"Unexpected character '{char}' in metadata filter")

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

    value = cast(MetadataValue, node.value)
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
        raise ValueError("NULL is not supported in Chroma metadata filters")

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
        raise ValueError("NULL is not supported in OpenAI metadata filters")

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


def _sql_literal(value: MetadataValue) -> str:
    if value is None:
        return "NULL"
    return _sql_literal_scalar(value)


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'
