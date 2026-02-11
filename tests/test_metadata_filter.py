import pytest

from raghilda._metadata import (
    compile_filter_to_chroma_where,
    compile_filter_to_openai_filters,
    compile_filter_to_sql,
)


def test_compile_filter_to_sql_with_and_or():
    sql = compile_filter_to_sql(
        "tenant = 'docs' AND (priority >= 2 OR is_public = TRUE)",
        allowed_columns={"tenant", "priority", "is_public"},
    )
    assert sql == '("tenant" = \'docs\' AND ("priority" >= 2 OR "is_public" = TRUE))'


def test_compile_filter_to_sql_null_via_equality():
    sql = compile_filter_to_sql(
        "tenant = NULL",
        allowed_columns={"tenant"},
    )
    assert sql == '"tenant" IS NULL'


def test_compile_filter_to_chroma_where():
    where = compile_filter_to_chroma_where(
        "tenant = 'docs' AND priority IN (1, 2, 3)",
        allowed_columns={"tenant", "priority"},
    )
    assert where == {
        "$and": [
            {"tenant": {"$eq": "docs"}},
            {"priority": {"$in": [1, 2, 3]}},
        ]
    }


def test_compile_filter_to_openai_filters():
    filters = compile_filter_to_openai_filters(
        "tenant = 'docs' AND score >= 0.75",
        allowed_columns={"tenant", "score"},
    )
    assert filters == {
        "type": "and",
        "filters": [
            {"type": "eq", "key": "tenant", "value": "docs"},
            {"type": "gte", "key": "score", "value": 0.75},
        ],
    }


def test_compile_filter_rejects_unknown_column():
    with pytest.raises(ValueError, match="Unknown metadata column 'unknown'"):
        compile_filter_to_sql(
            "unknown = 'x'",
            allowed_columns={"tenant"},
        )


def test_compile_filter_to_sql_from_mapping_ast_with_in():
    sql = compile_filter_to_sql(
        {
            "type": "and",
            "filters": [
                {"type": "in", "key": "tenant", "value": ["docs", "blog"]},
                {"type": "gte", "key": "priority", "value": 2},
            ],
        },
        allowed_columns={"tenant", "priority"},
    )
    assert sql == "(\"tenant\" IN ('docs', 'blog') AND \"priority\" >= 2)"


def test_compile_filter_to_chroma_where_from_mapping_ast():
    where = compile_filter_to_chroma_where(
        {
            "type": "and",
            "filters": [
                {"type": "eq", "key": "tenant", "value": "docs"},
                {"type": "in", "key": "priority", "value": [1, 2, 3]},
            ],
        },
        allowed_columns={"tenant", "priority"},
    )
    assert where == {
        "$and": [
            {"tenant": {"$eq": "docs"}},
            {"priority": {"$in": [1, 2, 3]}},
        ]
    }


def test_compile_filter_to_openai_filters_from_mapping_ast():
    filters = compile_filter_to_openai_filters(
        {
            "type": "or",
            "filters": [
                {"type": "eq", "column": "tenant", "value": "docs"},
                {"type": "gt", "key": "score", "value": 0.9},
            ],
        },
        allowed_columns={"tenant", "score"},
    )
    assert filters == {
        "type": "or",
        "filters": [
            {"type": "eq", "key": "tenant", "value": "docs"},
            {"type": "gt", "key": "score", "value": 0.9},
        ],
    }
