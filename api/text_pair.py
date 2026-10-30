#!/usr/bin/env python3
"""Routing and search code for sequence alignment"""

import bisect
import configparser
import html
import itertools
import math
import os
import re
import sqlite3
import time
from collections import Counter, OrderedDict, defaultdict, namedtuple
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

import networkx as nx
import numpy as np
import orjson
import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse
from pgvector.psycopg2 import register_vector
from philologic.runtime.DB import DB
from philologic.runtime.get_text import get_text_obj
from psycopg2 import sql
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import Response
from textpair.web_loader import FACET_TABLES, LENGTH_RANGES, OPEN_LENGTH_RANGE, relation_columns
from textpair.web_settings import SettingsError, load_app_config

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GLOBAL_CONFIG = configparser.ConfigParser()
GLOBAL_CONFIG.read("/etc/text-pair/global_settings.ini")
APP_PATH = GLOBAL_CONFIG["WEB_APP"]["web_app_path"]

PHILO_REQUEST = namedtuple("PHILO_REQUEST", ["byte", "start_byte", "end_byte", "passages"])
PHILO_CONFIG = namedtuple(
    "PHILO_CONFIG",
    [
        "db_path",
        "page_images_url_root",
        "page_image_extension",
        "page_external_page_images",
    ],
)

BOOLEAN_ARGS = re.compile(r"""(NOT \w+)|(OR \w+)|(\w+)|("")""")


class FormArguments:
    """Special dict to handle form arguments"""

    def __init__(self):
        self.dict = OrderedDict()

    def __getitem__(self, item) -> str | int | list:
        if item in self.dict:
            return self.dict[item]
        elif item == "page":
            return 1
        elif item == "id_anchor":
            return 0
        elif item == "direction":
            return "next"
        elif item == "timeSeriesInterval":
            return 1
        elif item == "directionSelected":
            return "source"
        elif item == "classification_filter":
            return []
        else:
            return ""

    def __getattr__(self, attr) -> str | int:
        return self.__getitem__(attr)

    def __setitem__(self, item, value):
        if value:
            self.dict[item] = value

    def __iter__(self):
        for k in self.dict:
            yield k

    def __bool__(self):
        if self.dict:
            return True
        return False

    def __contains__(self, key):
        if key in self.dict:
            return True
        return False

    def items(self):
        """Mimic items method of dict"""
        for k, v in self.dict.items():
            yield k, v

    def values(self):
        """Mimic values method of dict"""
        for _, v in self.dict.items():
            yield v

    def __str__(self):
        return repr(self.dict)


@contextmanager
def pg_connection():
    """A PostgreSQL connection, closed when the block ends, however it ends, and its transaction with it.

    psycopg2's own `with` ends the transaction but leaves the connection open.
    """
    conn = psycopg2.connect(
        user=GLOBAL_CONFIG["DATABASE"]["database_user"],
        password=GLOBAL_CONFIG["DATABASE"]["database_password"],
        database=GLOBAL_CONFIG["DATABASE"]["database_name"],
    )
    try:
        yield conn
    finally:
        conn.close()


def get_pg_type(conn, table_name):
    """Find PostgreSQL field type, empty for a name that is not a table of the database"""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = %s AND table_schema = current_schema()",
        (table_name,),
    )
    return {field: field_type.upper() for field, field_type in cursor}


def quote_ident(name) -> str:
    """A table or field name as an SQL identifier. Names come from get_pg_type, never from the request alone."""
    return '"' + str(name).replace('"', '""') + '"'


def db_directory(name: str) -> str:
    """The directory of a web app under APP_PATH, or a 404 for a name that is not one."""
    if not name or name not in os.listdir(APP_PATH) or not os.path.isdir(os.path.join(APP_PATH, name)):
        raise HTTPException(404, "No such database")
    return os.path.join(APP_PATH, name)


def side_of(direction) -> str:
    """A side of the alignments, source or target. Anything else is a 400."""
    if direction not in ("source", "target"):
        raise HTTPException(400, "directionSelected must be source or target")
    return direction


def web_app_file(db_path: str, folder: str, resource: str) -> str:
    """The path of a file served from a web app's dist folder, or a 404 for a name outside it or a file not there,
    such as an old bundle asked for by a page loaded before a rebuild."""
    if resource in ("", ".", "..") or "/" in resource or "\0" in resource:
        raise HTTPException(404, "No such file")
    path = os.path.join(db_directory(db_path), "dist", folder, resource)
    if not os.path.isfile(path):
        raise HTTPException(404, "No such file")
    return path


def has_passage_families(table) -> bool:
    """Whether alignments were grouped into passage families, as the network view needs."""
    with pg_connection() as conn:
        return get_pg_type(conn, table).get("group_id") == "ARRAY" and bool(get_pg_type(conn, f"{table}_groups"))


def text_data_path(db_dir: str, config: dict, direction: str) -> str | None:
    """The PhiloLogic database the full source or target texts are read from, if still on disk."""
    for path in (config.get(f"{direction}PhiloDBPath", ""), os.path.join(db_dir, f"{direction}_data")):
        if path and os.path.isfile(os.path.join(path, "data", "toms.db")):
            return path
    return None


def parse_request(request, conn):
    """Split URL args into the fields to filter on, the other args and the table's field types.

    The request's table must be one of the database's, or the request is a 404. Nothing here
    reaches SQL: the endpoints that filter go on to build_conditions.
    """
    query_args = FormArguments()
    other_args = FormArguments()
    other_args_keys = [
        "facet",
        "direction",
        "source",
        "target",
        "stats_field",
        "db_table",
        "filter_field",
        "filter_value",
        "page",
        "id_anchor",
        "directionSelected",
        "timeSeriesInterval",
        "field",
        "value",
        "philo_db",
        "philo_url",
        "philo_id",
        "doc_id",
        "philo_path",
        "div1_id",
        "passage",
        "by_section",
        "classification_filter_1",
        "classification_filter_2",
        "classification_filter_3",
        "aggregation_field",
        "graph",
        "focus",
        "focus_role",
        "min_threshold",
        "max_nodes",
        "node_id",
        "node_type",
        "limit",
        "offset",
        "centrality",
        "banality",
        "scope",  # in the links of the sorted view's earlier tabs; ignored
        "sort",
    ]
    for key, value in request.query_params.items():
        if key in other_args_keys:
            if key == "id_anchor":
                other_args[key] = value
            elif key in (
                "page",
                "timeSeriesInterval",
                "start_byte",
                "end_byte",
            ):
                if value.isdigit():
                    other_args[key] = int(value)
            elif key == "direction":
                other_args["direction"] = value or "next"
            elif key == "directionSelected":
                other_args["directionSelected"] = value or "source"
            elif key == "classification_filter":
                # Handle multiple values for classification_filter
                if key in other_args and isinstance(other_args[key], list):
                    other_args[key].append(value)
                else:
                    other_args[key] = [value]  # Initialize as list with first value
            else:
                other_args[key] = value
        else:
            if value:
                query_args[key] = value

    metadata_field_types = get_pg_type(conn, other_args["db_table"])
    if not metadata_field_types:
        raise HTTPException(404, "No such database")
    metadata_field_types["rowid"] = "INTEGER"
    return query_args, other_args, metadata_field_types


def parse_conditions(request, conn):
    """Parse URL args into the filter conditions, the other args and the table's field types"""
    query_args, other_args, metadata_field_types = parse_request(request, conn)
    return build_conditions(query_args, other_args, metadata_field_types), other_args, metadata_field_types


def parse_args(request, conn):
    """Parse URL args"""
    conditions, other_args, metadata_field_types = parse_conditions(request, conn)
    sql_fields, sql_values = join_conditions(conditions)
    return sql_fields, sql_values, other_args, list(metadata_field_types.keys())


# Results are ordered by year then position, and paged through with a keyset
# cursor rather than a materialized rank table. COALESCE reproduces the NULLS
# LAST that plain ASC gives, so the sort key is never NULL and the cursor
# comparison stays a plain row comparison an index can serve.
NULL_SORT_SENTINEL = 2147483647
PAGE_SORT_EXPRESSIONS = (
    f"COALESCE(source_year, {NULL_SORT_SENTINEL})",
    f"COALESCE(target_year, {NULL_SORT_SENTINEL})",
    "source_start_byte",
    "target_start_byte",
    "rowid",
)
PAGE_CURSOR_FIELDS = ("source_year", "target_year", "source_start_byte", "target_start_byte", "rowid")


def page_cursor(row) -> str:
    """Encode a row's position in the result order, for the next request to anchor on."""
    return ".".join(str(NULL_SORT_SENTINEL if row[f] is None else row[f]) for f in PAGE_CURSOR_FIELDS)


def parse_page_cursor(value) -> list[int] | None:
    """Decode a cursor, or None if it is absent or malformed (i.e. start from the top)."""
    try:
        anchor = [int(part) for part in str(value).split(".")]
    except (TypeError, ValueError):
        return None
    return anchor if len(anchor) == len(PAGE_CURSOR_FIELDS) else None


class Condition(NamedTuple):
    """One WHERE condition, the field it filters, and the values of its placeholders"""

    field: str
    sql: str
    values: list


def join_conditions(conditions: list[Condition]) -> tuple[str, list]:
    """The WHERE clause that ANDs conditions together, with its values"""
    return " AND ".join(c.sql for c in conditions), [value for c in conditions for value in c.values]


def build_conditions(query_args, other_args, field_types) -> list[Condition]:
    """Takes query arguments and returns the conditions of an SQL WHERE clause.

    A field the table does not have is a 400. The names that reach SQL are the table's own, quoted.
    """
    conditions: list[Condition] = []
    for field, value in query_args.items():
        if field not in field_types:
            raise HTTPException(400, f"Unknown field {field}")
        value: str = value.strip()
        field_type = field_types[field].upper()
        column = quote_ident(field)
        query = ""

        # --- Handle group_id specifically first ---
        if field == "group_id":
            try:
                # Assume user provides a single integer ID to filter by
                filter_gid = int(value)
                if field_type == "INTEGER":
                    query = f"{column} = %s"
                    params = [filter_gid]
                elif field_type == "ARRAY":  # Covers INTEGER[]
                    # Use array containment operator
                    query = f"{column} @> ARRAY[%s]::integer[]"
                    params = [filter_gid]
                elif field_type == "JSONB":
                    # Assumes JSONB stores an array of numbers
                    # Use JSONB containment operator
                    query = f"{column} @> %s::jsonb"
                    # Pass the integer as a JSON array string containing that number
                    params = [f"[{filter_gid}]"]
                else:  # Fallback to integer equality for unknown types
                    query = f"{column} = %s"
                    params = [filter_gid]

                conditions.append(Condition(field, query, params))

            except (ValueError, TypeError):
                # Ignore if the provided group_id filter value is not an integer
                continue  # Skip to the next field
            continue  # Skip the rest of the loop for group_id

        # --- Existing logic for other types ---
        if field_type == "TEXT":
            for not_query, or_query, regular_query, empty_query in BOOLEAN_ARGS.findall(value):
                if not_query != "":
                    value = not_query
                elif or_query != "":
                    value = or_query
                elif empty_query != "":
                    value = empty_query
                else:
                    value = regular_query
                if value.startswith('"'):
                    if value == '""':
                        query = f"{column} = %s"
                        params = [""]
                    else:
                        query = f"{column}=%s"
                        params = [value[1:-1]]
                elif value.startswith("NOT "):
                    split_value = " ".join(value.split()[1:]).strip()
                    query = f"{column} !~* %s"
                    params = [rf"\m{split_value}\M"]
                # elif value.startswith("OR "):  ## TODO: add support to OR queries by changing the join logic at the end of the function
                #     split_value = " ".join(value.split()[1:]).strip()
                #     query = "{} !~* %s".format(field)
                #     sql_values.append('\m{}\M'.format(split_value))
                else:
                    query = f"{column} ~* %s"
                    params = [rf"\m{value}\M"]
                conditions.append(Condition(field, query, params))
        elif field_type in ("INTEGER", "FLOAT"):
            value = value.replace('"', "")
            if "-" in value:
                values = [v for v in re.split(r"(-)", value) if v]
                if values[0] == "-":
                    query = f"{column} <= %s"
                    params = [values[1]]
                elif values[-1] == "-":
                    query = f"{column} >= %s"
                    params = [values[0]]
                else:
                    query = f"{column} BETWEEN %s AND %s"
                    params = [values[0], values[2]]
            else:
                query = f"{column} = %s"
                params = [value]
            conditions.append(Condition(field, query, params))
        else:
            continue
    if other_args.banality != "" and "banality" in field_types:  # the column exists only where banalities are stored
        conditions.append(
            Condition("banality", "banality=%s", [str(other_args.banality).lower() in ("true", "t", "1", "yes")])
        )

    # Add classification filter condition
    for i in range(1, 4):  # Handle classification_filter_1, classification_filter_2, classification_filter_3
        filter_key = f"classification_filter_{i}"
        if filter_key in other_args and other_args[filter_key]:
            # Each filter value must match in at least one classification field
            condition = "(target_first_class=%s OR target_second_class=%s OR target_third_class=%s)"
            # Add the same filter value three times (once for each field)
            filter_value = other_args[filter_key]
            conditions.append(Condition("classification", condition, [filter_value] * 3))

    return conditions


def check_access_control(request: Request):
    """Check if user has access to a particular database"""
    return True  # Placeholder


@app.get("/")
@app.get("/text-pair")
@app.get("/text-pair/")
def list_dir():
    """List Text-PAIR databases"""
    textpair_dbs = sorted(Path(APP_PATH).iterdir(), key=os.path.getmtime, reverse=True)
    links = "<h3>Text-PAIR databases</h3><hr/><table style='font-size: 130%'>"
    for db in textpair_dbs:
        date_components = time.ctime(os.path.getmtime(db)).split()
        date = f"{' '.join(date_components[0:3])} {date_components[-1]} {date_components[3]}"
        links += f'<tr><td><a href="{db.name}">{db.name}</a></td><td>{date}</td></tr>'
    links += "</table>"
    return HTMLResponse(links)


@app.get("/{db_path}/css/{resource}")
@app.get("/text-pair/{db_path}/css/{resource}")
def get_css_resource(db_path: str, resource: str):
    """Retrieve CSS resources"""
    return FileResponse(web_app_file(db_path, "css", resource), media_type="text/css")


@app.get("/{db_path}/js/{resource}")
@app.get("/text-pair/{db_path}/js/{resource}")
def get_js_resource(db_path: str, resource: str):
    """Retrieve JS resources"""
    return FileResponse(web_app_file(db_path, "js", resource), media_type="application/javascript")


@app.get("/{db_path}/assets/{resource}")
@app.get("/text-pair/{db_path}/assets/{resource}")
def get_ressource(db_path: str, resource: str):
    """Retrieve built assets: scripts, styles, and any other file typed by its extension"""
    media_type = "application/javascript" if resource.endswith(".js") else None
    return FileResponse(web_app_file(db_path, "assets", resource), media_type=media_type)


@app.get("/{db_path}/favicon.ico")
@app.get("/text-pair/{db_path}/favicon.ico")
def get_favicon(db_path: str):
    """Retrieve favicon"""
    return FileResponse(web_app_file(db_path, "", "favicon.ico"), media_type="image/x-icon")


def group_counts(cursor, groups_table: str, group_ids) -> dict[int, int]:
    """Each group's stored `count`: the documents its passage appears in. The table's name is quoted."""
    if not group_ids:
        return {}
    cursor.execute(f'SELECT group_id, "count" FROM {groups_table} WHERE group_id = ANY(%s)', (list(group_ids),))
    return {row[0]: row[1] for row in cursor}


def facing(row, side: str) -> dict:
    """An alignment with `side` presented as its target, source and target fields swapped if need be."""
    if side == "target":
        return {**row, "direction": "target"}
    flipped = {}
    for key, value in row.items():
        if key.startswith("source_"):
            flipped["target_" + key[7:]] = value
        elif key.startswith("target_"):
            flipped["source_" + key[7:]] = value
        else:
            flipped[key] = value
    flipped["direction"] = "target"
    return flipped


def occurrences(rows, group: dict) -> list[dict]:
    """Every place the group's passage is reused, once each, as an alignment facing it.

    Both sides of every alignment count, except the anchor's own document. Overlapping
    spans in one document are one place; its alignment is the one tied most closely to
    the group's passage, which is what the page diffs the reuse against: one with the
    passage itself, else one with another document reusing it, else one with another
    passage of the anchor's document.
    """
    anchor_doc = group["source_doc_id"]
    start, end = group["source_start_byte"], group["source_end_byte"]
    by_file: defaultdict[str, list] = defaultdict(list)
    for row in rows:
        for side, other in (("target", "source"), ("source", "target")):
            if row[f"{side}_doc_id"] == anchor_doc:
                continue
            if row[f"{other}_doc_id"] == anchor_doc:
                tie = min(row[f"{other}_end_byte"], end) - max(row[f"{other}_start_byte"], start)
                rank = (0, -tie) if tie > 0 else (2, -tie)
            else:
                rank = (1, 0)
            by_file[row[f"{side}_filename"]].append(
                (row[f"{side}_start_byte"], row[f"{side}_end_byte"], rank, row, side))
    found = []
    for spans in by_file.values():
        spans.sort(key=lambda item: (item[0], item[1]))
        place, reach = [], -1
        for item in spans + [None]:
            if place and (item is None or item[0] >= reach):
                best = min(place, key=lambda candidate: candidate[2])
                found.append(facing(best[3], best[4]))
                place, reach = [], -1
            if item is not None:
                place.append(item)
                reach = max(reach, item[1])
    return found


@app.get("/search_alignments/")
@app.get("/text-pair-api/search_alignments/")
def search_alignments(request: Request):
    """Search alignments according to URL params"""
    with pg_connection() as conn:
        sql_fields, sql_values, other_args, column_names = parse_args(request, conn)
        field_types = get_pg_type(conn, other_args.db_table)
        group_id_type = field_types.get("group_id", "INTEGER")
        forward = other_args.direction == "next"
        sort_key = ", ".join(PAGE_SORT_EXPRESSIONS)
        conditions = [sql_fields] if sql_fields else []
        sql_values = list(sql_values)

        anchor = parse_page_cursor(other_args.id_anchor)
        if anchor is not None:
            placeholders = ", ".join("%s" for _ in PAGE_SORT_EXPRESSIONS)
            conditions.append(f"({sort_key}) {'>' if forward else '<'} ({placeholders})")
            sql_values.extend(anchor)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        order_by = sort_key if forward else ", ".join(f"{e} DESC" for e in PAGE_SORT_EXPRESSIONS)
        table = quote_ident(other_args.db_table)
        query = f"SELECT * FROM {table} {where_clause} ORDER BY {order_by} LIMIT 50"
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute(query, sql_values)
        alignments = []
        group_ids = []
        for row in cursor:
            metadata = {key: row[key] for key in column_names}
            metadata["page_cursor"] = page_cursor(row)
            try:
                metadata["group_id"] = row["group_id"]
                if group_id_type == "ARRAY":
                    group_ids.extend([int(gid) for gid in metadata["group_id"]])
                else:
                    group_ids.append(metadata["group_id"])
            except KeyError:
                pass
            alignments.append(metadata)

        if other_args.direction == "previous":
            alignments.reverse()
            group_ids.reverse()

        # Check if groups table exists
        cursor.execute(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)", (f"{other_args.db_table}_groups",)
        )
        if cursor.fetchone()[0] is True:
            counts_per_group = group_counts(cursor, quote_ident(f"{other_args.db_table}_groups"), set(group_ids))
            for alignment in alignments:
                ids = alignment["group_id"] if group_id_type == "ARRAY" else [alignment["group_id"]]
                # in more than one family, an alignment is as reused as the most reused of them
                alignment["count"] = max((counts_per_group.get(int(gid), 0) for gid in ids), default=0)

    previous_url = ""
    current_path = re.sub(r"&(page|id_anchor|direction)=[^&]*", "", request.url.path)
    if other_args.page > 1 and alignments:  # type: ignore
        previous_url = (
            f"{current_path}&page={other_args.page - 1}&id_anchor={alignments[0]['page_cursor']}&direction=previous"  # type: ignore
        )
    try:
        next_url = (
            f"{current_path}&page={other_args.page + 1}&id_anchor={alignments[-1]['page_cursor']}&direction=next"  # type: ignore
        )
    except IndexError:
        next_url = ""
    start_position = 0
    if other_args.page > 1:  # type: ignore
        start_position = 50 * (other_args.page - 1)  # type: ignore
    return {
        "alignments": alignments,
        "page": other_args.page,
        "next_url": next_url,
        "previous_url": previous_url,
        "start_position": start_position,
    }


@app.get("/{db_path}/retrieve_all_docs/")
@app.get("/text-pair/{db_path}/retrieve_all_docs/")
def retrieve_all(request: Request):
    """Retrieve all docs and only return metadata"""
    with pg_connection() as conn:
        sql_fields, sql_values, other_args, column_names = parse_args(request, conn)
        if other_args.field not in column_names:
            raise HTTPException(400, f"Unknown field {other_args.field}")
        if other_args.field.startswith("source_"):  # type: ignore
            direction = "source_"
        else:
            direction = "target_"
        filtered_columns = {
            "source_filename",
            "source_passage",
            "source_context_before",
            "source_context_after",
            "source_start_byte",
            "source_end_byte",
            "target_filename",
            "target_passage",
            "target_context_before",
            "target_context_after",
            "target_start_byte",
            "target_end_byte",
            "rowid",
        }
        column_names = [
            column_name
            for column_name in column_names
            if column_name not in filtered_columns and column_name.startswith(direction)
        ]
        docs_found = {}
        doc_id = f"{direction}doc_id"
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        field = quote_ident(other_args.field)
        if sql_values:
            query = f"""SELECT * FROM {quote_ident(other_args.db_table)} WHERE {field}=%s AND {sql_fields}"""
            cursor.execute(query, [other_args.value, *sql_values])
        else:
            query = f"""SELECT * FROM {quote_ident(other_args.db_table)} WHERE {field}=%s"""
            cursor.execute(query, [other_args.value])
        for row in cursor:
            if row[doc_id] not in docs_found:
                docs_found[row[doc_id]] = {
                    "count": 0,
                    **{key: row[key] for key in column_names},
                }
            docs_found[row[doc_id]]["count"] += 1
    return list(docs_found.values())


@app.get("/retrieve_all_passage_pairs/")
@app.get("/text-pair-api/retrieve_all_passage_pairs/")
def retrieve_all_passage_pairs(request: Request):
    """Retrieve all passage pair metadata matching a particular query
    NOTE that this does not retrieve passages themselves"""
    with pg_connection() as conn:
        sql_fields, sql_values, other_args, column_names = parse_args(request, conn)
        filtered_columns = {
            "source_passage",
            "source_context_before",
            "source_context_after",
            "source_filename",
            "target_passage",
            "target_context_before",
            "target_context_after",
            "target_filename",
        }
        column_names = [column_name for column_name in column_names if column_name not in filtered_columns]
        query = f"SELECT * FROM {quote_ident(other_args.db_table)} WHERE {sql_fields}"
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute(query, sql_values)
        results = [{key: row[key] for key in column_names} for row in cursor]
    return results


@app.get("/count_results/")
@app.get("/text-pair-api/count_results/")
def count_results(request: Request):
    """Search alignments according to URL params"""
    with pg_connection() as conn:
        sql_fields, sql_values, other_args, _ = parse_args(request, conn)
        if sql_fields:
            query = f"SELECT COUNT(*) FROM {quote_ident(other_args.db_table)} WHERE {sql_fields}"
        else:
            query = f"SELECT COUNT(*) FROM {quote_ident(other_args.db_table)}"
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute(query, sql_values)
        result_object = {"counts": cursor.fetchone()[0]}
    return result_object


# Landing page counts, by table and oid: loading a database again recreates its tables
DATABASE_STATS = {}


@app.get("/database_stats/")
@app.get("/text-pair-api/database_stats/")
def database_stats(request: Request):
    """Counts of passage pairs, and of the passage families they form, for the landing page"""
    with pg_connection() as conn:
        table = str(request.query_params.get("db_table", ""))
        if not get_pg_type(conn, table):  # the landing page of a database without tables reads the error
            return {"error": f"No table {table}"}
        cursor = conn.cursor()
        cursor.execute("SELECT to_regclass(%s)::oid, to_regclass(%s)::oid", (table, f"{table}_groups"))
        key = (table, *cursor.fetchone())
        if key not in DATABASE_STATS:
            cursor.execute(f"SELECT count(*) FROM {quote_ident(table)}")
            stats = {"pairs_count": cursor.fetchone()[0]}
            if key[2] is not None:
                group_types = get_pg_type(conn, f"{table}_groups")
                fields = [field for field in ("author", "title") if f"source_{field}" in group_types]
                counts = ", ".join(["count(*)", *(f"count(DISTINCT source_{field})" for field in fields)])
                cursor.execute(f"SELECT {counts} FROM {quote_ident(f'{table}_groups')}")
                group_count, *field_counts = cursor.fetchone()
                stats["group_count"] = group_count
                stats.update({f"{field}_group_count": count for field, count in zip(fields, field_counts)})
            DATABASE_STATS[key] = stats
    return DATABASE_STATS[key]


@app.post("/generate_time_series/")
@app.post("/text-pair-api/generate_time_series/")
def generate_time_series(request: Request):
    """Generate a time series from search results"""
    # TODO: don't assume year is the field to use
    with pg_connection() as conn:
        sql_fields, sql_values, other_args, _ = parse_args(request, conn)
        year = quote_ident(f"{side_of(other_args.directionSelected)}_year")
        if sql_fields:
            query = f"select interval AS year, COUNT(*) FROM \
                    (SELECT floor({year}/{other_args.timeSeriesInterval})*{other_args.timeSeriesInterval} \
                    AS interval FROM {quote_ident(other_args.db_table)} WHERE {sql_fields}) t \
                    GROUP BY interval ORDER BY interval"
        else:
            query = f"select interval AS year, COUNT(*) FROM \
                    (SELECT floor({year}/{other_args.timeSeriesInterval})*{other_args.timeSeriesInterval} \
                    AS interval FROM {quote_ident(other_args.db_table)}) t \
                    GROUP BY interval ORDER BY interval"
        results = []
        total_results = 0
        next_year = None
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute(query, sql_values)
        for year, count in cursor:
            if year is None:
                continue
            if next_year is not None:
                while year > next_year:
                    results.append({"year": next_year, "count": 0})
                    next_year += other_args.timeSeriesInterval
            results.append({"year": year, "count": count})
            next_year = year + other_args.timeSeriesInterval
            total_results += count
    return {"counts": total_results, "results": results}


@app.post("/facets/")
@app.post("/text-pair-api/facets/")
def facets(request: Request):
    """Retrieve facet result"""
    with pg_connection() as conn:
        sql_fields, sql_values, other_args, column_names = parse_args(request, conn)
        if other_args.facet not in column_names:
            raise HTTPException(400, f"Unknown field {other_args.facet}")
        results = []
        total_count = 0
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        facet = quote_ident(other_args.facet)
        if sql_fields:
            query = f"SELECT {facet}, COUNT(*) FROM {quote_ident(other_args.db_table)} \
                WHERE {sql_fields} GROUP BY {facet} ORDER BY COUNT(*) DESC"
        else:
            query = f"SELECT {facet}, COUNT(*) FROM {quote_ident(other_args.db_table)} \
                GROUP BY {facet} ORDER BY COUNT(*) DESC"
        cursor.execute(query, sql_values)
        if not other_args.facet.endswith("passage_length"):  # type: ignore
            for result in cursor:
                field_name, count = result
                results.append({"field": field_name, "count": count})
                total_count += count
        else:
            counts = Counter()
            for length, count in cursor:
                if length < 26:
                    counts["1-25"] += count
                if 25 < length < 101:
                    counts["26-100"] += count
                elif 100 < length < 251:
                    counts["101-250"] += count
                elif 250 < length < 501:
                    counts["251-500"] += count
                elif 500 < length < 1001:
                    counts["501-1000"] += count
                elif 1000 < length < 3001:
                    counts["1001-3000"] += count
                elif length > 3000:
                    counts["3001-"] += count
                total_count += count
            results = [
                {"field": interval, "count": count}
                for interval, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)
            ]
    return {"facet": other_args.facet, "results": results, "total_count": total_count}


@app.get("/metadata/")
@app.get("/text-pair-api/metadata/")
def metadata(request: Request):
    """Retrieve all searchable metadata fields"""
    with pg_connection() as conn:
        _, _, metadata_field_types = parse_request(request, conn)
    return list(metadata_field_types.keys())


@app.get("/group/{group_id}")
@app.get("/group/{group_id}/")
@app.get("/text-pair-api/group/{group_id}")
@app.get("/text-pair-api/group/{group_id}/")
def get_passage_group(request: Request, group_id: int):
    """Retrieve a passage group"""
    with pg_connection() as conn:
        alignment_table = request.query_params.get("db_table", "")
        field_types = get_pg_type(conn, alignment_table)
        if not field_types:
            raise HTTPException(404, "No such database")
        group_id_type = field_types.get("group_id", "INTEGER").upper()
        groups_table = quote_ident(f"{alignment_table}_groups")
        alignment_table = quote_ident(alignment_table)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Query groups_table
        cursor.execute(f"""SELECT * FROM {groups_table} WHERE group_id=%s""", (group_id,))
        original_passage = cursor.fetchone()
        if original_passage is None:
            raise HTTPException(404, "No such passage group")
        original_passage = dict(original_passage)
        anchor_doc = original_passage["source_doc_id"]

        # Query the main alignment_table using the correct operator based on its group_id type
        if group_id_type == "ARRAY":
            query = f"SELECT * FROM {alignment_table} WHERE group_id @> ARRAY[%s]::integer[]"
        else:  # Backward compatibility for INTEGER type
            query = f"SELECT * FROM {alignment_table} WHERE group_id=%s"
        cursor.execute(query, (group_id,))
        rows = cursor.fetchall()

    reuses = occurrences(rows, original_passage)
    passage_list = []
    results = defaultdict(list)
    for passage in reuses:
        results[passage["target_year"]].append(passage)
    for key, value in results.items():
        value.sort(key=lambda x: (x["target_title"], x["target_start_byte"]), reverse=True)
        passage_list.append({"year": key, "result": value})
    passage_list.sort(key=lambda x: x["year"] or 9999)
    full_results = {"passageList": passage_list, "original_passage": original_passage}
    return full_results


# As many passage families to a page as alignments to a page of search results
SORTED_PAGE_SIZE = 50
# Family columns the sorted results do not show: the cards need the passage and its citation
FAMILY_CARD_EXCLUDED = {"source_context_before", "source_context_after", "source_ngrams"}


def family_candidates(cursor, table: str, sql_fields: str, sql_values: list) -> tuple[str, tuple]:
    """A WHERE clause and its params for the passage families in the running.

    Without filters every family is. With them, those with an alignment that matches.
    """
    if not sql_fields:
        return "", ()
    cursor.execute(f"SELECT group_id FROM {quote_ident(table)} WHERE {sql_fields}", sql_values)
    candidates: set[int] = set()
    for row in cursor:
        gid_value = row["group_id"]
        if isinstance(gid_value, list):
            candidates.update(int(gid) for gid in gid_value if gid is not None)
        elif gid_value is not None:
            candidates.add(int(gid_value))
    return "WHERE group_id = ANY(%s)", (list(candidates),)


# Cached by the oid of the table that marks the family filter tables as complete, so a rebuild invalidates it
FAMILY_FILTER_COLUMNS: dict[int, dict[str, set[str]]] = {}


def family_filter_columns(cursor, table: str) -> dict[str, set[str]] | None:
    """The fields the family filter tables can answer for, or None if the database has no such tables."""
    cursor.execute("SELECT to_regclass(%s)::oid", (f"{table}_family_filters",))
    oid = cursor.fetchone()[0]
    if oid is None:
        return None
    if oid not in FAMILY_FILTER_COLUMNS:
        names = {
            "source": f"{table}_source_meta",
            "target": f"{table}_target_meta",
            "alignments": f"{table}_family_alignments",
        }
        cursor.execute(
            """SELECT table_name, column_name FROM information_schema.columns
            WHERE table_schema = current_schema() AND table_name = ANY(%s)""",
            (list(names.values()),),
        )
        found = cursor.fetchall()
        columns = {kind: {column for name, column in found if name == names[kind]} for kind in names}
        columns["alignments"] -= {"r", "s", "t"}
        FAMILY_FILTER_COLUMNS[oid] = columns
    return FAMILY_FILTER_COLUMNS[oid]


class RankOrder(NamedTuple):
    """The order a page of families is picked in, as SQL over the family rank table `f`"""

    sort: str  # reuse or earliest
    join: str = ""
    by: str = "f.rank"


def rank_order(cursor, table: str, sort: str) -> RankOrder:
    """The order of a sort: by family rank, or by the anchor's year, from the stored place or else the groups"""
    if sort == "reuse":
        return RankOrder(sort)
    cursor.execute(
        """SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = current_schema()
        AND table_name = %s AND column_name = 'chrono_rank'""",
        (f"{table}_family_rank",),
    )
    if cursor.fetchone()[0]:
        return RankOrder(sort, by="f.chrono_rank")
    groups = quote_ident(f"{table}_groups")
    by = 'go.source_year, go."count" DESC, go.group_id'
    return RankOrder(sort, f"JOIN {groups} go ON go.group_id = f.group_id", by)


def ranked_tail(table: str, order: RankOrder) -> str:
    """The end of a statement that makes `c`, family ranks: each rank, how many there are and the best, in page order"""
    if order.sort == "reuse":
        return "SELECT r, (SELECT COUNT(*) FROM c), (SELECT MIN(r) FROM c) FROM c ORDER BY r"
    return (
        f"SELECT f.rank, (SELECT COUNT(*) FROM c), (SELECT MIN(r) FROM c) FROM c "
        f"JOIN {quote_ident(f'{table}_family_rank')} f ON f.rank = c.r {order.join} ORDER BY {order.by}"
    )


def family_filter_query(
    cursor, table: str, conditions: list[Condition], order: RankOrder = RankOrder("reuse")
) -> tuple[str, list] | None:
    """A statement picking the families with an alignment matching every condition, by rank, or None.

    Family rank is the place in `"count" DESC, group_id` order. The statement takes LIMIT and OFFSET
    as its last two params and returns each family's rank, in the order given, with how many families
    match and the best rank among them. None when a condition needs the alignments table itself.
    """
    columns = family_filter_columns(cursor, table)
    if columns is None:
        return None
    by_kind: dict[str, list[Condition]] = {"source": [], "target": [], "alignments": []}
    for condition in conditions:
        for kind, fields in columns.items():
            if condition.field in fields and (kind == "alignments" or condition.field.startswith(f"{kind}_")):
                by_kind[kind].append(condition)
                break
        else:
            return None

    # What a side's conditions match is looked up in its small table of distinct documents.
    sides = {side: join_conditions(by_kind[side]) for side in ("source", "target") if by_kind[side]}
    rank_table = quote_ident(f"{table}_family_rank")
    if len(sides) == 1 and not by_kind["alignments"]:
        side = next(iter(sides))
        side_where, side_params = sides[side]
        column, meta, families = side[0], quote_ident(f"{table}_{side}_meta"), quote_ident(f"{table}_{side}_families")
        cursor.execute(f"SELECT COALESCE(SUM(n), 0) FROM {meta} WHERE {side_where}", side_params)
        matching = cursor.fetchone()[0]
        cursor.execute(f"SELECT COALESCE(SUM(n), 0) FROM {meta}")
        if cursor.fetchone()[0] < 2 * matching:
            # Most of the documents match, so find the families left out: those with every document outside
            cursor.execute(f"SELECT id FROM {meta} WHERE (({side_where}) IS NOT TRUE)", side_params)
            return (
                f"""WITH x AS MATERIALIZED (
                    SELECT r FROM {families} WHERE {column} = ANY(%s) GROUP BY r, n HAVING COUNT(*) = n)
                SELECT f.rank, (SELECT MAX(rank) FROM {rank_table}) - (SELECT COUNT(*) FROM x),
                    (SELECT g.rank FROM {rank_table} g WHERE NOT EXISTS (SELECT 1 FROM x WHERE x.r = g.rank)
                     ORDER BY g.rank LIMIT 1)
                FROM {rank_table} f {order.join} WHERE NOT EXISTS (SELECT 1 FROM x WHERE x.r = f.rank)
                ORDER BY {order.by} LIMIT %s OFFSET %s""",
                [[row[0] for row in cursor]],
            )
        cursor.execute(f"SELECT id FROM {meta} WHERE {side_where}", side_params)
        source, where, params = families, f"{column} = ANY(%s)", [[row[0] for row in cursor]]
    else:
        # Both sides, or a field of the alignment itself: the one alignment has to match them all
        source, matches, params = quote_ident(f"{table}_family_alignments"), [], []
        for side, (side_where, side_params) in sides.items():
            cursor.execute(f"SELECT id FROM {quote_ident(f'{table}_{side}_meta')} WHERE {side_where}", side_params)
            matches.append(f"{side[0]} = ANY(%s)")
            params.append([row[0] for row in cursor])
        if by_kind["alignments"]:
            alignment_where, alignment_params = join_conditions(by_kind["alignments"])
            matches.append(alignment_where)
            params.extend(alignment_params)
        where = " AND ".join(matches)
    return (
        f"""WITH c AS MATERIALIZED (SELECT r FROM {source} WHERE {where} GROUP BY r)
        {ranked_tail(table, order)} LIMIT %s OFFSET %s""",
        params,
    )


def ranked_page(cursor, statement: str, params: list, page: int) -> tuple[int, list[int], int, int | None]:
    """A page of the families a family filter query picks: its number, their ranks, how many in all, the best rank.

    A page past the last gives the last.
    """

    def pick(start: int, size: int):
        cursor.execute(statement, (*params, size, start))
        rows = cursor.fetchall()
        return [row[0] for row in rows], (rows[0][1] if rows else 0), (rows[0][2] if rows else None)

    ranks, total, best = pick((page - 1) * SORTED_PAGE_SIZE, SORTED_PAGE_SIZE)
    if not ranks and page > 1:
        _, total, best = pick(0, 1)
        page = max(1, math.ceil(total / SORTED_PAGE_SIZE))
        ranks, _, _ = pick((page - 1) * SORTED_PAGE_SIZE, SORTED_PAGE_SIZE)
    return page, ranks, total, best


class OriginParts(NamedTuple):
    """A search split by what it asks of a passage family, with the columns the family filter tables have"""

    source: list[Condition]  # the earlier text: the anchor
    target: list[Condition]  # the later text
    aligned: list[Condition]  # the alignment, such as its passage length
    columns: dict[str, set[str]]


def field_on_side(field: str, side: str) -> str:
    """The same field of the other side, or of this one: target_year on the source side is source_year"""
    return f"{side}_{field.split('_', 1)[1]}"


def on_side(condition: Condition, side: str) -> Condition:
    """A condition on a source_ or target_ field, made to apply to the same field of `side`"""
    field = field_on_side(condition.field, side)
    return Condition(field, condition.sql.replace(quote_ident(condition.field), quote_ident(field)), condition.values)


def has_family_anchors(cursor, table: str) -> bool:
    """Whether the family filter tables say which document each family's anchor is (add_family_anchors)"""
    cursor.execute(
        """SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = current_schema()
        AND table_name = %s AND column_name IN ('anchor_s', 'anchor_t')""",
        (f"{table}_family_rank",),
    )
    return cursor.fetchone()[0] == 2


def origin_parts(cursor, table: str, conditions: list[Condition], groups_columns) -> OriginParts | None:
    """The conditions sorted by what they ask of a family, or None if the family filter tables cannot answer them.

    The source side filters the anchor as the groups table has it, so passage text counts too, and the target
    side has to be found on both sides of the alignments, as a later occurrence is.
    """
    columns = family_filter_columns(cursor, table)
    if columns is None or not has_family_anchors(cursor, table):
        return None
    parts = OriginParts([], [], [], columns)
    for condition in conditions:
        field = condition.field
        if field in columns["alignments"]:
            parts.aligned.append(condition)
        elif field.startswith("source_") and field in groups_columns:
            parts.source.append(condition)
        elif field.startswith("target_") and field in columns["target"]:
            if field_on_side(field, "source") not in columns["source"]:
                return None
            parts.target.append(condition)
        else:
            return None
    return parts


def matching_ids(cursor, table: str, side: str, conditions: list[Condition]) -> list[int]:
    """The ids of the documents on a side of the alignments that meet the conditions"""
    where, params = join_conditions(conditions)
    cursor.execute(f"SELECT id FROM {quote_ident(f'{table}_{side}_meta')} WHERE {where}", params)
    return [row[0] for row in cursor]


def found_again_columns(cursor, table: str) -> set[str]:
    """The columns of `_found_again`, or none where the facet tables are missing or incomplete (load_facet_tables)"""
    cursor.execute("SELECT to_regclass(%s) IS NOT NULL", (f"{table}_{FACET_TABLES[-1]}",))
    return set(relation_columns(cursor, f"{table}_found_again")) if cursor.fetchone()[0] else set()


def found_again_rows(cursor, table: str, parts: OriginParts, found: set[str]) -> tuple[str, str, list]:
    """A CTE, or none, and a FROM clause for `x`, the rows of `_found_again` that meet a search on the later side.

    These are the documents found again that meet the later-side conditions, in the families whose anchor meets
    the earlier-side ones. Conditions on the anchor's year alone are read from the rows, others from the groups.
    """
    rows = quote_ident(f"{table}_found_again")
    source = matching_ids(cursor, table, "source", [on_side(c, "source") for c in parts.target])
    ids = source + [-target for target in matching_ids(cursor, table, "target", parts.target)]
    if not parts.source:
        return "", f"{rows} x WHERE x.u = ANY(%s)", [ids]
    where, params = join_conditions(parts.source)
    if "source_year" in found and all(c.field == "source_year" for c in parts.source):
        return "", f"{rows} x WHERE x.u = ANY(%s) AND {where}", [ids, *params]
    groups, rank = quote_ident(f"{table}_groups"), quote_ident(f"{table}_family_rank")
    return (
        f"s AS MATERIALIZED (SELECT f.rank AS r FROM {groups} g JOIN {rank} f USING (group_id) WHERE {where})",
        f"s JOIN {rows} x ON x.r = s.r WHERE x.u = ANY(%s)",
        [*params, ids],
    )


def origin_ranks(cursor, table: str, parts: OriginParts) -> tuple[str, list]:
    """A WITH clause that makes `c`, the ranks of the families that meet a search read by where they were first found.

    The anchor meets the source conditions, and some other document of the family, on either side of the
    alignments, the target conditions. Where an alignment condition is set too, that alignment has the later
    document at one end. Without one, the facet tables answer it where the database has them.
    """
    if parts.target and not parts.aligned and (found := found_again_columns(cursor, table)):
        cte, rows, params = found_again_rows(cursor, table, parts, found)
        return f"WITH {cte + ', ' if cte else ''}c AS MATERIALIZED (SELECT DISTINCT x.r FROM {rows})", params
    rank_table = quote_ident(f"{table}_family_rank")
    alignments = quote_ident(f"{table}_family_alignments")
    a_where, a_params = join_conditions(parts.aligned) if parts.aligned else ("TRUE", [])
    prefix = ""
    params: list = []
    # The families that meet the source conditions come first, and only their later occurrences are looked up
    anchored, anchor = "{rows} x JOIN " + rank_table + " f ON f.rank = x.r", "f"
    if parts.source:
        where, params = join_conditions(parts.source)
        groups_table = quote_ident(f"{table}_groups")
        prefix = (
            f"s AS MATERIALIZED (SELECT f.rank AS r, f.anchor_s, f.anchor_t FROM {groups_table} g "
            f"JOIN {rank_table} f USING (group_id) WHERE {where}), "
        )
        anchored, anchor = "s JOIN {rows} x ON x.r = s.r", "s"
    if parts.target:
        branches = []
        for side in ("target", "source"):
            ids = matching_ids(cursor, table, side, [on_side(c, side) for c in parts.target])
            column = side[0]
            rows = alignments if parts.aligned else quote_ident(f"{table}_{side}_families")
            branches.append(
                f"SELECT DISTINCT x.r FROM {anchored.format(rows=rows)} WHERE x.{column} = ANY(%s) "
                f"AND x.{column} IS DISTINCT FROM {anchor}.anchor_{column} AND {a_where}"
            )
            params += [ids, *a_params]
        body = " UNION ".join(branches)
    elif parts.aligned:
        body = f"SELECT DISTINCT x.r FROM {anchored.format(rows=alignments)} WHERE {a_where}"
        params += a_params
    else:
        body = "SELECT r FROM s"
    return f"WITH {prefix}c AS MATERIALIZED ({body})", params


def origin_family_query(cursor, table: str, parts: OriginParts, order: RankOrder) -> tuple[str, list]:
    """A statement like family_filter_query's, for the families origin_ranks picks"""
    ranks, params = origin_ranks(cursor, table, parts)
    return f"{ranks} {ranked_tail(table, order)} LIMIT %s OFFSET %s", params


def found_again(cursor, table: str, parts: OriginParts, ranks: list[int]) -> dict[int, dict]:
    """For each rank, the documents that meet the later-side conditions, as many as there are and the first two.

    The documents are the ones origin_family_query asks for, so every family on the page has some.
    """
    if not parts.target or not ranks:
        return {}
    columns = parts.columns
    fields = sorted(columns["target"] - {"id", "n"})
    rank_table = quote_ident(f"{table}_family_rank")
    alignments = quote_ident(f"{table}_family_alignments")
    a_where, a_params = join_conditions(parts.aligned) if parts.aligned else ("TRUE", [])
    branches, params = [], []
    for side in ("target", "source"):
        column = side[0]
        meta = quote_ident(f"{table}_{side}_meta")
        select = ", ".join(
            f"m.{quote_ident(field_on_side(field, side))} AS {quote_ident(field)}"
            if field_on_side(field, side) in columns[side]
            else f"NULL AS {quote_ident(field)}"
            for field in fields
        )
        rows = alignments if parts.aligned else quote_ident(f"{table}_{side}_families")
        branches.append(
            f"SELECT x.r, {select} FROM {rows} x JOIN {rank_table} f ON f.rank = x.r "
            f"JOIN {meta} m ON m.id = x.{column} WHERE x.r = ANY(%s) AND x.{column} = ANY(%s) "
            f"AND x.{column} IS DISTINCT FROM f.anchor_{column} AND {a_where}"
        )
        params += [ranks, matching_ids(cursor, table, side, [on_side(c, side) for c in parts.target]), *a_params]
    cursor.execute(" UNION ALL ".join(branches), params)
    documents: dict[int, dict[str, dict]] = defaultdict(dict)
    for row in cursor:
        documents[row["r"]][row["target_doc_id"]] = {field: row[field] for field in fields}

    def earliest(document: dict):
        year = document.get("target_year")
        year = year if isinstance(year, int) else None
        return (year is None, year or 0, str(document.get("target_title") or ""))

    return {
        rank: {"matching": len(found), "documents": sorted(found.values(), key=earliest)[:2]}
        for rank, found in documents.items()
    }


def origin_groups_where(conditions: list[Condition], groups_columns) -> tuple[str, list] | None:
    """A WHERE clause on the groups table for a search made only of conditions on the earlier text, or None.

    These are conditions on the anchor, which the groups table holds, so no family tables are needed.
    """
    if all(c.field.startswith("source_") and c.field in groups_columns for c in conditions):
        where, params = join_conditions(conditions)
        return f"WHERE {where}", params
    return None


class OriginFamilies(NamedTuple):
    """The families a search leaves in the running, as a statement by rank or a WHERE clause on the groups table"""

    applied: str  # the rule they were picked by: origin, or any where the search fell back to it
    ranked: tuple[str, list] | None = None
    where: tuple[str, list | tuple] | None = None
    parts: OriginParts | None = None  # what the ranked statement was made from, for the evidence
    selective: bool = False  # whether `where` may leave few of the groups


def origin_families(
    cursor, table: str, conditions: list[Condition], groups_columns, order: RankOrder = RankOrder("reuse")
) -> OriginFamilies:
    """Pick the families by where they were first found, or by any match if this database's tables cannot say"""
    if where := origin_groups_where(conditions, groups_columns):
        return OriginFamilies("origin", where=where, selective=True)
    if parts := origin_parts(cursor, table, conditions, groups_columns):
        # Parallel plans of these statements sometimes never finish
        cursor.execute("SET LOCAL max_parallel_workers_per_gather = 0")
        return OriginFamilies("origin", ranked=origin_family_query(cursor, table, parts, order), parts=parts)
    return OriginFamilies("any", ranked=family_filter_query(cursor, table, conditions, order))


@app.get("/sorted_results/")
@app.get("/text-pair-api/sorted_results/")
def get_sorted_results(request: Request):
    """One page of passage families, the most reused first or, with `sort=earliest`, the earliest first.

    The earlier text's fields are read as those of the family's anchor, its first appearance, and the later
    text's as those of another occurrence. A search this database's tables cannot read so is answered by any
    match of the passage, and the response says so with `scope`. Earliest first is by the anchor's year,
    undated last, then most reused; a sort that is not one of the two is the default, and `sort` says which
    was applied.
    """
    with pg_connection() as conn:
        conditions, other_args, _ = parse_conditions(request, conn)
        table = other_args.db_table
        groups_table = quote_ident(f"{table}_groups")
        groups_columns = get_pg_type(conn, f"{table}_groups")
        columns = ", ".join(f'g."{column}"' for column in groups_columns if column not in FAMILY_CARD_EXCLUDED)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        sort = "earliest" if other_args.sort == "earliest" and "source_year" in groups_columns else "reuse"
        order = rank_order(cursor, table, sort)
        served = (
            origin_families(cursor, table, conditions, groups_columns, order) if conditions else OriginFamilies("origin")
        )
        ranked, where, parts = served.ranked, served.where, served.parts

        evidence = {}
        if ranked is not None:
            page, ranks, total, best = ranked_page(cursor, *ranked, max(1, other_args.page))  # type: ignore
            start_position = (page - 1) * SORTED_PAGE_SIZE
            max_count = 0
            if best is not None:  # the best-ranked family has the highest count
                cursor.execute(
                    f'SELECT g."count" FROM {quote_ident(f"{table}_family_rank")} f JOIN {groups_table} g USING (group_id) WHERE f.rank = %s',
                    (best,),
                )
                max_count = cursor.fetchone()[0]
            if parts is not None:
                evidence = found_again(cursor, table, parts, ranks)
            cursor.execute(
                f"""SELECT {columns}, f.rank AS family_rank FROM {groups_table} g JOIN {quote_ident(f"{table}_family_rank")} f USING (group_id)
                WHERE f.rank = ANY(%s) ORDER BY f.rank""",
                (ranks,),
            )
            place = {rank: place for place, rank in enumerate(ranks)}
        else:
            if where is None:
                sql_fields, sql_values = join_conditions(conditions)
                where = family_candidates(cursor, table, sql_fields, sql_values)

            where, params = where
            cursor.execute(f'SELECT COUNT(*), MAX("count") FROM {groups_table} {where}', params)
            total, max_count = cursor.fetchone()
            page = min(max(1, other_args.page), max(1, math.ceil(total / SORTED_PAGE_SIZE)))  # type: ignore
            start_position = (page - 1) * SORTED_PAGE_SIZE
            # The page is picked from the index alone, so skipping to a deep one reads no rows
            prefix, source = "", f"{groups_table} {where}"
            by, kept = '"count" DESC, group_id', '"count"'
            if sort == "earliest":
                by, kept = f"source_year, {by}", f"{kept}, source_year"
            if served.selective:
                # The index is walked until the page is full, which takes the whole table for a rare author
                cursor.execute("SELECT reltuples::bigint FROM pg_class WHERE oid = %s::regclass", (groups_table,))
                if (start_position + SORTED_PAGE_SIZE) * cursor.fetchone()[0] > total * total:
                    prefix = f'WITH m AS MATERIALIZED (SELECT group_id, {kept} FROM {groups_table} {where}) '
                    source = "m"
            cursor.execute(
                f"""{prefix}SELECT {columns} FROM {groups_table} g
                JOIN (SELECT group_id FROM {source} ORDER BY {by} LIMIT %s OFFSET %s) p
                USING (group_id) ORDER BY {", ".join(f"g.{key}" for key in by.split(", "))}""",
                (*params, SORTED_PAGE_SIZE, start_position),
            )
        groups = [dict(row) for row in cursor]
        if ranked is not None:
            groups.sort(key=lambda group: place[group["family_rank"]])  # the earliest first is not in rank order
            for group in groups:
                rank = group.pop("family_rank")
                if rank in evidence:
                    group["found_again"] = evidence[rank]
        results = {
            "groups": groups,
            "page": page,
            "page_size": SORTED_PAGE_SIZE,
            "start_position": start_position,
            "total_count": total,
            "max_count": max_count or 0,
            "scope": served.applied,
            "sort": sort,
        }
    return results


FACET_VALUES = 100  # values a facet returns unless `limit` says otherwise, as many as the panel shows
LENGTH_LABELS = [label for _, label in LENGTH_RANGES] + [OPEN_LENGTH_RANGE]  # by range number


def length_range(column: str) -> str:
    """SQL that puts a passage length into the ranges of the pairs' facets"""
    cases = " ".join(f"WHEN {column} <= {top} THEN '{label}'" for top, label in LENGTH_RANGES)
    return f"CASE {cases} ELSE '{OPEN_LENGTH_RANGE}' END"


def length_label(number: str) -> str:
    """SQL that names a range of the pairs' facets by its number"""
    cases = " ".join(f"WHEN {n} THEN '{label}'" for n, label in enumerate(LENGTH_LABELS))
    return f"CASE {number} {cases} END"


def facet_error(facet: str, reason: str) -> HTTPException:
    return HTTPException(400, f"Passages cannot be counted by {facet}: {reason}")


def facet_ranks(cursor, table: str, served: OriginFamilies) -> tuple[str, list]:
    """CTEs that make `ca`, each family a search leaves in the running with its anchor's document ids.

    None for a search that leaves them all in the running, which the family rank table then stands for.
    """
    rank = quote_ident(f"{table}_family_rank")
    if served.parts is not None:
        ranks, params = origin_ranks(cursor, table, served.parts)
        return (
            f"{ranks.removeprefix('WITH ')}, "
            f"ca AS MATERIALIZED (SELECT f.rank AS r, f.anchor_s, f.anchor_t FROM c JOIN {rank} f ON f.rank = c.r)",
            params,
        )
    if served.where is None or not served.where[0]:
        return "", []
    where, params = served.where
    groups = quote_ident(f"{table}_groups")
    return (
        f"ca AS MATERIALIZED (SELECT f.rank AS r, f.anchor_s, f.anchor_t "
        f"FROM {groups} g JOIN {rank} f USING (group_id) {where})",
        params,
    )


def facet_values(table: str, name: str) -> str:
    """CTEs that number the distinct values of a document field, as arrays by the id of the documents of each side.

    The families are counted by these numbers, which are small integers, and named at the end.
    """
    return f"""vals AS MATERIALIZED (SELECT side, id, v, (DENSE_RANK() OVER (ORDER BY v))::int AS vid FROM (
            SELECT 's' AS side, id, {quote_ident(f"source_{name}")} AS v FROM {quote_ident(f"{table}_source_meta")}
            UNION ALL SELECT 't', id, {quote_ident(f"target_{name}")} FROM {quote_ident(f"{table}_target_meta")}) d),
        vs AS MATERIALIZED (SELECT array_agg(vid ORDER BY id) AS a FROM vals WHERE side = 's'),
        vt AS MATERIALIZED (SELECT array_agg(vid ORDER BY id) AS a FROM vals WHERE side = 't')"""


def anchor_facet(table: str, name: str, restrict: bool) -> str:
    """A statement that counts families by a document field of their anchor, as the family tables hold it.

    An anchor on the source side of the alignments has its field in the source documents, or else in the target ones.
    """
    rank = quote_ident(f"{table}_family_rank")
    source, target = quote_ident(f"source_{name}"), quote_ident(f"target_{name}")
    return f"""SELECT v, SUM(n)::bigint AS n FROM (
        SELECT CASE WHEN a.anchor_s IS NOT NULL THEN sm.{source} ELSE tm.{target} END AS v, a.n
        FROM (SELECT anchor_s, anchor_t, COUNT(*) AS n FROM {"ca" if restrict else rank} GROUP BY 1, 2) a
        LEFT JOIN {quote_ident(f"{table}_source_meta")} sm ON sm.id = a.anchor_s
        LEFT JOIN {quote_ident(f"{table}_target_meta")} tm ON tm.id = a.anchor_t) q GROUP BY v"""


def found_again_facet(cursor, table: str, parts: OriginParts, name: str, restrict: bool) -> tuple[str, str, list]:
    """A statement that counts families once for each value of a document field among the documents found again.

    These are the family's documents other than the anchor's, on either side of the alignments, that meet the
    later-side conditions, and the ones at an end of an alignment meeting the alignment conditions.
    """
    rank = quote_ident(f"{table}_family_rank")
    a_where, a_params = join_conditions(parts.aligned) if parts.aligned else ("TRUE", [])
    branches, params = [], []
    for side in ("target", "source"):
        column = side[0]
        rows = quote_ident(f"{table}_family_alignments") if parts.aligned else quote_ident(f"{table}_{side}_families")
        match = f" AND x.{column} = ANY(%s)" if parts.target else ""
        joined = "ca ON ca.r = x.r" if restrict else f"{rank} ca ON ca.rank = x.r"
        branches.append(
            f"SELECT x.r, (SELECT a FROM v{column})[x.{column}] AS vid FROM {rows} x JOIN {joined} "
            f"WHERE x.{column} IS DISTINCT FROM ca.anchor_{column}{match} AND {a_where}"
        )
        if parts.target:
            params.append(matching_ids(cursor, table, side, [on_side(c, side) for c in parts.target]))
        params += a_params
    return (
        f"{facet_values(table, name)}, p AS ({' UNION ALL '.join(branches)})",
        """SELECT d.v, g.n FROM (SELECT vid, COUNT(*)::bigint AS n FROM (SELECT DISTINCT r, vid FROM p) q GROUP BY vid) g
        JOIN (SELECT DISTINCT vid, v FROM vals) d USING (vid)""",
        params,
    )


def passage_length_facet(cursor, table: str, parts: OriginParts, side: str, restrict: bool) -> tuple[str, str, list]:
    """A statement that counts families once for each length range of their alignments' passages on a side.

    The alignments are those that meet the alignment conditions, and where there are later-side conditions,
    those with a document found again at an end, as for the other facets.
    """
    a_where, a_params = join_conditions(parts.aligned) if parts.aligned else ("TRUE", [])
    length = f"x.{side}_passage_length"
    found, found_params = "TRUE", []
    if parts.target:
        found = (
            "(x.t = ANY(%s) AND x.t IS DISTINCT FROM ca.anchor_t "
            "OR x.s = ANY(%s) AND x.s IS DISTINCT FROM ca.anchor_s)"
        )
        for other in ("target", "source"):
            found_params.append(matching_ids(cursor, table, other, [on_side(c, other) for c in parts.target]))
    rank = quote_ident(f"{table}_family_rank")
    joined = "ca ON ca.r = x.r" if restrict else f"{rank} ca ON ca.rank = x.r"
    return (
        "",
        f"""SELECT v, COUNT(*)::bigint AS n FROM (
        SELECT DISTINCT x.r, {length_range(length)} AS v FROM {quote_ident(f"{table}_family_alignments")} x
        {f"JOIN {joined}" if parts.target or restrict else ""}
        WHERE {length} IS NOT NULL AND {a_where} AND {found}) u GROUP BY v""",
        [*a_params, *found_params],
    )


def stored_facet(
    cursor, table: str, conditions: list[Condition], served: OriginFamilies, facet: str, found: set[str], limit: int
) -> tuple[str, list] | None:
    """A statement like passage_facets' that counts from the facet tables (load_facet_tables), or None.

    A search with no filters reads the counts of all the families, and one on the anchor's year alone those by
    year. One on the later side counts the rows of the documents found again that it matches, which hold their
    anchor's values too, and another the rows of the families it leaves, for a later-side field.
    """
    side, _, name = facet.partition("_")
    length = name == "passage_length"
    anchored = side == "source" and not length
    column = f"{side}_lengths" if length else f"anchor_{name}" if anchored else f"v_{name}"
    parts = served.parts
    if column not in found or (parts is not None and (parts.aligned or not parts.target)):
        return None
    by_year = served.where is not None and "source_year" in found and all(c.field == "source_year" for c in conditions)
    values, years = quote_ident(f"{table}_facet_values"), quote_ident(f"{table}_facet_years")
    named = length_label("v") if length else "v"
    cte, params, counted_params = "", [], []
    if served.where is None and parts is None:
        families = f"SELECT families::bigint FROM {quote_ident(f'{table}_family_filters')}"
        counted, counted_params = f"SELECT {named} AS v, n::bigint AS n FROM {values} WHERE facet = %s", [facet]
    elif by_year:
        where, where_params = join_conditions(conditions)
        families, params = f"SELECT COUNT(*)::bigint FROM {quote_ident(f'{table}_groups')} WHERE {where}", where_params
        counted = f"""SELECT {named} AS v, n FROM (SELECT v, SUM(n)::bigint AS n FROM {years}
            WHERE facet = %s AND {where} GROUP BY v) y"""
        counted_params = [facet, *where_params]
    elif parts is not None:
        cte, rows, params = found_again_rows(cursor, table, parts, found)
        cte += ", " if cte else ""
        if length:  # each family once, with the ranges of its alignments that end in a document found again
            cte += f"p AS MATERIALIZED (SELECT x.r, bit_or(x.{column}) AS m FROM {rows} GROUP BY x.r)"
            families = "SELECT COUNT(*)::bigint FROM p"
            ranges = ", ".join(f"({1 << n}, '{label}')" for n, label in enumerate(LENGTH_LABELS))
            counted = f"""SELECT l.v, COUNT(*)::bigint AS n FROM p
                JOIN (VALUES {ranges}) l(bit, v) ON p.m & l.bit <> 0 GROUP BY l.v"""
        else:
            cte += f"p AS MATERIALIZED (SELECT DISTINCT x.r, x.{column} AS v FROM {rows})"
            families = "SELECT COUNT(*)::bigint FROM (SELECT DISTINCT r FROM p) q"
            counted = "SELECT v, COUNT(*)::bigint AS n FROM p GROUP BY v"
    elif not length and not anchored:
        cte, params = facet_ranks(cursor, table, served)
        families = "SELECT COUNT(*)::bigint FROM ca"
        counted = f"""SELECT v, COUNT(*)::bigint AS n FROM (SELECT DISTINCT x.r, x.{column} AS v
            FROM ca JOIN {quote_ident(f"{table}_found_again")} x ON x.r = ca.r) q GROUP BY v"""
    else:
        return None
    value, reads, read_params = "g.v", "", []
    if not length:  # the values are numbered in their order, and read from a document that has them
        source, target = quote_ident(f"source_{name}"), quote_ident(f"target_{name}")
        value = f"CASE WHEN fv.doc > 0 THEN sm.{source} ELSE tm.{target} END"
        reads = f"""LEFT JOIN {values} fv ON fv.facet = %s AND fv.v = g.v
            LEFT JOIN {quote_ident(f"{table}_source_meta")} sm ON sm.id = fv.doc
            LEFT JOIN {quote_ident(f"{table}_target_meta")} tm ON tm.id = -fv.doc"""
        read_params = [facet]
    statement = f"""{"WITH " + cte if cte else ""} SELECT t.families, {value}, g.n, g.total, g.kinds FROM
        ({families}) t(families) LEFT JOIN
        (SELECT v, n, (SUM(n) OVER ())::bigint AS total, (COUNT(*) OVER ())::bigint AS kinds
         FROM ({counted}) q ORDER BY n DESC, v LIMIT %s) g ON TRUE {reads} ORDER BY g.n DESC NULLS LAST, g.v"""
    return statement, [*params, *counted_params, limit, *read_params]


@app.post("/passage_facets/")
@app.post("/text-pair-api/passage_facets/")
@app.get("/passage_facets/")
@app.get("/text-pair-api/passage_facets/")
def passage_facets(request: Request):
    """Count passage families by the values of a field, as /facets/ counts alignments, under the sorted view's rule.

    A source_ field is the family's anchor's, where the passage was first found, so each family counts once.
    A target_ field is that of the documents it was found again in, and a family counts once for each value.
    The passage length of a side is counted over the family's alignments, in the ranges of the pairs' facets.
    The text of a passage is not a facet: every passage is its own value. A database whose tables cannot
    read the later side, or a search they cannot answer, is counted by any match of the passage for the
    source_ fields (`scope` is "any"), and is an error for the others.
    """
    with pg_connection() as conn:
        conditions, other_args, field_types = parse_conditions(request, conn)
        table, facet = other_args.db_table, other_args.facet
        if facet not in field_types:
            raise HTTPException(400, f"Unknown field {facet}")
        side, _, name = facet.partition("_")
        if side not in ("source", "target") or not name:
            raise facet_error(facet, "it is not a field of the earlier or the later occurrence")
        if name == "passage":
            raise facet_error(facet, "each passage is its own value")
        try:
            values = min(max(1, int(other_args.limit or FACET_VALUES)), 1000)
        except ValueError:
            values = FACET_VALUES
        groups_columns = get_pg_type(conn, f"{table}_groups")
        if not groups_columns:
            raise facet_error(facet, "this database has no passage families")
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        served = origin_families(cursor, table, conditions, groups_columns) if conditions else OriginFamilies("origin")
        columns = family_filter_columns(cursor, table) if has_family_anchors(cursor, table) else None
        length = name == "passage_length"
        ranked = columns is not None and served.applied == "origin"
        if ranked and not length and not all(f"{other}_{name}" in columns[other] for other in ("source", "target")):
            raise facet_error(facet, "it is not a field of the documents of both sides")
        if ranked and length and f"{side}_passage_length" not in columns["alignments"]:
            raise facet_error(facet, "the family tables do not have the lengths of this database's passages")

        if ranked:
            cursor.execute("SET LOCAL max_parallel_workers_per_gather = 0")
            cursor.execute("SET LOCAL work_mem = '256MB'")  # the counts are hashed in memory, millions of rows of them
            found = found_again_columns(cursor, table)
            stored = stored_facet(cursor, table, conditions, served, facet, found, values)
            if stored is not None:
                statement, params = stored
            else:
                ranks, params = facet_ranks(cursor, table, served)
                restrict = bool(ranks)
                parts = served.parts or OriginParts([], [], [], columns)
                if length:
                    ctes, counted, counted_params = passage_length_facet(cursor, table, parts, side, restrict)
                elif side == "source":
                    ctes, counted, counted_params = "", anchor_facet(table, name, restrict), []
                else:
                    ctes, counted, counted_params = found_again_facet(cursor, table, parts, name, restrict)
                families = "ca" if restrict else quote_ident(f"{table}_family_rank")
                ctes = ", ".join(cte for cte in (ranks, ctes) if cte)
                statement = f"""{"WITH " + ctes if ctes else ""} SELECT t.families, g.v, g.n, g.total, g.kinds FROM
                    (SELECT COUNT(*)::bigint AS families FROM {families}) t LEFT JOIN
                    (SELECT v, n, (SUM(n) OVER ())::bigint AS total, (COUNT(*) OVER ())::bigint AS kinds
                     FROM ({counted}) q ORDER BY n DESC, v LIMIT %s) g ON TRUE ORDER BY g.n DESC NULLS LAST, g.v"""
                params = [*params, *counted_params, values]
            cursor.execute(statement, params)
            rows = cursor.fetchall()
            applied = "origin"
        else:
            if side == "target" or length or facet not in groups_columns:
                raise facet_error(facet, "this search or database cannot say where passages were found again")
            if served.applied == "any":
                where, params = family_candidates(cursor, table, *join_conditions(conditions))
            else:
                where, params = served.where or ("", [])
            cursor.execute(
                f"""SELECT SUM(n) OVER ()::bigint, v, n, SUM(n) OVER ()::bigint, COUNT(*) OVER ()::bigint
                FROM (SELECT {quote_ident(facet)} AS v, COUNT(*)::bigint AS n
                      FROM {quote_ident(f"{table}_groups")} {where} GROUP BY 1) q
                ORDER BY n DESC, v LIMIT %s""",
                [*params, values],
            )
            rows = cursor.fetchall()
            applied = served.applied
        results = [{"field": row[1], "count": row[2]} for row in rows if row[2] is not None]
        return {
            "facet": facet,
            "results": results,
            "total_count": rows[0][3] or 0 if rows else 0,
            "families": rows[0][0] or 0 if rows else 0,
            "values": rows[0][4] or 0 if rows else 0,
            "counted_once": side == "source" and not length,
            "scope": applied,
        }


def merge_passages(rows: list, direction: str) -> list[dict]:
    """The passages of a document on one side of its alignments, overlapping ones merged."""
    merged = []
    for row in sorted(rows, key=lambda row: row[f"{direction}_start_byte"]):
        start, end = row[f"{direction}_start_byte"], row[f"{direction}_end_byte"]
        if merged and start <= merged[-1]["end_byte"]:
            merged[-1]["end_byte"] = max(merged[-1]["end_byte"], end)
        else:
            merged.append({"start_byte": start, "end_byte": end})
    return merged


def load_divisions(philo_path: str, doc_id: str) -> list[dict]:
    """The div1 sections of a document, in order, as PhiloLogic's toms.db has them."""
    conn = sqlite3.connect(f"file:{os.path.join(philo_path, 'data', 'toms.db')}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT philo_id, head, CAST(start_byte AS INT), CAST(end_byte AS INT) FROM toms "
            "WHERE philo_type = 'div1' AND philo_id LIKE ? ORDER BY CAST(start_byte AS INT)",
            (f"{doc_id.split('_')[0]} %",),
        ).fetchall()
    except sqlite3.OperationalError:  # no head column: a database of documents only
        rows = []
    finally:
        conn.close()
    return [
        {"philo_id": "_".join(philo_id.split()[:2]), "head": (head or "").strip(), "start_byte": start, "end_byte": end}
        for philo_id, head, start, end in rows
    ]


def clip_passages(passages: list[dict], division: dict) -> list[tuple[int, dict]]:
    """The passages overlapping a section, with their global numbers, clipped to its bounds."""
    clipped = []
    for number, passage in enumerate(passages):
        if passage["end_byte"] > division["start_byte"] and passage["start_byte"] < division["end_byte"]:
            start = max(passage["start_byte"], division["start_byte"])
            end = min(passage["end_byte"], division["end_byte"])
            clipped.append((number, {"start_byte": start, "end_byte": end}))
    return clipped


def number_passages(text: str, clipped: list[tuple[int, dict]], passages: list[dict]) -> str:
    """Renumbers the markers PhiloLogic made, counting from 0 in the section, to the passages' global numbers,
    and gives them their whole offsets."""
    numbers = [number for number, _ in clipped]

    def marker(match):
        number = numbers[int(match["n"])]
        passage = passages[number]
        return f'{match["head"]}data-offsets="{passage["start_byte"]}-{passage["end_byte"]}" n="{number}"'

    def renumber(match):
        number = numbers[int(match["n"])]
        return f'{match["head"]}{number}{match["tail"]}{number}"'

    text = re.sub(r'(?P<head><span class="passage-marker" )data-offsets="[\d-]+" n="(?P<n>\d+)"', marker, text)
    text = re.sub(r'(?P<head><span n=")(?P<n>\d+)(?P<tail>" id="end-passage-)\d+"', renumber, text)
    return re.sub(r'(?P<head><span class="passage-)\d+(?P<tail>" n=")(?P<n>\d+)"', renumber, text)


@app.get("/text_view/")
@app.get("/text-pair-api/text_view/")
def text_view(request: Request):
    """A document's full text, marking the passages it shares with other texts on one side of the alignments"""
    with pg_connection() as conn:
        _, other_args, _ = parse_request(request, conn)
        access_control = check_access_control(request)
        if access_control is False:  # TODO check if user has access to this database
            return {"error": "You do not have access to this database"}
        table, direction = str(other_args.db_table), side_of(other_args.directionSelected)
        other = "target" if direction == "source" else "source"
        db_dir = db_directory(table)
        config = load_app_config(db_dir)
        philo_path = text_data_path(db_dir, config, direction)
        if philo_path is None:
            return {"error": "The full text of this document is no longer available."}
        other_path = text_data_path(db_dir, config, other)
        # When both sides are the same texts, a document has passages on each, under the same doc_id
        same_texts = other_path is not None and os.path.realpath(other_path) == os.path.realpath(philo_path)

        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        # The text object aligned, which a VSA passage's philo_id, that of its first sentence, is within
        doc_id = str(other_args.doc_id)
        if not doc_id:
            cursor.execute(f"SELECT {direction}_doc_id FROM {quote_ident(table)} WHERE {direction}_philo_id = %s LIMIT 1", (other_args.philo_id,))
            doc_id = (cursor.fetchone() or {}).get(f"{direction}_doc_id", "")
        if same_texts:
            cursor.execute(f"SELECT * FROM {quote_ident(table)} WHERE source_doc_id = %s OR target_doc_id = %s", (doc_id, doc_id))
        else:
            cursor.execute(f"SELECT * FROM {quote_ident(table)} WHERE {direction}_doc_id = %s", (doc_id,))
        rows = cursor.fetchall()
    if not rows:
        return {"error": "This text shares no passages with others."}
    sides = {side: [row for row in rows if row[f"{side}_doc_id"] == doc_id] for side in (direction, other)}
    passages = merge_passages(sides[direction], direction)
    passage_counts = {direction: len(passages)}
    if same_texts:
        passage_counts[other] = len(merge_passages(sides[other], other))

    # The other texts sharing each passage, once each, with the fields of their citation
    starts = [passage["start_byte"] for passage in passages]
    fields = [f"{other}_doc_id", f"{other}_philo_id", *(cite["field"] for cite in config.get(f"{other}Citation", []))]
    other_texts = [{} for _ in passages]
    for row in sides[direction]:
        passage = bisect.bisect_right(starts, row[f"{direction}_start_byte"]) - 1
        other_texts[passage].setdefault(row[f"{other}_doc_id"], {field: row.get(field) for field in fields})

    if sides[direction]:
        metadata = sides[direction][0]
    else:  # no passages on this side: the document's own fields, from the other
        own_fields = sides[other][0].items()
        metadata = {direction + key[len(other) :]: value for key, value in own_fields if key.startswith(f"{other}_")}

    philo_db = DB(f"{philo_path}/data", width=7)
    # Web app builds older than section navigation expect the whole text
    divisions = load_divisions(philo_path, doc_id) if "by_section" in request.query_params else []
    division = None
    shown_passages = list(enumerate(passages))
    object_id = " ".join((doc_id.split("_") + ["0"] * 7)[:7])
    if divisions:
        division_starts = [division["start_byte"] for division in divisions]
        try:
            start_byte = int(request.query_params["start_byte"])
        except (KeyError, ValueError):
            start_byte = None
        wanted = request.query_params.get("div1_id")
        index = next((i for i, div in enumerate(divisions) if div["philo_id"] == wanted), None)
        if index is None and start_byte is not None:
            index = max(bisect.bisect_right(division_starts, start_byte) - 1, 0)
        division = divisions[index or 0]
        division["index"] = index or 0
        shown_passages = clip_passages(passages, division)
        object_id = " ".join(division["philo_id"].split("_") + ["0"] * 5)
    philo_request = PHILO_REQUEST("", "", "", [passage for _, passage in shown_passages])
    philo_text_object, _ = get_text_obj(
        philo_db[object_id],
        PHILO_CONFIG(philo_path, "", "", ""),
        philo_request,
        philo_db.locals["token_regex"],
    )
    if division:
        philo_text_object = number_passages(philo_text_object, shown_passages, passages)

    # Find passage number in passage pairs for autoscroll
    passage_number = 0
    try:
        start_byte = int(request.query_params["start_byte"])
        for i, passage in enumerate(passages):
            if passage["start_byte"] <= start_byte <= passage["end_byte"]:
                passage_number = i
                break
    except (KeyError, ValueError):
        pass
    result = {
        "text": philo_text_object,
        "metadata": metadata,
        "direction": direction,
        "doc_id": doc_id,
        "passage_number": passage_number,
        "other_metadata": [list(texts.values()) for texts in other_texts],
        "passage_counts": passage_counts,
    }
    if division:
        # A passage belongs to the section it starts in
        passage_divisions = [max(bisect.bisect_right(division_starts, passage["start_byte"]) - 1, 0) for passage in passages]
        counts = Counter(passage_divisions)
        result["section"] = {
            **{key: division[key] for key in ("philo_id", "head", "start_byte", "end_byte", "index")},
            "passages": [number for number, _ in shown_passages],
            "previous": divisions[division["index"] - 1]["philo_id"] if division["index"] > 0 else None,
            "next": divisions[division["index"] + 1]["philo_id"] if division["index"] + 1 < len(divisions) else None,
        }
        result["sections"] = [
            {"philo_id": div["philo_id"], "head": div["head"], "passages": counts[i]} for i, div in enumerate(divisions)
        ]
        result["passage_sections"] = [divisions[index]["philo_id"] for index in passage_divisions]
    return result


@app.get("/get_passages/")
@app.get("/text-pair-api/get_passages/")
def get_passages(request: Request):
    """Retrieve aligned passages"""
    with pg_connection() as conn:
        _, other_args, _ = parse_request(request, conn)
        start_byte = int(request.query_params["start_byte"])
        end_byte = int(request.query_params["end_byte"])
        side = side_of(other_args.directionSelected)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute(
            f"SELECT * FROM {quote_ident(other_args.db_table)} WHERE {side}_filename = %s AND {side}_start_byte >=%s AND {side}_end_byte <=%s",
            (request.query_params["filename"], start_byte, end_byte),
        )
        passages = []
        for row in cursor:
            passages.append({field: value for field, value in row.items()})
    return {"passages": passages}


# Passages reused by more authors (or titles) than this say little about any pair
# of them, and the pairs they add grow quadratically.
CANON_MAX_MEMBERS = 20
# Sampled betweenness visits about this many edges, which bounds its time on large graphs.
BETWEENNESS_EDGE_VISITS = 20_000_000
NETWORK_CENTRALITIES = {
    "provenance": ("reused", "reuses", "authority", "bridging"),
    "canon": ("degree", "bridging"),
}


@contextmanager
def network_connection():
    """A connection whose network queries may hash and sort in memory."""
    with pg_connection() as conn:
        conn.cursor().execute("SET LOCAL work_mem = '256MB'")
        yield conn


def network_members(cursor, table, field, sql_fields, sql_values, families=None, nodes=None, name="members") -> str:
    """A relation of (group_id, doc_id, year, node): each passage family and document reusing it
    among the selected alignments, optionally only for some families (a (query, values) pair) or nodes."""
    if families is not None:
        # A literal list, not a subquery: the planner can then use the group_id indexes
        ids_cursor = cursor.connection.cursor()
        ids_cursor.execute(*families)
        families = [row[0] for row in ids_cursor]
    if not sql_fields and field in get_pg_type(cursor.connection, f"{table}_members"):
        conditions, values = [], []
        if families is not None:
            conditions.append("group_id = ANY(%s)")
            values.append(families)
        if nodes is not None:
            conditions.append(f"{quote_ident(field)} = ANY(%s)")
            values.append(nodes)
        query = f"SELECT group_id, doc_id, year, {quote_ident(field)} AS node FROM {quote_ident(f'{table}_members')}"
        if not conditions:
            return f"({query})"
        query += " WHERE " + " AND ".join(conditions)
    else:
        parts, values = [], []
        for direction in ("source", "target"):
            conditions = [sql_fields] if sql_fields else []
            values.extend(sql_values)
            if families is not None:
                conditions.append("group_id && %s::integer[] AND x = ANY(%s)")
                values.extend([families, families])
            column = quote_ident(f"{direction}_{field}")
            if nodes is not None:
                conditions.append(f"{column} = ANY(%s)")
                values.append(nodes)
            where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
            parts.append(
                f"SELECT x AS group_id, {direction}_doc_id AS doc_id, {direction}_year AS year, "
                f"{column} AS node FROM {quote_ident(table)}, unnest(group_id) x{where}"
            )
        query = " UNION ".join(parts)
    cursor.execute(f"CREATE TEMP TABLE {name} AS {query}", values)
    cursor.execute(f"ANALYZE {name}")
    return name


def network_links(graph, table, field, members, focus, focus_role, sizes=None) -> tuple[str, list]:
    """A query for one (group_id, source, target, weight) row per passage linking two nodes.

    Canon weighs each passage by the nodes reusing it in `sizes`, the whole corpus, or in
    `members` when that is the whole corpus already.
    """
    if graph == "provenance":
        anchor = f"g.{quote_ident(f'source_{field}')}"
        focus_clause, values = "", []
        if focus and focus_role == "anchor":
            focus_clause, values = f"AND {anchor} = %s", [focus]
        elif focus and focus_role == "reuser":
            focus_clause, values = "AND m.node = %s", [focus]
        return (
            f"""SELECT DISTINCT m.group_id, {anchor} AS source, m.node AS target, 1.0 AS weight
            FROM {members} m JOIN {quote_ident(f'{table}_groups')} g USING (group_id)
            WHERE m.node <> {anchor} AND m.node <> '' AND {anchor} <> '' {focus_clause}""",
            values,
        )
    focus_families, pair_clause, values = "", "a.node < b.node", []
    if focus:
        focus_families = f"AND group_id IN (SELECT group_id FROM {members} f WHERE f.node = %s)"
        pair_clause = "a.node <> b.node AND a.node = %s"
        values = [focus, focus]
    family_sizes = (
        "SELECT group_id, count(*) AS n FROM m GROUP BY group_id HAVING count(*)"
        if sizes is None
        else f"SELECT group_id, count(DISTINCT node) AS n FROM {sizes} s WHERE node <> '' GROUP BY group_id HAVING count(DISTINCT node)"
    )
    return (
        f"""WITH m AS (SELECT DISTINCT group_id, node FROM {members} z WHERE node <> '' {focus_families}),
        n AS ({family_sizes} BETWEEN 2 AND {CANON_MAX_MEMBERS}),
        mm AS (SELECT group_id, node, n FROM m JOIN n USING (group_id))
        SELECT a.group_id, a.node AS source, b.node AS target, 1.0 / (a.n - 1) AS weight
        FROM mm a JOIN mm b ON a.group_id = b.group_id AND {pair_clause}""",
        values,
    )


def network_centrality(G, mode) -> dict:
    """Each node's score for one measure of importance."""
    if mode == "reused":
        return dict(G.out_degree(weight="passages"))
    if mode == "reuses":
        return dict(G.in_degree(weight="passages"))
    if mode == "authority":
        return nx.pagerank(G.reverse(copy=False), weight="passages")  # credit flows from reuser to source
    if mode == "degree":
        return dict(G.degree(weight="weight"))
    visits = G.number_of_edges() * (1 if G.is_directed() else 2)
    k = max(50, BETWEENNESS_EDGE_VISITS // max(visits, 1))
    k = k if k < G.number_of_nodes() else None
    return nx.betweenness_centrality(G, k=k, weight="distance", seed=0)  # strong links are short


@app.get("/network_data/")
@app.get("/text-pair-api/network_data/")
def get_network_data(request: Request):
    """Network of authors (or titles) over passage families, for Sigma.js.

    provenance: from the author of a passage's earliest occurrence to each author reusing it.
    canon: between authors reusing the same passages, each weighted 1/(n-1) over the n authors
    reusing it in the whole corpus.
    """
    with network_connection() as conn:
        sql_fields, sql_values, other_args, _ = parse_args(request, conn)
        field = other_args.aggregation_field or "author"
        graph = other_args.graph or "provenance"
        if graph not in NETWORK_CENTRALITIES:
            return {"error": f"Unknown graph '{graph}'"}
        modes = NETWORK_CENTRALITIES[graph]
        centrality_mode = other_args.centrality if other_args.centrality in modes else modes[0]
        min_threshold = int(other_args.min_threshold or 5)
        max_nodes = int(other_args.max_nodes or 10000)
        table = str(other_args.db_table)
        field_types = get_pg_type(conn, table)
        if f"source_{field}" not in field_types or f"target_{field}" not in field_types:
            return {
                "error": f"Field '{field}' not available in database",
                "available_fields": list(field_types.keys()),
            }

        cursor = conn.cursor()
        members = network_members(cursor, table, field, sql_fields, sql_values)
        sizes = None
        if graph == "canon" and sql_fields:
            families = (f"SELECT DISTINCT group_id FROM {members}", [])
            sizes = network_members(cursor, table, field, "", [], families=families, name="corpus")
        links, values = network_links(graph, table, field, members, other_args.focus, other_args.focus_role, sizes)
        cursor.execute(f"CREATE TEMP TABLE link AS {links}", values)
        cursor.execute("ANALYZE link")
        cursor.execute(
            """CREATE TEMP TABLE edge AS SELECT source, target, count(*) AS passages, sum(weight)::float AS weight
            FROM link GROUP BY source, target HAVING count(*) >= %s""",
            (min_threshold,),
        )
        cursor.execute("SELECT source, target, passages, weight FROM edge")
        edges = [{"source": s, "target": t, "passages": p, "weight": w} for s, t, p, w in cursor]
        cursor.execute("SELECT count(DISTINCT group_id) FROM link")
        total_passages = cursor.fetchone()[0]

        # Limit to the most connected nodes
        strength = defaultdict(int)
        for edge in edges:
            strength[edge["source"]] += edge["passages"]
            strength[edge["target"]] += edge["passages"]
        if len(strength) > max_nodes:
            kept = set(sorted(strength, key=strength.get, reverse=True)[:max_nodes])
            edges = [e for e in edges if e["source"] in kept and e["target"] in kept]
        node_ids = sorted({node for edge in edges for node in (edge["source"], edge["target"])})

        cursor.execute(
            f"""SELECT node, percentile_disc(0.5) WITHIN GROUP (ORDER BY year) FROM
            (SELECT DISTINCT node, doc_id, year FROM {members} m WHERE year IS NOT NULL AND node = ANY(%s)) d
            GROUP BY node""",
            (node_ids,),
        )
        years = dict(cursor.fetchall())

    G = nx.DiGraph() if graph == "provenance" else nx.Graph()
    G.add_nodes_from(node_ids)
    for edge in edges:
        G.add_edge(edge["source"], edge["target"], passages=edge["passages"], weight=edge["weight"], distance=1 / edge["weight"])
    centrality = network_centrality(G, centrality_mode)

    nodes = []
    for node_id in node_ids:
        node = {
            "id": node_id,
            "label": node_id,
            "centrality": centrality.get(node_id, 0),
            "year": years.get(node_id),
            "passages": G.degree(node_id, weight="passages"),
            "type": field,
        }
        if graph == "provenance":
            node["reused"] = G.out_degree(node_id, weight="passages")
            node["reuses"] = G.in_degree(node_id, weight="passages")
        nodes.append(node)

    return {
        "nodes": nodes,
        "edges": edges,
        "graph": graph,
        "aggregation_field": field,
        "centrality_mode": centrality_mode,
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "total_passages": total_passages,
        "threshold": min_threshold,
    }


@app.get("/network_passages/")
@app.get("/text-pair-api/network_passages/")
def get_network_passages(request: Request):
    """The passages behind one network edge, most reused first."""
    with network_connection() as conn:
        sql_fields, sql_values, other_args, _ = parse_args(request, conn)
        field = other_args.aggregation_field or "author"
        graph = other_args.graph or "provenance"
        source, target = str(other_args.source), str(other_args.target)
        limit, offset = int(other_args.limit or 20), int(other_args.offset or 0)
        table = str(other_args.db_table)
        field_types = get_pg_type(conn, table)
        if not source or not target or f"source_{field}" not in field_types or f"target_{field}" not in field_types:
            return {"error": "source, target and an aggregation_field present in the database are required"}

        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        if graph == "provenance":
            anchored = (
                f"SELECT group_id FROM {quote_ident(f'{table}_groups')} WHERE {quote_ident(f'source_{field}')} = %s",
                [source],
            )
            members = network_members(cursor, table, field, sql_fields, sql_values, families=anchored, nodes=[target])
            cursor.execute(f"SELECT DISTINCT group_id FROM {members} m")
        else:
            pair = network_members(cursor, table, field, sql_fields, sql_values, nodes=[source, target], name="pair")
            shared = (f"SELECT group_id FROM {pair} p GROUP BY group_id HAVING count(DISTINCT node) = 2", [])
            corpus = network_members(cursor, table, field, "", [], families=shared, name="corpus")
            cursor.execute(
                f"SELECT group_id FROM {corpus} c WHERE node <> '' GROUP BY group_id HAVING count(DISTINCT node) <= %s",
                (CANON_MAX_MEMBERS,),
            )
        family_ids = [row["group_id"] for row in cursor]
        cursor.execute(
            f"""SELECT group_id, "count", source_author, source_title, source_year, left(source_passage, 400) AS source_passage
            FROM {quote_ident(f'{table}_groups')} WHERE group_id = ANY(%s) ORDER BY "count" DESC, group_id LIMIT %s OFFSET %s""",
            (family_ids, limit, offset),
        )
        passages = [dict(row) for row in cursor]
    return {"passages": passages, "total_count": len(family_ids), "source": source, "target": target}


# Registered with and without the trailing slash: a slashless request otherwise
# hits the /{db_path} catch-all that serves web apps.
@app.get("/alignment_by_rowid")
@app.get("/alignment_by_rowid/")
@app.get("/text-pair-api/alignment_by_rowid")
@app.get("/text-pair-api/alignment_by_rowid/")
def get_alignment_by_rowid(request: Request):
    """One alignment by rowid, for the semantic map's click-through.

    A dedicated route rather than reusing search_alignments with a rowid
    filter: that endpoint is built for paged browsing, so it needs a direction
    and an anchor it has no use for here, and it returns the embedding column.
    The embedding is 13 KB of a 16.5 KB response and the client has no use for
    it, since the map is already drawn from the precomputed projection.
    """
    with pg_connection() as conn:
        _, other_args, _ = parse_request(request, conn)
        db_table = str(other_args.db_table)
        try:
            rowid = int(request.query_params["rowid"])
        except (KeyError, TypeError, ValueError):
            return {"error": "rowid is required and must be an integer"}

        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        columns = [
            c
            for c in get_pg_type(conn, db_table)
            if c != "embedding"
        ]
        field_list = ", ".join(f'"{c}"' for c in columns)
        cursor.execute(
            sql.SQL("SELECT {fields} FROM {table} WHERE rowid = %s").format(
                fields=sql.SQL(field_list), table=sql.Identifier(db_table)
            ),
            (rowid,),
        )
        row = cursor.fetchone()
    if row is None:
        raise HTTPException(404, f"No alignment with rowid {rowid}")
    return {"alignment": dict(row)}


@app.get("/semantic_scatter_data/")
@app.get("/text-pair-api/semantic_scatter_data/")
def get_semantic_scatter_data(request: Request):
    """Alignment-level scatter for the semantic map.

    Serves the points HDBSCAN clustered, rather than aggregates of them: the
    (author, cluster) view leaves a median of four nodes per cluster, which
    cannot read as a cluster at any position, while at alignment level the
    clustering is visually unambiguous.

    Precomputed by textpair_graph, so this is a file read.
    """
    with pg_connection() as conn:
        _, other_args, _ = parse_request(request, conn)
    scatter_path = os.path.join(db_directory(str(other_args.db_table)), "graph_data", "scatter_data.json")
    if not os.path.exists(scatter_path):
        return {"error": "No semantic graph has been generated for this database. Build it with textpair --graph_only."}
    with open(scatter_path, "rb") as f:
        return Response(content=f.read(), media_type="application/json")


@app.get("/semantic_graph_data/")
@app.get("/text-pair-api/semantic_graph_data/")
def get_semantic_graph_data(request: Request):
    """Get semantic clustering graph data as simple nodes/edges arrays (like network_data).

    Returns nodes and edges for (author, cluster) pair visualization.
    The Vue component will build the Graphology graph client-side.
    """
    with pg_connection() as conn:
        sql_fields, sql_values, other_args, _ = parse_args(request, conn)

    # Path to precomputed graph model files
    graph_data_path = os.path.join(db_directory(str(other_args.db_table)), "graph_data")

    # Check if graph data exists
    if not os.path.exists(graph_data_path):
        return {"error": f"Graph data not found at {graph_data_path}. Please run build_graph_model.py first."}

    # Fast path: if no filters, load precomputed graph
    if not sql_fields:
        print("Loading precomputed full graph from disk...")
        precomputed_graph_path = os.path.join(graph_data_path, "precomputed_graph_api.json")

        if os.path.exists(precomputed_graph_path):
            with open(precomputed_graph_path, "rb") as f:
                graph_data = orjson.loads(f.read())

            print(
                f"✓ Loaded precomputed graph: {len(graph_data.get('nodes', []))} nodes, {len(graph_data.get('edges', []))} edges"
            )
            return graph_data
        else:
            print(f"⚠ Precomputed graph not found at {precomputed_graph_path}, falling back to filtered path")

    # Filtered path: rebuild graph from database with search filters
    print(f"Building filtered graph with query: {sql_fields or '(no filters)'}")

    # Load precomputed graph model data
    cluster_labels_modified = np.load(os.path.join(graph_data_path, "cluster_labels_modified.npy"))
    embeddings_umap_2d = np.load(os.path.join(graph_data_path, "embeddings_umap_2d.npy"))
    cluster_similarity = np.load(os.path.join(graph_data_path, "cluster_similarity_matrix.npy"))

    with open(os.path.join(graph_data_path, "author_to_id.json"), "rb") as f:
        author_to_id = orjson.loads(f.read())

    with open(os.path.join(graph_data_path, "cluster_metadata.json"), "rb") as f:
        metadata = orjson.loads(f.read())

    n_clusters = metadata["n_clusters"]
    total_clusters = metadata.get("total_clusters", metadata["n_clusters"])
    id_to_author = {v: k for k, v in author_to_id.items()}

    # Load cluster labels if available
    cluster_label_map = {}
    cluster_labels_path = os.path.join(graph_data_path, "cluster_labels.json")
    if os.path.exists(cluster_labels_path):
        with open(cluster_labels_path, "rb") as f:
            labels_data = orjson.loads(f.read())
            # Convert string keys back to integers
            cluster_label_map = {int(k): v for k, v in labels_data.items()}

    # Query alignments from database
    with pg_connection() as conn:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        if sql_fields:
            query = f"SELECT rowid, source_author, target_author FROM {quote_ident(other_args.db_table)} WHERE {sql_fields}"
        else:
            query = f"SELECT rowid, source_author, target_author FROM {quote_ident(other_args.db_table)}"
        cursor.execute(query, sql_values)

        # Build (author, cluster) pairs from filtered alignments
        from collections import defaultdict

        pair_passage_counts = defaultdict(int)
        pair_embeddings_2d = defaultdict(list)

        for row in cursor:
            rowid = row["rowid"]
            source_author = row["source_author"]
            target_author = row["target_author"]

            alignment_idx = rowid - 1
            if alignment_idx < 0 or alignment_idx >= len(cluster_labels_modified):
                continue

            cluster_id = int(cluster_labels_modified[alignment_idx])
            if cluster_id < 0:
                # Unthemed: HDBSCAN called it noise and the graph was built without
                # --merge-unthemed. Mirrors the same skip in build_precomputed_api_graph.
                continue
            embedding_2d = embeddings_umap_2d[alignment_idx]

            for author_name in [source_author, target_author]:
                if author_name not in author_to_id:
                    continue

                author_id = author_to_id[author_name]
                pair_key = (author_id, cluster_id)
                pair_passage_counts[pair_key] += 1
                pair_embeddings_2d[pair_key].append(embedding_2d)

    # Calculate mean positions
    pair_positions = {}
    for pair_key, embeddings_list in pair_embeddings_2d.items():
        mean_position = np.mean(embeddings_list, axis=0)
        pair_positions[pair_key] = mean_position

    # Build nodes array
    # Threshold: only include author-cluster pairs with at least 5 passages
    MIN_PASSAGES_THRESHOLD = 5
    nodes = []
    for (author_id, cluster_id), passage_count in pair_passage_counts.items():
        # Skip nodes with fewer than threshold passages
        if passage_count < MIN_PASSAGES_THRESHOLD:
            continue

        node_id = f"author_{author_id}_cluster_{cluster_id}"
        position = pair_positions[(author_id, cluster_id)]

        nodes.append(
            {
                "id": node_id,
                "label": id_to_author[author_id],
                "author_id": author_id,
                "author_name": id_to_author[author_id],
                "cluster_id": cluster_id,
                "cluster_label": cluster_label_map.get(cluster_id, ""),
                "passages": int(passage_count),
                "size": int(passage_count),
                "x": float(position[0]),
                "y": float(position[1]),
            }
        )

    # Group nodes by cluster
    cluster_nodes = defaultdict(list)
    for node in nodes:
        cluster_id = node["cluster_id"]
        cluster_nodes[cluster_id].append(node["id"])

    # Add cluster anchor nodes
    cluster_centroid_positions_2d = {}
    for cluster_id in cluster_nodes.keys():
        cluster_2d_positions = []
        for pair_key, position in pair_positions.items():
            if pair_key[1] == cluster_id:
                cluster_2d_positions.append(position)

        if cluster_2d_positions:
            mean_2d_position = np.mean(cluster_2d_positions, axis=0)
            cluster_centroid_positions_2d[cluster_id] = mean_2d_position

    for cluster_id, position_2d in cluster_centroid_positions_2d.items():
        anchor_node_id = f"anchor_cluster_{cluster_id}"
        nodes.append(
            {
                "id": anchor_node_id,
                "label": "",
                "node_type": "cluster_anchor",
                "cluster_id": cluster_id,
                "cluster_label": cluster_label_map.get(cluster_id, ""),
                "size": 0.01,
                "x": float(position_2d[0]),
                "y": float(position_2d[1]),
                "hidden": True,
            }
        )

    # Build edges array
    edges = []

    # 1. Add intra-cluster edges (connect nodes within same cluster)
    for cluster_id, node_ids in cluster_nodes.items():
        # Only add intra-cluster edges if there are multiple nodes
        if len(node_ids) > 1:
            # Connect each node to cluster anchor (star topology, not complete graph)
            anchor_id = f"anchor_cluster_{cluster_id}"
            for node_id in node_ids:
                edges.append(
                    {
                        "source": node_id,
                        "target": anchor_id,
                        "weight": 1.0,
                        "edge_type": "intra_cluster",
                        "color": "#666666",
                        "size": 0.5,
                    }
                )

    # Add centroid similarity edges (connect cluster anchors)
    # No filtering - send all edges to let ForceAtlas2 see full structure
    similarity_threshold = 0

    # Only add centroid edges for clusters that exist in our filtered graph
    filtered_cluster_ids = set(cluster_nodes.keys())

    for cluster_i in range(n_clusters):
        if cluster_i not in filtered_cluster_ids:
            continue

        for cluster_j in range(cluster_i + 1, n_clusters):
            if cluster_j not in filtered_cluster_ids:
                continue

            similarity = cluster_similarity[cluster_i, cluster_j]

            if similarity > similarity_threshold:
                anchor_i = f"anchor_cluster_{cluster_i}"
                anchor_j = f"anchor_cluster_{cluster_j}"

                edges.append(
                    {
                        "source": anchor_i,
                        "target": anchor_j,
                        "weight": float(similarity * 10),
                        "edge_type": "centroid_similarity",
                        "color": "#999999",
                        "size": 1.0,
                    }
                )

    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "n_clusters": n_clusters,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "min_passages_threshold": MIN_PASSAGES_THRESHOLD,
        },
    }


# Each page with and without a trailing slash: behind a proxy, Starlette's redirect between them names the API's address
@app.get("/{db_path}/search")
@app.get("/{db_path}/search/")
@app.get("/text-pair/{db_path}/search")
@app.get("/text-pair/{db_path}/search/")
@app.get("/{db_path}/time")
@app.get("/{db_path}/time/")
@app.get("/text-pair/{db_path}/time")
@app.get("/text-pair/{db_path}/time/")
@app.get("/{db_path}")
@app.get("/{db_path}/")
@app.get("/text-pair/{db_path}")
@app.get("/text-pair/{db_path}/")
@app.get("/text-pair/{db_path}/group/{id}")
@app.get("/text-pair/{db_path}/group/{id}/")
@app.get("/{db_path}/group/{id}")
@app.get("/{db_path}/group/{id}/")
@app.get("/text-pair/{db_path}/sorted-results")
@app.get("/text-pair/{db_path}/sorted-results/")
@app.get("/{db_path}/sorted-results")
@app.get("/{db_path}/sorted-results/")
@app.get("/text-pair/{db_path}/text-view")
@app.get("/text-pair/{db_path}/text-view/")
@app.get("/{db_path}/text-view")
@app.get("/{db_path}/text-view/")
@app.get("/text-pair/{db_path}/network")
@app.get("/text-pair/{db_path}/network/")
@app.get("/{db_path}/network")
@app.get("/{db_path}/network/")
@app.get("/text-pair/{db_path}/semantic")
@app.get("/text-pair/{db_path}/semantic/")
@app.get("/{db_path}/semantic")
@app.get("/{db_path}/semantic/")
def index(db_path: str):
    """Return index.html, carrying the database's current app config"""
    db_dir = db_directory(db_path)
    try:
        config = load_app_config(db_dir)
    except SettingsError as error:
        return HTMLResponse(f"<p>The settings of this database have a mistake: {html.escape(str(error))}</p>", 500)
    config["hasPassageFamilies"] = has_passage_families(config["databaseName"])
    config["showNetworkGraph"] = config["showNetworkGraph"] and config["hasPassageFamilies"]
    for direction in ("source", "target"):
        config[f"{direction}TextAvailable"] = text_data_path(db_dir, config, direction) is not None
    config_script = orjson.dumps(config).decode().replace("</", "<\\/")
    with open(os.path.join(db_dir, "dist/index.html"), encoding="utf8") as index_file:
        index_html = index_file.read()
    return HTMLResponse(index_html.replace("</head>", f"<script>window.textpairConfig = {config_script}</script></head>", 1))
