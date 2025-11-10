#!/usr/bin/env python3
"""Routing and search code for sequence alignment"""

import configparser
import itertools
import os
import re
import time
from collections import Counter, OrderedDict, defaultdict, namedtuple
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import orjson
import psycopg2
import psycopg2.extras
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pgvector.psycopg2 import register_vector
from philologic.runtime.DB import DB
from philologic.runtime.get_text import get_text_obj
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import Response

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
    "PHILO_CONFIG", ["db_path", "page_images_url_root", "page_image_extension", "page_external_page_images"]
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


def get_pg_type(table_name):
    """Find PostgreSQL field type"""
    with psycopg2.connect(
        user=GLOBAL_CONFIG["DATABASE"]["database_user"],
        password=GLOBAL_CONFIG["DATABASE"]["database_password"],
        database=GLOBAL_CONFIG["DATABASE"]["database_name"],
    ) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name ='{table_name}'"
        )
        field_types = {field: field_type.upper() for field, field_type in cursor}
    return field_types


def parse_args(request):
    """Parse URL args"""
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
        "philo_path",
        "classification_filter_1",
        "classification_filter_2",
        "classification_filter_3",
        "aggregation_field",
        "min_threshold",
        "max_nodes",
        "node_id",
        "node_type",
        "limit",
        "offset",
        "centrality",
    ]
    for key, value in request.query_params.items():
        if key in other_args_keys:
            if key in ("page", "id_anchor", "timeSeriesInterval", "start_byte", "end_byte"):
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

    # Handle network graph node expansion: remove source/target filters before query_builder
    aggregation_field = request.query_params.get("aggregation_field", "author")
    source_param = request.query_params.get(f"source_{aggregation_field}", "").strip('"')
    target_param = request.query_params.get(f"target_{aggregation_field}", "").strip('"')

    if source_param and target_param and source_param == target_param:
        # This is a node expansion request - remove source/target filters from query_args
        # so query_builder doesn't process them (we'll add OR logic in get_network_data)
        if f"source_{aggregation_field}" in query_args:
            del query_args.dict[f"source_{aggregation_field}"]
        if f"target_{aggregation_field}" in query_args:
            del query_args.dict[f"target_{aggregation_field}"]

    metadata_field_types = get_pg_type(other_args["db_table"])
    metadata_field_types["rowid"] = "INTEGER"
    sql_fields, sql_values = query_builder(query_args, other_args, metadata_field_types)
    return sql_fields, sql_values, other_args, list(metadata_field_types.keys())


def query_builder(query_args, other_args, field_types) -> tuple[str, list[str]]:
    """Takes query arguments and returns an SQL WHERE clause"""
    sql_fields: list[str] = []
    sql_values: list[str] = []
    for field, value in query_args.items():
        value: str = value.strip()
        field_type = field_types.get(field, "TEXT").upper()
        query = ""

        # --- Handle group_id specifically first ---
        if field == "group_id":
            try:
                # Assume user provides a single integer ID to filter by
                filter_gid = int(value)
                if field_type == "INTEGER":
                    query = f"{field} = %s"
                    sql_values.append(filter_gid)
                elif field_type == "ARRAY": # Covers INTEGER[]
                    # Use array containment operator
                    query = f"{field} @> ARRAY[%s]::integer[]"
                    sql_values.append(filter_gid)
                elif field_type == "JSONB":
                    # Assumes JSONB stores an array of numbers
                    # Use JSONB containment operator
                    query = f"{field} @> %s::jsonb"
                    # Pass the integer as a JSON array string containing that number
                    sql_values.append(f'[{filter_gid}]')
                else: # Fallback to integer equality for unknown types
                    query = f"{field} = %s"
                    sql_values.append(filter_gid)

                if query: # Only append if a query was successfully constructed
                    sql_fields.append(query)

            except (ValueError, TypeError):
                # Ignore if the provided group_id filter value is not an integer
                continue # Skip to the next field
            continue # Skip the rest of the loop for group_id

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
                        query = f"{field} = %s"
                        sql_values.append("")
                    else:
                        query = f"{field}=%s"
                        sql_values.append(value[1:-1])
                elif value.startswith("NOT "):
                    split_value = " ".join(value.split()[1:]).strip()
                    query = f"{field} !~* %s"
                    sql_values.append(fr"\m{split_value}\M")
                # elif value.startswith("OR "):  ## TODO: add support to OR queries by changing the join logic at the end of the function
                #     split_value = " ".join(value.split()[1:]).strip()
                #     query = "{} !~* %s".format(field)
                #     sql_values.append('\m{}\M'.format(split_value))
                else:
                    query = f"{field} ~* %s"
                    sql_values.append(fr"\m{value}\M")
                sql_fields.append(query)
        elif field_type in ("INTEGER", "FLOAT"):
            value = value.replace('"', "")
            if "-" in value:
                values = [v for v in re.split(r"(-)", value) if v]
                if values[0] == "-":
                    query = f"{field} <= %s"
                    sql_values.append(values[1])
                elif values[-1] == "-":
                    query = f"{field} >= %s"
                    sql_values.append(values[0])
                else:
                    query = f"{field} BETWEEN %s AND %s"
                    sql_values.extend([values[0], values[2]])
            else:
                query = f"{field} = %s"
                sql_values.append(value)
            sql_fields.append(query)
        else:
            continue
    if other_args.banality != "":
        sql_fields.append("banality=%s")
        sql_values.append(other_args.banality)

    # Add classification filter condition
    for i in range(1, 4):  # Handle classification_filter_1, classification_filter_2, classification_filter_3
        filter_key = f"classification_filter_{i}"
        if filter_key in other_args and other_args[filter_key]:
            # Each filter value must match in at least one classification field
            condition = "(target_first_class=%s OR target_second_class=%s OR target_third_class=%s)"
            sql_fields.append(condition)
            # Add the same filter value three times (once for each field)
            filter_value = other_args[filter_key]
            sql_values.extend([filter_value, filter_value, filter_value])

    return " AND ".join(sql_fields), sql_values


def check_access_control(request: Request):
    """Check if user has access to a particular database"""
    return True  # Placeholder


@app.get("/")
@app.get("/text-pair")
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
    with open(os.path.join(APP_PATH, db_path, "dist/css", resource)) as resource_file:
        resource = resource_file.read()
    return Response(resource, media_type="text/css")


@app.get("/{db_path}/js/{resource}")
@app.get("/text-pair/{db_path}/js/{resource}")
def get_js_resource(db_path: str, resource: str):
    """Retrieve JS resources"""
    with open(os.path.join(APP_PATH, db_path, "dist/js", resource), encoding="utf8") as resource_file:
        resource = resource_file.read()
    return Response(resource, media_type="application/javascript")


@app.get("/{db_path}/assets/{resource}")
@app.get("/text-pair/{db_path}/assets/{resource}")
def get_ressource(db_path: str, resource: str):
    """Retrieve JS resources"""
    with open(os.path.join(APP_PATH, db_path, "dist/assets", resource), encoding="utf8") as resource_file:
        resource_content = resource_file.read()
    if resource.endswith(".js"):
        return Response(resource_content, media_type="application/javascript")
    elif resource.endswith(".css"):
        return Response(resource_content, media_type="text/css")


@app.get("/{db_path}/favicon.ico")
@app.get("/text-pair/{db_path}/favicon.ico")
def get_favicon(db_path: str):
    """Retrieve favicon"""
    with open(os.path.join(APP_PATH, db_path, "dist/favicon.ico"), "rb") as resource_file:
        resource_content = resource_file.read()
    return Response(resource_content, media_type="image/x-icon")


@app.get("/search_alignments/")
@app.get("/text-pair-api/search_alignments/")
def search_alignments(request: Request):
    """Search alignments according to URL params"""
    sql_fields, sql_values, other_args, column_names = parse_args(request)
    field_types = get_pg_type(other_args.db_table)
    group_id_type = field_types.get("group_id", "INTEGER")
    if other_args.direction == "next":
        if sql_fields:
            query = f"SELECT o.rowid_ordered, m.* FROM {other_args.db_table} m, {other_args.db_table}_ordered o WHERE {sql_fields} AND o.source_year_target_year=m.rowid and \
                    o.rowid_ordered > {other_args.id_anchor} ORDER BY o.rowid_ordered LIMIT 50"
        else:
            query = f"SELECT o.rowid_ordered, m.* FROM {other_args.db_table} m, {other_args.db_table}_ordered o WHERE o.source_year_target_year=m.rowid and \
                    o.rowid_ordered > {other_args.id_anchor} ORDER BY o.rowid_ordered LIMIT 50"
    else:
        if sql_fields:
            query = f"SELECT o.rowid_ordered, m.* FROM {other_args.db_table} m, {other_args.db_table}_ordered o WHERE {sql_fields} AND o.source_year_target_year=m.rowid and \
                    o.rowid_ordered < {other_args.id_anchor} ORDER BY o.rowid_ordered desc LIMIT 50"
        else:
            query = f"SELECT o.rowid_ordered, m.* FROM {other_args.db_table} m, {other_args.db_table}_ordered o WHERE o.source_year_target_year=m.rowid and \
                    o.rowid_ordered < {other_args.id_anchor} ORDER BY o.rowid_ordered desc LIMIT 50"
    conn = psycopg2.connect(
        user=GLOBAL_CONFIG["DATABASE"]["database_user"],
        password=GLOBAL_CONFIG["DATABASE"]["database_password"],
        database=GLOBAL_CONFIG["DATABASE"]["database_name"],
    )
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cursor.execute(query, sql_values)
    alignments = []
    group_ids = []
    for row in cursor:
        metadata = {key: row[key] for key in column_names}
        metadata["rowid_ordered"] = row["rowid_ordered"]
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
    cursor.execute(f"""SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{other_args.db_table}_groups')""")
    if cursor.fetchone()[0] is True:
        counts_per_group: dict[int, int] = {}
        for group_id in set(group_ids):
            cursor.execute(f"""SELECT source_doc_id FROM {other_args.db_table}_groups WHERE group_id=%s""", (group_id,))
            try:
                source_doc_id = cursor.fetchone()[0]
            except:
                return {"group_id": group_id, "error": "No source_doc_id found for this group_id"}
            if group_id_type == "ARRAY":
                count_query = f"SELECT COUNT(*) FROM {other_args.db_table} WHERE group_id @> ARRAY[%s]::integer[] AND source_doc_id=%s"
                params = (group_id, source_doc_id)
            else: # Assume INTEGER
                count_query = f"SELECT COUNT(*) FROM {other_args.db_table} WHERE group_id=%s AND source_doc_id=%s"
                params = (group_id, source_doc_id)
            cursor.execute(count_query, params)
            counts_per_group[group_id] = cursor.fetchone()[0]
        for alignment in alignments:
            if group_id_type == "ARRAY":
                group_ids = [int(gid) for gid in alignment["group_id"]]
                alignment["count"] = sum(counts_per_group[gid] for gid in group_ids)
            else:
                alignment["count"] = counts_per_group[alignment["group_id"]]
    conn.rollback()
    conn.close()

    previous_url = ""
    current_path = re.sub(r"&(page|id_anchor|direction)=(previous|next|\d*)", "", request.url.path)
    if other_args.page > 1:  # type: ignore
        previous_url = f"{current_path}&page={other_args.page - 1}&id_anchor={alignments[0]['rowid_ordered']}&direction=previous"  # type: ignore
    try:
        next_url = f"{current_path}&page={other_args.page + 1}&id_anchor={alignments[-1]['rowid_ordered']}&direction=next"  # type: ignore
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
    sql_fields, sql_values, other_args, column_names = parse_args(request)
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
    conn = psycopg2.connect(
        user=GLOBAL_CONFIG["DATABASE"]["database_user"],
        password=GLOBAL_CONFIG["DATABASE"]["database_password"],
        database=GLOBAL_CONFIG["DATABASE"]["database_name"],
    )
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    if sql_values:
        query = (
            f"""SELECT * FROM {other_args.db_table} WHERE {other_args.field}='{other_args.value}' AND {sql_fields}"""
        )
        cursor.execute(query, sql_values)
    else:
        query = f"""SELECT * FROM {other_args.db_table} WHERE {other_args.field}='{other_args.value}'"""
        cursor.execute(query)
    for row in cursor:
        if row[doc_id] not in docs_found:
            docs_found[row[doc_id]] = {"count": 0, **{key: row[key] for key in column_names}}
        docs_found[row[doc_id]]["count"] += 1
    conn.rollback()
    conn.close()
    return list(docs_found.values())


@app.get("/retrieve_all_passage_pairs/")
@app.get("/text-pair-api/retrieve_all_passage_pairs/")
def retrieve_all_passage_pairs(request: Request):
    """Retrieve all passage pair metadata matching a particular query
    NOTE that this does not retrieve passages themselves"""
    sql_fields, sql_values, other_args, column_names = parse_args(request)
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
    query = f"SELECT * FROM {other_args.db_table} WHERE {sql_fields}"
    conn = psycopg2.connect(
        user=GLOBAL_CONFIG["DATABASE"]["database_user"],
        password=GLOBAL_CONFIG["DATABASE"]["database_password"],
        database=GLOBAL_CONFIG["DATABASE"]["database_name"],
    )
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cursor.execute(query, sql_values)
    results = [{key: row[key] for key in column_names} for row in cursor]
    conn.rollback()
    conn.close()
    return results


@app.get("/count_results/")
@app.get("/text-pair-api/count_results/")
def count_results(request: Request):
    """Search alignments according to URL params"""
    sql_fields, sql_values, other_args, _ = parse_args(request)
    if sql_fields:
        query = f"SELECT COUNT(*) FROM {other_args.db_table} WHERE {sql_fields}"
    else:
        query = f"SELECT COUNT(*) FROM {other_args.db_table}"
    conn = psycopg2.connect(
        user=GLOBAL_CONFIG["DATABASE"]["database_user"],
        password=GLOBAL_CONFIG["DATABASE"]["database_password"],
        database=GLOBAL_CONFIG["DATABASE"]["database_name"],
    )
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cursor.execute(query, sql_values)
    result_object = {"counts": cursor.fetchone()[0]}
    conn.rollback()
    conn.close()
    return result_object


@app.post("/generate_time_series/")
@app.post("/text-pair-api/generate_time_series/")
def generate_time_series(request: Request):
    """Generate a time series from search results"""
    # TODO: don't assume year is the field to use
    sql_fields, sql_values, other_args, _ = parse_args(request)
    if sql_fields:
        query = f"select interval AS year, COUNT(*) FROM \
                (SELECT floor({other_args.directionSelected}_year/{other_args.timeSeriesInterval})*{other_args.timeSeriesInterval} \
                AS interval FROM {other_args.db_table} WHERE {sql_fields}) t \
                GROUP BY interval ORDER BY interval"
    else:
        query = f"select interval AS year, COUNT(*) FROM \
                (SELECT floor({other_args.directionSelected}_year/{other_args.timeSeriesInterval})*{other_args.timeSeriesInterval} \
                AS interval FROM {other_args.db_table}) t \
                GROUP BY interval ORDER BY interval"
    results = []
    total_results = 0
    next_year = None
    conn = psycopg2.connect(
        user=GLOBAL_CONFIG["DATABASE"]["database_user"],
        password=GLOBAL_CONFIG["DATABASE"]["database_password"],
        database=GLOBAL_CONFIG["DATABASE"]["database_name"],
    )
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
    conn.rollback()
    conn.close()
    return {"counts": total_results, "results": results}


@app.post("/facets/")
@app.post("/text-pair-api/facets/")
def facets(request: Request):
    """Retrieve facet result"""
    sql_fields, sql_values, other_args, _ = parse_args(request)
    results = []
    total_count = 0
    conn = psycopg2.connect(
        user=GLOBAL_CONFIG["DATABASE"]["database_user"],
        password=GLOBAL_CONFIG["DATABASE"]["database_password"],
        database=GLOBAL_CONFIG["DATABASE"]["database_name"],
    )
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    if sql_fields:
        query = f"SELECT {other_args.facet}, COUNT(*) FROM {other_args.db_table} \
            WHERE {sql_fields} GROUP BY {other_args.facet} ORDER BY COUNT(*) DESC"
    else:
        query = f"SELECT {other_args.facet}, COUNT(*) FROM {other_args.db_table} \
            GROUP BY {other_args.facet} ORDER BY COUNT(*) DESC"
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
    conn.rollback()
    conn.close()
    return {"facet": other_args.facet, "results": results, "total_count": total_count}


@app.get("/metadata/")
@app.get("/text-pair-api/metadata/")
def metadata(request: Request):
    """Retrieve all searchable metadata fields"""
    _, _, _, metadata_fields = parse_args(request)
    return metadata_fields


@app.get("/group/{group_id}")
@app.get("/text-pair-api/group/{group_id}")
def get_passage_group(request: Request, group_id: int):
    """Retrieve a passage group"""
    alignment_table = request.query_params["db_table"]
    field_types = get_pg_type(alignment_table)
    group_id_type = field_types.get("group_id", "INTEGER").upper()
    groups_table = f"{alignment_table}_groups"
    filtered_passages: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    original_passage: dict[str, Any] = {}
    conn = psycopg2.connect(
        user=GLOBAL_CONFIG["DATABASE"]["database_user"],
        password=GLOBAL_CONFIG["DATABASE"]["database_password"],
        database=GLOBAL_CONFIG["DATABASE"]["database_name"],
    )
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # Query groups_table
    cursor.execute(f"""SELECT * FROM {groups_table} WHERE group_id=%s""", (group_id,))
    original_passage_result = cursor.fetchone()
    original_passage = {k: v for k, v in original_passage_result.items()}


    # Query the main alignment_table using the correct operator based on its group_id type
    if group_id_type == "ARRAY":
        query = f"SELECT * FROM {alignment_table} WHERE group_id @> ARRAY[%s]::integer[] AND source_doc_id=%s"
        params = (group_id, original_passage["source_doc_id"])
    else: # Backward compatibility for INTEGER type
        query = f"SELECT * FROM {alignment_table} WHERE group_id=%s AND source_doc_id=%s"
        params = (group_id, original_passage["source_doc_id"])

    cursor.execute(query, params)

    # Process results from the alignment_table query
    for row in cursor:
        filtered_passages[row["target_filename"]].append(
            {
                **row,
                "direction": "target",
            }
        )
    conn.rollback()
    conn.close()
    passage_list = []
    results = defaultdict(list)
    for passages in filtered_passages.values():
        for passage in passages:
            results[passage["target_year"]].append(passage)
    for key, value in results.items():
        value.sort(key=lambda x: (x["target_title"], x["target_start_byte"]), reverse=True)
        passage_list.append({"year": key, "result": value})
    passage_list.sort(key=lambda x: x["year"] or 9999)
    full_results = {"passageList": passage_list, "original_passage": original_passage}
    return full_results


@app.get("/sorted_results/")
@app.get("/text-pair-api/sorted_results/")
def get_sorted_results(request: Request):
    """Sort results based on the number of passages within each group"""
    sql_fields, sql_values, other_args, _ = parse_args(request)
    field_types = get_pg_type(other_args.db_table)
    group_id_type = field_types.get("group_id", "INTEGER").upper()

    conn = psycopg2.connect(
        user=GLOBAL_CONFIG["DATABASE"]["database_user"],
        password=GLOBAL_CONFIG["DATABASE"]["database_password"],
        database=GLOBAL_CONFIG["DATABASE"]["database_name"],
    )
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    # Select group_id and source_doc_id, applying filters if any
    query = f"SELECT source_doc_id, group_id FROM {other_args.db_table}"
    if sql_fields:
        query += f" WHERE {sql_fields}"
    cursor.execute(query, sql_values)

    # Store potential group IDs and their source docs
    # Check type of retrieved value here
    potential_groups: dict[int, str] = {} # group_id -> source_doc_id
    for row in cursor:
        gid_value = row["group_id"]
        source_doc_id = row["source_doc_id"]
        if isinstance(gid_value, int): # Check if it's an integer
            if gid_value is not None:
                 potential_groups[gid_value] = source_doc_id
        elif isinstance(gid_value, list): # Check if it's a list (from ARRAY)
             for single_gid in gid_value:
                 if single_gid is not None:
                     try:
                         potential_groups[int(single_gid)] = source_doc_id
                     except (ValueError, TypeError):
                         continue
        elif gid_value is None:
             continue

    counts_per_group: dict[int, int] = {}
    groups_table = f"{other_args.db_table}_groups"

    for group_id in potential_groups:
        # Verify against the _groups table for the canonical source_doc_id
        cursor.execute(f"""SELECT source_doc_id FROM {groups_table} WHERE group_id=%s""", (group_id,))
        group_source_result = cursor.fetchone()
        if not group_source_result:
            continue
        group_source_doc_id = group_source_result[0]

        # Count occurrences in the main table using the correct query type
        if group_id_type == "ARRAY": # Check for ARRAY type
            count_query = f"SELECT COUNT(*) FROM {other_args.db_table} WHERE group_id @> ARRAY[%s]::integer[] AND source_doc_id=%s"
            params = (group_id, group_source_doc_id)
        else: # Assume INTEGER if not ARRAY
            count_query = f"SELECT COUNT(*) FROM {other_args.db_table} WHERE group_id=%s AND source_doc_id=%s"
            params = (group_id, group_source_doc_id)

        cursor.execute(count_query, params)
        count_result = cursor.fetchone()
        if count_result:
            counts_per_group[group_id] = count_result[0]

    results = {"total_count": len(counts_per_group), "groups": []}
    sorted_group_ids = sorted(counts_per_group.items(), key=lambda x: x[1], reverse=True)[:100]

    for group_id, count in sorted_group_ids:
        cursor.execute(f"""SELECT * FROM {groups_table} WHERE group_id=%s""", (group_id,))
        group_data = cursor.fetchone()
        if group_data:
            results["groups"].append({**group_data, "count": count})

    conn.rollback()
    conn.close()
    return results


@app.get("/text_view/")
@app.get("/text-pair-api/text_view/")
def text_view(request: Request):
    """Retrieve a text object from PhiloLogic4"""
    _, _, other_args, _ = parse_args(request)
    access_control = check_access_control(request)
    if access_control is False:  # TODO check if user has access to this database
        return {"error": "You do not have access to this database"}

    # Get passage pairs offsets
    conn = psycopg2.connect(
        user=GLOBAL_CONFIG["DATABASE"]["database_user"],
        password=GLOBAL_CONFIG["DATABASE"]["database_password"],
        database=GLOBAL_CONFIG["DATABASE"]["database_name"],
    )
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute(
        f"SELECT {other_args.directionSelected}_start_byte, {other_args.directionSelected}_end_byte FROM {other_args.db_table} WHERE {other_args.directionSelected}_philo_id=%s",
        (other_args.philo_id,),
    )
    passage_pairs = [
        {
            "start_byte": row[f"{other_args.directionSelected}_start_byte"],
            "end_byte": row[f"{other_args.directionSelected}_end_byte"],
        }
        for row in cursor
    ]

    # Merge passage pairs based on overlapping offsets
    passage_pairs.sort(key=lambda x: x["start_byte"])
    merged_passage_pairs = []
    for passage_pair in passage_pairs:
        if not merged_passage_pairs:
            merged_passage_pairs.append(passage_pair)
        else:
            last_merged_passage_pair = merged_passage_pairs[-1]
            if passage_pair["start_byte"] <= last_merged_passage_pair["end_byte"]:
                last_merged_passage_pair["end_byte"] = passage_pair["end_byte"]
            else:
                merged_passage_pairs.append(passage_pair)

    if other_args.philo_path:
        philo_config = PHILO_CONFIG(other_args.philo_path, "", "", "")
    else:
        philo_path = os.path.join(APP_PATH, other_args.db_table)
        philo_config = PHILO_CONFIG(philo_path, "", "", "")
    philo_db = DB(f"{philo_config.db_path}/data", width=7)
    philo_object = philo_db[other_args.philo_id]
    philo_request = PHILO_REQUEST("", "", "", merged_passage_pairs)
    philo_text_object, _ = get_text_obj(philo_object, philo_config, philo_request, philo_db.locals["token_regex"])

    # Get metadata
    cursor.execute(
        f"SELECT * FROM {other_args.db_table} WHERE {other_args.directionSelected}_philo_id=%s", (other_args.philo_id,)
    )
    metadata_fields: psycopg2.extras.RealDictCursor = cursor.fetchone()  # type: ignore

    # Get metadata for other direction
    other_metadata_fields = []
    for passage_pair in merged_passage_pairs:
        cursor.execute(f"SELECT * FROM {other_args.db_table} WHERE {other_args.directionSelected}_filename = %s AND {other_args.directionSelected}_start_byte >=%s AND {other_args.directionSelected}_end_byte <=%s", (metadata_fields[f"{other_args.directionSelected}_filename"], passage_pair["start_byte"], passage_pair["end_byte"]))  # type: ignore
        other_metadata_fields.append([])
        for row in cursor:
            other_metadata_fields[-1].append(
                {field: value for field, value in row.items() if not field.startswith(other_args.directionSelected)}
            )
    conn.rollback()
    conn.close()

    # Find passage number in passage pairs for autoscroll
    passage_number = 0
    try:
        start_byte = int(request.query_params["start_byte"])
        for i, passage_pair in enumerate(merged_passage_pairs):
            if passage_pair["start_byte"] <= start_byte <= passage_pair["end_byte"]:
                passage_number = i
                break
    except KeyError:
        pass
    return {
        "text": philo_text_object,
        "metadata": metadata_fields,
        "direction": other_args.directionSelected,
        "passage_number": passage_number,
        "other_metadata": other_metadata_fields,
    }


@app.get("/get_passages/")
@app.get("/text-pair-api/get_passages/")
def get_passages(request: Request):
    """Retrieve aligned passages"""
    _, _, other_args, _ = parse_args(request)
    start_byte = int(request.query_params["start_byte"])
    end_byte = int(request.query_params["end_byte"])
    conn = psycopg2.connect(
        user=GLOBAL_CONFIG["DATABASE"]["database_user"],
        password=GLOBAL_CONFIG["DATABASE"]["database_password"],
        database=GLOBAL_CONFIG["DATABASE"]["database_name"],
    )
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute(f"SELECT * FROM {other_args.db_table} WHERE {other_args.directionSelected}_filename = %s AND {other_args.directionSelected}_start_byte >=%s AND {other_args.directionSelected}_end_byte <=%s", (request.query_params["filename"], start_byte, end_byte))
    passages = []
    for row in cursor:
        passages.append({field: value for field, value in row.items()})
    conn.rollback()
    conn.close()
    return {"passages": passages}


@app.get("/network_data/")
@app.get("/text-pair-api/network_data/")
def get_network_data(request: Request):
    """Get network graph data aggregated by author or title
    Returns nodes and edges for visualization with Sigma.js"""
    sql_fields, sql_values, other_args, _ = parse_args(request)

    # Get parameters for network configuration
    aggregation_field = request.query_params.get("aggregation_field", "author")  # author or title
    min_threshold = int(request.query_params.get("min_threshold", 5))
    max_nodes = int(request.query_params.get("max_nodes", 10000))

    conn = psycopg2.connect(
        user=GLOBAL_CONFIG["DATABASE"]["database_user"],
        password=GLOBAL_CONFIG["DATABASE"]["database_password"],
        database=GLOBAL_CONFIG["DATABASE"]["database_name"],
    )
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    # Determine which fields to aggregate by
    if aggregation_field == "author":
        source_field = "source_author"
        target_field = "target_author"
    else:  # title
        source_field = "source_title"
        target_field = "target_title"

    # Check if the requested fields exist in the table
    field_types = get_pg_type(other_args.db_table)
    if source_field not in field_types or target_field not in field_types:
        conn.close()
        return {
            "error": f"Field '{aggregation_field}' not available in database",
            "available_fields": list(field_types.keys())
        }

    # Build aggregation query
    # Special case: if filtering by same node in both source and target, use OR logic
    source_param = request.query_params.get(f"source_{aggregation_field}", "").strip('"')
    target_param = request.query_params.get(f"target_{aggregation_field}", "").strip('"')

    # At this point, sql_fields and sql_values are clean (parse_args removed source/target filters for node expansion)
    where_clause = f"WHERE {sql_fields}" if sql_fields else ""

    if source_param and target_param and source_param == target_param:
        # User wants all connections for a specific node (both as source and target)
        # Add OR filter to existing filters
        node_filter = f"({source_field} = %s OR {target_field} = %s)"
        if where_clause:
            where_clause += f" AND {node_filter}"
            sql_values = list(sql_values) + [source_param, source_param]
        else:
            where_clause = f"WHERE {node_filter}"
            sql_values = [source_param, source_param]

    # Query to get aggregated edges (author/title pairs) - DIRECTED
    edge_query = f"""
        SELECT
            {source_field} as source,
            {target_field} as target,
            COUNT(*) as weight
        FROM {other_args.db_table}
        {where_clause}
        GROUP BY {source_field}, {target_field}
        HAVING COUNT(*) >= %s
            AND {source_field} IS NOT NULL
            AND {target_field} IS NOT NULL
            AND {source_field} != ''
            AND {target_field} != ''
            AND {source_field} != {target_field}
        ORDER BY weight DESC
    """

    # Execute with threshold parameter added to sql_values
    threshold_params = list(sql_values) + [min_threshold]
    cursor.execute(edge_query, threshold_params)

    edges = []
    node_weights_total = defaultdict(int)  # Track total connections per node
    node_weights_as_source = defaultdict(int)  # Track times as source
    node_weights_as_target = defaultdict(int)  # Track times as target
    node_set = set()

    for row in cursor:
        source = row["source"]
        target = row["target"]
        weight = row["weight"]

        # Skip empty strings (extra safety check)
        if not source or not target:
            continue

        edges.append({
            "source": source,
            "target": target,
            "weight": weight
        })

        node_set.add(source)
        node_set.add(target)
        node_weights_total[source] += weight
        node_weights_total[target] += weight
        node_weights_as_source[source] += weight
        node_weights_as_target[target] += weight

    # Limit to top N nodes by total weight (most connected)
    if len(node_set) > max_nodes:
        top_nodes = set(sorted(node_weights_total.keys(), key=lambda x: node_weights_total[x], reverse=True)[:max_nodes])
        # Filter edges to only include top nodes
        edges = [e for e in edges if e["source"] in top_nodes and e["target"] in top_nodes]

    # Always recalculate node_set based on final edges (handles self-loop filtering and top-node filtering)
    node_set = set()
    for edge in edges:
        node_set.add(edge["source"])
        node_set.add(edge["target"])

    # Calculate centrality metrics using NetworkX
    centrality_mode = request.query_params.get("centrality", "degree")  # degree, eigenvector, or betweenness

    # Build NetworkX graph for centrality calculations
    G = nx.DiGraph()
    for edge in edges:
        G.add_edge(edge["source"], edge["target"], weight=edge["weight"])

    # Calculate centrality based on mode
    if centrality_mode == "eigenvector":
        try:
            centrality = nx.eigenvector_centrality(G, max_iter=100, weight='weight')
        except:
            # Fall back to degree centrality if eigenvector fails (e.g., disconnected graph)
            centrality = dict(G.degree(weight='weight'))
    elif centrality_mode == "betweenness":
        centrality = nx.betweenness_centrality(G, weight='weight')
    else:  # degree (default)
        centrality = dict(G.degree(weight='weight'))

    # Create nodes list with metadata
    nodes = []
    for node_id in node_set:
        nodes.append({
            "id": node_id,
            "label": node_id,
            "size": centrality.get(node_id, 0),  # Size based on centrality metric
            "centrality": centrality.get(node_id, 0),  # Also keep as separate field for reference
            "total_alignments": node_weights_total[node_id],
            "as_source": node_weights_as_source[node_id],
            "as_target": node_weights_as_target[node_id],
            "type": aggregation_field
        })

    conn.rollback()
    conn.close()

    return {
        "nodes": nodes,
        "edges": edges,
        "aggregation_field": aggregation_field,
        "centrality_mode": centrality_mode,
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "threshold": min_threshold
    }




@app.get("/count_by_year/")



@app.get("/network_data/edge_details/")
@app.get("/text-pair-api/network_data/edge_details/")
def get_edge_details(request: Request):
    """Get sample alignments for a specific edge (source->target pair)
    Returns paginated list of actual passage alignments"""
    sql_fields, sql_values, other_args, _ = parse_args(request)

    source_node = request.query_params.get("source")
    target_node = request.query_params.get("target")
    node_type = request.query_params.get("node_type", "author")
    limit = int(request.query_params.get("limit", 50))
    offset = int(request.query_params.get("offset", 0))

    if not source_node or not target_node:
        return {"error": "source and target parameters are required"}

    conn = psycopg2.connect(
        user=GLOBAL_CONFIG["DATABASE"]["database_user"],
        password=GLOBAL_CONFIG["DATABASE"]["database_password"],
        database=GLOBAL_CONFIG["DATABASE"]["database_name"],
    )
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    # Determine field names
    if node_type == "author":
        source_field = "source_author"
        target_field = "target_author"
    else:
        source_field = "source_title"
        target_field = "target_title"

    # Build query to get specific alignments (DIRECTED - matches get_network_data logic)
    edge_where = f"({source_field} = %s AND {target_field} = %s)"
    if sql_fields:
        where_clause = f"WHERE {sql_fields} AND {edge_where}"
        query_params = list(sql_values) + [source_node, target_node]
    else:
        where_clause = f"WHERE {edge_where}"
        query_params = [source_node, target_node]

    # Get total count
    count_query = f"SELECT COUNT(*) FROM {other_args.db_table} {where_clause}"
    cursor.execute(count_query, query_params)
    total_count = cursor.fetchone()[0]

    # Get sample alignments
    alignments_query = f"""
        SELECT * FROM {other_args.db_table}
        {where_clause}
        ORDER BY rowid
        LIMIT %s OFFSET %s
    """
    cursor.execute(alignments_query, query_params + [limit, offset])

    alignments = []
    for row in cursor:
        alignments.append(dict(row))

    conn.rollback()
    conn.close()

    return {
        "alignments": alignments,
        "total_count": total_count,
        "limit": limit,
        "offset": offset,
        "source": source_node,
        "target": target_node
    }

@app.get("/get_clusters")
@app.get("/text-pair-api/get_clusters")
def get_clusters(request: Request):
    """Return semantic graph using precomputed cluster model.

    Two modes:
    1. Full graph (no filters): Fast path - delegates to get_full_graph()
    2. Filtered graph (with search params): Rebuild graph from database with filters

    Returns nodes and edges for (author, cluster) pair visualization.
    """
    sql_fields, sql_values, other_args, _ = parse_args(request)

    # Path to precomputed graph model files
    db_name = str(other_args.db_table)
    graph_data_path = os.path.join(APP_PATH, db_name, "graph_data")

    # Check if graph data exists
    if not os.path.exists(graph_data_path):
        return {"error": f"Graph data not found at {graph_data_path}. Please run build_graph_model.py first."}

    # Fast path: if no filters, use the existing get_full_graph function
    if not sql_fields:
        print("Loading full pre-built graph from disk...")
        return get_full_graph(request, db_name)

    # Filtered path: rebuild graph from database with search filters
    print(f"Building filtered graph with query: {sql_fields}")

    # Load precomputed graph model data
    print("Loading precomputed graph model...")
    cluster_labels_modified = np.load(os.path.join(graph_data_path, 'cluster_labels_modified.npy'))
    cluster_centroids_umap = np.load(os.path.join(graph_data_path, 'cluster_centroids_umap.npy'))
    embeddings_umap_2d = np.load(os.path.join(graph_data_path, 'embeddings_umap_2d.npy'))
    cluster_similarity = np.load(os.path.join(graph_data_path, 'cluster_similarity_matrix.npy'))

    with open(os.path.join(graph_data_path, 'author_to_id.json'), 'rb') as f:
        author_to_id = orjson.loads(f.read())

    with open(os.path.join(graph_data_path, 'cluster_metadata.json'), 'rb') as f:
        metadata = orjson.loads(f.read())

    n_clusters = metadata['n_clusters']
    total_clusters = metadata['total_clusters']
    id_to_author = {v: k for k, v in author_to_id.items()}

    # Query alignments from database based on search parameters
    conn = psycopg2.connect(
        user=GLOBAL_CONFIG["DATABASE"]["database_user"],
        password=GLOBAL_CONFIG["DATABASE"]["database_password"],
        database=GLOBAL_CONFIG["DATABASE"]["database_name"],
    )
    register_vector(conn)
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    # Build query to fetch alignments with embeddings
    query = f"SELECT rowid, source_author, target_author, embedding FROM {other_args.db_table} WHERE {sql_fields}"
    cursor.execute(query, sql_values)


    # Process query results - map alignments to (author, cluster) pairs
    from collections import defaultdict
    pair_passage_counts = defaultdict(int)
    pair_embeddings_2d = defaultdict(list)

    for row in cursor:
        rowid = row['rowid']
        source_author = row['source_author']
        target_author = row['target_author']

        # Get cluster assignment for this alignment
        # The rowid should correspond to the index in cluster_labels_modified
        alignment_idx = rowid - 1  # Adjust for 0-indexing if needed
        if alignment_idx < 0 or alignment_idx >= len(cluster_labels_modified):
            continue

        cluster_id = int(cluster_labels_modified[alignment_idx])

        # Get 2D UMAP position for this alignment
        embedding_2d = embeddings_umap_2d[alignment_idx]

        # Track both source and target authors
        for author_name in [source_author, target_author]:
            if author_name not in author_to_id:
                continue

            author_id = author_to_id[author_name]
            pair_key = (author_id, cluster_id)
            pair_passage_counts[pair_key] += 1
            pair_embeddings_2d[pair_key].append(embedding_2d)

    conn.close()

    # Calculate mean 2D positions for each pair
    pair_positions = {}
    for pair_key, embeddings_list in pair_embeddings_2d.items():
        mean_position = np.mean(embeddings_list, axis=0)
        pair_positions[pair_key] = mean_position

    # Build graph structure
    print(f"Building graph with {len(pair_passage_counts)} (author, cluster) pairs...")

    # Create nodes list (colors assigned client-side)
    nodes = []
    for (author_id, cluster_id), passage_count in pair_passage_counts.items():
        node_id = f"author_{author_id}_cluster_{cluster_id}"
        position = pair_positions[(author_id, cluster_id)]

        nodes.append({
            'id': node_id,
            'label': id_to_author[author_id],
            'author_id': author_id,
            'author_name': id_to_author[author_id],
            'cluster_id': cluster_id,
            'size': int(passage_count),
            'x': float(position[0]),
            'y': float(position[1])
        })

    # Group nodes by cluster for edge creation
    cluster_nodes = defaultdict(list)
    for node in nodes:
        cluster_id = node['cluster_id']
        cluster_nodes[cluster_id].append(node['id'])

    # Add invisible cluster anchor nodes
    cluster_centroid_positions_2d = {}
    for cluster_id in range(total_clusters):
        cluster_2d_positions = []
        for pair_key, position in pair_positions.items():
            if pair_key[1] == cluster_id:
                cluster_2d_positions.append(position)

        if cluster_2d_positions:
            mean_2d_position = np.mean(cluster_2d_positions, axis=0)
            cluster_centroid_positions_2d[cluster_id] = mean_2d_position

    for cluster_id, position_2d in cluster_centroid_positions_2d.items():
        anchor_node_id = f"anchor_cluster_{cluster_id}"
        nodes.append({
            'id': anchor_node_id,
            'label': '',
            'node_type': 'cluster_anchor',
            'cluster_id': cluster_id,
            'size': 0.01,
            'x': float(position_2d[0]),
            'y': float(position_2d[1]),
            'hidden': True
        })

    # Create edges list
    edges = []

    # 1. Intra-cluster edges (complete subgraphs)
    intra_weight = 100.0
    for cluster_id, node_ids in cluster_nodes.items():
        if len(node_ids) > 1:
            for node1, node2 in itertools.combinations(node_ids, 2):
                edges.append({
                    'id': f"{node1}_{node2}",
                    'source': node1,
                    'target': node2,
                    'weight': intra_weight,
                    'edge_type': 'intra_cluster',
                    'size': 2.0,
                    'color': '#666666'
                })

    # 2. Anchor connection edges (pairs to anchors)
    anchor_weight = 50.0
    for cluster_id, node_ids in cluster_nodes.items():
        anchor_node_id = f"anchor_cluster_{cluster_id}"
        for node_id in node_ids:
            edges.append({
                'id': f"{node_id}_{anchor_node_id}",
                'source': node_id,
                'target': anchor_node_id,
                'weight': anchor_weight,
                'edge_type': 'anchor_connection',
                'size': 0.5,
                'color': '#444444'
            })

    # 3. Centroid similarity edges (between cluster anchors)
    similarity_threshold = np.percentile(cluster_similarity[np.triu_indices_from(cluster_similarity, k=1)], 75)
    for cluster_i in range(n_clusters):
        for cluster_j in range(cluster_i + 1, n_clusters):
            similarity = cluster_similarity[cluster_i, cluster_j]

            if similarity > similarity_threshold:
                anchor_i = f"anchor_cluster_{cluster_i}"
                anchor_j = f"anchor_cluster_{cluster_j}"

                edges.append({
                    'id': f"{anchor_i}_{anchor_j}",
                    'source': anchor_i,
                    'target': anchor_j,
                    'weight': float(similarity * 10),
                    'edge_type': 'centroid_similarity',
                    'size': 1.5,
                    'color': '#999999'
                })

    return {
        'nodes': nodes,
        'edges': edges,
        'total_nodes': len(nodes),
        'total_edges': len(edges),
        'num_clusters': n_clusters,
        'similarity_threshold': float(similarity_threshold),
        'metadata': metadata  # Include cluster metadata (n_clusters, n_noise, etc)
    }


@app.get("/semantic_graph_data/")
@app.get("/text-pair-api/semantic_graph_data/")
def get_semantic_graph_data(request: Request):
    """Get semantic clustering graph data as simple nodes/edges arrays (like network_data).

    Returns nodes and edges for (author, cluster) pair visualization.
    The Vue component will build the Graphology graph client-side.
    """
    sql_fields, sql_values, other_args, _ = parse_args(request)

    # Path to precomputed graph model files
    db_name = str(other_args.db_table)
    graph_data_path = os.path.join(APP_PATH, db_name, "graph_data")

    # Check if graph data exists
    if not os.path.exists(graph_data_path):
        return {"error": f"Graph data not found at {graph_data_path}. Please run build_graph_model.py first."}

    # Fast path: if no filters, load from full_graph.json
    # if not sql_fields:
    #     print("Loading full pre-built graph from disk...")
    #     full_graph_path = os.path.join(graph_data_path, 'full_graph.json')

    #     if not os.path.exists(full_graph_path):
    #         return {"error": f"Full graph not found at {full_graph_path}"}

    #     with open(full_graph_path, 'rb') as f:
    #         graph_data = orjson.loads(f.read())

    #     print(f"✓ Loaded full graph: {len(graph_data.get('nodes', []))} nodes, {len(graph_data.get('edges', []))} edges")
    #     return graph_data

    # Filtered path: rebuild graph from database with search filters
    print(f"Building filtered graph with query: {sql_fields}")

    # Load precomputed graph model data
    cluster_labels_modified = np.load(os.path.join(graph_data_path, 'cluster_labels_modified.npy'))
    embeddings_umap_2d = np.load(os.path.join(graph_data_path, 'embeddings_umap_2d.npy'))
    cluster_similarity = np.load(os.path.join(graph_data_path, 'cluster_similarity_matrix.npy'))

    with open(os.path.join(graph_data_path, 'author_to_id.json'), 'rb') as f:
        author_to_id = orjson.loads(f.read())

    with open(os.path.join(graph_data_path, 'cluster_metadata.json'), 'rb') as f:
        metadata = orjson.loads(f.read())

    n_clusters = metadata['n_clusters']
    total_clusters = metadata['total_clusters']
    id_to_author = {v: k for k, v in author_to_id.items()}

    # Load cluster labels if available
    cluster_label_map = {}
    cluster_labels_path = os.path.join(graph_data_path, 'cluster_labels.json')
    if os.path.exists(cluster_labels_path):
        with open(cluster_labels_path, 'rb') as f:
            labels_data = orjson.loads(f.read())
            # Convert string keys back to integers
            cluster_label_map = {int(k): v for k, v in labels_data.items()}

    # Query alignments from database
    conn = psycopg2.connect(
        user=GLOBAL_CONFIG["DATABASE"]["database_user"],
        password=GLOBAL_CONFIG["DATABASE"]["database_password"],
        database=GLOBAL_CONFIG["DATABASE"]["database_name"],
    )
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    if sql_fields:
        query = f"SELECT rowid, source_author, target_author FROM {other_args.db_table} WHERE {sql_fields}"
    else:
        query = f"SELECT rowid, source_author, target_author FROM {other_args.db_table}"
    cursor.execute(query, sql_values)


    # Build (author, cluster) pairs from filtered alignments
    from collections import defaultdict
    pair_passage_counts = defaultdict(int)
    pair_embeddings_2d = defaultdict(list)

    for row in cursor:
        rowid = row['rowid']
        source_author = row['source_author']
        target_author = row['target_author']

        alignment_idx = rowid - 1
        if alignment_idx < 0 or alignment_idx >= len(cluster_labels_modified):
            continue

        cluster_id = int(cluster_labels_modified[alignment_idx])
        embedding_2d = embeddings_umap_2d[alignment_idx]

        for author_name in [source_author, target_author]:
            if author_name not in author_to_id:
                continue

            author_id = author_to_id[author_name]
            pair_key = (author_id, cluster_id)
            pair_passage_counts[pair_key] += 1
            pair_embeddings_2d[pair_key].append(embedding_2d)

    conn.close()

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

        nodes.append({
            'id': node_id,
            'label': id_to_author[author_id],
            'author_id': author_id,
            'author_name': id_to_author[author_id],
            'cluster_id': cluster_id,
            'cluster_label': cluster_label_map.get(cluster_id, ''),
            'passages': int(passage_count),
            'size': int(passage_count),
            'x': float(position[0]),
            'y': float(position[1])
        })

    # Group nodes by cluster
    cluster_nodes = defaultdict(list)
    for node in nodes:
        cluster_id = node['cluster_id']
        cluster_nodes[cluster_id].append(node['id'])

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
        nodes.append({
            'id': anchor_node_id,
            'label': '',
            'node_type': 'cluster_anchor',
            'cluster_id': cluster_id,
            'cluster_label': cluster_label_map.get(cluster_id, ''),
            'size': 0.01,
            'x': float(position_2d[0]),
            'y': float(position_2d[1]),
            'hidden': True
        })

    # Build edges array
    edges = []

    # 1. Add intra-cluster edges (connect nodes within same cluster)
    for cluster_id, node_ids in cluster_nodes.items():
        # Only add intra-cluster edges if there are multiple nodes
        if len(node_ids) > 1:
            # Connect each node to cluster anchor (star topology, not complete graph)
            anchor_id = f"anchor_cluster_{cluster_id}"
            for node_id in node_ids:
                edges.append({
                    'source': node_id,
                    'target': anchor_id,
                    'weight': 1.0,
                    'edge_type': 'intra_cluster',
                    'color': '#666666',
                    'size': 0.5
                })

    # 2. Add centroid similarity edges (connect cluster anchors)
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

                edges.append({
                    'source': anchor_i,
                    'target': anchor_j,
                    'weight': float(similarity * 10),
                    'edge_type': 'centroid_similarity',
                    'color': '#999999',
                    'size': 1.0
                })

    return {
        'nodes': nodes,
        'edges': edges,
        'metadata': {
            'n_clusters': n_clusters,
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'min_passages_threshold': MIN_PASSAGES_THRESHOLD
        }
    }


@app.get("/text-pair/{db_path}/get_full_graph")
@app.get("/{db_path}/get_full_graph")
def get_full_graph(request: Request, db_path: str):
    """
    Load and return the pre-built full corpus graph.

    This endpoint loads the complete graph that was pre-computed by build_graph_model.py
    from all alignments in the corpus. Much faster than building on-the-fly since it's
    just loading a JSON file from disk.

    Returns:
        - nodes: List of (author, cluster) pair nodes with positions
        - edges: List of edges (intra-cluster, anchor connections, centroid similarity)
        - metadata: Graph statistics
    """
    _, _, other_args, _ = parse_args(request)

    # Path to precomputed graph files
    db_name = str(other_args.db_table)
    graph_data_path = os.path.join(APP_PATH, db_name, "graph_data")
    full_graph_path = os.path.join(graph_data_path, 'full_graph.json')

    # Check if full graph exists
    if not os.path.exists(full_graph_path):
        return {
            "error": f"Full graph not found at {full_graph_path}. Please run build_graph_model.py with full graph generation enabled."
        }

    # Load the pre-built graph
    print(f"Loading full graph from {full_graph_path}...")
    with open(full_graph_path, 'rb') as f:
        graph_data = orjson.loads(f.read())

    print(f"✓ Loaded full graph: {len(graph_data.get('nodes', []))} nodes, {len(graph_data.get('edges', []))} edges")

    return graph_data


@app.get("/text-pair/{db_path}/get_full_graph_graphology")
@app.get("/{db_path}/get_full_graph_graphology")
def get_full_graph_graphology(request: Request, db_path: str):
    """
    Load and return the pre-built full corpus graph in Graphology format.

    This endpoint returns the graph in Graphology's native JSON format, which can be
    loaded directly by Sigma.js for optimal performance. This is faster than the
    standard format because Sigma.js doesn't need to transform the data.

    Usage in JavaScript:
        const response = await fetch('/text-pair/mydb/get_full_graph_graphology');
        const graphData = await response.json();
        const graph = new graphology.Graph();
        graph.import(graphData);
        const renderer = new Sigma(graph, container);

    Returns:
        Graphology-formatted graph with nodes, edges, and attributes ready for Sigma.js
    """
    _, _, other_args, _ = parse_args(request)

    # Path to precomputed graph files
    db_name = str(other_args.db_table)
    graph_data_path = os.path.join(APP_PATH, db_name, "graph_data")
    graphology_path = os.path.join(graph_data_path, 'full_graph_graphology.json')

    # Check if Graphology format exists
    if not os.path.exists(graphology_path):
        return {
            "error": f"Graphology graph not found at {graphology_path}. Please run build_graph_model.py to generate it."
        }

    # Load the Graphology-formatted graph
    print(f"Loading Graphology graph from {graphology_path}...")
    with open(graphology_path, 'rb') as f:
        graph_data = orjson.loads(f.read())

    print(f"✓ Loaded Graphology graph: {len(graph_data.get('nodes', []))} nodes, {len(graph_data.get('edges', []))} edges")

    return graph_data


@app.get("/{db_path}/search")
@app.get("/text-pair/{db_path}/search")
@app.get("/{db_path}/time")
@app.get("/text-pair/{db_path}/time")
@app.get("/{db_path}")
@app.get("/text-pair/{db_path}")
@app.get("/text-pair/{db_path}/group/{id}")
@app.get("/{db_path}/group/{id}")
@app.get("/text-pair/{db_path}/sorted-results")
@app.get("/{db_path}/sorted-results")
@app.get("/text-pair/{db_path}/text-view")
@app.get("/{db_path}/text-view")
@app.get("/text-pair/{db_path}/network")
@app.get("/{db_path}/network")
def index(db_path: str):
    """Return index.html which lists available POOLs"""
    with open(os.path.join(APP_PATH, db_path, "dist/index.html"), encoding="utf8") as html:
        index_html = html.read()
    return HTMLResponse(index_html)