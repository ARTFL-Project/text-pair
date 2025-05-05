#!/usr/bin/env python3
"""Routing and search code for sequence alignment"""

import configparser
import os
import re
import time
from collections import Counter, OrderedDict, defaultdict, namedtuple
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
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
def index(db_path: str):
    """Return index.html which lists available POOLs"""
    with open(os.path.join(APP_PATH, db_path, "dist/index.html"), encoding="utf8") as html:
        index_html = html.read()
    return HTMLResponse(index_html)