#!/usr/bin/env python3
"""Routing and search code for sequence alignment"""

import configparser
import os
import re
from collections import Counter, OrderedDict
from typing import Dict
from pathlib import Path
import time

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
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


BOOLEAN_ARGS = re.compile(r"""(NOT \w+)|(OR \w+)|(\w+)|("")""")


class formArguments:
    """Special dict to handle form arguments"""

    def __init__(self):
        self.dict = OrderedDict()

    def __getitem__(self, item):
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
        else:
            return ""

    def __getattr__(self, attr):
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
    query_args = formArguments()
    other_args = formArguments()
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
    ]
    for key, value in request.query_params.items():
        if key in other_args_keys:
            if key in ("page", "id_anchor", "timeSeriesInterval"):
                if value.isdigit():
                    other_args[key] = int(value)
            elif key == "direction":
                other_args["direction"] = value or "next"
            elif key == "directionSelected":
                other_args["directionSelected"] = value or "source"
            else:
                other_args[key] = value
        else:
            if value:
                query_args[key] = value
    metadata_field_types = get_pg_type(other_args["db_table"])
    metadata_field_types["rowid"] = "INTEGER"
    sql_fields, sql_values = query_builder(query_args, other_args, metadata_field_types)
    return sql_fields, sql_values, other_args, list(metadata_field_types.keys())


def query_builder(query_args, other_args, field_types):
    """Takes query arguments and returns an SQL WHERE clause"""
    sql_fields = []
    sql_values = []
    for field, value in query_args.items():
        value = value.strip()
        field_type = field_types.get(field, "TEXT").upper()
        query = ""
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
                    sql_values.append(f"\m{split_value}\M")
                # elif value.startswith("OR "):  ## TODO: add support to OR queries by changing the join logic at the end of the function
                #     split_value = " ".join(value.split()[1:]).strip()
                #     query = "{} !~* %s".format(field)
                #     sql_values.append('\m{}\M'.format(split_value))
                else:
                    query = f"{field} ~* %s"
                    sql_values.append(f"\m{value}\M")
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
    return " AND ".join(sql_fields), sql_values


@app.get("/")
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
def get_css_resource(db_path: str, resource: str):
    """Retrieve CSS resources"""
    with open(os.path.join(APP_PATH, db_path, "dist/css", resource)) as resource_file:
        resource = resource_file.read()
    return Response(resource, media_type="text/css")


@app.get("/{db_path}/js/{resource}")
def get_js_resource(db_path: str, resource: str):
    """Retrieve JS resources"""
    with open(os.path.join(APP_PATH, db_path, "dist/js", resource)) as resource_file:
        resource = resource_file.read()
    return Response(resource, media_type="application/javascript")


@app.get("/{db_path}/search")
@app.get("/{db_path}/time")
@app.get("/{db_path}")
def index(db_path: str):
    """Return index.html which lists available POOLs"""
    with open(os.path.join(APP_PATH, db_path, "dist/index.html")) as html:
        index_html = html.read()
    return HTMLResponse(index_html)


@app.get("/search_alignments/")
def search_alignments(request: Request):
    """Search alignments according to URL params"""
    sql_fields, sql_values, other_args, column_names = parse_args(request)
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
    with psycopg2.connect(
        user=GLOBAL_CONFIG["DATABASE"]["database_user"],
        password=GLOBAL_CONFIG["DATABASE"]["database_password"],
        database=GLOBAL_CONFIG["DATABASE"]["database_name"],
    ) as conn:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute(query, sql_values)
        alignments = []
        for row in cursor:
            metadata = {key: row[key] for key in column_names}
            metadata["rowid_ordered"] = row["rowid_ordered"]
            alignments.append(metadata)
    if other_args.direction == "previous":
        alignments.reverse()
    previous_url = ""
    current_path = re.sub(r"&(page|id_anchor|direction)=(previous|next|\d*)", "", request.url.path)
    if other_args.page > 1:
        previous_url = (
            f"{current_path}&page={other_args.page - 1}&id_anchor={alignments[0]['rowid_ordered']}&direction=previous"
        )
    try:
        next_url = (
            f"{current_path}&page={other_args.page + 1}&id_anchor={alignments[-1]['rowid_ordered']}&direction=next"
        )
    except IndexError:
        next_url = ""
    start_position = 0
    if other_args.page > 1:
        start_position = 50 * (other_args.page - 1)
    return {
        "alignments": alignments,
        "page": other_args.page,
        "next_url": next_url,
        "previous_url": previous_url,
        "start_position": start_position,
    }


@app.get("/{db_path}/retrieve_all_docs/")
def retrieve_all(request: Request):
    """Retrieve all docs and only return metadata"""
    sql_fields, sql_values, other_args, column_names = parse_args(request)
    if other_args.field.startswith("source_"):
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
    with psycopg2.connect(
        user=GLOBAL_CONFIG["DATABASE"]["database_user"],
        password=GLOBAL_CONFIG["DATABASE"]["database_password"],
        database=GLOBAL_CONFIG["DATABASE"]["database_name"],
    ) as conn:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        if sql_values:
            query = f"""SELECT * FROM {other_args.db_table} WHERE {other_args.field}='{other_args.value}' AND {sql_fields}"""
            cursor.execute(query, sql_values)
        else:
            query = f"""SELECT * FROM {other_args.db_table} WHERE {other_args.field}='{other_args.value}'"""
            cursor.execute(query)
        for row in cursor:
            if row[doc_id] not in docs_found:
                docs_found[row[doc_id]] = {"count": 0, **{key: row[key] for key in column_names}}
            docs_found[row[doc_id]]["count"] += 1
    return list(docs_found.values())


@app.get("/retrieve_all_passage_pairs/")
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
    with psycopg2.connect(
        user=GLOBAL_CONFIG["DATABASE"]["database_user"],
        password=GLOBAL_CONFIG["DATABASE"]["database_password"],
        database=GLOBAL_CONFIG["DATABASE"]["database_name"],
    ) as conn:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute(query, sql_values)
        results = [{key: row[key] for key in column_names} for row in cursor]
    return results


@app.get("/count_results/")
def count_results(request: Request):
    """Search alignments according to URL params"""
    sql_fields, sql_values, other_args, _ = parse_args(request)
    if sql_fields:
        query = f"SELECT COUNT(*) FROM {other_args.db_table} WHERE {sql_fields}"
    else:
        query = f"SELECT COUNT(*) FROM {other_args.db_table}"
    with psycopg2.connect(
        user=GLOBAL_CONFIG["DATABASE"]["database_user"],
        password=GLOBAL_CONFIG["DATABASE"]["database_password"],
        database=GLOBAL_CONFIG["DATABASE"]["database_name"],
    ) as conn:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute(query, sql_values)
        result_object = {"counts": cursor.fetchone()[0]}
    return result_object


@app.post("/generate_time_series/")
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
    with psycopg2.connect(
        user=GLOBAL_CONFIG["DATABASE"]["database_user"],
        password=GLOBAL_CONFIG["DATABASE"]["database_password"],
        database=GLOBAL_CONFIG["DATABASE"]["database_name"],
    ) as conn:
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
def facets(request: Request):
    """Retrieve facet result"""
    sql_fields, sql_values, other_args, _ = parse_args(request)
    results = []
    total_count = 0
    with psycopg2.connect(
        user=GLOBAL_CONFIG["DATABASE"]["database_user"],
        password=GLOBAL_CONFIG["DATABASE"]["database_password"],
        database=GLOBAL_CONFIG["DATABASE"]["database_name"],
    ) as conn:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        if sql_fields:
            query = f"SELECT {other_args.facet}, COUNT(*) FROM {other_args.db_table} \
                WHERE {sql_fields} GROUP BY {other_args.facet} ORDER BY COUNT(*) DESC"
        else:
            query = f"SELECT {other_args.facet}, COUNT(*) FROM {other_args.db_table} \
                GROUP BY {other_args.facet} ORDER BY COUNT(*) DESC"
        cursor.execute(query, sql_values)
        if not other_args.facet.endswith("passage_length"):
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
def metadata(request: Request):
    """Retrieve all searchable metadata fields"""
    _, _, _, metadata_fields = parse_args(request)
    return metadata_fields
