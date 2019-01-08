#!/usr/bin/env python3
"""Routing and search code for sequence alignment"""

import configparser
import re
from collections import OrderedDict, Counter

import psycopg2
import psycopg2.extras
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

application = Flask(__name__)
CORS(application)

GLOBAL_CONFIG = configparser.ConfigParser()
GLOBAL_CONFIG.read("/etc/text-pair/global_settings.ini")

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
        else:
            return False

    def __contains__(self, key):
        if key in self.dict:
            return True
        else:
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
    database = psycopg2.connect(
        user=GLOBAL_CONFIG["DATABASE"]["database_user"],
        password=GLOBAL_CONFIG["DATABASE"]["database_password"],
        database=GLOBAL_CONFIG["DATABASE"]["database_name"],
    )
    cursor = database.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cursor.execute(f"select * from {table_name}")
    type_mapping = {23: "INTEGER", 25: "TEXT"}
    field_types = {column.name: type_mapping[column.type_code] for column in cursor.description}
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
    ]
    for key, value in request.args.items():
        if key in other_args_keys:
            if key == "page" or key == "id_anchor" or key == "timeSeriesInterval":
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
                        query = "{} = %s".format(field)
                        sql_values.append("")
                    else:
                        query = "{}=%s".format(field)
                        sql_values.append(value[1:-1])
                elif value.startswith("NOT "):
                    split_value = " ".join(value.split()[1:]).strip()
                    query = "{} !~* %s".format(field)
                    sql_values.append("\m{}\M".format(split_value))
                # elif value.startswith("OR "):  ## TODO: add support to OR queries by changing the join logic at the end of the function
                #     split_value = " ".join(value.split()[1:]).strip()
                #     query = "{} !~* %s".format(field)
                #     sql_values.append('\m{}\M'.format(split_value))
                else:
                    query = "{} ~* %s".format(field)
                    sql_values.append("\m{}\M".format(value))
                sql_fields.append(query)
        elif field_type == "INTEGER":
            if "-" in value:
                values = [v for v in re.split(r"(-)", value) if v]
                if values[0] == "-":
                    query = "{} <= %s".format(field)
                    sql_values.append(values[1])
                elif values[-1] == "-":
                    query = "{} >= %s".format(field)
                    sql_values.append(values[0])
                else:
                    query = "{} BETWEEN %s AND %s".format(field)
                    sql_values.extend([values[0], values[2]])
            else:
                query = "{} = %s".format(field)
                sql_values.append(value)
            sql_fields.append(query)
        else:
            continue
    if other_args.banality != "":
        sql_fields.append("banality=%s")
        sql_values.append(other_args.banality)
    return " AND ".join(sql_fields), sql_values


@application.route("/")
def index():
    """Return index.html which lists available databases"""
    return render_template("index.html")


@application.route("/search_alignments/", methods=["GET", "POST"])
def search_alignments():
    """Search alignments according to URL params"""
    sql_fields, sql_values, other_args, column_names = parse_args(request)
    if other_args.direction == "next":
        if sql_fields:
            query = "SELECT o.rowid_ordered, m.* FROM {} m, {}_ordered o WHERE {} AND o.source_year_target_year=m.rowid and \
                    o.rowid_ordered > {} ORDER BY o.rowid_ordered LIMIT 50".format(
                other_args.db_table, other_args.db_table, sql_fields, other_args.id_anchor
            )
        else:
            query = "SELECT o.rowid_ordered, m.* FROM {} m, {}_ordered o WHERE o.source_year_target_year=m.rowid and \
                    o.rowid_ordered > {} ORDER BY o.rowid_ordered LIMIT 50".format(
                other_args.db_table, other_args.db_table, other_args.id_anchor
            )
    else:
        if sql_fields:
            query = "SELECT o.rowid_ordered, m.* FROM {} m, {}_ordered o WHERE {} AND o.source_year_target_year=m.rowid and \
                    o.rowid_ordered < {} ORDER BY o.rowid_ordered desc LIMIT 50".format(
                other_args.db_table, other_args.db_table, sql_fields, other_args.id_anchor
            )
        else:
            query = "SELECT o.rowid_ordered, m.* FROM {} m, {}_ordered o WHERE o.source_year_target_year=m.rowid and \
                    o.rowid_ordered < {} ORDER BY o.rowid_ordered desc LIMIT 50".format(
                other_args.db_table, other_args.db_table, other_args.id_anchor
            )
    database = psycopg2.connect(
        user=GLOBAL_CONFIG["DATABASE"]["database_user"],
        password=GLOBAL_CONFIG["DATABASE"]["database_password"],
        database=GLOBAL_CONFIG["DATABASE"]["database_name"],
    )
    cursor = database.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cursor.execute(query, sql_values)
    alignments = []
    for row in cursor:
        metadata = {key: row[key] for key in column_names}
        metadata["rowid_ordered"] = row["rowid_ordered"]
        alignments.append(metadata)
    if other_args.direction == "previous":
        alignments.reverse()
    previous_url = ""
    current_path = re.sub(r"&(page|id_anchor|direction)=(previous|next|\d*)", "", request.path)
    if other_args.page > 1:
        previous_url = "{}&page={}&id_anchor={}&direction=previous".format(
            current_path, other_args.page - 1, alignments[0]["rowid_ordered"]
        )
    try:
        next_url = "{}&page={}&id_anchor={}&direction=next".format(
            current_path, other_args.page + 1, alignments[-1]["rowid_ordered"]
        )
    except IndexError:
        next_url = ""
    start_position = 0
    if other_args.page > 1:
        start_position = 50 * (other_args.page - 1)
    result_object = {
        "alignments": alignments,
        "page": other_args.page,
        "next_url": next_url,
        "previous_url": previous_url,
        "start_position": start_position,
    }
    response = jsonify(result_object)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@application.route("/count_results/", methods=["GET", "POST"])
def count_results():
    """Search alignments according to URL params"""
    sql_fields, sql_values, other_args, _ = parse_args(request)
    if sql_fields:
        query = "SELECT COUNT(*) FROM {} WHERE {}".format(other_args.db_table, sql_fields)
    else:
        query = "SELECT COUNT(*) FROM {}".format(other_args.db_table)
    database = psycopg2.connect(
        user=GLOBAL_CONFIG["DATABASE"]["database_user"],
        password=GLOBAL_CONFIG["DATABASE"]["database_password"],
        database=GLOBAL_CONFIG["DATABASE"]["database_name"],
    )
    cursor = database.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cursor.execute(query, sql_values)
    result_object = {"counts": cursor.fetchone()[0]}
    response = jsonify(result_object)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@application.route("/generate_time_series/", methods=["GET", "POST"])
def generate_time_series():
    """Generate a time series from search results"""
    # TODO: don't assume year is the field to use
    sql_fields, sql_values, other_args, _ = parse_args(request)
    if sql_fields:
        query = "select interval AS year, COUNT(*) FROM \
                (SELECT floor({}_year/{})*{} AS interval FROM {} WHERE {}) t \
                GROUP BY interval ORDER BY interval".format(
            other_args.directionSelected,
            other_args.timeSeriesInterval,
            other_args.timeSeriesInterval,
            other_args.db_table,
            sql_fields,
        )
    else:
        query = "select interval AS year, COUNT(*) FROM \
                (SELECT floor({}_year/{})*{} AS interval FROM {}) t \
                GROUP BY interval ORDER BY interval".format(
            other_args.directionSelected,
            other_args.timeSeriesInterval,
            other_args.timeSeriesInterval,
            other_args.db_table,
        )
    database = psycopg2.connect(
        user=GLOBAL_CONFIG["DATABASE"]["database_user"],
        password=GLOBAL_CONFIG["DATABASE"]["database_password"],
        database=GLOBAL_CONFIG["DATABASE"]["database_name"],
    )
    cursor = database.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cursor.execute(query, sql_values)
    results = []
    total_results = 0
    next_year = None
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
    response = jsonify({"counts": total_results, "results": results})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@application.route("/facets/", methods=["GET", "POST"])
def facets():
    """Retrieve facet result"""
    sql_fields, sql_values, other_args, _ = parse_args(request)
    database = psycopg2.connect(
        user=GLOBAL_CONFIG["DATABASE"]["database_user"],
        password=GLOBAL_CONFIG["DATABASE"]["database_password"],
        database=GLOBAL_CONFIG["DATABASE"]["database_name"],
    )
    cursor = database.cursor(cursor_factory=psycopg2.extras.DictCursor)
    if sql_fields:
        query = "SELECT {}, COUNT(*) FROM {} WHERE {} GROUP BY {} ORDER BY COUNT(*) DESC".format(
            other_args.facet, other_args.db_table, sql_fields, other_args.facet
        )
    else:
        query = "SELECT {}, COUNT(*) FROM {} GROUP BY {} ORDER BY COUNT(*) DESC".format(
            other_args.facet, other_args.db_table, other_args.facet
        )
    cursor.execute(query, sql_values)
    results = []
    total_count = 0
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
    response = jsonify({"facet": other_args.facet, "results": results, "total_count": total_count})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@application.route("/metadata/", methods=["GET"])
def metadata():
    """Retrieve all searchable metadata fields"""
    _, _, _, metadata_fields = parse_args(request)
    response = jsonify(metadata_fields)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response
