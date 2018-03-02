#!/usr/bin/env python3
"""Routing and search code for sequence alignment"""

import configparser
import re
from ast import literal_eval as eval
from collections import OrderedDict

import psycopg2
import psycopg2.extras
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

application = Flask(__name__)
CORS(application)

GLOBAL_CONFIG = configparser.ConfigParser()
GLOBAL_CONFIG.read("/etc/text-align/global_settings.ini")


class formArguments():
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
        elif item == "full":
            return False
        elif item == "direction":
            return "next"
        elif item == "interval":
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


def query_builder(query_args, field_types):
    """Takes query arguments and returns an SQL WHERE clause"""
    sql_fields = []
    sql_values = []
    for field, value in query_args.items():
        value = value.strip()
        field_type = field_types.get(field, "TEXT").upper()
        query = ""
        if field_type == "TEXT":
            if value.startswith('"'):
                query = "{}=%s".format(field)
                sql_values.append(value[1:-1])
            elif value.startswith("NOT "):
                split_value = " ".join(value.split()[1:]).strip()
                query = "{} ~* %s".format(field)
                sql_values.append('[^\\m{}\\M]'.format(split_value))
            else:
                query = "{} ~* %s".format(field)
                sql_values.append('\m{}\M'.format(value))
        elif field_type == "INTEGER":
            if "-" in value:
                values = [v for v in re.split(r"(-)", value) if v]
                if values[0] == "-":
                    query = "{} < %s".format(field)
                    sql_values.append(values[1])
                elif values[-1] == "-":
                    query = "{} > %s".format(field)
                    sql_values.append(values[0])
                else:
                    query = "{} BETWEEN %s AND %s".format(field)
                    sql_values.extend([values[0], values[2]])
            else:
                query = '{} = %s'.format(field)
                sql_values.append(value)
        else:
            continue
        sql_fields.append(query)
    import sys
    print(sql_fields, file=sys.stderr)
    return " AND ".join(sql_fields), sql_values


@application.route("/")
def index():
    """Return index.html which lists available databases"""
    return render_template("index.html")

@application.route("/search_alignments/", methods=["GET", "POST"])
def search_alignments():
    """Search alignments according to URL params"""
    query_args, other_args = parse_args(request)
    metadata_field_types = request.get_json()["metadata"]
    sql_fields, sql_values = query_builder(query_args, metadata_field_types)
    if other_args.direction == "next":
        if query_args:
            query = "SELECT o.rowid_ordered, m.* FROM {} m, {}_ordered o WHERE {} AND o.source_year_target_year=m.rowid and \
                    o.rowid_ordered > {} ORDER BY o.rowid_ordered LIMIT 50".format(other_args.db_table, other_args.db_table,
                    sql_fields, other_args.id_anchor)
        else:
            query = "SELECT o.rowid_ordered, m.* FROM {} m, {}_ordered o WHERE o.source_year_target_year=m.rowid and \
                    o.rowid_ordered > {} ORDER BY o.rowid_ordered LIMIT 50".format(other_args.db_table, other_args.db_table,
                    other_args.id_anchor)
    else:
        if query_args:
            query = "SELECT o.rowid_ordered, m.* FROM {} m, {}_ordered o WHERE {} AND o.source_year_target_year=m.rowid and \
                    o.rowid_ordered < {} ORDER BY o.rowid_ordered desc LIMIT 50".format(other_args.db_table, other_args.db_table,
                    sql_fields, other_args.id_anchor)
        else:
            query = "SELECT o.rowid_ordered, m.* FROM {} m, {}_ordered o WHERE o.source_year_target_year=m.rowid and \
                    o.rowid_ordered < {} ORDER BY o.rowid_ordered desc LIMIT 50".format(other_args.db_table, other_args.db_table,
                    other_args.id_anchor)
    database = psycopg2.connect(user=GLOBAL_CONFIG["DATABASE"]["database_user"],
                                password=GLOBAL_CONFIG["DATABASE"]["database_password"],
                                database=GLOBAL_CONFIG["DATABASE"]["database_name"])
    cursor = database.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cursor.execute(query, sql_values)
    column_names = [desc[0] for desc in cursor.description]
    alignments = []
    for row in cursor:
        metadata = {key: row[key] for key in column_names}
        alignments.append(metadata)
    if other_args.direction == "previous":
        alignments.reverse()
    previous_url = ""
    current_path = re.sub(r'&(page|id_anchor|direction)=(previous|next|\d*)', '', request.path)
    if other_args.page > 1:
        previous_url = "{}&page={}&id_anchor={}&direction=previous".format(current_path, other_args.page-1, alignments[0]["rowid_ordered"])
    try:
        next_url = "{}&page={}&id_anchor={}&direction=next".format(current_path, other_args.page+1, alignments[-1]["rowid_ordered"])
    except IndexError:
        next_url = ""
    start_position = 0
    if other_args.page > 1:
        start_position = 50 * (other_args.page - 1)
    result_object = {"alignments": alignments, "page": other_args.page, "next_url": next_url, "previous_url": previous_url, "start_position": start_position}
    response = jsonify(result_object)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@application.route("/generate_time_series/", methods=["GET", "POST"])
def generate_time_series():
    """Generate a time series from search results"""
    #TODO: don't assume year is the field to use
    query_args, other_args = parse_args(request)
    metadata_field_types = request.get_json()["metadata"]
    sql_fields, sql_values = query_builder(query_args, metadata_field_types)
    if query_args:
        query = "select interval AS year, COUNT(*) FROM \
                (SELECT floor({}_year/{})*{} AS interval FROM {} WHERE {}) t \
                GROUP BY interval ORDER BY interval".format(other_args.directionSelected, other_args.interval, other_args.interval, other_args.db_table, sql_fields)
    else:
        query = "select interval AS year, COUNT(*) FROM \
                (SELECT floor({}_year/{})*{} AS interval FROM {}) t \
                GROUP BY interval ORDER BY interval".format(other_args.directionSelected, other_args.interval, other_args.interval, other_args.db_table)
    database = psycopg2.connect(user=GLOBAL_CONFIG["DATABASE"]["database_user"],
                                password=GLOBAL_CONFIG["DATABASE"]["database_password"],
                                database=GLOBAL_CONFIG["DATABASE"]["database_name"])
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
                next_year += other_args.interval
        results.append({"year": year, "count": count})
        next_year = year + other_args.interval
        total_results += count
    response = jsonify({"counts": total_results, "results": results})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@application.route("/facets/", methods=["POST"])
def facets():
    """Retrieve facet result"""
    query_args, other_args = parse_args(request)
    metadata_field_types = request.get_json()["metadata"]
    sql_fields, sql_values = query_builder(query_args, metadata_field_types)
    database = psycopg2.connect(user=GLOBAL_CONFIG["DATABASE"]["database_user"],
                                password=GLOBAL_CONFIG["DATABASE"]["database_password"],
                                database=GLOBAL_CONFIG["DATABASE"]["database_name"])
    cursor = database.cursor(cursor_factory=psycopg2.extras.DictCursor)
    if query_args:
        query = "SELECT {}, COUNT(*) FROM {} WHERE {} GROUP BY {} ORDER BY COUNT(*) DESC".format(
            other_args.facet, other_args.db_table, sql_fields, other_args.facet)
    else:
        query = "SELECT {}, COUNT(*) FROM {} GROUP BY {} ORDER BY COUNT(*) DESC".format(
            other_args.facet, other_args.db_table, other_args.facet)
    cursor.execute(query,sql_values)
    results = []
    for result in cursor:
        field_name, count = result
        results.append({
            "field": field_name,
            "count": count
        })
    response = jsonify({"facet": other_args.facet, "results": results})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

def parse_args(api_request):
    """Parse URL args"""
    query_args = formArguments()
    other_args = formArguments()
    other_args_keys = ["facet", "direction", "source", "target", "stats_field", "db_table",
                       "filter_field", "filter_value", "page", "id_anchor", "directionSelected",
                       "timeSeriesInterval"]
    for key, value in api_request.args.items():
        if key in other_args_keys:
            if key == "full":
                try:
                    other_args["full"] = eval(value.title())
                except ValueError:
                    pass
            elif key == "page":
                try:
                    other_args["page"] = int(value)
                except TypeError:
                    pass
            elif key == "id_anchor":
                try:
                    other_args["id_anchor"] = int(value)
                except TypeError:
                    pass
            elif key == "direction":
                other_args["direction"] = value or "next"
            elif key == "directionSelected":
                other_args["directionSelected"] = value or "source"
            elif key == "timeSeriesInterval":
                other_args["interval"] = int(value)
            else:
                other_args[key] = value
        else:
            if value:
                query_args[key] = value
    return query_args, other_args
