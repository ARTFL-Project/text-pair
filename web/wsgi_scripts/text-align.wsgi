#!/usr/bin/env python3
"""Routing and search code for sequence alignment"""

import json
import os
import re
import sys
from ast import literal_eval as eval
from collections import OrderedDict, Counter, defaultdict
from pathlib import Path

import psycopg2
import psycopg2.extras
from flask import (Flask, jsonify, redirect, render_template, request,
                   send_from_directory)
from philologic.runtime.link import byte_range_to_link

ROOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./")
application = Flask(__name__,
                    template_folder=ROOT_PATH,
                    static_folder=os.path.join(ROOT_PATH, "assets"))


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
        for k, v in self.dict.items():
            yield k, v

    def values(self):
        for _, v in self.dict.items():
            yield v

    def __str__(self):
        return repr(self.dict)


@application.route("/")
def index():
    return render_template("index.html")

# @application.route('/<path:path>', methods=['GET'])
# def return_app(path):
#     main_config = load_json("{}/db_configs.json".format(ROOT_PATH))
#     path = path.split('/')[0]
#     return redirect(main_config[path])

@application.route("/search_alignments/", methods=["GET"])
def search_alignments():
    search_args, other_args = parse_args(request)
    if other_args.direction == "next":
        query = "SELECT o.rowid_ordered, m.* FROM {} m, {}_ordered o WHERE {} AND o.source_year_target_year=m.rowid and \
                o.rowid_ordered > {} ORDER BY o.rowid_ordered LIMIT 50".format(other_args.db_table, other_args.db_table,
                " and ".join([i + " ilike %s " for i in search_args if search_args[i]]), other_args.id_anchor)
    else:
        query = "SELECT o.rowid_ordered, m.* FROM {} m, {}_ordered o WHERE {} AND o.source_year_target_year=m.rowid and \
                o.rowid_ordered < {} ORDER BY o.rowid_ordered desc LIMIT 50".format(other_args.db_table, other_args.db_table,
                " and ".join([i + " ilike %s " for i in search_args if search_args[i]]), other_args.id_anchor)
    DATABASE = psycopg2.connect(user="alignments", password="martini", database="alignments")
    CURSOR = DATABASE.cursor(cursor_factory=psycopg2.extras.DictCursor)
    CURSOR.execute(query, ["%{}%".format(v) for v in search_args.values() if v])
    column_names = [desc[0] for desc in CURSOR.description]
    alignments = []
    for pos, row in enumerate(CURSOR):
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

@application.route("/search_alignments_full/", methods=["GET"])
def search_alignments_full():
    search_args, other_args = parse_args(request)
    database = psycopg2.connect(user="alignments", password="martini", database="alignments")
    cursor = database.cursor(cursor_factory=psycopg2.extras.DictCursor)
    if search_args is True:
        query = "SELECT {}, {} FROM {} WHERE {} != '' and {} != '' and {}".format(
            other_args.source, other_args.target, other_args.db_table, other_args.source, other_args.target,
            " and ".join([i + " ilike %s " for i in search_args if search_args[i]]))
        cursor.execute(query, ["%{}%".format(v) for v in search_args.values() if v])
    else:
        query = "SELECT {}, {} FROM {} WHERE {} != '' and {} != ''".format(
            other_args.source, other_args.target, other_args.db_table, other_args.source, other_args.target)
        cursor.execute(query)
    results = []
    for source, target in cursor:
        results.append((source, target))
    response = jsonify({
        "source": other_args.source,
        "target": other_args.target,
        "results": results
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@application.route("/facets/", methods=["GET"])
def facets():
    search_args, other_args = parse_args(request)
    database = psycopg2.connect(user="alignments", password="martini", database="alignments")
    cursor = database.cursor(cursor_factory=psycopg2.extras.DictCursor)
    query = "SELECT {}, COUNT(*) FROM {} WHERE {} GROUP BY {} ORDER BY COUNT(*) DESC".format(
        other_args.facet, other_args.db_table, " and ".join([i + " ilike %s " for i in search_args if search_args[i]]), other_args.facet)
    cursor.execute(query, ["%{}%".format(v) for v in search_args.values() if v])
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

@application.route("/stats/", methods=["GET"])
def stats():
    search_args, other_args = parse_args(request)
    database = psycopg2.connect(user="alignments", password="martini", database="alignments")
    cursor = database.cursor(cursor_factory=psycopg2.extras.DictCursor)
    if search_args:
        query = "SELECT {}, COUNT(*) FROM {} WHERE {} AND earliest=%s GROUP BY {} ORDER BY COUNT(*) DESC LIMIT 100".format(
            other_args.stats_field, other_args.db_table, " and ".join([i + " ilike %s " for i in search_args if search_args[i]]), other_args.stats_field
        )
    else:
        query = "SELECT {}, COUNT(*) FROM {} WHERE earliest=%s GROUP BY {} ORDER BY COUNT(*) DESC LIMIT 100".format(
            other_args.stats_field, other_args.db_table, other_args.stats_field
        )
    if other_args.direction == "source":
        cursor.execute(query, ["%{}%".format(v) for v in search_args.values() if v] + ["yes"])
    else:
        cursor.execute(query, ["%{}%".format(v) for v in search_args.values() if v] + ["no"])  # give instances where passage is a reuse
    results = []
    for result in cursor:
        field_name, count = result
        results.append({
            "label": field_name,
            "count": count
        })
    # results = sorted([{"label": field_name, "count": result["count"], "passage_id": result["passage_id"]} for field_name, result in results.items()], key=lambda x: x["count"], reverse=True)
    response = jsonify({"stats_field": other_args.stats_field, "results": results})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@application.route("/get_most_reused_passages/", methods=["GET"])
def get_most_reused_passages():
    search_args, other_args = parse_args(request)
    database = psycopg2.connect(user="alignments", password="martini", database="alignments")
    cursor = database.cursor(cursor_factory=psycopg2.extras.DictCursor)
    query = "SELECT * FROM {} WHERE {}".format(
        other_args.db_table, " and ".join([i + " ilike %s " for i in search_args if search_args[i]])
    )
    cursor.execute(query, ["%{}%".format(v) for v in search_args.values() if v])
    column_names = [desc[0] for desc in cursor.description]
    results = {}
    for pos, row in enumerate(cursor):
        metadata = {key: row[key] for key in column_names}
        passage_id = metadata["passage_id"]
        if passage_id not in results:
            results[passage_id] = {
                "metadata": metadata,
                "count": 0
            }
        results[passage_id]["count"] += 1
    results = [v for k, v in sorted(results.items(), key=lambda x: x[1]["count"], reverse=True)]
    response = jsonify({"results": results})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@application.route("/link_to_philologic", methods=["GET"])
def link_to_philologic():
    _, other_args = parse_args(request)
    db_path = str(Path(request.args["filename"]).parent).replace("data/TEXT", "")
    print(db_path, file=sys.stderr)
    philologic_link = byte_range_to_link(db_path, request.args["doc_id"], int(request.args["start_byte"]), int(request.args["end_byte"]))
    response = jsonify({"link": philologic_link})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

def parse_args(request):
    search_args = formArguments()
    other_args = formArguments()
    other_args_keys = ["facet", "direction", "source", "target", "stats_field", "db_table", "filter_field", "filter_value", "page", "id_anchor"]
    for key, value in request.args.items():
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
            else:
                other_args[key] = value
        else:
            search_args[key] = value
    return search_args, other_args

def load_json(path):
    with open(path) as p:
        return json.load(p)
