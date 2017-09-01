#!/usr/bin/env python3
"""Routing and search code for sequence alignment"""

import os
import re
import sys
from collections import OrderedDict
from pathlib import Path

from flask import Flask, redirect
from flask import render_template
from flask import request

from philologic.runtime.link import byte_range_to_link
import psycopg2
import psycopg2.extras

application = Flask(__name__)


class formArguments():
    """Special dict to handle form arguments"""

    def __init__(self):
        self.dict = OrderedDict()

    def __getitem__(self, item):
        if item in self.dict:
            return self.dict[item]
        else:
            return ""

    def __getattr__(self, attr):
        return self.__getitem__(attr)

    def __setitem__(self, item, value):
        self.dict[item] = value

    def __iter__(self):
        for k in self.dict:
            yield k

    def items(self):
        for k, v in self.dict.items():
            yield k, v

    def values(self):
        for _, v in self.dict.items():
            yield v

    def __str__(self):
        return repr(self.dict)


@application.route("/<path:db_table>")
def search(db_table):
    form_args = formArguments()
    return render_template("search.html", db_table=db_table, form_args=form_args)

@application.route("/<path:db_table>/results", methods=["GET"])
def results(db_table):
    search_args = formArguments()
    page = 1
    id_anchor = 0
    direction = "next"
    for key, value in request.args.items():
        if key == "page":
            try:
                page = int(value)
            except TypeError:
                pass
        elif key == "id_anchor":
            try:
                id_anchor = int(value)
            except TypeError:
                pass
        elif key == "direction":
            direction = value or "next"
        else:
            search_args[key] = value
    if direction == "next":
        query = "SELECT o.rowid_ordered, m.* FROM {} m, {}_ordered o WHERE {} AND o.source_year_target_year=m.rowid and \
                 o.rowid_ordered > {} ORDER BY o.rowid_ordered LIMIT 50".format(db_table, db_table,
                " and ".join([i + " ilike %s " for i in search_args if search_args[i]]), id_anchor)
    else:
        query = "SELECT o.rowid_ordered, m.* FROM {} m, {}_ordered o WHERE {} AND o.source_year_target_year=m.rowid and \
                 o.rowid_ordered < {} ORDER BY o.rowid_ordered desc LIMIT 50".format(db_table, db_table,
                " and ".join([i + " ilike %s " for i in search_args if search_args[i]]), id_anchor)
    DATABASE = psycopg2.connect(user="alignments", password="martini", database="alignments")
    CURSOR = DATABASE.cursor(cursor_factory=psycopg2.extras.DictCursor)
    CURSOR.execute(query, ["%{}%".format(v) for v in search_args.values() if v])
    column_names = [desc[0] for desc in CURSOR.description]
    alignments = []
    for pos, row in enumerate(CURSOR):
        metadata = {key: row[key] for key in column_names}
        metadata["source_link_to_philologic"] = "{}/link_to_philologic?filename={}&doc_id={}&start_byte={}&end_byte={}".format(
            db_table,
            metadata["source_filename"],
            metadata["source_doc_id"].replace("_", " "),
            metadata["source_start_byte"],
            metadata["source_end_byte"]
        )
        metadata["target_link_to_philologic"] = "{}/link_to_philologic?filename={}&doc_id={}&start_byte={}&end_byte={}".format(
            db_table,
            metadata["target_filename"],
            metadata["target_doc_id"].replace("_", " "),
            metadata["target_start_byte"],
            metadata["target_end_byte"]
        )
        alignments.append(metadata)
    if direction == "previous":
        alignments.reverse()
    previous_url = ""
    current_path = re.sub(r'&(page|id_anchor|direction)=(previous|next|\d*)', '', request.full_path)
    full_url = "{}{}".format(request.script_root, current_path)
    if page > 1:
        previous_url = "{}&page={}&id_anchor={}&direction=previous".format(full_url, page-1, alignments[0]["rowid_ordered"])
    try:
        next_url = "{}&page={}&id_anchor={}&direction=next".format(full_url, page+1, alignments[-1]["rowid_ordered"])
    except IndexError:
        next_url = ""
    start_position = 0
    if page > 1:
        start_position = 50 * (page - 1)
    return render_template("results.html", db_table="", alignments=alignments, form_args=search_args, page=page,
                           previous_url=previous_url, next_url=next_url, start_position=start_position)

@application.route("/<path:db_table>/link_to_philologic", methods=["GET"])
def link_to_philologic(db_table):
    db_path = str(Path(request.args["filename"]).parent).replace("data/TEXT", "")
    print(db_path, file=sys.stderr)
    philologic_link = byte_range_to_link(db_path, request.args["doc_id"], int(request.args["start_byte"]), int(request.args["end_byte"]))
    return redirect(philologic_link[0])
