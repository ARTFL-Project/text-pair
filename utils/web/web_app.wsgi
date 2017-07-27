#!/usr/bin/env python3

import os
import sys
from collections import OrderedDict

from flask import Flask
from flask import render_template
from flask import request

import psycopg2
import psycopg2.extras

application = Flask(__name__)

DATABASE = psycopg2.connect(user="alignments", password="martini", database="alignments")
CURSOR = DATABASE.cursor(cursor_factory=psycopg2.extras.DictCursor)

@application.route("/")
def search():
    return render_template("search.html", name="search")

@application.route("/results", methods=["GET"])
def results():
    search_args = OrderedDict()
    for key, value in request.args.items():
        search_args[key] = value
    query = "SELECT * FROM frantext WHERE {}".format(" ".join([i + " ilike %s" for i in search_args.keys() if search_args[i]]))
    print(query, list(search_args.values()), file=sys.stderr)
    CURSOR.execute(query, ["%{}%".format(i) for i in search_args.values() if i])
    alignments = []
    fields_extract = ["source_author", "source_title", "source_year", "source_context_before", "source_passage", "source_context_after",
                      "target_author", "target_title", "target_year", "target_context_before", "target_passage", "target_context_after"]
    for row in CURSOR:
        alignments.append({key: row[key] for key in fields_extract})
    return render_template("results.html", alignments=alignments)
