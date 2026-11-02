#!/usr/bin/env python3
"""Fields the two sides name otherwise must go together, as app_settings.toml pairs them.

    test_paired_fields.py

A database of two collections has texts of both where a passage was first found and where it was found
again, so a field of one side must be read, for the other side's texts, from the field the settings pair
it with, or else the one named alike. Runs over web app directories made for the test, so it needs no
database.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import text_pair  # noqa: E402
from text_pair import Condition, field_on_side, on_side, paired_fields, two_collections  # noqa: E402
from textpair.web_loader import document_fields  # noqa: E402
from textpair.web_settings import SettingsError, read_app_settings, write_app_settings  # noqa: E402

FAILURES = []

PAIRS = '''paired_fields = [
    { source = "source_author", target = "target_kafauth" },
    { source = "source_title", target = "target_head" },
]
'''


def check(label, ok):
    print(f"{'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        FAILURES.append(label)


def philo_db(root, name):
    path = os.path.join(root, name)
    os.makedirs(os.path.join(path, "data"))
    open(os.path.join(path, "data", "toms.db"), "w").close()
    return path


def app_dir(root, name, source, target, settings=""):
    path = os.path.join(root, "apps", name)
    os.makedirs(path)
    with open(os.path.join(path, "appConfig.json"), "w", encoding="utf8") as config:
        json.dump({"databaseName": name, "sourcePhiloDBPath": source, "targetPhiloDBPath": target}, config)
    with open(os.path.join(path, "app_settings.toml"), "w", encoding="utf8") as config:
        config.write(settings)


def main():
    root = tempfile.mkdtemp(prefix="paired_")
    frantext, encyclopedie = philo_db(root, "frantext"), philo_db(root, "encyclopedie")
    app_dir(root, "two", frantext, encyclopedie, PAIRS)
    app_dir(root, "one", frantext, frantext + "/")
    app_dir(root, "broken", frantext, encyclopedie, "paired_fields = [ { source = \"source_author\" } ]\n")
    text_pair.APP_PATH = os.path.join(root, "apps")

    pairs = paired_fields("two")
    check("the settings pair fields each way",
          pairs == {"source_author": "target_kafauth", "target_kafauth": "source_author",
                    "source_title": "target_head", "target_head": "source_title"})
    check("a database without pairs, or with broken settings, or none at all, pairs nothing",
          paired_fields("one") == {} and paired_fields("broken") == {} and paired_fields("missing") == {})

    check("a field of the other side reads the one it is paired with",
          field_on_side("target_kafauth", "source", pairs) == "source_author"
          and field_on_side("source_title", "target", pairs) == "target_head")
    check("... or the one named alike",
          field_on_side("target_year", "source", pairs) == "source_year"
          and field_on_side("target_author", "source", pairs) == "source_author")
    check("... and a field of its own side is itself", field_on_side("target_kafauth", "target", pairs) == "target_kafauth")
    condition = on_side(Condition("target_kafauth", '"target_kafauth" ILIKE %s', ["%Voltaire%"]), "source", pairs)
    check("a condition moves to the paired field",
          condition.field == "source_author" and condition.sql == '"source_author" ILIKE %s' and condition.values == ["%Voltaire%"])

    check("texts of two PhiloLogic databases are two collections", two_collections("two"))
    check("... and those of one, however its path is written, one", not two_collections("one"))

    columns = {"source_doc_id": "text", "source_author": "text", "source_head": "text", "source_year": "integer",
               "source_passage": "text", "source_start_byte": "integer", "source_context_after": "text",
               "source_embedding": "USER-DEFINED", "target_kafauth": "text", "group_id": "ARRAY"}
    check("a side's document fields leave out the passage's, the id, those kept and the other side's",
          document_fields(columns, "source", ["source_author"]) == ["source_head", "source_year"])

    path = os.path.join(root, "written.toml")
    write_app_settings({}, path)
    with open(path, encoding="utf8") as written:
        check("a new settings file lists the pairs, empty", "\npaired_fields = []\n" in written.read())
    try:
        read_app_settings(os.path.join(root, "apps", "broken", "app_settings.toml"))
        check("a pair without its target is a mistake", False)
    except SettingsError as error:
        check("a pair without its target is a mistake", "no target" in str(error))

    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
