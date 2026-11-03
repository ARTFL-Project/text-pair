#!/usr/bin/env python3
"""The fields of the reused passages must read each text by its own collection's, as app_settings.toml names them.

    test_reused_fields.py

A database of two collections has texts of both where a passage was first found and where it was found
again, so a field of the reused passages names the field of each collection it reads, or of one only, or
else goes with the one named alike. Runs over web app directories made for the test, so it needs no
database.
"""
import json
import os
import sys
import tempfile
import tomllib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import text_pair  # noqa: E402
from text_pair import Condition, field_on_side, on_side, reused_names, two_collections  # noqa: E402
from textpair.web_loader import document_fields  # noqa: E402
from textpair.web_settings import (  # noqa: E402
    SettingsError,
    fields_by_name,
    load_app_config,
    read_app_settings,
    write_app_settings,
)

FAILURES = []

FIELDS = '''reused_search_fields = [
    { label = "Author", fields = ["source_author", "target_kafauth"] },
    { label = "Title", fields = ["source_title", "target_head"] },
    { label = "Year", fields = ["source_year", "target_year"] },
    { label = "Genre", fields = ["source_genre"] },
]
'''
METADATA = {
    "metadataFields": {
        "source": [{"value": "source_author", "label": "Author"}, {"value": "source_year", "label": "Year"}],
        "target": [{"value": "target_kafauth", "label": "Author"}, {"value": "target_year", "label": "Date"}],
    },
    "facetsFields": {
        "source": [{"value": "source_passage", "label": "Passage text"}, {"value": "source_year", "label": "Year"}],
        "target": [{"value": "target_passage", "label": "Passage text"}, {"value": "target_year", "label": "Year"}],
    },
}


def check(label, ok):
    print(f"{'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        FAILURES.append(label)


def philo_db(root, name):
    path = os.path.join(root, name)
    os.makedirs(os.path.join(path, "data"))
    open(os.path.join(path, "data", "toms.db"), "w").close()
    return path


def app_dir(root, name, source, target, settings=None):
    path = os.path.join(root, "apps", name)
    os.makedirs(path)
    with open(os.path.join(path, "appConfig.json"), "w", encoding="utf8") as config:
        json.dump({"databaseName": name, "sourcePhiloDBPath": source, "targetPhiloDBPath": target, **METADATA}, config)
    if settings is not None:
        with open(os.path.join(path, "app_settings.toml"), "w", encoding="utf8") as config:
            config.write(settings)
    return path


def settings_error(root, settings) -> str:
    path = os.path.join(root, "mistake.toml")
    with open(path, "w", encoding="utf8") as config:
        config.write(settings)
    try:
        read_app_settings(path)
    except SettingsError as error:
        return str(error)
    return ""


def main():
    root = tempfile.mkdtemp(prefix="reused_")
    frantext, encyclopedie = philo_db(root, "frantext"), philo_db(root, "encyclopedie")
    app_dir(root, "two", frantext, encyclopedie, FIELDS)
    one = app_dir(root, "one", frantext, frantext + "/")
    app_dir(root, "broken", frantext, encyclopedie, '[[reused_search_fields]]\nlabel = "Author"\n')
    text_pair.APP_PATH = os.path.join(root, "apps")

    names = reused_names("two")
    author = {"source": "source_author", "target": "target_kafauth"}
    title = {"source": "source_title", "target": "target_head"}
    genre = {"source": "source_genre", "target": None}
    check("each field reads the field it names of each collection, by its name on either side",
          names == {"source_author": author, "target_kafauth": author, "source_title": title, "target_head": title,
                    "source_genre": genre, "target_genre": genre})
    check("a database without settings for them, or with broken settings, or none at all, reads them by name",
          reused_names("one") == {} and reused_names("broken") == {} and reused_names("missing") == {})

    check("a field of the other side reads the one named with it",
          field_on_side("target_kafauth", "source", names) == "source_author"
          and field_on_side("source_title", "target", names) == "target_head")
    check("... or none, for a collection the settings name none of",
          field_on_side("target_genre", "target", names) is None
          and field_on_side("target_genre", "source", names) == "source_genre")
    check("... or the one named alike",
          field_on_side("target_year", "source", names) == "source_year"
          and field_on_side("target_author", "source", names) == "source_author")
    check("... and a field of its own side is itself", field_on_side("target_kafauth", "target", names) == "target_kafauth")
    condition = Condition("target_kafauth", '"target_kafauth" ILIKE %s', ["%Voltaire%"])
    moved = on_side(condition, "source", names)
    check("a condition moves to the field named with it",
          moved.field == "source_author" and moved.sql == '"source_author" ILIKE %s' and moved.values == ["%Voltaire%"])
    check("... and is none on a side with no field for it",
          on_side(Condition("source_genre", '"source_genre" ILIKE %s', ["%roman%"]), "target", names) is None)

    reused = load_app_config(one)["reusedSearchFields"]
    check("without settings for them, the fields of the two collections go with those named alike",
          reused == [{"label": "Author", "fields": ["source_author", "target_author"]},
                     {"label": "Year", "fields": ["source_year", "target_year"]},
                     {"label": "Author", "fields": ["source_kafauth", "target_kafauth"]}])
    loaded = fields_by_name(METADATA["metadataFields"], {"source_author", "source_year", "target_kafauth", "target_year"})
    check("... and where a database is loaded, a field the other collection has none of is one collection's",
          loaded == [{"label": "Author", "fields": ["source_author"]},
                     {"label": "Year", "fields": ["source_year", "target_year"]},
                     {"label": "Author", "fields": ["target_kafauth"]}])
    check("... and passage text is not among those counted",
          load_app_config(one)["reusedFacets"] == [{"label": "Year", "fields": ["source_year", "target_year"]}])

    check("texts of two PhiloLogic databases are two collections", two_collections("two"))
    check("... and those of one, however its path is written, one", not two_collections("one"))

    columns = {"source_doc_id": "text", "source_author": "text", "source_head": "text", "source_year": "integer",
               "source_passage": "text", "source_start_byte": "integer", "source_context_after": "text",
               "source_embedding": "USER-DEFINED", "target_kafauth": "text", "group_id": "ARRAY"}
    check("a side's document fields leave out the passage's, the id, those kept and the other side's",
          document_fields(columns, "source", ["source_author"]) == ["source_head", "source_year"])

    path = os.path.join(root, "written.toml")
    write_app_settings(METADATA, path)
    with open(path, encoding="utf8") as written:
        text = written.read()
    check("a new settings file has the fields by name, each on one line",
          '    { label = "Year", fields = ["source_year", "target_year"] },\n' in text)
    with open(path, "rb") as written:
        check("... which read back as written", tomllib.load(written)["reused_facets"]
              == [{"label": "Year", "fields": ["source_year", "target_year"]}])

    entry = 'reused_search_fields = [ {{ label = "Author", fields = {} }} ]\n'
    check("a field without a list of fields is a mistake", "should have a list" in settings_error(root, entry.format('"source_author"')))
    check("... as is one not of a collection", "source_ or target_" in settings_error(root, entry.format('["author"]')))
    check("... or two of one collection", "two fields of one" in settings_error(root, entry.format('["source_author", "source_name"]')))
    clash = ('reused_search_fields = [ { label = "Title", fields = ["source_title", "target_head"] },\n'
             '    { label = "Headword", fields = ["source_head"] } ]\n')
    check("... or two fields that go by one name", "go by target_head" in settings_error(root, clash))
    differ = ('reused_search_fields = [ { label = "Author", fields = ["source_author", "target_kafauth"] } ]\n'
              'reused_facets = [ { label = "Author", fields = ["source_author", "target_author"] } ]\n')
    check("... or a name the search and the counts read otherwise", "go by source_author" in settings_error(root, differ))

    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
