"""Web app settings edited by hand: app_settings.toml, next to appConfig.json in each database's directory.

The API reads both whenever it serves the app, so edits apply on reload without a rebuild.
"""

import json
import os
import re
import tomllib
from dataclasses import dataclass, field

SETTINGS_FILE = "app_settings.toml"


@dataclass
class Option:
    name: str  # in app_settings.toml
    path: tuple[str, ...]  # in appConfig.json
    default: bool | str | list
    record: dict[str, str] | None = None  # for a list of records: its keys -> those of appConfig.json
    optional: dict = field(default_factory=dict)  # record keys that may be left out, with their defaults


FIELD = {"field": "value", "label": "label"}
CITATION = {"field": "field", "style": "style"}
REUSED = {"label": "label", "fields": "fields"}
BLOCKS = [
    (
        "Title shown in the page header. May contain HTML.",
        [Option("database_label", ("databaseLabel",), "")],
    ),
    (
        "Names of the source and target texts throughout the interface. May contain HTML.",
        [
            Option("source_label", ("sourceLabel",), "Earlier occurrence"),
            Option("target_label", ("targetLabel",), "Later occurrence"),
        ],
    ),
    (
        "Show the network graph view: true or false. It stays hidden anyway if passages\n"
        "were not grouped into families when the database was loaded.",
        [Option("network_graph", ("showNetworkGraph",), True)],
    ),
    (
        "Show the semantic graph view: true or false. Turned on when a semantic graph is built;\n"
        "if it is on without one, the view says so.",
        [Option("semantic_graph", ("showSemanticGraph",), False)],
    ),
    (
        "Links at the top left of the page header.",
        [Option("branding", ("branding",), [], {"label": "label", "link": "link"})],
    ),
    (
        "Fields of the search for passage pairs, in display order: source_ fields are the source collection's,\n"
        "target_ ones the target collection's. Where passages are grouped into families, the form has the\n"
        'reused passages\' fields below instead. "field" is the column in the database, "label" what the form\n'
        "shows. Delete a line to remove a field.",
        [
            Option("source_search_fields", ("metadataFields", "source"), [], FIELD),
            Option("target_search_fields", ("metadataFields", "target"), [], FIELD),
        ],
    ),
    (
        'Fields of the "Count shared passages by" panel for passage pairs, in the search form\'s format.',
        [
            Option("source_facets", ("facetsFields", "source"), [], FIELD),
            Option("target_facets", ("facetsFields", "target"), [], FIELD),
        ],
    ),
    (
        'Fields of the search form and of the "Count reused passages by" panel for reused passages, alike for\n'
        'where a passage was first found and where it was found again. "fields" names the field of each\n'
        'collection, as fields = ["source_title", "target_head"]: each text is read by its own collection\'s,\n'
        "and with one field the other collection's texts have none. Left empty, the fields above go together\n"
        "by name.",
        [
            Option("reused_search_fields", ("reusedSearchFields",), [], REUSED),
            Option("reused_facets", ("reusedFacets",), [], REUSED),
        ],
    ),
    (
        'Fields of the citation above each passage, in order. "style" takes CSS properties,\n'
        'as in style = { font-style = "italic" }, and may be left out.',
        [
            Option("source_citation", ("sourceCitation",), [], CITATION, {"style": {}}),
            Option("target_citation", ("targetCitation",), [], CITATION, {"style": {}}),
        ],
    ),
    (
        "Intervals offered by the timeline: a label and a number of years.",
        [Option("time_series_intervals", ("timeSeriesIntervals",), [], {"label": "label", "years": "value"})],
    ),
]
OPTIONS = {option.name: option for _, options in BLOCKS for option in options}
# appConfig.json keys that app_settings.toml holds instead
SETTINGS_KEYS = {option.path[0] for option in OPTIONS.values()}
KINDS = {bool: "true or false", str: "text in quotes", list: "a list in [ brackets ]"}


class SettingsError(ValueError):
    """A mistake in app_settings.toml."""


def toml_value(value, inline=False) -> str:
    """A value in TOML syntax, a list on one line if `inline`."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")  # JSON escapes are TOML's
    if isinstance(value, dict):
        keys = (key if re.fullmatch(r"[A-Za-z0-9_-]+", key) else json.dumps(key) for key in value)
        return "{ " + ", ".join(f"{key} = {toml_value(item, True)}" for key, item in zip(keys, value.values())) + " }"
    if not value:
        return "[]"
    if inline:
        return "[" + ", ".join(toml_value(item, True) for item in value) + "]"
    return "[\n" + "".join(f"    {toml_value(item)},\n" for item in value) + "]"


def write_app_settings(config: dict, path: str):
    """Write the hand-editable settings of an app config to a TOML file."""
    lines = [
        "# Settings of this database's web app. Changes apply when the page is reloaded.",
        "# Text goes in quotes, and lists hold one entry per line, in { braces }.",
    ]
    config = with_reused_fields(config)
    for comment, options in BLOCKS:
        lines.append("")
        lines.extend(f"# {line}" for line in comment.split("\n"))
        for option in options:
            value = config
            for key in option.path:
                value = value.get(key, {})
            value = option.default if value == {} else value
            if option.record:
                value = [settings_entry(option, entry) for entry in value]
            lines.append(f"{option.name} = {toml_value(value)}")
    with open(path, "w", encoding="utf8") as settings_file:
        settings_file.write("\n".join(lines) + "\n")


def settings_entry(option: Option, entry: dict) -> dict:
    """An entry of a list of records, with the keys of app_settings.toml."""
    settings = {}
    for key, config_key in option.record.items():
        if key in option.optional and entry.get(config_key, option.optional[key]) == option.optional[key]:
            continue  # left out, as it has the default value
        settings[key] = entry[config_key]
    return settings


def read_app_settings(path: str) -> dict:
    """The app config keys set in a TOML settings file."""
    with open(path, "rb") as settings_file:
        try:
            settings = tomllib.load(settings_file)
        except tomllib.TOMLDecodeError as error:
            raise SettingsError(f"{SETTINGS_FILE}: {error}") from None
    for name in settings:
        if name not in OPTIONS:
            raise SettingsError(f"{SETTINGS_FILE}: unknown setting {name}")
    config = {}
    for option in OPTIONS.values():
        value = settings.get(option.name, option.default)
        if type(value) is not type(option.default):
            raise SettingsError(f"{SETTINGS_FILE}: {option.name} should be {KINDS[type(option.default)]}")
        if option.record:
            value = [record_entry(option, entry) for entry in value]
        if len(option.path) == 1:
            config[option.path[0]] = value
        else:
            config.setdefault(option.path[0], {})[option.path[1]] = value
    check_reused_fields(config)
    return config


def record_entry(option: Option, entry) -> dict:
    """An entry of a list of records, with the keys of appConfig.json."""
    if not isinstance(entry, dict):
        raise SettingsError(f"{SETTINGS_FILE}: each entry of {option.name} should be in {{ braces }}")
    for key in entry:
        if key not in option.record:
            raise SettingsError(f"{SETTINGS_FILE}: unknown key {key} in {option.name}")
    for key in option.record:
        if key not in entry and key not in option.optional:
            raise SettingsError(f"{SETTINGS_FILE}: an entry of {option.name} has no {key}")
    return {config_key: entry.get(key, option.optional.get(key)) for key, config_key in option.record.items()}


def role_key(fields: list[str], role: str) -> str:
    """The name a field of the reused passages goes by where they were first found (`role` source) or found
    again (target): the field of that collection, or else one named after the other's."""
    return next((field for field in fields if field.startswith(f"{role}_")), f"{role}_{fields[0].split('_', 1)[1]}")


def check_reused_fields(config: dict):
    """Each field of the reused passages names at most one field of each collection, and goes by names no other
    field of them takes otherwise."""
    taken = {}
    for option in (OPTIONS["reused_search_fields"], OPTIONS["reused_facets"]):
        names = set()
        for entry in config[option.path[0]]:
            fields, where = entry["fields"], f"{SETTINGS_FILE}: {entry['label']} in {option.name}"
            if not isinstance(fields, list) or not fields or not all(isinstance(field, str) for field in fields):
                raise SettingsError(f"{where} should have a list of fields")
            sides = [field.split("_", 1)[0] for field in fields]
            if any(side not in ("source", "target") or field == f"{side}_" for side, field in zip(sides, fields)):
                raise SettingsError(f"{where} should have fields that start with source_ or target_")
            if len(set(sides)) < len(sides):
                raise SettingsError(f"{where} has two fields of one collection")
            for role in ("source", "target"):
                name = role_key(fields, role)
                if name in names or taken.setdefault(name, sorted(fields)) != sorted(fields):
                    raise SettingsError(f"{SETTINGS_FILE}: two fields of the reused passages go by {name}")
                names.add(name)


def fields_by_name(fields: dict, columns=None) -> list[dict]:
    """Fields of the reused passages from those of the two collections, each with the field of either named alike
    after source_ and target_, of those in `columns` if given"""
    labels = {}
    for side in ("source", "target"):
        for field in fields.get(side, []):
            labels.setdefault(field["value"].split("_", 1)[1], field["label"])
    reused = []
    for name, label in labels.items():
        named = [f"{side}_{name}" for side in ("source", "target") if columns is None or f"{side}_{name}" in columns]
        if named:
            reused.append({"label": label, "fields": named})
    return reused


def with_reused_fields(config: dict, columns=None) -> dict:
    """The config, with fields of the reused passages by name (fields_by_name) where it has none. Passage text is
    not among those counted: every passage is its own value of it."""
    config = dict(config)
    if not config.get("reusedSearchFields"):
        config["reusedSearchFields"] = fields_by_name(config.get("metadataFields", {}), columns)
    if not config.get("reusedFacets"):
        facets = fields_by_name(config.get("facetsFields", {}), columns)
        config["reusedFacets"] = [entry for entry in facets if not entry["fields"][0].endswith("_passage")]
    return config


def reused_fields(config: dict) -> dict[str, dict[str, str | None]]:
    """The field of each collection that each name of a field of the reused passages reads, where these are not
    the fields named alike: source_author -> {"source": "source_author", "target": "target_kafauth"}"""
    names = {}
    for entry in config.get("reusedSearchFields", []) + config.get("reusedFacets", []):
        fields = {field.split("_", 1)[0]: field for field in entry["fields"]}
        read = {"source": fields.get("source"), "target": fields.get("target")}
        for role in ("source", "target"):
            name = role_key(entry["fields"], role)
            suffix = name.split("_", 1)[1]
            if read != {"source": f"source_{suffix}", "target": f"target_{suffix}"}:
                names[name] = read
    return names


def show_view(db_dir: str, name: str):
    """Turn a view on in a database's settings file, if it has one."""
    settings_path = os.path.join(db_dir, SETTINGS_FILE)
    if os.path.exists(settings_path):
        with open(settings_path, encoding="utf8") as settings_file:
            settings = settings_file.read()
        with open(settings_path, "w", encoding="utf8") as settings_file:
            settings_file.write(re.sub(rf"^{name}\s*=.*$", f"{name} = true", settings, flags=re.MULTILINE))


def load_app_config(db_dir: str) -> dict:
    """A database's app config: appConfig.json, with app_settings.toml over it when there is one."""
    with open(os.path.join(db_dir, "appConfig.json"), encoding="utf8") as config_file:
        config = json.load(config_file)
    settings_path = os.path.join(db_dir, SETTINGS_FILE)
    if os.path.exists(settings_path):
        config.update(read_app_settings(settings_path))
    for option in (OPTIONS["network_graph"], OPTIONS["semantic_graph"]):
        config.setdefault(option.path[0], option.default)
    return with_reused_fields(config)
