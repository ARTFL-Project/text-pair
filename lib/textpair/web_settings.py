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
        'Fields of the search form, in display order. "field" is the column in the database,\n'
        '"label" what the form shows. Delete a line to remove a field.',
        [
            Option("source_search_fields", ("metadataFields", "source"), [], FIELD),
            Option("target_search_fields", ("metadataFields", "target"), [], FIELD),
        ],
    ),
    (
        'Fields of the "Count shared passages by" and "Count reused passages by" panels, '
        "in the search form's format.",
        [
            Option("source_facets", ("facetsFields", "source"), [], FIELD),
            Option("target_facets", ("facetsFields", "target"), [], FIELD),
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


def toml_value(value) -> str:
    """A value in TOML syntax."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")  # JSON escapes are TOML's
    if isinstance(value, dict):
        keys = (key if re.fullmatch(r"[A-Za-z0-9_-]+", key) else json.dumps(key) for key in value)
        return "{ " + ", ".join(f"{key} = {toml_value(item)}" for key, item in zip(keys, value.values())) + " }"
    if not value:
        return "[]"
    return "[\n" + "".join(f"    {toml_value(item)},\n" for item in value) + "]"


def write_app_settings(config: dict, path: str):
    """Write the hand-editable settings of an app config to a TOML file."""
    lines = [
        "# Settings of this database's web app. Changes apply when the page is reloaded.",
        "# Text goes in quotes, and lists hold one entry per line, in { braces }.",
    ]
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
    return config
