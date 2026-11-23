---
title: The web application
nav_order: 5
---

# The web application
{: .no_toc }

`textpair` builds a web application for each database in the `web_app_path` directory of `global_settings.ini` (`/var/www/html/text-pair/database_name` by default), starting from the defaults in `/var/lib/text-pair/config/appConfig.json`. Its settings are in two files in that directory:

- `app_settings.toml` holds the settings you can edit, each with a comment on what it does. Changes apply when the page is reloaded, without rebuilding the application. If the file has a mistake, the page says which and where.
- `appConfig.json` holds the settings `textpair` sets when it loads the database, such as the database name, the matching algorithm, and the application path. Do not edit it.

{: .note }
Metadata fields are prefixed with `source_` for the source texts and `target_` for the target texts: `source_author`, `target_title`, and so on.

1. TOC
{:toc}

## Labels

| Setting | Description |
|:--------|:------------|
| `database_label` | Title of the database, in the page header. When the texts come from PhiloLogic databases, it starts as the `dbname` of their `web_config.cfg`, or as "source vs target" when the two databases differ. |
| `source_label`, `target_label` | Names of the source and target texts throughout the interface. They start as "Earlier occurrence" and "Later occurrence". |

Labels may contain HTML.

## Views

| Setting | Description |
|:--------|:------------|
| `network_graph` | Show the network graph view: `true` or `false`. It stays hidden anyway when passages were not grouped into families. |
| `semantic_graph` | Show the semantic graph view. It is set to `true` when a graph is built, with `build_graph = true` in `[GRAPH]` or with `--graph_only`; if it is `true` without a graph, the view says so. |

## Header links

`branding` lists the links in the page header:

```toml
branding = [
    { label = "ARTFL Project", link = "https://artfl-project.uchicago.edu" },
]
```

## Search fields and facets

For passage pairs, `source_search_fields` and `target_search_fields` are the fields of the search form, and `source_facets` and `target_facets` those of the faceted browser, in order. `field` is the name of the metadata field in the database:

```toml
source_search_fields = [
    { field = "source_author", label = "Author" },
    { field = "source_title", label = "Title" },
]
```

Where passages were grouped into families, the search form has the fields of the reused passages instead: `reused_search_fields`, and `reused_facets` for the faceted browser. Since a reused passage can be found in texts of either collection, `fields` names the field of each collection, and each text is searched and counted by its own:

```toml
reused_search_fields = [
    { label = "Title", fields = ["source_title", "target_head"] },
]
```

With one field, as `fields = ["target_head"]`, the other collection's texts have none. Left empty, the fields of the passage pairs go together by name, `source_year` with `target_year`.

{: .note }
Databases loaded before these settings existed need their family tables rebuilt to search and count by fields their search form did not have at load: `extras/add_family_filter_tables.py --force --tables database_name`.

## Citations

`source_citation` and `target_citation` are the fields of the citation shown with each passage, in order. The style takes CSS properties and may be left out:

```toml
source_citation = [
    { field = "source_author" },
    { field = "source_title", style = { font-style = "italic" } },
]
```

Where the source and target texts are two collections, each text is cited by its own collection's citation, on whichever side of a passage family it is.

## Timeline

`time_series_intervals` lists the intervals offered by the timeline:

```toml
time_series_intervals = [
    { label = "Decade", years = 10 },
    { label = "Half century", years = 50 },
]
```
