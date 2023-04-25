"""Parse files using Philologic parser"""

import os
from typing import Set

from philologic.loadtime.Loader import Loader, setup_db_dir
from philologic.loadtime import LoadFilters
from philologic.loadtime import Parser as XMLParser
from philologic.loadtime import PlainTextParser

PHILO_TEXT_OBJECT_LEVELS = {"doc": 1, "div1": 2, "div2": 3, "div3": 4, "para": 5, "sent": 6, "word": 7}


def parse_files(
    input_file_path: str,
    file_type: str,
    metadata: str,
    output_path: str,
    words_to_index: str,
    object_level: str,
    workers: int,
    debug: bool,
):
    """Parse files using Philologic parser"""
    output_path = os.path.abspath(output_path)
    os.makedirs(output_path, exist_ok=True)
    setup_db_dir(output_path, False, force_delete=True)
    word_list: Set[str] = set()
    if words_to_index != "all":
        with open(words_to_index, encoding="utf8") as fh:
            for line in fh:
                word_list.add(line.strip())
    navigable_objects = [
        text_object
        for text_object, depth in PHILO_TEXT_OBJECT_LEVELS.items()
        if PHILO_TEXT_OBJECT_LEVELS[object_level] >= depth
    ]
    if file_type == "tei":
        loader = Loader.set_class_attributes(
            {
                "post_filters": [],
                "debug": debug,
                "words_to_index": word_list,
                "data_destination": output_path,
                "db_destination": "",
                "default_object_level": object_level,
                "token_regex": XMLParser.TOKEN_REGEX,
                "url_root": "",
                "cores": workers,
                "ascii_conversion": True,
                "doc_xpaths": XMLParser.DEFAULT_DOC_XPATHS,
                "metadata_sql_types": {},
                "metadata_to_parse": XMLParser.DEFAULT_METADATA_TO_PARSE,
                "tag_to_obj_map": XMLParser.DEFAULT_TAG_TO_OBJ_MAP,
                "parser_factory": XMLParser.XMLParser,
                "load_filters": LoadFilters.set_load_filters(navigable_objects=navigable_objects),
                "file_type": file_type,
                "bibliography": metadata,
                "load_config": "",
            }
        )
    else:
        loader = Loader.set_class_attributes(
            {
                "post_filters": [],
                "debug": debug,
                "words_to_index": word_list,
                "data_destination": output_path,
                "db_destination": "",
                "default_object_level": object_level,
                "token_regex": PlainTextParser.TOKEN_REGEX,
                "url_root": "",
                "cores": workers,
                "ascii_conversion": True,
                "doc_xpaths": XMLParser.DEFAULT_DOC_XPATHS,
                "metadata_sql_types": {},
                "metadata_to_parse": XMLParser.DEFAULT_METADATA_TO_PARSE,
                "tag_to_obj_map": XMLParser.DEFAULT_TAG_TO_OBJ_MAP,
                "parser_factory": PlainTextParser.PlainTextParser,
                "load_filters": LoadFilters.set_load_filters(navigable_objects=navigable_objects),
                "file_type": file_type,
                "bibliography": metadata,
                "load_config": "",
            }
        )
    loader.tables = ["toms"]  # We just want the toms (metadata) table.
    loader.add_files([f.path for f in os.scandir(input_file_path)])
    if metadata != "":
        doc_metadata = loader.parse_bibliography_file(metadata, ["year", "author", "title", "filename"])
    else:
        doc_metadata = loader.parse_metadata(["year", "author", "title", "filename"], header="tei", verbose=False)
    loader.set_file_data(doc_metadata, loader.textdir, loader.workdir)
    loader.parse_files(workers, verbose=False)
    loader.merge_files("toms", verbose=False)
    loader.setup_sql_load(verbose=False)
    loader.post_processing(verbose=False)
    os.chdir("../../..")
