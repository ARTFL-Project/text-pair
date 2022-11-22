"""Parse files using Philologic parser"""

import os

from philologic.loadtime.Loader import Loader, setup_db_dir
from philologic.loadtime import LoadFilters
from philologic.loadtime.Parser import (
    DEFAULT_DOC_XPATHS,
    DEFAULT_METADATA_TO_PARSE,
    DEFAULT_TAG_TO_OBJ_MAP,
    TOKEN_REGEX,
    XMLParser,
)

PHILO_TEXT_OBJECT_LEVELS = {"doc": 1, "div1": 2, "div2": 3, "div3": 4, "para": 5, "sent": 6, "word": 7}


def parse_tei(
    input_file_path: str, output_path: str, words_to_index: str, object_level: str, workers: int, debug: bool
):
    output_path = os.path.abspath(output_path)
    setup_db_dir(output_path, False, force_delete=True)
    word_list = set()
    if words_to_index != "all":
        with open(words_to_index) as fh:
            for line in fh:
                word_list.add(line.strip())
    navigable_objects = [
        text_object
        for text_object, depth in PHILO_TEXT_OBJECT_LEVELS.items()
        if PHILO_TEXT_OBJECT_LEVELS[object_level] >= depth
    ]
    loader = Loader.set_class_attributes(
        {
            "post_filters": [],
            "debug": debug,
            "words_to_index": word_list,
            "data_destination": output_path,
            "db_destination": "",
            "default_object_level": object_level,
            "token_regex": TOKEN_REGEX,
            "url_root": "",
            "cores": workers,
            "ascii_conversion": True,
            "doc_xpaths": DEFAULT_DOC_XPATHS,
            "metadata_sql_types": {},
            "metadata_to_parse": DEFAULT_METADATA_TO_PARSE,
            "tag_to_obj_map": DEFAULT_TAG_TO_OBJ_MAP,
            "parser_factory": XMLParser,
            "load_filters": LoadFilters.set_load_filters(navigable_objects=navigable_objects),
            "file_type": "xml",
            "load_config": "",
        }
    )
    loader.tables = ["toms"]  # We just want the toms (metadata) table.
    loader.add_files([f.path for f in os.scandir(input_file_path)])
    doc_metadata = loader.parse_metadata(["year", "author", "title", "filename"], header="tei", verbose=False)
    loader.set_file_data(doc_metadata, loader.textdir, loader.workdir)
    loader.parse_files(workers, verbose=False)
    loader.merge_files("toms", verbose=False)
    loader.setup_sql_load(verbose=False)
    loader.post_processing(verbose=False)
    os.chdir("../../..")
