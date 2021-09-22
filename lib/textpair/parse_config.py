#!/usr/bin/env python3
"""Parse textpair config file"""

import configparser
import os
from typing import Dict, Any


def parse_config(textpair_config, output_path="./output", skip_web_app=False):
    """Read config file and store into 4 dicts for each phase of the alignment"""
    tei_parsing = {}
    preprocessing_params = {"source": {}, "target": {}}
    matching_params: Dict[str, Any] = {"matching_algorithm": "sa"}
    web_app_config = {"skip_web_app": skip_web_app}
    config = configparser.ConfigParser()
    config.read(textpair_config)
    file_paths = {
        "source_files": config["FILE_PATHS"]["source_file_path"] or "",
        "target_files": config["FILE_PATHS"]["target_file_path"] or "",
    }
    for key, value in dict(config["TEI_PARSING"]).items():
        if key.startswith("parse"):
            if value.lower() == "yes" or value.lower() == "true":
                tei_parsing[key] = True
            else:
                tei_parsing[key] = False
        else:
            if not value:
                if key.startswith("output_source"):
                    value = os.path.join(output_path, "source")
                else:
                    value = os.path.join(output_path, "target")
            tei_parsing[key] = value
    for key, value in dict(config["PREPROCESSING"]).items():
        if value:
            if key in ("skipgram", "numbers", "word_order", "modernize", "ascii", "stemmer"):
                if value.lower() == "yes" or value.lower() == "true":
                    value = True
                else:
                    value = False
            if key.endswith("object_level"):
                if key.startswith("source"):
                    preprocessing_params["source"]["text_object_level"] = value
                else:
                    preprocessing_params["target"]["text_object_level"] = value
            elif key in ("ngram", "gap", "minimum_word_length", "n_chunk", "min_text_object_length",):
                preprocessing_params["source"][key] = int(value)
                preprocessing_params["target"][key] = int(value)
            elif key == "pos_to_keep":
                preprocessing_params["source"][key] = [i.strip() for i in value.split(",")]
                preprocessing_params["target"][key] = preprocessing_params["source"][key]
            elif key in ("min_freq", "max_freq"):
                preprocessing_params["source"][key] = float(value)
                preprocessing_params["target"][key] = float(value)
            else:
                preprocessing_params["source"][key] = value
                preprocessing_params["target"][key] = value
    for key, value in dict(config["MATCHING"]).items():
        if value or key not in matching_params:
            if key == "flex_gap":
                if value.lower() == "yes" or value.lower() == "true":
                    value = "true"
                else:
                    value = "false"
            elif key == "min_similarity":
                value = float(value)
            elif key == "min_matching_words":
                value = int(value)
            matching_params[key] = value
    if skip_web_app is False:
        web_app_config["field_types"] = {}
        for key, value in dict(config["WEB_APPLICATION"]).items():
            if key == "api_server" or key == "source_philo_db_link" or key == "target_philo_db_link":
                web_app_config[key] = value
            else:
                web_app_config["field_types"][key] = value
    global_config = configparser.ConfigParser()
    global_config.read("/etc/text-pair/global_settings.ini")
    web_app_config["web_application_directory"] = global_config["WEB_APP"]["web_app_path"]

    return file_paths, tei_parsing, preprocessing_params, matching_params, web_app_config
