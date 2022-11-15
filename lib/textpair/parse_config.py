#!/usr/bin/env python3
"""Parse textpair config file"""

import configparser
import argparse
import os
from typing import Dict, Any, Union
from collections import defaultdict


class TextPairConfig:
    """TextPAIR parameters returned from parsing CLI arguments"""

    def __init__(self, cli_args: Dict[str, Any]):
        self.__cli_args: Dict[str, Any] = cli_args
        self.__file_paths: Dict[str, str] = {}
        self.tei_parsing: Dict[str, Union[bool, str]] = {}
        self.preprocessing_params: Dict[str, Any] = {"source": {}, "target": {}}
        self.matching_params: Dict[str, Any] = defaultdict(str)
        self.matching_params["matching_algorithm"] = "sa"
        self.web_app_config: Dict[str, Any] = {"skip_web_app": self.__cli_args["skip_web_app"]}
        self.only_web_app = self.__cli_args["load_only_web_app"]
        self.paths: Dict[str, Dict[str, Any]] = {"source": {}, "target": defaultdict(str)}
        self.__parse_config()
        self.__set_params()

    def __getattr__(self, attr):
        return self.__cli_args[attr]

    def __parse_config(self):
        """Read config file and store into 4 dicts for each phase of the alignment"""
        config = configparser.ConfigParser()
        config.read(self.__cli_args["config"])
        if self.only_align is False:
            self.__file_paths = {
                "source_files": config["FILE_PATHS"]["source_file_path"] or "",
                "target_files": config["FILE_PATHS"]["target_file_path"] or "",
            }
        else:
            self.__file_paths["source_files"] = os.path.join(self.output_path, "source")
            if config["FILE_PATHS"]["target_file_path"]:
                self.__file_paths["target_files"] = os.path.join(self.output_path, "target")
            else:
                self.__file_paths["target_files"] = ""
        for key, value in dict(config["TEI_PARSING"]).items():
            if key.startswith("parse"):
                if value.lower() == "yes" or value.lower() == "true":
                    self.tei_parsing[key] = True
                else:
                    self.tei_parsing[key] = False
            else:
                if not value:
                    if key.startswith("output_source"):
                        value = os.path.join(self.output_path, "source")
                    else:
                        value = os.path.join(self.output_path, "target")
                self.tei_parsing[key] = value
        for key, value in dict(config["PREPROCESSING"]).items():
            if value:
                if key in ("skipgram", "numbers", "word_order", "modernize", "ascii", "stemmer"):
                    if value.lower() == "yes" or value.lower() == "true":
                        value = True
                    else:
                        value = False
                if key.endswith("object_level"):
                    if key.startswith("source"):
                        self.preprocessing_params["source"]["text_object_level"] = value
                    else:
                        self.preprocessing_params["target"]["text_object_level"] = value
                elif key in (
                    "ngram",
                    "gap",
                    "minimum_word_length",
                    "n_chunk",
                    "min_text_object_length",
                ):
                    self.preprocessing_params["source"][key] = int(value)
                    self.preprocessing_params["target"][key] = int(value)
                elif key == "pos_to_keep":
                    self.preprocessing_params["source"][key] = [i.strip() for i in value.split(",")]
                    self.preprocessing_params["target"][key] = self.preprocessing_params["source"][key]
                elif key in ("min_freq", "max_freq"):
                    self.preprocessing_params["source"][key] = float(value)
                    self.preprocessing_params["target"][key] = float(value)
                elif key == "target_language":
                    if value:
                        self.preprocessing_params["target"]["language"] = value
                    else:  # assumes the language key comes before the target_language key...
                        self.preprocessing_params["target"]["language"] = self.preprocessing_params["source"][
                            "language"
                        ]
                else:
                    self.preprocessing_params["source"][key] = value
                    self.preprocessing_params["target"][key] = value
        for key, value in dict(config["MATCHING"]).items():
            if value or key not in self.matching_params:
                if key in ("flex_gap", "banality_auto_detection", "store_banalities"):
                    if value.lower() == "yes" or value.lower() == "true":
                        value = True
                    else:
                        value = False
                elif key == "min_similarity":
                    value = float(value)
                elif key in ("min_matching_words", "source_batch", "target_batch"):
                    value = int(value)
                self.matching_params[key] = value
        if self.__cli_args["skip_web_app"] is False:
            self.web_app_config["field_types"] = {}
            for key, value in dict(config["WEB_APPLICATION"]).items():
                if key == "api_server" or key == "source_philo_db_link" or key == "target_philo_db_link":
                    self.web_app_config[key] = value
                else:
                    self.web_app_config["field_types"][key] = value
        global_config = configparser.ConfigParser()
        global_config.read("/etc/text-pair/global_settings.ini")
        self.web_app_config["web_application_directory"] = global_config["WEB_APP"]["web_app_path"]

    def __set_params(self):
        """Set parameters for alignment"""
        self.web_app_config["skip_web_app"] = self.__cli_args["skip_web_app"]
        if self.__cli_args["only_align"] is False:
            if self.tei_parsing["parse_source_files"] is True:
                self.paths["source"]["tei_input_files"] = self.__file_paths["source_files"]
                self.paths["source"]["parse_output"] = os.path.join(self.output_path, "source")
                self.paths["source"]["input_files_for_ngrams"] = os.path.join(self.output_path, "source/texts")
                self.paths["source"]["ngram_output_path"] = os.path.join(self.output_path, "source/")
                self.paths["source"]["metadata_path"] = os.path.join(self.output_path, "source/metadata/metadata.json")
                self.paths["source"]["is_philo_db"] = False
            else:
                if self.__cli_args["is_philo_db"] is True:
                    self.paths["source"]["input_files_for_ngrams"] = os.path.join(
                        self.__file_paths["source_files"], "data/words_and_philo_ids"
                    )
                else:
                    self.paths["source"]["input_files_for_ngrams"] = self.__file_paths["source_files"]
                self.paths["source"]["ngram_output_path"] = os.path.join(self.output_path, "source/")
                self.paths["source"]["metadata_path"] = self.__cli_args["source_metadata"] or os.path.join(
                    self.output_path, "source/metadata/metadata.json"
                )
                self.paths["source"]["is_philo_db"] = self.__cli_args["is_philo_db"]
            self.paths["source"]["common_ngrams"] = os.path.join(
                self.output_path, "source/index/most_common_ngrams.txt"
            )
            self.matching_params["ngram_index"] = os.path.join(self.output_path, "source/index/index.tab")
            if self.__file_paths["target_files"]:
                if self.tei_parsing["parse_target_files"] is True:
                    self.paths["target"]["tei_input_files"] = self.__file_paths["target_files"]
                    self.paths["target"]["parse_output"] = os.path.join(self.output_path, "target")
                    self.paths["target"]["input_files_for_ngrams"] = os.path.join(self.output_path, "target/texts")
                    self.paths["target"]["ngram_output_path"] = os.path.join(self.output_path, "target/")
                    self.paths["target"]["metadata_path"] = os.path.join(
                        self.output_path, "target/metadata/metadata.json"
                    )
                    self.paths["target"]["is_philo_db"] = False
                else:
                    if self.__cli_args["is_philo_db"] is True:
                        self.paths["target"]["input_files_for_ngrams"] = os.path.join(
                            self.__file_paths["target_files"], "data/words_and_philo_ids"
                        )
                    else:
                        self.paths["target"]["input_files_for_ngrams"] = self.__file_paths["target_files"]
                    self.paths["target"]["ngram_output_path"] = os.path.join(self.output_path, "target/")
                    self.paths["target"]["metadata_path"] = self.__cli_args["target_metadata"] or os.path.join(
                        self.output_path, "target/metadata/metadata.json"
                    )
                    self.paths["target"]["is_philo_db"] = self.__cli_args["is_philo_db"]
                self.paths["target"]["common_ngrams"] = os.path.join(
                    self.output_path, "target/index/most_common_ngrams.txt"
                )
        else:
            self.paths["source"]["ngram_output_path"] = os.path.join(self.output_path, "source")
            self.paths["source"]["metadata_path"] = os.path.join(self.output_path, "source/metadata/metadata.json")
            self.paths["source"]["common_ngrams"] = os.path.join(
                self.output_path, "source/index/most_common_ngrams.txt"
            )
            self.matching_params["ngram_index"] = os.path.join(self.output_path, "source/index/index.tab")
            if self.__file_paths["target_files"]:
                self.paths["target"]["ngram_output_path"] = os.path.join(self.output_path, "target")
                self.paths["target"]["metadata_path"] = os.path.join(self.output_path, "target/metadata/metadata.json")
                self.paths["target"]["common_ngrams"] = os.path.join(
                    self.output_path, "target/index/most_common_ngrams.txt"
                )
            else:
                self.paths["target"] = self.paths["source"]


def get_config() -> TextPairConfig:
    """Command line parsing function"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dbname",
        help="Name of TextPAIR database. This will also be used for the PostgreSQL table name. Must not be longer than 30 characters",
    )
    parser.add_argument(
        "--config", help="configuration file used to override defaults", type=str, default="", required=True
    )
    parser.add_argument("--source_files", help="path to source files from which to compare", type=str)
    parser.add_argument("--target_files", help="path to target files to compared to source files", type=str, default="")
    parser.add_argument(
        "--is_philo_db", help="define if files are from a PhiloLogic instance", action="store_true", default=False
    )
    parser.add_argument(
        "--source_metadata", help="path to source metadata if not from PhiloLogic instance", type=str, default=""
    )
    parser.add_argument(
        "--target_metadata", help="path to target metadata if not from PhiloLogic instance", type=str, default=""
    )
    parser.add_argument(
        "--only_align",
        help="skip parsing or ngram generation phase to go straight to the aligner",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--source_common_ngrams", help="path to source common ngrams when using --only_align", type=str, default=""
    )
    parser.add_argument(
        "--target_common_ngrams", help="path to target common ngrams when using --only_align", type=str, default=""
    )
    parser.add_argument(
        "--ngram_index", help="path to ngram index when using --only_align with debug", type=str, default=""
    )
    parser.add_argument(
        "--skip_web_app",
        help="define whether to load results into a database and build a corresponding web app",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--load_only_web_app",
        help="define whether to load results into a database and build a corresponding web app",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--update_db", help="update database without rebuilding web_app", action="store_true", default=False
    )
    parser.add_argument("--file", help="alignment file to load", type=str, default=None)
    parser.add_argument(
        "--output_path", help="output path for ngrams and sequence alignment", type=str, default="output"
    )
    parser.add_argument(
        "--workers", help="How many threads or cores to use for preprocessing and matching", type=int, default=4
    )
    parser.add_argument("--debug", help="add debugging", action="store_true", default=False)
    args = vars(parser.parse_args())
    if args["config"]:
        if not os.path.exists(args["config"]):
            print(f"""config file does not exist at the location {args["config"]} you provided.""")
            print("Exiting...")
            exit()
    else:
        print("No config file provided.")
        print("Exiting...")
        exit()
    text_pair_config = TextPairConfig(args)
    return text_pair_config
