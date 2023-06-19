#!/usr/bin/env python3
"""Parse textpair config file"""

import configparser
import argparse
import os
from typing import Any
from collections import defaultdict


class TextPairConfig:
    """TextPAIR parameters returned from parsing CLI arguments"""

    def __init__(self, cli_args: dict[str, Any]):
        self.__cli_args: dict[str, Any] = cli_args
        self.is_philo_db: bool = self.__cli_args["is_philo_db"]
        self.__file_paths: dict[str, str] = {}
        self.text_parsing: dict[str, bool | str] = {}
        self.preprocessing_params: dict[str, Any] = {"source": {}, "target": {}}
        self.matching_params: defaultdict[str, Any] = defaultdict(str)
        self.matching_params["matching_algorithm"] = "sa"
        self.web_app_config: dict[str, Any] = {"skip_web_app": self.__cli_args["skip_web_app"]}
        self.only_web_app = self.__cli_args["load_only_web_app"]
        self.paths: dict[str, dict[str, Any]] = {"source": {}, "target": defaultdict(str)}
        self.source_against_source = False
        self.__parse_config()
        self.__set_params()

    def __getattr__(self, attr):
        return self.__cli_args[attr]

    def __parse_config(self):
        """Read config file and store into 4 dicts for each phase of the alignment"""
        global_config = configparser.ConfigParser()
        global_config.read("/etc/text-pair/global_settings.ini")
        self.web_app_config["web_application_directory"] = global_config["WEB_APP"]["web_app_path"]
        self.web_app_config["api_server"] = global_config["WEB_APP"]["api_server"]
        config = configparser.ConfigParser()
        config.read(self.__cli_args["config"])
        self.web_app_config["source_url"] = config["TEXT_SOURCES"]["source_url"]
        self.web_app_config["target_url"] = config["TEXT_SOURCES"]["target_url"]
        if self.__cli_args["is_philo_db"] is True:
            self.web_app_config["source_philo_db_path"] = config["TEXT_SOURCES"]["source_file_path"]
            self.web_app_config["target_philo_db_path"] = config["TEXT_SOURCES"]["target_file_path"] or config["TEXT_SOURCES"]["source_file_path"]
        else:
            self.web_app_config["source_philo_db_path"] = os.path.join(global_config["WEB_APP"]["web_app_path"], self.__cli_args["dbname"], "source_data")
            self.web_app_config["target_philo_db_path"] = os.path.join(global_config["WEB_APP"]["web_app_path"], self.__cli_args["dbname"], "target_data")
        if self.only_align is False:
            self.__file_paths = {
                "source_files": config["TEXT_SOURCES"]["source_file_path"] or "",
                "input_source_metadata": config["TEXT_SOURCES"]["source_metadata"] or "",
                "target_files": config["TEXT_SOURCES"]["target_file_path"] or "",
                "input_target_metadata": config["TEXT_SOURCES"]["target_metadata"] or "",
            }
        else:
            self.__file_paths["source_files"] = os.path.join(self.output_path, "source")
            if config["TEXT_SOURCES"]["target_file_path"]:
                self.__file_paths["target_files"] = os.path.join(self.output_path, "target")
            else:
                self.__file_paths["target_files"] = ""

        for key, value in dict(config["TEXT_PARSING"]).items():
            if key.startswith("parse"):
                if value.lower() == "yes" or value.lower() == "true":
                    self.text_parsing[key] = True
                else:
                    self.text_parsing[key] = False
            else:
                if not value:
                    if key.startswith("output_source"):
                        value = os.path.join(self.output_path, "source")
                    else:
                        value = os.path.join(self.output_path, "target")
                self.text_parsing[key] = value
        for key, value in dict(config["PREPROCESSING"]).items():
            if not value:
                continue
            match key:
                case "skipgram" | "numbers" | "word_order" | "modernize" | "ascii" | "stemmer":
                    if value.lower() == "yes" or value.lower() == "true":
                        value = True
                    else:
                        value = False
                    self.preprocessing_params["source"][key] = value
                    self.preprocessing_params["target"][key] = value
                case "source_text_object_type":
                    self.preprocessing_params["source"]["text_object_type"] = value
                case "target_text_object_type":
                    self.preprocessing_params["target"]["text_object_type"] = value
                case "ngram" | "gap" | "minimum_word_length" | "n_chunk" | "min_text_object_length":
                    self.preprocessing_params["source"][key] = int(value)
                    self.preprocessing_params["target"][key] = int(value)
                case "pos_to_keep":
                    self.preprocessing_params["source"][key] = [i.strip() for i in value.split(",")]
                    self.preprocessing_params["target"][key] = self.preprocessing_params["source"][key]
                case "min_freq" | "max_freq":
                    self.preprocessing_params["source"][key] = float(value)
                    self.preprocessing_params["target"][key] = float(value)
                case "target_language":
                    if value:
                        self.preprocessing_params["target"]["language"] = value
                    else:  # assumes the language key comes before the target_language key...
                        self.preprocessing_params["target"]["language"] = self.preprocessing_params["source"][
                            "language"
                        ]
                case _:
                    self.preprocessing_params["source"][key] = value
                    self.preprocessing_params["target"][key] = value
        for key, value in dict(config["MATCHING"]).items():
            if value or key not in self.matching_params:
                match key:
                    case "flex_gap" | "banality_auto_detection" | "store_banalities":
                        if value.lower() == "yes" or value.lower() == "true":
                            value = True
                        else:
                            value = False
                    case "min_similarity" | "most_common_ngram_proportion" | "common_ngram_threshold":
                        value = float(value)
                    case "min_matching_words" | "source_batch" | "target_batch":
                        value = int(value)
                self.matching_params[key] = value
        if not config["TEXT_SOURCES"]["target_file_path"]:
            self.source_against_source = True

    def __set_params(self):
        """Set parameters for alignment"""
        self.web_app_config["skip_web_app"] = self.__cli_args["skip_web_app"]
        if self.__cli_args["only_align"] is False and self.__cli_args["update_db"] is False:
            if self.text_parsing["parse_source_files"] is True:
                self.paths["source"]["input_files"] = self.__file_paths["source_files"]
                self.paths["source"]["input_source_metadata"] = self.__file_paths["input_source_metadata"]
                self.paths["source"]["parse_output"] = os.path.join(self.output_path, "source")
                self.paths["source"]["input_files_for_ngrams"] = os.path.join(
                    self.output_path, "source/words_and_philo_ids/"
                )
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
                self.paths["source"]["metadata_path"] = os.path.join(self.output_path, "source/metadata/metadata.json")
                self.paths["source"]["is_philo_db"] = self.__cli_args["is_philo_db"]
            self.paths["source"]["common_ngrams"] = os.path.join(
                self.output_path, "source/index/most_common_ngrams.txt"
            )
            self.matching_params["ngram_index"] = os.path.join(self.output_path, "source/index/index.tab")
            if self.__file_paths["target_files"]:
                if self.text_parsing["parse_target_files"] is True:
                    self.paths["target"]["input_files"] = self.__file_paths["target_files"]
                    self.paths["target"]["input_target_metadata"] = self.__file_paths["input_target_metadata"]
                    self.paths["target"]["parse_output"] = os.path.join(self.output_path, "target")
                    self.paths["target"]["input_files_for_ngrams"] = os.path.join(
                        self.output_path, "target/words_and_philo_ids/"
                    )
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
                    self.paths["target"]["metadata_path"] = os.path.join(
                        self.output_path, "target/metadata/metadata.json"
                    )
                    self.paths["target"]["is_philo_db"] = self.__cli_args["is_philo_db"]
                self.paths["target"]["common_ngrams"] = os.path.join(
                    self.output_path, "target/index/most_common_ngrams.txt"
                )
        elif self.__cli_args["update_db"] is True:
            self.paths["source"]["metadata_path"] = self.__cli_args["source_metadata"]
            self.paths["target"]["metadata_path"] = self.__cli_args["target_metadata"]
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
    parser.add_argument(
        "--is_philo_db", help="define if files are from a PhiloLogic instance", action="store_true", default=False
    )
    parser.add_argument(
        "--only_align",
        help="skip parsing or ngram generation phase to go straight to the aligner",
        action="store_true",
        default=False,
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
        "--source_metadata",
        help="source metadata needed for loading database. Used only with --update_db.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--target_metadata",
        help="target metadata needed for loading database. Used only with --update_db.",
        type=str,
        default=None,
    )
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
