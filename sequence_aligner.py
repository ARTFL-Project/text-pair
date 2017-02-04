#!/usr/bin/env python3
"""Sequence aligner script"""

import argparse
import configparser
import os
import re
from ast import literal_eval
from glob import glob

from compare_ngrams import SequenceAligner
from generate_ngrams import Ngrams

TRIM_LAST_SLASH = re.compile(r'/\Z')


def parse_command_line():
    """Command line parsing function"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="configuration file used to override defaults",
                        type=str, default="")
    parser.add_argument("--source_path", help="path to source files from which to compare",
                        type=str)
    parser.add_argument("--target_path", help="path to target files to compare to source files",
                        type=str, default="")
    parser.add_argument("--is_philo_db", help="define is files are from a PhiloLogic4 instance",
                        type=literal_eval, default=True)
    parser.add_argument("--output_path", help="output path of ngrams",
                        type=str, default="./")
    parser.add_argument("--output_type", help="output format: html, json (see docs for proper decoding), xml, or tab",
                        type=str, default="html")
    parser.add_argument("--debug", help="add debugging", action='store_true', default=False)
    args = vars(parser.parse_args())
    if args["is_philo_db"]:
        args["source_path"] = TRIM_LAST_SLASH.sub("", args["source_path"])
        args["source_files"] = sorted(glob(os.path.join(args["source_path"], "data/words_and_philo_ids/*")))
        args["source_output_path"] = os.path.join(args["output_path"], os.path.basename(args["source_path"]))
        if args["target_path"]:
            args["target_path"] = TRIM_LAST_SLASH.sub("", args["target_path"])
            args["target_files"] = sorted(glob(os.path.join(args["target_path"], "data/words_and_philo_ids/*")))
            args["target_output_path"] = os.path.join(args["output_path"], os.path.basename(args["target_path"]))
        else:
            args["target_files"] = None
    args["preprocessing"] = {}
    args["matching"] = {}
    if args["config"]:
        if os.path.exists(args["config"]):
            config = configparser.ConfigParser()
            config.read(args["config"])
            for key, value in dict(config["Preprocessing"]).items():
                if key == "skipgram" or key == "stemmer" or key == "lowercase":
                    args["preprocessing"][key] = config.getboolean("Preprocessing", key)
                elif key == "ngram":
                    args["preprocessing"][key] = config.getint("Preprocessing", key)
                else:
                    args["preprocessing"][key] = value
            for key, value in dict(config["Matching"]).items():
                if key == "matching_window_size" or key == "minimum_matching_ngrams_in_window = 4" or key == "max_gap" \
                or key == "minimum_matching_ngrams" or key == "common_ngrams_in_docs":
                    args["matching"][key] = config.getint("Matching", key)
                elif key == "common_ngrams_limit":
                    args["matching"][key] = config.getfloat("Matching", key)
                else:
                    args["matching"][key] = value
    return args

if __name__ == '__main__':
    ARGS = parse_command_line()
    NGRAMS = Ngrams(**ARGS["preprocessing"])
    print("### Generating source ngrams ###")
    NGRAMS.generate(ARGS["source_files"], ARGS["source_output_path"])
    SOURCE_JSON = sorted(NGRAMS.json_output_files)
    SOURCE_INDEX = NGRAMS.ngram_index
    if ARGS["target_files"] is not None:
        print("\n### Generating target ngrams ###")
        NGRAMS.generate(ARGS["target_files"], ARGS["target_output_path"])
        TARGET_JSON = sorted(NGRAMS.json_output_files)
        TARGET_INDEX = NGRAMS.ngram_index
    else:
        TARGET_JSON = None
        TARGET_INDEX = None
    print("\n### Starting sequence alignment ###")
    ALIGNER = SequenceAligner(SOURCE_JSON, SOURCE_INDEX, target_files=TARGET_JSON, target_ngram_index=TARGET_INDEX,
                              output=ARGS["output_type"], debug=ARGS["debug"], source_db_path=ARGS["source_path"],
                              target_db_path=ARGS["target_path"], **ARGS["matching"])
    ALIGNER.compare()
