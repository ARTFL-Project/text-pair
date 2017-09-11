#!/usr/bin/env python3
"""Sequence aligner script"""

import argparse
import configparser
import gc
import os
import re
from ast import literal_eval
from glob import glob
from pathlib import Path

# from compare_ngrams import SequenceAligner
from generate_ngrams import Ngrams

TRIM_LAST_SLASH = re.compile(r'/\Z')


def parse_command_line():
    """Command line parsing function"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="configuration file used to override defaults",
                        type=str, default="")
    parser.add_argument("--source_files", help="path to source files from which to compare",
                        type=str)
    parser.add_argument("--target_files", help="path to target files to compared to source files",
                        type=str, default="")
    parser.add_argument("--is_philo_db", help="define is files are from a PhiloLogic instance",
                        type=literal_eval, default=True)
    parser.add_argument("--source_metadata", help="path to source metadata if not from PhiloLogic instance",
                        type=str, default="")
    parser.add_argument("--target_metadata", help="path to target metadata if not from PhiloLogic instance",
                        type=str, default="")
    parser.add_argument("--output_path", help="output path for ngrams and sequence alignment",
                        type=str, default="./")
    parser.add_argument("--threads", help="How many threads or cores to use for preprocessing and matching",
                        type=int, default=4)
    parser.add_argument("--debug", help="add debugging", action='store_true', default=False)
    args = vars(parser.parse_args())
    args["preprocessing"] = {}
    args["matching"] = {}
    if args["config"]:
        if os.path.exists(args["config"]):
            config = configparser.ConfigParser()
            config.read(args["config"])
            for key, value in dict(config["PREPROCESSING"]).items():
                if key.endswith("object_level"):
                    args[key] = value
                elif value or key not in args["preprocessing"]:
                    if key == "ngram":
                        args["preprocessing"][key] = int(value)
                    else:
                        args["preprocessing"][key] = value
            for key, value in dict(config["MATCHING"]).items():
                if value or key not in args["matching"]:
                    args["matching"][key] = value
    args["source_files"] = glob(str(Path(args["source_files"]).joinpath("*")))
    args["source_files_path"] = Path(args["output_path"]).joinpath("source_ngrams")
    args["source_metadata_path"] = Path(args["source_files_path"]).joinpath("metadata/metadata.json")
    if args["target_files"]:
        args["target_files"] = glob(str(Path(args["target_files"]).joinpath("*")))
        args["target_files_path"] = Path(args["output_path"]).joinpath("target_ngrams")
        args["target_metadata_path"] = Path(args["target_files_path"]).joinpath("metadata/metadata.json")
    else:
        args["target_files"] = ""
        args["target_files_path"] = ""
        args["target_metadata_path"] = ""
    return args

if __name__ == '__main__':
    ARGS = parse_command_line()
    NGRAMS = Ngrams(**ARGS["preprocessing"])
    print("### Generating source ngrams ###")
    NGRAMS.generate(ARGS["source_files"], ARGS["source_files_path"], text_object_level=ARGS["source_text_object_level"], workers=ARGS["threads"])
    if ARGS["target_files"]:
        print("\n### Generating target ngrams ###")
        NGRAMS.generate(ARGS["target_files"], ARGS["target_files_path"], text_object_level=ARGS["target_text_object_level"], workers=ARGS["threads"])
    print("\n### Starting sequence alignment ###")
    if ARGS["config"]:
        os.system("./compareNgrams \
                  --output_path={} \
                  --threads={} \
                  --source_files={} \
                  --target_files={} \
                  --source_metadata={} \
                  --target_metadata={} \
                  --sort_by={} \
                  --source_batch={} \
                  --target_batch={} \
                  --source_common_ngrams={} \
                  --target_common_ngrams={} \
                  --most_common_ngram_threshold={} \
                  --common_ngrams_limit={} \
                  --matching_window_size={} \
                  --max_gap={} \
                  --minimum_matching_ngrams={} \
                  --minimum_matching_ngrams_in_window={} \
                  --minimum_matching_ngrams_in_docs={} \
                  --context_size={} \
                  --banal_ngrams={} \
                  --duplicate_threshold={} \
                  --merge_passages_on_byte_distance={} \
                  --merge_passages_on_ngram_distance={} \
                  --passage_distance_multiplier={} \
                  --one_way_matching={} \
                  --debug={} \
                  --ngram_index={}"
                  .format(
                      Path(ARGS["output_path"]).joinpath("results"),
                      ARGS["threads"],
                      ARGS["source_files_path"],
                      ARGS["target_files_path"],
                      ARGS["source_metadata_path"],
                      ARGS["target_metadata_path"],
                      ARGS["matching"]["sort_by"],
                      ARGS["matching"]["source_batch"],
                      ARGS["matching"]["target_batch"],
                      ARGS["matching"]["source_common_ngrams"],
                      ARGS["matching"]["target_common_ngrams"],
                      ARGS["matching"]["most_common_ngram_threshold"],
                      ARGS["matching"]["common_ngrams_limit"],
                      ARGS["matching"]["matching_window_size"],
                      ARGS["matching"]["max_gap"],
                      ARGS["matching"]["minimum_matching_ngrams"],
                      ARGS["matching"]["minimum_matching_ngrams_in_window"],
                      ARGS["matching"]["minimum_matching_ngrams_in_docs"],
                      ARGS["matching"]["context_size"],
                      ARGS["matching"]["banal_ngrams"],
                      ARGS["matching"]["duplicate_threshold"],
                      ARGS["matching"]["merge_passages_on_byte_distance"],
                      ARGS["matching"]["merge_passages_on_ngram_distance"],
                      ARGS["matching"]["passage_distance_multiplier"],
                      str(ARGS["matching"]["one_way_matching"]).lower(),
                      str(ARGS["matching"]["debug"]).lower(),
                      ARGS["matching"]["ngram_index"],
                  ))
    else:
        os.system("./compareNgrams \
                  --output_path={} \
                  --source_files={} \
                  --target_files={} \
                  --source_metadata={}/metadata/metadata.json \
                  --target_metadata={}/metadata/metadata.json \
                  --debug={}"
                  .format(
                      Path(ARGS["output_path"]).joinpath("results"),
                      ARGS["source_files_path"],
                      ARGS["target_files_path"],
                      ARGS["source_metadata"],
                      ARGS["target_metadata"],
                      str(ARGS["debug"]).lower()
                  ))
