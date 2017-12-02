#!/usr/bin/env python3
"""Sequence aligner script"""

import argparse
import configparser
import os
import re
from ast import literal_eval
from collections import defaultdict

from .xml_parser import TEIParser
from .generate_ngrams import Ngrams

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
                        type=literal_eval, default=False)
    parser.add_argument("--source_metadata", help="path to source metadata if not from PhiloLogic instance",
                        type=str, default="")
    parser.add_argument("--target_metadata", help="path to target metadata if not from PhiloLogic instance",
                        type=str, default="")
    parser.add_argument("--output_path", help="output path for ngrams and sequence alignment",
                        type=str, default="./output")
    parser.add_argument("--workers", help="How many threads or cores to use for preprocessing and matching",
                        type=int, default=4)
    parser.add_argument("--debug", help="add debugging", action='store_true', default=False)
    args = vars(parser.parse_args())
    tei_parsing = {}
    preprocessing_params = {"source": {}, "target": {}}
    matching_params = {}
    if args["config"]:
        if os.path.exists(args["config"]):
            config = configparser.ConfigParser()
            config.read(args["config"])
            for key, value in dict(config["TEI_PARSING"]).items():
                if key.startswith("parse"):
                    if value.lower() == "yes" or value.lower() == "true":
                        tei_parsing[key] = True
                    else:
                        tei_parsing[key] = False
                else:
                    if not value:
                        if key.startswith("output_source"):
                            value = os.path.join(args["output_path"], "source")
                        else:
                            value = os.path.join(args["output_path"], "target")
                    tei_parsing[key] = value
            for key, value in dict(config["PREPROCESSING"]).items():
                if value:
                    if key == "skipgram" or key == "numbers" or key == "order":
                        if value.lower() == "yes" or value.lower() == "true":
                            value = True
                        else:
                            value = False
                    if key.endswith("object_level"):
                        if key.startswith("source"):
                            preprocessing_params["source"]["text_object_level"] = value
                        else:
                            preprocessing_params["target"]["text_object_level"] = value
                    elif key == "ngram":
                        preprocessing_params["source"][key] = int(value)
                        preprocessing_params["target"][key] = int(value)
                    elif key == "gap":
                        preprocessing_params["source"][key] = int(value)
                        preprocessing_params["target"][key] = int(value)
                    else:
                        preprocessing_params["source"][key] = value
                        preprocessing_params["target"][key] = value
            for key, value in dict(config["MATCHING"]).items():
                if value or key not in matching_params:
                    matching_params[key] = value
    paths = {"source": {}, "target": defaultdict(str)}
    if tei_parsing["parse_source_files"] is True:
        paths["source"]["tei_input_files"] = args["source_files"]
        paths["source"]["parse_output"] = os.path.join(args["output_path"], "source")
        paths["source"]["input_files_for_ngrams"] = os.path.join(args["output_path"], "source/texts")
        paths["source"]["ngram_output_path"] = os.path.join(args["output_path"], "source/")
        paths["source"]["metadata_path"] = os.path.join(args["output_path"], "source/metadata/metadata.json")
        paths["source"]["is_philo_db"] = False
    else:
        paths["source"]["input_files_for_ngrams"] = args["source_files"]
        paths["source"]["ngram_output_path"] = os.path.join(args["output_path"], "source/")
        paths["source"]["metadata_path"] = args["source_metadata"] or os.path.join(args["output_path"], "source/metadata/metadata.json")
        paths["source"]["is_philo_db"] = args["is_philo_db"]
    if args["target_files"]:
        if tei_parsing["parse_target_files"] is True:
            paths["target"]["tei_input_files"] = args["target_files"]
            paths["target"]["parse_output"] = os.path.join(args["output_path"], "target")
            paths["target"]["input_files_for_ngrams"] = os.path.join(args["output_path"], "target/texts")
            paths["target"]["ngram_output_path"] = os.path.join(args["output_path"], "target/")
            paths["target"]["metadata_path"] = os.path.join(args["output_path"], "target/metadata/metadata.json")
            paths["target"]["is_philo_db"] = False
        else:
            paths["target"]["input_files_for_ngrams"] = args["target_files"]
            paths["target"]["ngram_output_path"] = os.path.join(args["output_path"], "target/")
            paths["target"]["metadata_path"] = args["target_metadata"] or os.path.join(args["output_path"], "target/metadata/metadata.json")
            paths["target"]["is_philo_db"] = args["is_philo_db"]
    return paths, tei_parsing, preprocessing_params, matching_params, args["output_path"], args["workers"], args["debug"]

def run_alignment():
    """Main function to start sequence alignment"""
    paths, tei_parsing, preprocessing_params, matching_params, output_path, workers, debug = parse_command_line()
    if tei_parsing["parse_source_files"] is True:
        print("\n### Parsing source TEI files ###")
        parser = TEIParser(paths["source"]["tei_input_files"], output_path=paths["source"]["parse_output"],
                           words_to_keep=tei_parsing["source_words_to_keep"], cores=workers, debug=debug)
        parser.get_metadata()
        parser.get_text()
    print("\n### Generating source ngrams ###")
    ngrams = Ngrams(**preprocessing_params["source"], debug=debug)
    ngrams.generate(paths["source"]["input_files_for_ngrams"], paths["source"]["ngram_output_path"],
                    metadata=paths["source"]["metadata_path"], is_philo_db=paths["source"]["is_philo_db"],
                    workers=workers)
    if paths["target"]:
        if tei_parsing["parse_target_files"] is True:
            print("\n### Parsing target TEI files ###")
            parser = TEIParser(paths["target"]["tei_input_files"], output_path=paths["target"]["parse_output"], cores=workers,
                               words_to_keep=tei_parsing["target_words_to_keep"], debug=debug)
            parser.get_metadata()
            parser.get_text()
        print("\n### Generating target ngrams ###")
        ngrams = Ngrams(**preprocessing_params["target"], debug=debug)
        ngrams.generate(paths["target"]["input_files_for_ngrams"], paths["target"]["ngram_output_path"],
                        metadata=paths["target"]["metadata_path"], is_philo_db=paths["target"]["is_philo_db"], workers=workers)
    print("\n### Starting sequence alignment ###")
    if paths["target"]["ngram_output_path"] == "":  # if path not defined make target like source
        paths["target"]["ngram_output_path"] = paths["source"]["ngram_output_path"]
    if matching_params:
        os.system("./compareNgrams \
                  --output_path={}/results \
                  --threads={} \
                  --source_files={}/ngrams \
                  --target_files={}/ngrams \
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
                      output_path,
                      workers,
                      paths["source"]["ngram_output_path"],
                      paths["target"]["ngram_output_path"],
                      paths["source"]["metadata_path"],
                      paths["target"]["metadata_path"],
                      matching_params["sort_by"],
                      matching_params["source_batch"],
                      matching_params["target_batch"],
                      matching_params["source_common_ngrams"],
                      matching_params["target_common_ngrams"],
                      matching_params["most_common_ngram_threshold"],
                      matching_params["common_ngrams_limit"],
                      matching_params["matching_window_size"],
                      matching_params["max_gap"],
                      matching_params["minimum_matching_ngrams"],
                      matching_params["minimum_matching_ngrams_in_window"],
                      matching_params["minimum_matching_ngrams_in_docs"],
                      matching_params["context_size"],
                      matching_params["banal_ngrams"],
                      matching_params["duplicate_threshold"],
                      matching_params["merge_passages_on_byte_distance"],
                      matching_params["merge_passages_on_ngram_distance"],
                      matching_params["passage_distance_multiplier"],
                      str(matching_params["one_way_matching"]).lower(),
                      str(debug).lower(),
                      matching_params["ngram_index"],
                  ))
    else:
        os.system("./compareNgrams \
                  --output_path={}/results \
                  --source_files={}/ngrams \
                  --target_files={}/ngrams \
                  --source_metadata={}/metadata/metadata.json \
                  --target_metadata={}/metadata/metadata.json \
                  --debug={}"
                  .format(
                      output_path,
                      paths["source"]["ngram_output_path"],
                      paths["target"]["ngram_output_path"],
                      paths["source"]["metadata_path"],
                      paths["target"]["metadata_path"],
                      str(debug).lower()
                  ))

if __name__ == '__main__':
    run_alignment()
