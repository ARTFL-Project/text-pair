#!/usr/bin/env python3
"""Sequence aligner script"""

import argparse
from glob import glob
from ast import literal_eval
import os
import re

from compare_ngrams import SequenceAligner
from generate_ngrams import Ngrams

TRIM_LAST_SLASH = re.compile(r'/\Z')


def parse_command_line():
    """Command line parsing function"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", help="path to source files from which to compare",
                        type=str)
    parser.add_argument("--target_path", help="path to target files to compare to source files",
                        type=str, default="")
    parser.add_argument("--is_philo_db", help="define is files are from a PhiloLogic4 instance",
                        type=literal_eval, default=True)
    parser.add_argument("--language", help="define language for preprocessing",
                        type=str, default="french")
    parser.add_argument("--ngrams", help="define how many tokens constitute an ngram, or specify skipgram for skipgrams",
                        type=str, default="3")
    parser.add_argument("--output_path", help="output path of ngrams",
                        type=str, default="./")
    parser.add_argument("--output_type", help="output format: html, json (see docs for proper decoding), xml, or tab",
                        type=str, default="html")
    parser.add_argument("--stem", help="define whether to stem input text", action='store_true', default=False)
    parser.add_argument("--stopwords", help="define path to stopwords", type=str, default="")
    parser.add_argument("--debug", help="add debugging", action='store_true', default=False)
    args = vars(parser.parse_args())
    if args["is_philo_db"]:
        args["source_path"] = TRIM_LAST_SLASH.sub("", args["source_path"])
        args["source_files"] = glob(os.path.join(args["source_path"], "data/words_and_philo_ids/*"))
        args["source_output_path"] = os.path.join(args["output_path"], os.path.basename(args["source_path"]))
        if args["target_path"]:
            args["target_path"] = TRIM_LAST_SLASH.sub("", args["target_path"])
            args["target_files"] = glob(os.path.join(args["target_path"], "data/words_and_philo_ids/*"))
            args["target_output_path"] = os.path.join(args["output_path"], os.path.basename(args["target_path"]))
        else:
            args["target_files"] = None
    if args["ngrams"] == "skipgram":
        args["skipgram"] = True
        args["ngrams"] = 3
    else:
        args["skipgram"] = False
        args["ngrams"] = int(args["ngrams"])

    return args

if __name__ == '__main__':
    ARGS = parse_command_line()
    NGRAMS = Ngrams(ngram=ARGS["ngrams"], skipgram=ARGS["skipgram"], stem=ARGS["stem"], stopwords=ARGS["stopwords"],
                    language=ARGS["language"])
    print("Generating source ngrams...")
    NGRAMS.generate(ARGS["source_files"], ARGS["source_output_path"])
    SOURCE_JSON = NGRAMS.json_output_files
    SOURCE_INDEX = NGRAMS.ngram_index
    if ARGS["target_files"] is not None:
        print("Generating target ngrams...")
        NGRAMS.generate(ARGS["target_files"], ARGS["target_output_path"])
        TARGET_JSON = NGRAMS.json_output_files
        TARGET_INDEX = NGRAMS.ngram_index
    else:
        TARGET_JSON = None
        TARGET_INDEX = None
    print("\n### Starting sequence alignment ###")
    ALIGNER = SequenceAligner(SOURCE_JSON, SOURCE_INDEX, target_files=TARGET_JSON, target_ngram_index=TARGET_INDEX,
                              output=ARGS["output_type"], debug=ARGS["debug"], source_db_path=ARGS["source_path"],
                              target_db_path=ARGS["target_path"])
    ALIGNER.compare()
