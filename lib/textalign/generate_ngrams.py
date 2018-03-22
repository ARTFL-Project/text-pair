#!/usr/bin/env python3
"""N-gram generator"""

import argparse
import configparser
import html
import json
import os
import re
import sys
import unicodedata
from ast import literal_eval
from collections import defaultdict, deque
from glob import glob
from itertools import combinations, permutations
from math import floor

from multiprocess import Pool
from tqdm import tqdm

from mmh3 import hash as hash32
from text_preprocessing import PreProcessor, modernize

try:
    from philologic.DB import DB
except ImportError:
    DB = None


# See https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python/266162#266162
TRIM_LAST_SLASH = re.compile(r'/\Z')

PHILO_TEXT_OBJECT_LEVELS = {'doc': 1, 'div1': 2, 'div2': 3, 'div3': 4, 'para': 5, 'sent': 6, 'word': 7}


class Ngrams:
    """Generate Ngrams"""

    def __init__(self, text_object_level="doc", ngram=3, gap=0, stemmer=True, lemmatizer="", stopwords=None, numbers=False, language="french",
                 lowercase=True, minimum_word_length=2, word_order=True, modernize=True, debug=False):
        self.config = {
            "ngram": ngram,
            "window": ngram + gap,
            "word_order": word_order,
            "numbers": numbers,
            "stemmer": stemmer,
            "modernize": modernize,
            "language": language,
            "lowercase": lowercase,
            "minimum_word_length": minimum_word_length,
            "lemmatizer": lemmatizer,
            "stopwords": stopwords,
            "text_object_level": text_object_level
        }
        self.debug = debug
        self.input_path = ""
        self.output_path = ""
        self.metadata_done = False
        self.db_name = ""
        self.db_path = ""

    def __write_to_disk(self, ngrams, text_id):
        with open("%s/debug/%s_ngrams.json" % (self.output_path, text_id), "w") as output:
            json.dump(list(ngrams), output)

    def __build_text_index(self, ngrams, text_id):
        """Build a file representation used for ngram comparisons"""
        text_index = defaultdict(list)
        index_pos = 0
        for ngram, start_byte, end_byte in ngrams:
            text_index[ngram].append((index_pos, start_byte, end_byte))
            index_pos += 1
        with open("%s/ngrams/%s.json" % (self.output_path, text_id), "w") as json_file:
            json.dump(dict(text_index), json_file)

    def __get_metadata(self, text_id):
        """Pull metadata from PhiloLogic DB based on position of ngrams in file"""
        metadata = {}
        philo_db = DB(os.path.join(self.input_path, "data"), cached=False)
        try:
            text_object = philo_db[text_id.split('_')]
            for field in philo_db.locals["metadata_fields"]:
                metadata[field] = str(text_object[field])
        except AttributeError:
            pass
        metadata["filename"] = os.path.join(self.input_path, "data/TEXT", metadata["filename"])
        return metadata

    def __dump_config(self, output_path):
        with open(os.path.join(output_path, "config/ngram_config.ini"), "w") as ini_file:
            ngram_config = configparser.ConfigParser()
            ngram_config.add_section('PREPROCESSING')
            for param, value in self.config.items():
                ngram_config.set("PREPROCESSING", param, repr(value))
            ngram_config.write(ini_file)

    def generate(self, file_path, output_path, is_philo_db=False, db_path=None, metadata=None, workers=4, ram="50%"):
        """Generate n-grams."""
        if os.path.isfile(file_path):
            files = [file_path]
        else:
            files = glob(os.path.join(file_path, "*"))
        os.system('rm -rf {}/ngrams'.format(output_path))
        os.system('mkdir -p {}/ngrams'.format(output_path))
        if self.debug:
            os.system("mkdir {}/debug".format(output_path))
        os.system('mkdir -p {}/metadata'.format(output_path))
        os.system("mkdir -p {}/index".format(output_path))
        os.system("mkdir -p {}/config".format(output_path))
        os.system('mkdir -p {}/temp'.format(output_path))
        if db_path is None and is_philo_db is True:
            self.input_path = os.path.dirname(os.path.abspath(files[0])).replace("data/words_and_philo_ids", "")
        else:
            self.input_path = db_path
            self.db_path = db_path
        self.output_path = output_path
        if is_philo_db:
            combined_metadata = {}
        elif os.path.isfile(metadata):
            self.metadata_done = True
            combined_metadata = metadata
        else:
            print("No metadata provided: exiting...")
            exit()

        print("\nGenerating ngrams...", flush=True)
        pool = Pool(workers)
        with tqdm(total=len(files)) as pbar:
            for local_metadata in pool.imap_unordered(self.process_file, files):
                if self.metadata_done is False:
                    combined_metadata.update(local_metadata)
                pbar.update()
        pool.close()
        pool.join()

        mem_usage = floor(int(ram.replace('%', '')) / 2)
        if mem_usage >= 50:
            mem_usage = 45
        print("Saving ngram index and most common ngrams (this can take a while)...", flush=True)
        os.system(r'''for i in {}/temp/*; do cat $i; done | sort -S {}% | uniq -c | sort -rn -S {}% | awk '{{print $2"\t"$3}}' |
                tee {}/index/index.tab | awk '{{print $2}}' > {}/index/most_common_ngrams.txt'''
                .format(output_path, mem_usage, mem_usage, output_path, output_path))

        print("Saving metadata...")

        if self.metadata_done is False:
            with open("%s/metadata/metadata.json" % self.output_path, "w") as metadata_output:
                json.dump(combined_metadata, metadata_output)
        else:
            os.system("cp {} {}/metadata/metadata.json".format(metadata, self.output_path))

        self.__dump_config(output_path)

        print("Cleaning up...")
        os.system("rm -r {}/temp".format(self.output_path))

    def process_file(self, input_file):
        """Convert each file into an inverted index of ngrams"""
        preprocessor = PreProcessor(language=self.config["language"], stemmer=self.config["stemmer"],
                                    lemmatizer=self.config["lemmatizer"], modernize=True, lowercase=self.config["lowercase"],
                                    min_word_length=self.config["minimum_word_length"], strip_numbers=self.config["numbers"],
                                    stopwords=self.config["stopwords"])
        doc_ngrams = []
        metadata = {}
        with open(input_file) as filehandle:
            ngrams = deque([])
            ngram_obj = deque([])
            current_text_id = None
            for line in filehandle:
                word_obj = json.loads(line.strip())
                word = word_obj["token"]
                if self.config["modernize"] is True:
                    word = modernize(word, self.config["language"])
                word = preprocessor.lemmatizer.get(word, word)
                word = preprocessor.normalize(word)
                if word == "":
                    continue
                position = word_obj["position"]
                if self.config["text_object_level"] == 'doc':
                    text_id = position.split()[0]
                else:
                    text_id = '_'.join(position.split()[:PHILO_TEXT_OBJECT_LEVELS[self.config["text_object_level"]]])
                if current_text_id is None:
                    current_text_id = text_id
                if current_text_id != text_id:
                    if self.debug:
                        self.__write_to_disk(ngrams, current_text_id)
                    if self.metadata_done is False:
                        metadata[current_text_id] = self.__get_metadata(current_text_id)
                    self.__build_text_index(ngrams, current_text_id)
                    ngrams = deque([])
                    ngram_obj = deque([])
                    current_text_id = text_id
                ngram_obj.append((word, position, word_obj["start_byte"], word_obj["end_byte"]))
                if len(ngram_obj) == self.config["window"]:   # window is ngram+gap
                    if self.config["word_order"] is True:
                        iterator = combinations(ngram_obj, self.config["ngram"])
                    else:
                        iterator = permutations(ngram_obj)
                    for value in iterator:
                        current_ngram_list, _, start_bytes, end_bytes = zip(*value)
                        current_ngram = "_".join(current_ngram_list)
                        hashed_ngram = hash32(current_ngram)
                        ngrams.append((hashed_ngram, start_bytes[0], end_bytes[-1]))
                        doc_ngrams.append("\t".join((current_ngram, str(hashed_ngram))))
                    ngram_obj.popleft()
            if self.config["text_object_level"] == "doc" and current_text_id is not None:  # make sure the file is not empty (no lines so never entered loop)
                if self.debug:
                    self.__write_to_disk(ngrams, current_text_id)
                if self.metadata_done is False:
                    metadata[current_text_id] = self.__get_metadata(current_text_id)
                self.__build_text_index(ngrams, current_text_id)
            with open("{}/temp/{}".format(self.output_path, os.path.basename(input_file)), "w") as output:
                output.write("\n".join(sorted(doc_ngrams)))
        return metadata

def parse_command_line():
    """Command line parsing function"""
    parser = argparse.ArgumentParser(prog="generate_ngrams")
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument("--config", help="configuration file used to override defaults",
                          type=str, default="")
    optional.add_argument("--cores", help="number of cores used for parsing and generating ngrams",
                          type=int, default=4)
    required.add_argument("--file_path", help="path to files",
                          type=str)
    optional.add_argument("--lemmatizer", help="path to a file where each line contains a token/lemma pair separated by a tab ")
    optional.add_argument("--mem_usage", help="how much max RAM to use: expressed in percentage, no higher than 90%%",
                          type=str, default="80%%")
    optional.add_argument("--is_philo_db", help="define is files are from a PhiloLogic4 instance",
                          type=literal_eval, default=False)
    optional.add_argument("--metadata", help="metadata for input files", default=None)
    optional.add_argument("--text_object_level", help="type of object to split up docs in",
                          type=str, default="doc")
    optional.add_argument("--output_path", help="output path of ngrams",
                          type=str, default="./output")
    optional.add_argument("--debug", help="add debugging", action='store_true', default=False)
    optional.add_argument("--stopwords", help="path to stopword list", type=str, default=None)
    optional.add_argument("--ngram", help="number of grams", type = int, default=3)
    optional.add_argument("--gap", help="number of gap", action='store_true', default=0)
    optional.add_argument("--word_order", help="words order must be respected", action='store_true', default=True)
    args = vars(parser.parse_args())
    if len(sys.argv[1:]) == 0:  # no command line args were provided
        parser.print_help()
        exit()
    return args

if __name__ == '__main__':
    ARGS = parse_command_line()
    NGRAM_GENERATOR = Ngrams(stopwords=ARGS["stopwords"], lemmatizer=ARGS["lemmatizer"], text_object_level=ARGS["text_object_level"], gap=ARGS["gap"])
    NGRAM_GENERATOR.generate(ARGS["file_path"], ARGS["output_path"], is_philo_db=ARGS["is_philo_db"], metadata=ARGS["metadata"],
                             workers=ARGS["cores"], ram=ARGS["mem_usage"], db_path=ARGS["db_path"])
