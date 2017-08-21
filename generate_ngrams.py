#!/usr/bin/env python3
"""N-gram generator"""

import argparse
import html
import json
import os
import re
import sys
import unicodedata
from ast import literal_eval
from collections import defaultdict, deque
from glob import glob
from pathlib import Path
from math import floor

from multiprocess import Pool
from tqdm import tqdm

from mmh3 import hash as hash32
from Stemmer import Stemmer
from unidecode import unidecode

try:
    from philologic.DB import DB
except ImportError:
    DB = None


# See https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python/266162#266162
PUNCTUATION_MAP = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))
NUMBER_MAP = {ord(ch): None for ch in '0123456789'}
TRIM_LAST_SLASH = re.compile(r'/\Z')

PHILO_TEXT_OBJECT_LEVELS = {'doc': 1, 'div1': 2, 'div2': 3, 'div3': 4, 'para': 5, 'sent': 6, 'word': 7}


class Ngrams:
    """Generate Ngrams"""

    def __init__(self, ngram=3, skipgram=False, stemmer=True, lemmatizer="", stopwords=None, numbers=False, language="french",
                 lowercase=True, is_philo_db=True, text_object_level="doc", debug=False):
        self.ngram = ngram
        self.skipgram = skipgram
        self.numbers = numbers
        self.stemmer = stemmer
        self.language = language
        self.lowercase = lowercase
        if lemmatizer:
            self.lemmatize = True
            self.lemmatizer_path = lemmatizer
        else:
            self.lemmatize = False
        if stopwords is not None and os.path.isfile(stopwords):
            self.stopwords = self.__get_stopwords(stopwords)
        else:
            self.stopwords = set()
        self.is_philo_db = is_philo_db
        if DB is None:
            self.is_philo_db = False
        self.text_object_level = text_object_level
        self.debug = debug
        self.input_path = ""
        self.output_path = ""
        self.metadata_done = False

    def __get_stopwords(self, path):
        stopwords = set([])
        stemmer = Stemmer(self.language)
        if self.lemmatize:
            lemmatizer = self.__get_lemmatizer()
        else:
            lemmatizer = None
        with open(path) as stopword_file:
            for line in stopword_file:
                stopwords.add(self.__normalize(line.strip(), stemmer, lemmatizer))
        return stopwords

    def __get_lemmatizer(self):
        lemmas = {}
        with open(self.lemmatizer_path) as input_file:
            for line in input_file:
                word, lemma = line.strip().split("\t")
                lemmas[word] = lemma
        return lemmas

    def __normalize(self, input_str, stemmer, lemmatizer):
        input_str = input_str.translate(PUNCTUATION_MAP)
        input_str = input_str.translate(NUMBER_MAP)
        if input_str == "":
            return ""
        input_str = html.unescape(input_str)
        if self.lowercase:
            input_str = input_str.lower()
        if self.lemmatize:
            try:
                input_str = lemmatizer[input_str]
            except KeyError:
                try:
                    input_str = lemmatizer[unidecode(input_str)]
                except KeyError:
                    pass
        if self.stemmer:
            input_str = stemmer.stemWord(input_str)
        return unidecode(input_str)

    def __write_to_disk(self, ngrams, text_id):
        with open("%s/debug/%s_ngrams.json" % (self.output_path, text_id), "w") as output:
            json.dump(ngrams, output)

    def __build_text_index(self, ngrams, text_id):
        """Build a file representation used for ngram comparisons"""
        text_index = defaultdict(list)
        index_pos = 0
        for ngram, start_byte, end_byte in ngrams:
            text_index[ngram].append((index_pos, start_byte, end_byte))
            index_pos += 1
        with open("%s/%s.json" % (self.output_path, text_id), "w") as json_file:
            json.dump(dict(text_index), json_file)

    def __get_metadata(self, text_id):
        """Pull metadata from PhiloLogic DB based on position of ngrams in file"""
        metadata = {}
        if self.is_philo_db is True:
            philo_db = DB(os.path.join(self.input_path, "data"), cached=False)
            try:
                text_object = philo_db[text_id.split('_')]
                for field in philo_db.locals["metadata_fields"]:
                    metadata[field] = str(text_object[field])
            except AttributeError:
                pass
        metadata["filename"] = os.path.join(self.input_path, "data/TEXT", metadata["filename"])
        return metadata

    def generate(self, files, output_path, db_path=None, metadata=None, workers=4, ram="20%"):
        """Generate n-grams. Takes a list of files as an argument."""
        os.system('rm -rf %s/' % output_path)
        os.system('mkdir -p %s' % output_path)
        os.system("mkdir %s/index" % output_path)
        os.system('mkdir {}/temp'.format(output_path))
        if db_path is None and self.is_philo_db is True:
            self.input_path = os.path.dirname(os.path.abspath(files[0])).replace("data/words_and_philo_ids", "")
        else:
            self.input_path = db_path
        self.output_path = output_path
        if metadata is None:
            if self.is_philo_db is False:
                print("No metadata provided, only the filename will be used as metadata")
                combined_metadata = {os.path.basename(i): {"filename": os.path.basename(i)} for i in files}
                self.metadata_done = True
            else:
                combined_metadata = {}
        else:
            self.metadata_done = True
            combined_metadata = metadata

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
        os.system("mkdir %s/metadata" % output_path)
        if self.metadata_done is False:
            with open("%s/metadata/metadata.json" % self.output_path, "w") as metadata_output:
                json.dump(combined_metadata, metadata_output)
        else:
            os.system("cp {} {}/metadata/metadata.json".format(metadata, self.output_path))

        print("Cleaning up...")
        os.system("rm -r {}/temp".format(self.output_path))

        ngram_index_path = os.path.join(self.output_path, "index/index.tab")
        return ngram_index_path

    def process_file(self, input_file):
        """Convert each file into an inverted index of ngrams"""
        if self.stemmer:
            stemmer = Stemmer(self.language)  # we initialize here since it creates a deadlock with in self
        else:
            stemmer = None
        if self.lemmatize:
            lemmatizer = self.__get_lemmatizer()  # we initialize here since it is faster than copying between procs
        else:
            lemmatizer = None
        doc_ngrams = []
        metadata = {}
        with open(input_file) as filehandle:
            ngrams = deque([])
            ngram_obj = deque([])
            current_text_id = None
            for line in filehandle:
                word_obj = json.loads(line.strip())
                word = word_obj["token"]
                word = self.__normalize(word, stemmer, lemmatizer)
                if len(word) < 2 or word in self.stopwords:
                    continue
                position = word_obj["philo_id"]
                if self.text_object_level == 'doc':
                    text_id = position.split()[0]
                else:
                    text_id = '_'.join(position.split()[:PHILO_TEXT_OBJECT_LEVELS[self.text_object_level]])
                if current_text_id is None:
                    current_text_id = text_id
                if current_text_id != text_id:
                    if self.debug:
                        self.__write_to_disk(ngrams, current_text_id)
                    print("Storing %s: %s..." %(self.text_object_level, current_text_id))
                    if self.metadata_done is False:
                        metadata[current_text_id] = self.__get_metadata(current_text_id)
                    self.__build_text_index(ngrams, current_text_id)
                    ngrams = deque([])
                    ngram_obj = deque([])
                    current_text_id = text_id
                ngram_obj.append((word, position, word_obj["start_byte"], word_obj["end_byte"]))
                if len(ngram_obj) == self.ngram:
                    if self.skipgram:
                        ngram_obj_to_store = [ngram_obj[0], ngram_obj[-1]]
                    else:
                        ngram_obj_to_store = list(ngram_obj)
                    current_ngram_list, _, start_bytes, end_bytes = zip(*ngram_obj_to_store)
                    current_ngram = "_".join(current_ngram_list)
                    hashed_ngram = hash32(current_ngram)
                    ngrams.append((hashed_ngram, start_bytes[0], end_bytes[-1]))
                    doc_ngrams.append("\t".join((current_ngram, str(hashed_ngram))))
                    ngram_obj.popleft()
            if self.text_object_level == "doc" and current_text_id is not None:  # make sure the file is not empty (no lines so never entered loop)
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
    required.add_argument("--file_path", help="path to source files",
                          type=str)
    optional.add_argument("--lemmatizer", help="path to a file where each line contains a token/lemma pair separated by a tab ")
    optional.add_argument("--mem_usage", help="how much max RAM to use: expressed in percentage, no higher than 90%%",
                          type=str, default="20%%")
    optional.add_argument("--is_philo_db", help="define is files are from a PhiloLogic4 instance",
                          type=literal_eval, default=True)
    optional.add_argument("--metadata", help="metadata for input files", default=None)
    optional.add_argument("--text_object_level", help="type of object to split up docs in",
                          type=str, default="doc")
    optional.add_argument("--output_path", help="output path of ngrams",
                          type=str, default="./ngrams")
    optional.add_argument("--debug", help="add debugging", action='store_true', default=False)
    optional.add_argument("--stopwords", help="path to stopword list", type=str, default=None)
    optional.add_argument("--skipgram", help="use skipgrams", action='store_true', default=False)
    args = vars(parser.parse_args())
    if len(sys.argv[1:]) == 0:  # no command line args were provided
        parser.print_help()
        exit()
    if args["is_philo_db"] is True:
        args["file_path"] = TRIM_LAST_SLASH.sub("", args["file_path"])
        file_path = str(Path(args["file_path"]).joinpath("*"))
        args["files"] = sorted(glob(file_path))
    else:
        args["files"] = glob(Path(args["file_path"]).joinpath("*"))
    return args


if __name__ == '__main__':
    ARGS = parse_command_line()
    print(ARGS["output_path"])
    NGRAM_GENERATOR = Ngrams(stopwords=ARGS["stopwords"], text_object_level=ARGS["text_object_level"], lemmatizer=ARGS["lemmatizer"], skipgram=ARGS["skipgram"])
    NGRAM_GENERATOR.generate(ARGS["files"], ARGS["output_path"], metadata=ARGS["metadata"], workers=ARGS["cores"], ram=ARGS["mem_usage"])
