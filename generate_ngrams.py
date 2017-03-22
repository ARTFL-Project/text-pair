#!/usr/bin/env python3
"""N-gram generator"""

import argparse
import html
import json
import os
import re
import sqlite3
import unicodedata
from ast import literal_eval
from collections import Counter, defaultdict, deque
from glob import glob
from json import dump, loads

from philologic.DB import DB

from Stemmer import Stemmer

PUNCTUATION = re.compile(r'[,?;.:!]*')
NUMBERS = re.compile(r'[0-9]+')
TRIM_LAST_SLASH = re.compile(r'/\Z')

PHILO_TEXT_OBJECT_LEVELS = {'doc': 1, 'div1': 2, 'div2': 3, 'div3': 4, 'para': 5, 'sent': 6, 'word': 7}

class NgramIndex:
    """SQlite representation of ngram to ints"""

    def __init__(self, path, prior_db=None):
        self.save_path = os.path.join(path, "index/")
        self.database = sqlite3.connect(":memory:")
        self.cursor = self.database.cursor()
        self.cursor.execute("CREATE TABLE ngram_index (ngram text PRIMARY KEY, ngram_int INTEGER, count INTEGER) WITHOUT ROWID")
        self.current_index = 0
        if prior_db is not None:
            self.__load_db(prior_db)

    def __load_db(self, prior_db):
        with open(prior_db) as prior_database:
            for line in prior_database:
                ngram, ngram_int = line.strip().split('\t')[:2]
                ngram_int = int(ngram_int)
                if ngram_int > self.current_index:
                    self.current_index = ngram_int
                self.cursor.execute("INSERT INTO ngram_index (ngram, ngram_int, count) VALUES (?, ?, ?)", (ngram, ngram_int, 0))

    def __getitem__(self, item):
        self.cursor.execute('SELECT ngram_int, count FROM ngram_index WHERE ngram=?', (item,))
        result = self.cursor.fetchone()
        if result is None:
            self.current_index += 1
            self.cursor.execute("INSERT INTO ngram_index (ngram, ngram_int, count) VALUES (?, ?, ?)", (item, self.current_index, 1))
            # self.database.commit()
            return self.current_index
        else:
            self.cursor.execute("UPDATE ngram_index SET count=? WHERE ngram=?", (int(result[1])+1, item))
            # self.database.commit()
            return int(result[0])

    def get_most_frequent(self, limit):
        """Get most frequent ngrams and save to JSON"""
        self.cursor.execute("SELECT ngram_int FROM ngram_index where count > 1 order by count desc limit %d" % limit)
        results = []
        for i in self.cursor:
            results.append(i[0])
        with open(os.path.join(self.save_path, "ngram_count.json"), "w") as ngram_count_output:
            json.dump(results, ngram_count_output)

    def save_db(self):
        """Save in memory DB to disk"""
        with open(os.path.join(self.save_path, "index.tab"), "w") as output:
            self.cursor.execute("SELECT ngram, ngram_int, count FROM ngram_index")
            for i in self.cursor:
                print("\t".join(str(j) for j in i), file=output)


class Ngrams:
    """Generate Ngrams"""

    def __init__(self, ngram=3, skipgram=False, stemmer=True, stopwords=None, numbers=False, language="french",
                 lowercase=True, is_philo_db=True, text_object_level="doc", debug=False):
        self.ngram = ngram
        self.skipgram = skipgram
        self.numbers = numbers
        if stemmer:
            try:
                self.stemmer = Stemmer(language)
                self.stemmer.maxCacheSize = 50000
            except Exception:
                self.stemmer = False
        else:
            self.stemmer = False
        self.lowercase = lowercase
        if stopwords is not None and os.path.isfile(stopwords):
            self.stopwords = self.__get_stopwords(stopwords)
        else:
            self.stopwords = []
        self.is_philo_db = is_philo_db
        self.text_object_level = text_object_level
        self.debug = debug
        self.input_path = ""
        self.output_path = ""
        self.metadata = {}

    def __get_stopwords(self, path):
        stopwords = set([])
        with open(path) as stopword_file:
            for line in stopword_file:
                stopwords.add(self.__normalize(line.strip()))
        return stopwords

    def __normalize(self, input_str):
        input_str = PUNCTUATION.sub("", input_str)
        input_str = NUMBERS.sub("", input_str)
        if input_str == "":
            return ""
        input_str = html.unescape(input_str)
        if self.lowercase:
            input_str = input_str.lower()
        if self.stemmer:
            input_str = self.stemmer.stemWord(input_str)
        nkfd_form = unicodedata.normalize('NFKD', input_str)
        return "".join([c for c in nkfd_form if not unicodedata.combining(c)])

    def __write_to_disk(self, ngrams, text_id):
        with open("%s/debug/%s_ngrams.json" % (self.output_path, text_id), "w") as output:
            dump(ngrams, output)

    def __build_text_index(self, ngrams, text_id):
        """Build a file representation used for ngram comparisons"""
        text_index = defaultdict(list)
        index_pos = 0
        for ngram, start_byte, end_byte in ngrams:
            text_index[ngram].append((index_pos, start_byte, end_byte))
            index_pos += 1
        with open("%s/%s.json" % (self.output_path, text_id), "w") as json_file:
            dump(text_index, json_file)

    def __get_metadata(self, text_id):
        """Pull metadata from PhiloLogic DB based on position of ngrams in file"""
        metadata = {}
        philo_db = DB(os.path.join(self.input_path, "data"), cached=False)
        text_object = philo_db[text_id.split('_')]
        for field in philo_db.locals["metadata_fields"]:
            metadata[field] = str(text_object[field])
        metadata["filename"] = os.path.join(self.input_path, "data/TEXT", metadata["filename"])
        return metadata

    def generate(self, files, output_path, ngram_index=None, db_path=None):
        """Generate n-grams. Takes a list of files as an argument."""
        os.system('rm -rf %s/*' % output_path)
        os.system('mkdir -p %s' % output_path)
        os.system("mkdir %s/metadata" % output_path)
        os.system("mkdir %s/index" % output_path)
        if db_path is None and self.is_philo_db:
            self.input_path = os.path.dirname(os.path.abspath(files[0])).replace("data/words_and_philo_ids", "")
        else:
            self.input_path = db_path
        self.output_path = output_path
        self.metadata = {}
        if ngram_index is None:
            ngram_index = NgramIndex(self.output_path)
        else:
            ngram_index = NgramIndex(self.output_path, prior_db=ngram_index)
        for input_file in files:
            print("Processing document %s..." % input_file)
            with open(input_file) as filehandle:
                ngrams = deque([])
                ngram_obj = deque([])
                current_text_id = None
                for line in filehandle:
                    word_obj = loads(line.strip())
                    word = word_obj["token"]
                    word = self.__normalize(word)
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
                        self.metadata[current_text_id] = self.__get_metadata(current_text_id)
                        self.__build_text_index(ngrams, current_text_id)
                        ngrams = deque([])
                        ngram_obj = deque([])
                        current_ngram = ""
                        current_text_id = text_id
                    ngram_obj.append((word, position, word_obj["start_byte"], word_obj["end_byte"]))
                    if len(ngram_obj) == self.ngram:
                        if self.skipgram:
                            ngram_obj_to_store = [ngram_obj[0], ngram_obj[-1]]
                        else:
                            ngram_obj_to_store = list(ngram_obj)
                        current_ngram_list, philo_ids, start_bytes, end_bytes = zip(*ngram_obj_to_store)
                        current_ngram = " ".join(current_ngram_list)
                        current_ngram_index = ngram_index[current_ngram]
                        ngrams.append((current_ngram_index, start_bytes[0], end_bytes[-1]))
                        ngram_obj.popleft()
                if self.text_object_level == "doc":
                    if self.debug:
                        self.__write_to_disk(ngrams, current_text_id)
                    self.metadata[current_text_id] = self.__get_metadata(current_text_id)
                    self.__build_text_index(ngrams, current_text_id)
        print("Finished processing files...")
        print("Saving metadata...")
        with open("%s/metadata/metadata.json" % self.output_path, "w") as metadata_output:
            dump(self.metadata, metadata_output)
        print("Saving most common ngrams...")
        ngram_index.get_most_frequent(10000)
        print("Saving index...")
        ngram_index.save_db()
        ngram_index_path = os.path.join(self.output_path, "index/index.db")
        return ngram_index_path

def parse_command_line():
    """Command line parsing function"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="configuration file used to override defaults",
                        type=str, default="")
    parser.add_argument("--file_path", help="path to source files",
                        type=str)
    parser.add_argument("--prior_index", help="Use ngram index generated from another set of files for cross dataset comparison",
                        type=str, default="")
    parser.add_argument("--is_philo_db", help="define is files are from a PhiloLogic4 instance",
                        type=literal_eval, default=True)
    parser.add_argument("--text_object_level", help="type of object to split up docs in",
                        type=str, default="doc")
    parser.add_argument("--output_path", help="output path of ngrams",
                        type=str, default="./")
    parser.add_argument("--output_type", help="output format: html, json (see docs for proper decoding), xml, or tab",
                        type=str, default="html")
    parser.add_argument("--debug", help="add debugging", action='store_true', default=False)
    args = vars(parser.parse_args())
    if args["is_philo_db"]:
        args["file_path"] = TRIM_LAST_SLASH.sub("", args["file_path"])
        args["files"] = sorted(glob(os.path.join(args["file_path"], "data/words_and_philo_ids/*")))
    return args


if __name__ == '__main__':
    ARGS = parse_command_line()
    NGRAM_GENERATOR = Ngrams(stopwords="stopwords.txt", text_object_level=ARGS["text_object_level"])
    if ARGS["prior_index"]:
        NGRAM_GENERATOR.generate(ARGS["files"], ARGS["output_path"], ngram_index=ARGS["prior_index"])
    else:
        NGRAM_GENERATOR.generate(ARGS["files"], ARGS["output_path"])
