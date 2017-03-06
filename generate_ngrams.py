#/usr/bin/env python3
"""N-gram generator"""

import html
import os
import re
import sys
import unicodedata
from collections import defaultdict
from json import dump, loads
from glob import glob

from nltk.stem.snowball import SnowballStemmer
from philologic.DB import DB

PUNCTUATION = re.compile(r'[,?;.:!]*')
NUMBERS = re.compile(r'\d+')

PHILO_TEXT_OBJECT_LEVELS = {'doc': 1, 'div1': 2, 'div2': 3, 'div3': 4, 'para': 5, 'sent': 6, 'word': 7}

class Ngrams:
    """Generate Ngrams"""

    def __init__(self, ngram=3, skipgram=False, stemmer=True, stopwords=None, numbers=False, language="french",
                 lowercase=True, is_philo_db=True, text_object_level="doc", debug=False):
        self.ngram = ngram
        self.skipgram = skipgram
        self.numbers = numbers
        if stemmer:
            try:
                self.stemmer = SnowballStemmer(language)
            except Exception:
                self.stemmer = False
        else:
            self.stemmer = False
        if stopwords is not None and os.path.isfile(stopwords):
            self.stopwords = self.__get_stopwords(stopwords)
        else:
            self.stopwords = []
        self.lowercase = lowercase
        self.is_philo_db = is_philo_db
        self.text_object_level = text_object_level
        self.debug = debug

    def __getattr__(self, attr):
        if attr == "output_files":
            return glob(os.path.join(self.output_path, "*pickle"))
        elif attr == "ngram_index":
            return os.path.join(self.output_path, "index/ngram_index.pickle")

    def __get_stopwords(self, path):
        stopwords = set([])
        with open(path) as stopword_file:
            for line in stopword_file:
                stopwords.add(self.__normalize(line.strip()))
        return stopwords

    def __normalize(self, input_str):
        word = PUNCTUATION.sub("", input_str)
        word = NUMBERS.sub("", word)
        input_str = html.unescape(input_str)
        if self.lowercase:
            input_str = input_str.lower()
        if self.stemmer:
            input_str = self.stemmer.stem(input_str)
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

    def __get_column_names(self, path):
        philo_db = DB(os.path.join(self.input_path, "data"), cached=False)
        cursor = philo_db.dbh.cursor()
        cursor.execute('select %s from toms limit 1' % ",".join(philo_db.locals["metadata_fields"]))
        return list(map(lambda x: x[0], cursor.description))

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
        if ngram_index == None:
            ngram_index = {}
            ngram_index_count = 0
        else:
            ngram_index_count = max(ngram_index.values()) + 1
        for input_file in files:
            print("Processing document %s..." % input_file)
            with open(input_file) as fh:
                ngrams = []
                ngram_obj = []
                current_text_id = None
                for line in fh:
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
                        self.metadata[current_text_id] = self.__get_metadata(current_text_id)
                        self.__build_text_index(ngrams, current_text_id)
                        ngrams = []
                        ngram_obj = []
                        current_ngram = []
                        current_text_id = text_id
                    ngram_obj.append((word, position, word_obj["start_byte"], word_obj["end_byte"]))
                    if len(ngram_obj) == self.ngram:
                        if self.skipgram:
                            ngram_obj_to_store = [ngram_obj[0], ngram_obj[-1]]
                        else:
                            ngram_obj_to_store = ngram_obj
                        current_ngram, philo_ids, start_bytes, end_bytes = zip(*ngram_obj_to_store)
                        current_ngram = " ".join(current_ngram)
                        if current_ngram not in ngram_index:
                            ngram_index[current_ngram] = ngram_index_count
                            current_ngram_index = ngram_index_count
                            ngram_index_count += 1
                        else:
                            current_ngram_index = ngram_index[current_ngram]
                        ngrams.append((current_ngram_index, start_bytes[0], end_bytes[-1]))
                        ngram_obj = ngram_obj[1:]
                if self.text_object_level == "doc":
                    if self.debug:
                        self.__write_to_disk(ngrams, current_text_id)
                    self.metadata[current_text_id] = self.__get_metadata(current_text_id)
                    self.__build_text_index(ngrams, current_text_id)
        with open("%s/metadata/metadata.json" % self.output_path, "w") as metadata_output:
            dump(self.metadata, metadata_output)
        with open("%s/index/ngram_index.json" % self.output_path, "w") as ngram_index_output:
            dump(ngram_index, ngram_index_output)
        return ngram_index


if __name__ == '__main__':
    OUTPUT_PATH = sys.argv[1]
    FILES = sys.argv[2:]
    NGRAM_GENERATOR = Ngrams(stopwords="stopwords.txt")
    NGRAM_INDEX = NGRAM_GENERATOR.generate(FILES, OUTPUT_PATH)
