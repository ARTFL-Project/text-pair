#/usr/bin/env python3
"""N-gram generator"""

import html
import os
import re
import sys
import unicodedata
from collections import defaultdict, namedtuple
from json import dump, loads
import pickle
from glob import glob

from nltk.stem.snowball import SnowballStemmer

PUNCTUATION = re.compile(r'[,?;.:!]*')
NUMBERS = re.compile(r'\d+')

PHILO_TEXT_OBJECT_LEVELS = {'doc': 1, 'div1': 2, 'div2': 3, 'div3': 4, 'para': 5, 'sent': 6, 'word': 7}

IndexedNgram = namedtuple("IndexedNgram", "index, position")
DocObject = namedtuple("DocObject", "doc_id, ngram_object")

class Ngrams:
    """Generate Ngrams"""

    def __init__(self, ngram=3, skipgram=False, stemmer=True, stopwords=None, numbers=False, language="french",
                 lowercase=True, is_philo_id=True, text_object_level="doc", debug=False):
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
        self.is_philo_id = is_philo_id
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
        return u"".join([c for c in nkfd_form if not unicodedata.combining(c)])

    def __write_to_disk(self, ngrams, text_id):
        with open("%s/debug/%s_ngrams.json" % (self.output_path, text_id), "w") as output:
            dump(ngrams, output)

    def __build_text_index(self, ngrams, text_id):
        """Build a file representation used for ngram comparisons"""
        text_index = defaultdict(list)
        index_pos = 0
        for ngram_obj in ngrams:
            ngram, philo_ids = zip(*ngram_obj)
            try:
                text_index[repr(ngram)].append(IndexedNgram(index_pos, philo_ids))
                index_pos += 1
            except KeyError:
                pass
        with open("%s/%s.pickle" % (self.output_path, text_id), "wb") as file_to_pickle:
            pickle.dump(text_index, file_to_pickle, pickle.HIGHEST_PROTOCOL)

    def generate(self, files, output_path):
        """Generate n-grams. Takes a list of files as an argument."""
        ngram_count = defaultdict(int)
        os.system('rm -rf %s/*' % output_path)
        os.system('mkdir -p %s/index' % output_path)
        self.output_path = output_path
        unique_ngrams_per_text = defaultdict(set)
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
                    position = word_obj["position"]
                    if self.text_object_level == 'doc':
                        text_id = position.split()[0]
                    else:
                        text_id = '_'.join(position.split()[:PHILO_TEXT_OBJECT_LEVELS[self.text_object_level]])
                    if current_text_id is None:
                        current_text_id = text_id
                    if current_text_id != text_id:
                        if self.debug:
                            self.__write_to_disk(ngrams, current_text_id)
                        self.__build_text_index(ngrams, current_text_id)
                        ngrams = []
                        ngram_obj = []
                        current_text_id = text_id
                    ngram_obj.append((word, position))
                    if len(ngram_obj) == self.ngram:
                        if self.skipgram:
                            skipgram_obj = [ngram_obj[0], ngram_obj[-1]]
                            ngrams.append(skipgram_obj)
                            ngram = repr([i for i, j in skipgram_obj])
                        else:
                            ngrams.append(ngram_obj[:])
                            ngram = repr([i for i, j in ngram_obj])
                        ngram_count[ngram] += 1
                        unique_ngrams_per_text[current_text_id].add(ngram)
                        ngram_obj = ngram_obj[1:]
                if self.text_object_level == "doc":
                    if self.debug:
                        self.__write_to_disk(ngrams, current_text_id)
                    self.__build_text_index(ngrams, current_text_id)
        with open("%s/index/ngram_index.pickle" % self.output_path, "wb") as unique_ngrams_ouput:
            pickle.dump(unique_ngrams_per_text, unique_ngrams_ouput)


if __name__ == '__main__':
    output_path = sys.argv[1]
    files = sys.argv[2:]
    ngram_generator = Ngrams(stopwords="stopwords.txt")
    ngram_generator.generate(files, output_path)
