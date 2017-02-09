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

PUNCTUATION = re.compile(r'[,?;.:!]*')
NUMBERS = re.compile(r'\d+')

PHILO_TEXT_OBJECT_LEVELS = {'doc': 1, 'div1': 2, 'div2': 3, 'div3': 4, 'para': 5, 'sent': 6, 'word': 7}


class Ngrams:
    """Generate Ngrams"""

    def __init__(self, ngram=3, skipgram=False, stemmer=True, stopwords=None, numbers=False, language="french",
                 lowercase=True, is_philo_id=True, text_object_level="doc"):
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

    def __getattr__(self, attr):
        if attr == "json_output_files":
            return glob(os.path.join(self.output_path, "ngrams/*json"))
        elif attr == "ngram_index":
            return os.path.join(self.output_path, "ngram_count.json")

    def __get_stopwords(self, path):
        stopwords = set([])
        with open(path) as fh:
            for line in fh:
                stopwords.add(self.__normalize(line.strip()))
        return stopwords

    def __normalize(self, input_str):
        input_str = html.unescape(input_str)
        if self.lowercase:
            input_str = input_str.lower()
        if self.stemmer:
            input_str = self.stemmer.stem(input_str)
        nkfd_form = unicodedata.normalize('NFKD', input_str)
        return u"".join([c for c in nkfd_form if not unicodedata.combining(c)])

    def __filter_word(self, word):
        word = PUNCTUATION.sub("", word)
        word = NUMBERS.sub("", word)
        if len(word) < 2 or word in self.stopwords:
            return True
        else:
            return False

    def __write_to_disk(self, ngrams, text_id):
        with open("%s/ngrams/%s_ngrams.json" % (self.output_path, text_id), "w") as output:
            dump(ngrams, output)
            print("Done with %s %s" % (self.text_object_level, text_id))

    def generate(self, files, output_path):
        """Generate n-grams. Takes a list of files as an argument."""
        ngram_count = defaultdict(int)
        os.system('rm -rf %s' % output_path)
        os.system('mkdir -p %s/ngrams' % output_path)
        self.output_path = output_path
        for input_file in files:
            with open(input_file) as fh:
                ngrams = []
                ngram_obj = []
                current_text_id = None
                for line in fh:
                    word_obj = loads(line.strip())
                    word = word_obj["token"]
                    word = self.__normalize(word)
                    if self.__filter_word(word):
                        continue
                    position = word_obj["position"]
                    if self.text_object_level == 'doc':
                        text_id = position.split()[0]
                    else:
                        text_id = '_'.join(position.split()[:PHILO_TEXT_OBJECT_LEVELS[self.text_object_level]])
                    if current_text_id is None:
                        current_text_id = text_id
                    if current_text_id != text_id:
                        self.__write_to_disk(ngrams, current_text_id)
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
                        ngram_obj = ngram_obj[1:]
                if self.text_object_level == "doc":
                    self.__write_to_disk(ngrams, current_text_id)
        with open("%s/ngram_count.json" % self.output_path, "w") as ngram_count_ouput:
            dump(ngram_count, ngram_count_ouput)


if __name__ == '__main__':
    output_path = sys.argv[1]
    files = sys.argv[2:]
    ngram_generator = Ngrams(stopwords="stopwords.txt")
    ngram_generator.generate_ngrams(files, output_path)
