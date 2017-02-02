#/usr/bin/env python3
"""N-gram generator"""

import html
import os
import re
import sys
import unicodedata
from collections import defaultdict
from json import dump, loads

from nltk.stem.snowball import SnowballStemmer

PUNCTUATION = re.compile(r'[,?;.:!]*')
NUMBERS = re.compile(r'\d+')


class Ngrams:
    """Generate Ngrams"""

    def __init__(self, ngram=3, skipgram=False, stem=True, stopwords=None, NUMBERS=False, language="french"):
        if skipgram is False:
            self.skipgram = False
            self.ngram = ngram
        else:
            self.skipgram = True  # Skipgram are trigrams with middle token skipped.
            self.ngram = 3
        if stem:
            self.stemmer = SnowballStemmer(language)
        else:
            self.stemmer = False
        if stopwords is not None and os.path.isfile(stopwords):
            self.stopwords = self.__get_stopwords(stopwords)
        else:
            self.stopwords = []

    def generate_ngrams(self, files, output_path):
        """Generate n-grams. Takes a list of files as an argument."""
        ngram_count = defaultdict(int)
        os.system('rm -rf %s' % output_path)
        os.system('mkdir -p %s/ngrams' % output_path)
        for input_file in files:
            with open(input_file) as fh:
                ngrams = []
                ngram_obj = []
                for line in fh:
                    word_obj = loads(line.strip())
                    word = word_obj["token"]
                    philo_id = word_obj["position"]
                    word = self.__normalize(word)
                    if self.__filter_word(word):
                        continue
                    ngram_obj.append((word, philo_id))
                    if len(ngram_obj) == self.ngram:
                        if self.skipgram:
                            skipgram_obj = ngram_obj[0] + ngram_obj[2]
                            ngrams.append(skipgram_obj)
                            ngram = repr([i for i, j in skipgram_obj])
                        else:
                            ngrams.append(ngram_obj[:])
                            ngram = repr([i for i, j in ngram_obj])
                        ngram_count[ngram] += 1
                        ngram_obj = ngram_obj[1:]
            with open("%s/ngrams/%s_ngrams.json" % (output_path, os.path.basename(input_file)), "w") as output:
                dump(ngrams, output)
            print("Done with %s" % input_file)
        with open("%s/ngram_count.json" % output_path, "w") as ngram_count_ouput:
            dump(ngram_count, ngram_count_ouput)

    def __get_stopwords(self, path):
        stopwords = set([])
        with open(path) as fh:
            for line in fh:
                stopwords.add(self.__normalize(line.strip()))
        return stopwords

    def __normalize(self, input_str):
        input_str = html.unescape(input_str)
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


if __name__ == '__main__':
    output_path = sys.argv[1]
    files = sys.argv[2:]
    ngram_generator = Ngrams(stopwords="stopwords.txt")
    ngram_generator.generate_ngrams(files, output_path)
