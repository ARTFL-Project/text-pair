#!/usr/bin/env python3
"""N-gram generator"""

import configparser
import os
import shutil
import sqlite3
from glob import glob
from typing import Any, Dict, List

import orjson
from mmh3 import hash64
from tqdm import tqdm

from textpair.preprocessing import PreProcessor, TextObject

from . import ngram_binary, ngram_index

# https://github.com/tqdm/tqdm/issues/481
tqdm.monitor_interval = 0
PHILO_TEXT_OBJECT_LEVELS = {
    "doc": 1,
    "div1": 2,
    "div2": 3,
    "div3": 4,
    "para": 5,
    "sent": 6,
    "word": 7,
}


class Ngrams:
    """Generate Ngrams"""

    def __init__(
        self,
        text_object_type="doc",
        ngram=3,
        gap=0,
        stemmer=True,
        lemmatizer="",
        stopwords=False,
        numbers=False,
        language="french",
        lowercase=True,
        minimum_word_length=2,
        word_order=True,
        modernize=True,
        ascii=False,
        pos_to_keep=[],
        language_model="",
        debug=False,
        **kwargs,
    ):
        self.config = {
            "ngram": ngram,
            "window": ngram + gap,
            "gap": gap,
            "word_order": word_order,
            "numbers": numbers,
            "stemmer": stemmer,
            "modernize": modernize,
            "language": language,
            "lowercase": lowercase,
            "minimum_word_length": minimum_word_length,
            "lemmatizer": lemmatizer,
            "stopwords": stopwords,
            "text_object_type": text_object_type,
            "pos_to_keep": pos_to_keep,
            "ascii": ascii,
            # spaCy pipeline used for lemmatization and POS tagging. Without it
            # `lemmatizer = spacy` and `pos_to_keep` have no effect, and the GPU
            # is never touched, since there is no model to put on it.
            "language_model": language_model,
        }
        self.debug = debug
        self.input_path = ""
        self.output_path = ""
        self.db_name = ""
        self.db_path = ""
        self.is_philo_db = False

    def __dump_config(self, output_path):
        with open(os.path.join(output_path, "config/ngram_config.ini"), "w", encoding="utf-8") as ini_file:
            ngram_config = configparser.ConfigParser()
            ngram_config.add_section("PREPROCESSING")
            for param, value in self.config.items():
                ngram_config.set("PREPROCESSING", param, repr(value))
            ngram_config.write(ini_file)

    def generate(
        self,
        file_path: str,
        output_path: str,
        workers: int,
    ):
        """Generate n-grams."""
        if os.path.isfile(file_path):
            files = [file_path]
        else:
            # Sorted, not in directory order: documents are spilled into the
            # n-gram index as they come back, so the order they are submitted in
            # is the order their counts are totalled in.
            files = sorted(glob(os.path.join(file_path, "*")))
        # Use shutil/os rather than shelling out: unquoted paths passed to the
        # shell break (dangerously, for rm -rf) on paths containing spaces.
        shutil.rmtree(os.path.join(output_path, "ngrams"), ignore_errors=True)
        shutil.rmtree(os.path.join(output_path, "ngrams_in_order"), ignore_errors=True)
        os.makedirs(os.path.join(output_path, "ngrams"), exist_ok=True)
        if self.debug:
            os.makedirs(os.path.join(output_path, "debug"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "metadata"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "index"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "config"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "ngrams_in_order"), exist_ok=True)
        # temp/ holds the n-gram text, which only index.tab needs, which in turn
        # only the aligner's --debug tracer reads. Writing it means a sort of
        # every document's n-grams; skip it when nothing will read it.
        shutil.rmtree(os.path.join(output_path, "temp"), ignore_errors=True)
        if self.debug:
            os.makedirs(os.path.join(output_path, "temp"), exist_ok=True)
        self.input_path = os.path.abspath(os.path.join(files[0], "../../../"))
        self.output_path = output_path
        combined_metadata: dict[str, Any] = {}

        print("Generating ngrams...", flush=True)
        # word_order and the n-gram size/gap are read straight from self.config;
        # PreprocessConfig accepts both the config-file names and its own.
        preprocessor = PreProcessor(
            workers=workers,
            language=self.config["language"],
            stemmer=self.config["stemmer"],
            lemmatizer=self.config["lemmatizer"],
            modernize=self.config["modernize"],
            lowercase=self.config["lowercase"],
            strip_numbers=self.config["numbers"],
            stopwords=self.config["stopwords"],
            pos_to_keep=self.config["pos_to_keep"],
            language_model=self.config["language_model"],
            ngrams=self.config["ngram"],
            ngram_gap=self.config["gap"],
            ngram_word_order=self.config["word_order"],
            text_object_type=self.config["text_object_type"],
            min_word_length=self.config["minimum_word_length"],
            ascii=self.config["ascii"],
            # The n-gram text is only written for index.tab, which only the
            # aligner's --debug tracer reads; without it the keys are hashed
            # from the normalized forms and no n-gram string is built.
            keep_ngram_text=self.debug,
            post_processing_function=self.text_to_ngram,
        )
        philo_type_count = self.count_texts(files[0])
        # The index buckets each document while the workers are still running,
        # so what is left once they finish is the totalling rather than the
        # whole of it.
        index = ngram_index.IncrementalIndex(output_path)
        ngram_dir = os.path.join(output_path, "ngrams")
        with tqdm(total=philo_type_count, leave=False) as pbar:
            for local_metadata in preprocessor.process_texts(files):
                combined_metadata.update(local_metadata)
                for text_object_id in local_metadata:
                    index.add(os.path.join(ngram_dir, f"{text_object_id}.bin"))
                pbar.update()

        print("Saving ngram index and most common ngrams...", flush=True)
        distinct = index.finish(write_index_tab=self.debug)
        print(f"{distinct:,} distinct ngram keys indexed.", flush=True)

        print("Saving metadata...")
        with open(f"{self.output_path}/metadata/metadata.json", "wb") as metadata_output:
            metadata_output.write(orjson.dumps(combined_metadata))
        self.__dump_config(output_path)

        print("Cleaning up...")
        shutil.rmtree(os.path.join(self.output_path, "temp"), ignore_errors=True)

    def text_to_ngram(self, text_object: TextObject) -> Dict[str, Any]:
        """Transform one text object into its n-gram files. Runs in a worker."""
        metadata: Dict[str, Any] = {}
        # Make sure we only have strings in our metadata:
        for k, v in text_object.metadata.items():
            if not isinstance(v, str):
                text_object.metadata[k] = str(v)
        if "philo_id" not in text_object.metadata:
            print(f"WARNING: skipping text object with no philo_id: {list(text_object.metadata.keys())}", flush=True)
            return {}
        text_object_id = "_".join(
            text_object.metadata["philo_id"].split()[: PHILO_TEXT_OBJECT_LEVELS[self.config["text_object_type"]]]
        )
        metadata[text_object_id] = text_object.metadata
        # Three parallel columns rather than a dict of position lists: the binary
        # writer groups them by hash with one stable sort.
        forms = text_object.forms
        start_bytes = text_object.start_bytes
        # The key is the low 64 bits of MurmurHash3-128, signed. 32 bits used to be
        # enough; at a few hundred million distinct n-grams it is not, and the collision
        # rate is first-order in the population -- see NGRAM_KEY_COLLISIONS.md.
        #
        # `keys` is set when the reader interned its tokens, in which case the
        # n-grams were hashed from those without ever being built as strings.
        if text_object.keys is not None:
            hashes = text_object.keys
        else:
            hashes = [hash64(form)[0] for form in forms]
        ngram_binary.write_positions(
            f"{self.output_path}/ngrams/{text_object_id}.bin",
            hashes,
            start_bytes,
            text_object.end_bytes,
        )
        if self.debug:
            # Sorted here so the corpus index is a merge rather than a sort. The
            # trailing newline is load-bearing: sort -m treats a final incomplete
            # line as a line, but anything concatenating these files would weld it
            # to the next document's first n-gram.
            doc_ngrams = sorted(f"{form}\t{hashed}" for form, hashed in zip(forms, hashes))
            with open(f"{self.output_path}/temp/{text_object_id}", "w", encoding="utf-8") as output:
                output.write("\n".join(doc_ngrams))
                output.write("\n")
        # Binary rather than JSON: 12 bytes an n-gram against 30, and the banality
        # filter mmaps it instead of parsing 1.6MB of text per document.
        ngram_binary.write_order(
            f"{self.output_path}/ngrams_in_order/{text_object_id}.bin",
            hashes,
            start_bytes,
        )
        return metadata

    def count_texts(self, text: str) -> int:
        """Count number of texts in PhiloLogic database"""
        philo_db_path: str = os.path.abspath(os.path.join(text, os.pardir, os.pardir, "toms.db"))
        toms_db = sqlite3.connect(philo_db_path)
        cursor = toms_db.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM toms WHERE philo_type = ?",
            (self.config["text_object_type"],),
        )
        philo_type_count = cursor.fetchone()[0]
        return philo_type_count
