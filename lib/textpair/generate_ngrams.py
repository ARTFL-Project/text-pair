#!/usr/bin/env python3
"""N-gram generator"""

import configparser
import os
from collections import defaultdict
from glob import glob
from typing import Any, Dict, List, Tuple

import orjson
from mmh3 import hash as hash32
from text_preprocessing import PreProcessor, Tokens
from tqdm import tqdm

# https://github.com/tqdm/tqdm/issues/481
tqdm.monitor_interval = 0
PHILO_TEXT_OBJECT_LEVELS = {"doc": 1, "div1": 2, "div2": 3, "div3": 4, "para": 5, "sent": 6, "word": 7}


class Ngrams:
    """Generate Ngrams"""

    def __init__(
        self,
        text_object_type="doc",
        ngram=3,
        gap=0,
        stemmer=True,
        lemmatizer="",
        stopwords=None,
        numbers=False,
        language="french",
        lowercase=True,
        minimum_word_length=2,
        word_order=True,
        modernize=True,
        ascii=False,
        pos_to_keep=[],
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
            "stopwords": stopwords,  # TODO: generate error if file not found
            "text_object_type": text_object_type,
            "pos_to_keep": pos_to_keep,
            "ascii": ascii,
        }
        self.debug = debug
        self.input_path = ""
        self.output_path = ""
        self.db_name = ""
        self.db_path = ""
        self.is_philo_db = False
        if pos_to_keep:
            self.use_pos = True
        else:
            self.use_pos = False

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
            files = glob(os.path.join(file_path, "*"))
        os.system(f"rm -rf {output_path}/ngrams")
        os.system(f"rm -rf {output_path}/ngrams_in_order")
        os.system(f"mkdir -p {output_path}/ngrams")
        if self.debug:
            os.system(f"mkdir {output_path}/debug")
        os.system(f"mkdir -p {output_path}/metadata")
        os.system(f"mkdir -p {output_path}/index")
        os.system(f"mkdir -p {output_path}/config")
        os.system(f"mkdir -p {output_path}/temp")
        os.system(f"mkdir -p {output_path}/ngrams_in_order")
        self.input_path = os.path.abspath(os.path.join(files[0], "../../../"))
        # self.input_path = os.path.dirname(os.path.abspath(files[0])).replace("data/words_and_philo_ids", "")
        self.output_path = output_path
        combined_metadata: Dict[str, Any] = {}

        print("Generating ngrams...", flush=True)
        preprocessor = PreProcessor(
            language=self.config["language"],
            stemmer=self.config["stemmer"],
            lemmatizer=self.config["lemmatizer"],
            modernize=self.config["modernize"],
            lowercase=self.config["lowercase"],
            strip_numbers=self.config["numbers"],
            stopwords=self.config["stopwords"],
            pos_to_keep=self.config["pos_to_keep"],
            ngrams=self.config["ngram"],
            ngram_gap=self.config["gap"],
            text_object_type=self.config["text_object_type"],
            min_word_length=self.config["minimum_word_length"],
            ascii=self.config["ascii"],
            post_processing_function=self.text_to_ngram,
            is_philo_db=True,
            workers=workers,
            progress=False,
        )
        with tqdm(total=len(files), leave=False) as pbar:
            for local_metadata in preprocessor.process_texts(files, progress=False):
                combined_metadata.update(local_metadata)  # type: ignore
                pbar.update()

        print("Saving ngram index and most common ngrams (this can take a while)...", flush=True)
        os.system(
            rf"""for i in {output_path}/temp/*; do cat $i; done | sort -T {output_path} -S 25% | uniq -c |
            sort -rn -T {output_path} -S 25% | awk '{{print $2"\t"$3}}' | tee {output_path}/index/index.tab |
            awk '{{print $2}}' > {output_path}/index/most_common_ngrams.txt"""
        )

        print("Saving metadata...")
        with open(f"{self.output_path}/metadata/metadata.json", "wb") as metadata_output:
            metadata_output.write(orjson.dumps(combined_metadata))
        self.__dump_config(output_path)

        print("Cleaning up...")
        os.system(f"rm -r {self.output_path}/temp")

    def text_to_ngram(self, text_object: Tokens) -> Dict[str, Any]:
        """Tranform doc to inverted index of ngrams"""
        doc_ngrams: List[str] = []
        metadata: Dict[str, Any] = {}
        # Make sure we only have strings in our metadata:
        for k, v in text_object.metadata.items():
            if not isinstance(v, str):
                text_object.metadata[k] = str(v)
        text_object_id = "_".join(
            text_object.metadata["philo_id"].split()[: PHILO_TEXT_OBJECT_LEVELS[self.config["text_object_type"]]]
        )
        metadata[text_object_id] = text_object.metadata
        text_index = defaultdict(list)
        doc_ngrams_in_order: List[Tuple[int, int]] = []  # for banality filter
        for index_pos, ngram in enumerate(text_object):
            hashed_ngram = hash32(ngram)
            text_index[str(hashed_ngram)].append((index_pos, ngram.ext["start_byte"], ngram.ext["end_byte"]))
            doc_ngrams_in_order.append((ngram.ext["start_byte"], hashed_ngram))
            doc_ngrams.append("\t".join((ngram, str(hashed_ngram))))
        with open(f"{self.output_path}/ngrams/{text_object_id}.json", "wb") as json_file:
            json_file.write(orjson.dumps(dict(text_index)))
        with open(f"{self.output_path}/temp/{text_object_id}", "w", encoding="utf-8") as output:
            output.write("\n".join(sorted(doc_ngrams)))
        with open(f"{self.output_path}/ngrams_in_order/{text_object_id}.json", "wb") as json_file:
            json_file.write(orjson.dumps(doc_ngrams_in_order))
        return metadata
