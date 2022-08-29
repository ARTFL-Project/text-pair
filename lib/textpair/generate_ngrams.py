#!/usr/bin/env python3
"""N-gram generator"""

import configparser
import json
import os
from collections import defaultdict
from glob import glob
from math import floor
from typing import List, Dict, Optional, Any

from text_preprocessing import PreProcessor
from text_preprocessing import Tokens
from tqdm import tqdm

from mmh3 import hash as hash32

# https://github.com/tqdm/tqdm/issues/481
tqdm.monitor_interval = 0
PHILO_TEXT_OBJECT_LEVELS = {"doc": 1, "div1": 2, "div2": 3, "div3": 4, "para": 5, "sent": 6, "word": 7}


class Ngrams:
    """Generate Ngrams"""

    def __init__(
        self,
        text_object_level="doc",
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
            "text_object_level": text_object_level,
            "pos_to_keep": pos_to_keep,
            "ascii": ascii,
        }
        self.debug = debug
        self.input_path = ""
        self.output_path = ""
        self.metadata_done = False
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
        metadata: str,
        is_philo_db: bool,
        workers: int,
        db_path: Optional[str] = None,
        ram: str = "50%",
    ):
        """Generate n-grams."""
        if os.path.isfile(file_path):
            files = [file_path]
        else:
            files = glob(os.path.join(file_path, "*"))
        os.system(f"rm -rf {output_path}/ngrams")
        os.system(f"mkdir -p {output_path}/ngrams")
        if self.debug:
            os.system(f"mkdir {output_path}/debug")
        os.system(f"mkdir -p {output_path}/metadata")
        os.system(f"mkdir -p {output_path}/index")
        os.system(f"mkdir -p {output_path}/config")
        os.system(f"mkdir -p {output_path}/temp")
        if db_path is None and is_philo_db is True:
            self.input_path = os.path.dirname(os.path.abspath(files[0])).replace("data/words_and_philo_ids", "")
        else:
            self.input_path = db_path
            self.db_path = db_path
        self.output_path = output_path
        combined_metadata: Dict[str, Any] = {}
        if is_philo_db:
            combined_metadata = {}
        elif os.path.isfile(metadata):
            self.metadata_done = True
            combined_metadata = metadata
        else:
            print("No metadata provided: exiting...")
            exit()

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
            text_object_type=self.config["text_object_level"],
            min_word_length=self.config["minimum_word_length"],
            ascii=self.config["ascii"],
            post_processing_function=self.text_to_ngram,
            is_philo_db=True,
            workers=workers,
            progress=False,
        )
        with tqdm(total=len(files), leave=False) as pbar:
            for local_metadata in preprocessor.process_texts(files, progress=False):
                if self.metadata_done is False:
                    combined_metadata.update(local_metadata)  # type: ignore
                pbar.update()

        mem_usage = floor(int(ram.replace("%", "")) / 2)
        if mem_usage >= 50:
            mem_usage = 45
        print("Saving ngram index and most common ngrams (this can take a while)...", flush=True)
        os.system(
            rf"""for i in {output_path}/temp/*; do cat $i; done | sort -T {output_path} -S {mem_usage}% | uniq -c |
            sort -rn -T {output_path} -S {mem_usage}% | awk '{{print $2"\t"$3}}' | tee {output_path}/index/index.tab |
            awk '{{print $2}}' > {output_path}/index/most_common_ngrams.txt"""
        )

        print("Saving metadata...")

        if self.metadata_done is False:
            with open(f"{self.output_path}/metadata/metadata.json", "w", encoding="utf-8") as metadata_output:
                json.dump(combined_metadata, metadata_output)
        else:
            os.system(f"cp {metadata} {self.output_path}/metadata/metadata.json 2>/dev/null")

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
        if self.metadata_done is False:
            text_object_id = "_".join(
                text_object.metadata["philo_id"].split()[: PHILO_TEXT_OBJECT_LEVELS[self.config["text_object_level"]]]
            )
            metadata[text_object_id] = text_object.metadata
        else:
            text_object_id = text_object.metadata["filename"]
        text_index = defaultdict(list)
        for index_pos, ngram in enumerate(text_object):
            hashed_ngram = hash32(ngram)
            text_index[hashed_ngram].append((index_pos, ngram.ext["start_byte"], ngram.ext["end_byte"]))
            doc_ngrams.append("\t".join((ngram, str(hashed_ngram))))
        with open(f"{self.output_path}/ngrams/{text_object_id}.json", "w", encoding="utf-8") as json_file:
            json.dump(dict(text_index), json_file)
        with open(f"{self.output_path}/temp/{text_object_id}", "w", encoding="utf-8") as output:
            output.write("\n".join(sorted(doc_ngrams)))
        return metadata
