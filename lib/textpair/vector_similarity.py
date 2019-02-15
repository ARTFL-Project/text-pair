#! /usr/bin/env python3
"""Passage similarity detection"""

import json
import os
import re
from collections import deque
from html import unescape as unescape_html
from itertools import chain
from typing import Dict, List, Optional, Deque, Tuple, Any, Iterator, Iterable
from xml.sax.saxutils import unescape as unescape_xml

import numpy as np
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.textcorpus import TextCorpus
from gensim.models import TfidfModel
from gensim.models.phrases import Phraser, Phrases
from gensim.similarities import LevenshteinSimilarityIndex
from gensim.similarities.docsim import SparseMatrixSimilarity
from namedlist import namedlist
from text_preprocessing import PreProcessor, Tokens, Token
from tqdm import tqdm

TAGS = re.compile(r"<[^>]+>")

PASSAGE_GROUP = namedlist("PassageGroup", [("start_byte", 0), ("end_byte", 0), ("filename", None), ("metadata", {})])
MERGED_GROUP = namedlist("MergedGroup", [("source", PASSAGE_GROUP()), ("target", PASSAGE_GROUP())])

PHILO_TEXT_OBJECT_LEVELS = {"doc": 1, "div1": 2, "div2": 3, "div3": 4, "para": 5, "sent": 6, "word": 7}


class CorpusLoader(TextCorpus):
    """Subclass of gensim's TextCorpus"""

    # pylint: disable=W0231,W0223

    texts: Iterable[Tokens]
    vectors: List[List[Tuple[int, int]]]
    min_text_obj_length: int
    n_chunk: int
    length: int
    metadata: list
    dictionary: Dictionary
    phrase_model: Phraser
    text_object_definition: str

    def __init__(
        self,
        texts: Iterator[Tokens],
        dictionary: Dictionary = None,
        text_object_definition: str = "n_token",
        min_text_obj_length: int = 15,
        n_chunk: int = 3,
        text_object_level_split: str = "doc",
        phrase_model: Phrases = None,
    ):
        """Intialize CorpusLoader"""
        self.texts = texts
        self.vectors = []
        self.min_text_obj_length = min_text_obj_length
        self.n_chunk = n_chunk
        self.length = 0
        self.metadata = []
        self.text_object_level_split = text_object_level_split
        self.dictionary = dictionary
        self.phrase_model = phrase_model
        self.text_object_definition = text_object_definition
        self.load_texts(dictionary)

    def __iter__(self) -> Iterator[List[Tuple[int, int]]]:
        for vector in self.vectors:
            yield vector

    def __len__(self) -> int:
        return len(self.vectors)

    def load_texts(self, dictionary: Dictionary):
        """Load all texts and create gensim dictionary"""
        if dictionary is None:
            self.dictionary = Dictionary()
        chunk_group: Deque[Tokens] = deque(maxlen=self.n_chunk)
        min_chunk_length: int = self.n_chunk * self.min_text_obj_length
        current_text_level_id: str = "0"
        for text in self.texts:
            text_level_id: str = " ".join(
                text.metadata["philo_id"].split()[: PHILO_TEXT_OBJECT_LEVELS[self.text_object_level_split]]
            )
            if text_level_id != current_text_level_id:
                chunk_group_length: int = sum([len(t) for t in chunk_group])
                if chunk_group_length >= min_chunk_length:
                    vector = self.__vector_builder(chunk_group)
                    self.vectors.append(vector)
                chunk_group.clear()
            current_text_level_id = text_level_id
            if self.phrase_model is not None:
                text = Tokens(self.phrase_model[text], text.metadata)
            self.dictionary.add_documents([text])
            if self.text_object_definition == "text_object":
                if len(text) < self.min_text_obj_length:
                    try:
                        chunk_group[-1].extend(text)
                        continue
                    except IndexError:
                        pass
                if text:
                    chunk_group.append(text)
                if len(chunk_group) == self.n_chunk:
                    chunk_group_length = sum([len(t) for t in chunk_group])
                    if chunk_group_length >= min_chunk_length:
                        vector = self.__vector_builder(chunk_group)
                        self.vectors.append(vector)
            else:
                for chunk in text.split_tokens(self.min_text_obj_length):
                    if len(chunk) != self.min_text_obj_length:
                        try:
                            chunk_group[-1].extend(chunk)
                            self.vectors.pop()
                            vector = self.__vector_builder(chunk_group)
                            self.vectors.append(vector)
                            break
                        except IndexError:
                            pass
                    else:
                        chunk_group.append(chunk)
                    if len(chunk_group) == self.n_chunk:
                        vector = self.__vector_builder(chunk_group)
                        self.vectors.append(vector)

        self.length = len(self.vectors)

    def __vector_builder(self, chunk_group) -> List[Tuple[int, int]]:
        """Build vector representation of text chunks"""
        metadata: Dict[str, Any] = chunk_group[0].metadata
        metadata["start_byte"] = chunk_group[0][0].ext["start_byte"]
        metadata["end_byte"] = chunk_group[-1][-1].ext["end_byte"]
        self.metadata.append(metadata)
        chunks: List[Token] = []
        for chunk in chunk_group:
            chunks.extend(chunk.tokens)
        vector: List[Tuple[int, int]] = self.dictionary.doc2bow(chunks, allow_update=False)
        return vector

    def getstream(self) -> Iterator[List[Tuple[int, int]]]:
        """Yield vector when interating"""
        for vector in self.vectors:
            yield vector

    def update_with_tfidf(self, model) -> None:
        """Update vectors with TF-IDF score"""
        for pos, vector in enumerate(self.vectors):
            self.vectors[pos] = model[vector]


def phrase_detection(source_path, preproc, target_path=None, scoring="npmi") -> Phraser:
    """Detect phrases in texts"""
    if target_path is None:
        sentences = preproc.process_texts((i.path for i in os.scandir(source_path)))
    else:
        sentences = preproc.process_texts(
            chain((i.path for i in os.scandir(source_path)), (i.path for i in os.scandir(target_path)))
        )
    sentences = [sentence for sentence in sentences if sentence is not None]
    phrase_model = Phraser(Phrases(sentences, scoring=scoring, threshold=0.5))
    return phrase_model


def get_text(metadata, length=300) -> Tuple[str, str, str]:
    """Grab all texts"""
    start_byte: int
    end_byte: int
    filename: str
    start_byte, end_byte, filename = metadata["start_byte"], metadata["end_byte"], metadata["filename"]
    passages: List[str] = []
    byte_ranges: List[Tuple[int, int]] = [
        (start_byte - 300, start_byte),
        (start_byte, end_byte),
        (end_byte, end_byte + 300),
    ]
    for start, end in byte_ranges:
        if start < 0:
            start = 0
        length = end - start
        with open(filename, "rb") as text_file:
            text_file.seek(start)
            text: str = text_file.read(length).decode("utf8", "ignore")
        text = TAGS.sub("", text)
        text = unescape_xml(text)
        text = unescape_html(text)
        text = text.strip()
        passages.append(text)
    return passages[0], passages[1], passages[2]


def post_process_passages(
    source_passage: str,
    target_passage: str,
    preproc: PreProcessor,
    dictionary: Dictionary,
    model: SparseMatrixSimilarity,
) -> Tuple[str, str, float]:
    """Post process function to highlight matching words in HTML tags"""
    source_tokens = preproc.process_string(source_passage)
    target_tokens = preproc.process_string(target_passage)
    source_map = {token.surface_form: token.text for token in source_tokens}
    target_map = {token.surface_form: token.text for token in target_tokens}
    source_target_intersect = set(
        filter(lambda w: len(w) >= preproc.min_word_length, source_map.values())
    ).intersection(target_map.values())
    source_passage_with_matches = []
    for token in source_tokens:
        if source_map[token.surface_form] and source_map[token.surface_form] in source_target_intersect:
            source_passage_with_matches.append(f'<span class="token-match">{token.surface_form}</span>')
        elif source_map[token.surface_form] == "":
            source_passage_with_matches.append(f'<span class="filtered-token">{token.surface_form}</span>')
        else:
            source_passage_with_matches.append(token.surface_form)
    target_passage_with_matches = []
    for token in target_tokens:
        if target_map[token.surface_form] and target_map[token.surface_form] in source_target_intersect:
            target_passage_with_matches.append(f'<span class="token-match">{token.surface_form}</span>')
        elif target_map[token.surface_form] == "":
            target_passage_with_matches.append(f'<span class="filtered-token">{token.surface_form}</span>')
        else:
            target_passage_with_matches.append(token.surface_form)

    # Get similarity of merged passage
    source_vector = model[dictionary.doc2bow([w for w in source_tokens if w], allow_update=False)]
    target_vector = model[dictionary.doc2bow([w for w in target_tokens if w], allow_update=False)]
    index = SparseMatrixSimilarity([source_vector], num_features=len(dictionary), num_docs=1)
    similarity = index[target_vector][0]

    return "".join(source_passage_with_matches), "".join(target_passage_with_matches), similarity


def merge_passages(matches: List[namedlist]):
    """Merge all passages into bigger passages"""
    # pylint: disable=E1101
    matches.sort(
        key=lambda x: (
            x[0].filename,
            x[1].filename,
            x[0].start_byte,
            x[0].start_byte - x[0].end_byte,
            x[1].start_byte,
            x[1].start_byte - x[1].end_byte,
        )
    )  # sort by smaller start byte and bigger end_byte
    merged_group: namedlist = MERGED_GROUP()
    saved_groups: List[namedlist] = []
    total_matches: int = len(matches)
    for pos, match in enumerate(matches):
        source: namedlist
        target: namedlist
        source, target = match
        if merged_group.source.filename is None:
            merged_group = MERGED_GROUP(source, target)
            continue
        if source.filename != merged_group.source.filename or target.filename != merged_group.target.filename:
            saved_groups.append(merged_group)
            merged_group = MERGED_GROUP(source, target)
            continue
        if source.start_byte <= merged_group.source.end_byte and target.start_byte <= merged_group.target.end_byte:
            if source.end_byte > merged_group.source.end_byte and target.end_byte > merged_group.target.end_byte:
                merged_group.source.end_byte = source.end_byte
                merged_group.source.metadata["end_byte"] = source.end_byte
                merged_group.target.end_byte = target.end_byte
                merged_group.target.metadata["end_byte"] = target.end_byte
        else:
            saved_groups.append(merged_group)
            merged_group = MERGED_GROUP(source, target)
        if pos + 1 == total_matches:
            saved_groups.append(merged_group)
    return saved_groups


def run_vsm(config: Dict[str, Any]):
    """Main function"""

    if config["text_object_definition"] == "n_token":
        config["text_object_type"] = config["text_object_level_split"]
    preproc: PreProcessor = PreProcessor(
        language=config["language"],
        stemmer=config["stemmer"],
        lemmatizer=config["lemmatizer"],
        modernize=config["modernize"],
        lowercase=config["lowercase"],
        strip_numbers=config["numbers"],
        stopwords=config["stopwords"],
        pos_to_keep=config["pos_to_keep"],
        ngrams=config["ngram"],
        ngram_gap=config["gap"],
        text_object_type=config["text_object_type"],
        min_word_length=config["minimum_word_length"],
        ascii=config["ascii"],
        is_philo_db=True,
        workers=config["workers"],
    )
    # phrase_model = phrase_detection(config["source_path"], preproc, config["target_path"])
    phrase_model: Optional[Phraser] = None
    source_texts: Iterator[Tokens] = preproc.process_texts((file.path for file in os.scandir(config["source_path"])))
    source_corpus: CorpusLoader = CorpusLoader(
        source_texts,
        text_object_definition=config["text_object_definition"],
        min_text_obj_length=config["min_text_obj_length"],
        n_chunk=config["n_chunk"],
        text_object_level_split=config["text_object_level_split"],
        phrase_model=phrase_model,
    )
    target_texts: Iterator[Tokens] = preproc.process_texts((file.path for file in os.scandir(config["target_path"])))
    target_corpus: CorpusLoader = CorpusLoader(
        target_texts,
        text_object_definition=config["text_object_definition"],
        dictionary=source_corpus.dictionary,
        min_text_obj_length=config["min_text_obj_length"],
        n_chunk=config["n_chunk"],
        text_object_level_split=config["text_object_level_split"],
        phrase_model=phrase_model,
    )
    source_corpus.dictionary = target_corpus.dictionary
    model: TfidfModel = TfidfModel(chain(source_corpus.vectors, target_corpus.vectors), smartirs="atc")
    source_corpus.update_with_tfidf(model)
    target_corpus.update_with_tfidf(model)
    if config["similarity_metric"] == "cosine":
        index: SparseMatrixSimilarity = SparseMatrixSimilarity(
            source_corpus, num_features=len(source_corpus.dictionary), num_docs=len(source_corpus)
        )
    elif config["similarity_metric"] == "soft_cosine":
        _ = LevenshteinSimilarityIndex(source_corpus.dictionary)
    count: int = 0
    matches: List[namedlist] = []
    with tqdm(total=source_corpus.length, leave=False) as pbar:
        results: np.array = index[target_corpus]
        for source_pos, source_vector_results in enumerate(results.T):
            filtered_results: np.array = np.where(source_vector_results > config["min_similarity"])[0]
            count += len(filtered_results)
            for target_pos in filtered_results:
                matches.append(
                    (
                        PASSAGE_GROUP(
                            source_corpus.metadata[source_pos]["start_byte"],
                            source_corpus.metadata[source_pos]["end_byte"],
                            source_corpus.metadata[source_pos]["filename"],
                            source_corpus.metadata[source_pos],
                        ),
                        PASSAGE_GROUP(
                            target_corpus.metadata[target_pos]["start_byte"],
                            target_corpus.metadata[target_pos]["end_byte"],
                            target_corpus.metadata[target_pos]["filename"],
                            target_corpus.metadata[target_pos],
                        ),
                    )
                )
            pbar.update()
    print(f"{count} found...")
    print("Merging matches...")
    # passage_groups = merge_passages(matches)
    with open("alignments.jsonl", "w") as output:
        for source, target in matches:
            source_context_before, source_passage, source_context_after = get_text(source.metadata)
            target_context_before, target_passage, target_context_after = get_text(target.metadata)
            source_passage_with_matches, target_passage_with_matches, similarity = post_process_passages(
                source_passage, target_passage, preproc, source_corpus.dictionary, model
            )
            result_object: str = json.dumps(
                {
                    "source_context_before": source_context_before,
                    "source_passage": source_passage,
                    "source_context_after": source_context_after,
                    "source_passage_with_matches": source_passage_with_matches,
                    "target_context_before": target_context_before,
                    "target_passage": target_passage,
                    "target_context_after": target_context_after,
                    "target_passage_with_matches": target_passage_with_matches,
                    "similarity": float(similarity),
                    **{f"source_{field}": value for field, value in source.metadata.items()},
                    **{f"target_{field}": value for field, value in target.metadata.items()},
                }
            )
            print(result_object, file=output)
    # print(f"Found {len(passage_groups)}...")


if __name__ == "__main__":
    config: Dict[str, Any] = {
        "source_path": "/var/www/html/philologic/rousseau_politics/data/words_and_philo_ids",
        "target_path": "/var/www/html/philologic/robesAPext2/data/words_and_philo_ids",
        "language": "french",
        "text_object_type": "sent",
        "is_philo_db": True,
        "modernize": True,
        "ascii": True,
        "stemmer": True,
        "lowercase": True,
        "numbers": True,
        "ngram": None,
        "gap": 0,
        "text_object_level_split": "div1",
        "text_object_definition": "n_token",
        "min_text_obj_length": 20,
        "minimum_word_length": 3,
        "lemmatizer": "/home/clovis/french_lemmas",
        "stopwords": "/shared/PhiloLogic4/extras/FrenchStopwords.txt",
        "workers": 32,
        "pos_to_keep": ["NOUN", "PROPN", "ADJ"],
        "n_chunk": 2,
        "min_similarity": 0.2,
        "similarity_metric": "cosine",
    }
    run_vsm(config)
