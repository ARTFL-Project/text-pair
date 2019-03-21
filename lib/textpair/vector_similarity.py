#! /usr/bin/env python3
"""Passage similarity detection"""

import json
import os
import re
import sys
from collections import deque
from html import unescape as unescape_html
from itertools import chain
from typing import Any, Deque, Dict, Iterable, Iterator, List, Optional, Tuple
from xml.sax.saxutils import unescape as unescape_xml

from dill import load, dump
import numpy as np
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.textcorpus import TextCorpus
from gensim.models import TfidfModel
from gensim.models.phrases import Phraser, Phrases
from gensim.similarities import LevenshteinSimilarityIndex
from gensim.similarities.docsim import SparseMatrixSimilarity
from namedlist import namedlist
from text_preprocessing import PreProcessor, Token, Tokens
from tqdm import tqdm

TAGS = re.compile(r"<[^>]+>")

PASSAGE_GROUP = namedlist(
    "PassageGroup", [("vector", []), ("start_byte", 0), ("end_byte", 0), ("filename", None), ("metadata", {})]
)
MERGED_GROUP = namedlist("MergedGroup", [("source", PASSAGE_GROUP()), ("target", PASSAGE_GROUP()), ("similarity", 0.0)])

PHILO_TEXT_OBJECT_LEVELS = {"doc": 1, "div1": 2, "div2": 3, "div3": 4, "para": 5, "sent": 6, "word": 7}

TEMP_DIR = os.getcwd()


class CorpusLoader(TextCorpus):
    """Subclass of gensim's TextCorpus"""

    # pylint: disable=W0231,W0223
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
        self.texts: Iterable[Tokens] = texts
        self.vectors: List[List[Tuple[int, int]]] = []
        self.min_text_obj_length: int = min_text_obj_length
        self.n_chunk: int = n_chunk
        self.length: int = 0
        self.metadata: list = []
        self.text_object_level_split = text_object_level_split
        self.dictionary: Dictionary = dictionary
        self.phrase_model: Phraser = phrase_model
        self.text_object_definition: str = text_object_definition
        os.system(f"mkdir -p {TEMP_DIR}/tmp_docs")
        self.tmp_dir = os.path.abspath(f"{TEMP_DIR}/tmp_docs/")
        self.load_texts(dictionary)

    def __iter__(self) -> Iterator[List[Tuple[int, int]]]:
        for vector in self.vectors:
            yield vector

    def __getitem__(self, item: int) -> List[Tuple[int, int]]:
        return self.vectors[item]

    def __len__(self) -> int:
        return len(self.vectors)

    def load_texts(self, dictionary: Dictionary):
        """Load all texts and create gensim dictionary"""
        if dictionary is None:
            self.dictionary = Dictionary()
        chunk_group: Deque[Tokens] = deque(maxlen=self.n_chunk)
        min_chunk_length: int = self.n_chunk * self.min_text_obj_length
        current_text_level_id: str = "0"
        full_doc = Tokens([], {})
        current_doc_id = None
        text = None
        for text in self.texts:
            doc_id = text.metadata["philo_id"].split()[0]
            if doc_id != current_doc_id and current_doc_id is not None:
                self.__save_doc(full_doc)
                current_doc_id = doc_id
                full_doc = Tokens([], text.metadata)
            full_doc.extend(text)
            text.purge()
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
                            self.metadata.pop()
                            vector = self.__vector_builder(chunk_group)
                            self.vectors[-1] = vector
                            break
                        except IndexError:
                            pass
                    else:
                        chunk_group.append(chunk)
                    if len(chunk_group) == self.n_chunk:
                        vector = self.__vector_builder(chunk_group)
                        self.vectors.append(vector)
        self.__save_doc(full_doc)
        self.length = len(self.vectors)

    def __save_doc(self, doc: Tokens):
        """Save doc to tmp dir for later retrieval"""
        byte_index = {}
        for pos, token in enumerate(doc):
            byte_index[token.ext["start_byte"]] = pos
            byte_index[token.ext["end_byte"]] = pos
        cached_file_path = os.path.join(self.tmp_dir, str(hash(doc.metadata["parsed_filename"])))
        with open(cached_file_path, "wb") as output:
            dump((doc, byte_index), output)

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


def get_text(start_byte: int, end_byte: int, filename: str, length: int = 300) -> str:
    """Grab all texts"""
    if start_byte < 0:
        start_byte = 0
    length = end_byte - start_byte
    with open(filename, "rb") as text_file:
        text_file.seek(start_byte)
        text: str = text_file.read(length).decode("utf8", "ignore")
    text = TAGS.sub("", text)
    text = unescape_xml(text)
    text = unescape_html(text)
    text = text.strip()
    return text


def vectorize(model: SparseMatrixSimilarity, dictionary: Dictionary, tokens: List[Token]) -> List[Tuple[int, int]]:
    """Vectorize list of tokens"""
    return model[dictionary.doc2bow([w for w in tokens if w], allow_update=False)]


def get_similarity(
    dictionary: Dictionary, source_vector: List[Tuple[int, int]], target_vector: List[Tuple[int, int]]
) -> float:
    """Get similarity score"""
    index = SparseMatrixSimilarity([source_vector], num_features=len(dictionary))
    similarity = index[target_vector][0]
    return similarity


def evaluate_score(start_score: float, new_score: float, min_score: float) -> bool:
    """Evaluate if new score is within 2/3 of start score"""
    if new_score / start_score > 0.66 or new_score >= min_score:
        return True
    return False

def get_docs_with_matches(matches: List[Tuple[namedlist, namedlist, float]]) -> Dict[str, Tokens]:
    """Fetch all documents with matches"""
    docs_with_matches: Dict[str, Tokens] = {}
    for source, target, _ in matches:
        if source.metadata["parsed_filename"] not in docs_with_matches:
            with open(
                os.path.join(f"{TEMP_DIR}/tmp_docs", str(hash(source.metadata["parsed_filename"]))), "rb"
            ) as input_doc:
                docs_with_matches[source.metadata["parsed_filename"]] = load(input_doc)
        if target.metadata["parsed_filename"] not in docs_with_matches:
            with open(
                os.path.join(f"{TEMP_DIR}/tmp_docs", str(hash(target.metadata["parsed_filename"]))), "rb"
            ) as input_doc:
                docs_with_matches[target.metadata["parsed_filename"]] = load(input_doc)
    return docs_with_matches


def get_passage(doc: Tokens, start_byte: int, end_byte: int) -> Tokens:
    """Get passage within Tokens object"""
    text, byte_index = doc
    start_index = byte_index[start_byte]
    end_index = byte_index[end_byte] + 1
    return text[start_index:end_index]


def merge_passages(
    matches: List[namedlist],
    preproc: PreProcessor,
    model: SparseMatrixSimilarity,
    dictionary: Dictionary,
    min_score: float,
    max_iter: int,
) -> List[namedlist]:
    """Merge all passages into bigger passages"""
    # pylint: disable=E1101
    last_count = len(matches)
    print(f"Merging matches: {last_count} matches before iteration 1", end="", flush=True)
    docs_with_matches: Dict[str, Tokens] = get_docs_with_matches(matches)
    for iteration in range(max_iter):
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
        start_score: float = 0.0
        for pos, match in enumerate(matches):
            source: namedlist
            target: namedlist
            source, target, similarity = match
            merged_source: bool = False
            merged_target: bool = False
            if merged_group.source.filename is None:
                start_score = similarity
                merged_group = MERGED_GROUP(source, target, start_score)
                continue
            if source.filename != merged_group.source.filename or target.filename != merged_group.target.filename:
                saved_groups.append(merged_group)
                start_score = similarity
                merged_group = MERGED_GROUP(source, target, start_score)
                continue
            if source.start_byte <= merged_group.source.end_byte:
                if source.end_byte > merged_group.source.end_byte:
                    source_tokens = get_passage(
                        docs_with_matches[source.metadata["parsed_filename"]],
                        merged_group.source.start_byte,
                        source.end_byte,
                    )
                    source_vector = vectorize(model, dictionary, source_tokens)
                    new_score = get_similarity(dictionary, source_vector, merged_group.target.vector)
                    if evaluate_score(start_score, new_score, min_score) is True:
                        merged_group.source.end_byte = source.end_byte
                        merged_group.source.metadata["end_byte"] = source.end_byte
                        merged_group.source.vector = source_vector
                        merged_group.similarity = new_score
                        merged_source = True
            if target.start_byte <= merged_group.target.end_byte:
                if target.end_byte > merged_group.target.end_byte:
                    target_tokens = get_passage(
                        docs_with_matches[target.metadata["parsed_filename"]],
                        merged_group.target.start_byte,
                        target.end_byte,
                    )
                    target_vector = vectorize(model, dictionary, target_tokens)
                    new_score = get_similarity(dictionary, merged_group.source.vector, target_vector)
                    if evaluate_score(start_score, new_score, min_score) is True:
                        merged_group.target.end_byte = target.end_byte
                        merged_group.target.metadata["end_byte"] = target.end_byte
                        merged_group.target.vector = target_vector
                        merged_group.similarity = new_score
                        merged_target = True
            if merged_source is False and merged_target is False:
                saved_groups.append(merged_group)
                start_score = similarity
                merged_group = MERGED_GROUP(source, target, start_score)
            if pos + 1 == total_matches:
                saved_groups.append(merged_group)
        current_count = len(saved_groups)
        if last_count / current_count <= 1.0:  # we stop iterating if there's minimal change between iterations
            print()
            break
        print(f"\rMerging matches: {current_count} matches after iteration {iteration+1}...", end="", flush=True)
        last_count = current_count
        matches = saved_groups
    return matches


def optimize_matches(
    matches: List[Tuple[namedlist, namedlist, float]], model: SparseMatrixSimilarity, dictionary: Dictionary
) -> Tuple[List[Tuple[namedlist, namedlist, float]], Dict[str, Tokens]]:
    """Optimize match by trimming on left and right side of matches until best score is reached"""
    print("Optimizing matches...")
    docs_with_matches: Dict[str, Tokens] = get_docs_with_matches(matches)
    optimized_matches: List[Tuple[namedlist, namedlist, float]] = []
    for source, target, best_score in matches:
        source_tokens = get_passage(
            docs_with_matches[source.metadata["parsed_filename"]], source.start_byte, source.end_byte
        )
        target_tokens = get_passage(
            docs_with_matches[target.metadata["parsed_filename"]], target.start_byte, target.end_byte
        )
        # print(target_tokens)
        while True:
            token = source_tokens.popleft()
            source_vector = vectorize(model, dictionary, source_tokens)
            new_score = get_similarity(dictionary, source_vector, target.vector)
            if new_score >= best_score:
                best_score = new_score
                source.start_byte = source_tokens.metadata["start_byte"]
            else:
                source_tokens.appendleft(token)
                break
        while True:
            token = source_tokens.pop()
            source_vector = vectorize(model, dictionary, source_tokens)
            new_score = get_similarity(dictionary, source_vector, target.vector)
            if new_score >= best_score:
                best_score = new_score
                source.end_byte = source_tokens.metadata["end_byte"]
            else:
                source_tokens.append(token)
                break
        source.vector = vectorize(model, dictionary, source_tokens)
        while True:
            token = target_tokens.popleft()
            target_vector = vectorize(model, dictionary, target_tokens)
            new_score = get_similarity(dictionary, source.vector, target_vector)
            # print(token, best_score, new_score)
            if new_score >= best_score:
                best_score = new_score
                target.start_byte = target_tokens.metadata["start_byte"]
            else:
                target_tokens.appendleft(token)
                break
        while True:
            token = target_tokens.pop()
            target_vector = vectorize(model, dictionary, target_tokens)
            new_score = get_similarity(dictionary, target_vector, source.vector)
            if new_score >= best_score:
                best_score = new_score
                target.end_byte = target_tokens.metadata["end_byte"]
            else:
                target_tokens.append(token)
                break
        target.vector = vectorize(model, dictionary, target_tokens)
        optimized_matches.append((source, target, best_score))
    return optimized_matches, docs_with_matches


def get_tokens(passage: namedlist, preproc: PreProcessor) -> List[Token]:
    """Get tokens while making sure we grab a full sentence for POS tagging"""
    sentence_boundaries: set = {".", "!", "?"}
    text: str = " "
    start_byte: int = passage.start_byte
    end_byte: int = passage.end_byte
    while text[0] not in sentence_boundaries:
        start_byte -= 1
        with open(passage.filename, "rb") as text_file:
            text_file.seek(start_byte)
            text = text_file.read(end_byte-start_byte).decode("utf8", "ignore")
        if start_byte == 0:
            break
    start_byte += 1 # We don't want the sentence boundarie in our string
    while text[-1] not in sentence_boundaries:
        end_byte += 1
        with open(passage.filename, "rb") as text_file:
            text_file.seek(start_byte)
            text = text_file.read(end_byte-start_byte).decode("utf8", "ignore")
    tokens: List[Token] = []
    for token in preproc.process_string(text):
        end_position = start_byte + len(token.surface_form.encode("utf8"))
        if start_byte >= passage.start_byte and end_position <= passage.end_byte:
            tokens.append(token)
        if end_position > passage.end_byte:
            break
        start_byte = end_position
    return tokens


def post_process_passages(source: namedlist, target: namedlist, preproc: PreProcessor) -> Tuple[str, str]:
    """Post process function to highlight matching words in HTML tags"""
    source_tokens = get_tokens(source, preproc)
    target_tokens = get_tokens(target, preproc)
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

    return " ".join(source_passage_with_matches), " ".join(target_passage_with_matches)


def run_vsm(config: Dict[str, Any]):
    """Main function"""
    if config["text_object_definition"] not in ("n_token", "text_object"):
        print("Error: Only valid values for text object definition are 'n_token' and 'text_object'")
        exit()
    if config["text_object_definition"] == "n_token":
        config["text_object_type"] = config["text_object_level_split"]
    if config["max_iter"] == "inf":
        config["max_iter"] = sys.maxsize ** 10
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
    source_texts: Iterator[Tokens] = preproc.process_texts(
        (file.path for file in os.scandir(config["source_path"])), keep_all=True
    )
    source_corpus: CorpusLoader = CorpusLoader(
        source_texts,
        text_object_definition=config["text_object_definition"],
        min_text_obj_length=config["min_text_obj_length"],
        n_chunk=config["n_chunk"],
        text_object_level_split=config["text_object_level_split"],
        phrase_model=phrase_model,
    )
    target_texts: Iterator[Tokens] = preproc.process_texts(
        (file.path for file in os.scandir(config["target_path"])), keep_all=True
    )
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
                            source_corpus[source_pos],
                            source_corpus.metadata[source_pos]["start_byte"],
                            source_corpus.metadata[source_pos]["end_byte"],
                            source_corpus.metadata[source_pos]["filename"],
                            source_corpus.metadata[source_pos],
                        ),
                        PASSAGE_GROUP(
                            target_corpus[target_pos],
                            target_corpus.metadata[target_pos]["start_byte"],
                            target_corpus.metadata[target_pos]["end_byte"],
                            target_corpus.metadata[target_pos]["filename"],
                            target_corpus.metadata[target_pos],
                        ),
                        source_vector_results[target_pos],
                    )
                )
            pbar.update()
    print(f"{count} matches found.")
    matches = merge_passages(
        matches, preproc, model, source_corpus.dictionary, config["min_similarity"], config["max_iter"]
    )
    matches, docs_with_matches = optimize_matches(matches, model, source_corpus.dictionary)
    print("Writing out results...")
    with open("alignments.jsonl", "w") as output:
        for source, target, best_score in matches:
            source_context_before = get_text(source.start_byte - 300, source.start_byte, source.metadata["filename"])
            source_passage = get_text(source.start_byte, source.end_byte, source.metadata["filename"])
            source_context_after = get_text(source.end_byte, source.end_byte + 300, source.metadata["filename"])
            target_context_before = get_text(target.start_byte - 300, target.start_byte, target.metadata["filename"])
            target_passage = get_text(target.start_byte, target.end_byte, target.metadata["filename"])
            target_context_after = get_text(target.end_byte, target.end_byte + 300, target.metadata["filename"])
            source_passage_with_matches, target_passage_with_matches = post_process_passages(source, target, preproc)
            result_object: str = json.dumps(
                {
                    "source_doc_id": source.metadata["philo_id"].split()[0],
                    "source_context_before": source_context_before,
                    "source_passage": source_passage,
                    "source_context_after": source_context_after,
                    "source_passage_with_matches": source_passage_with_matches,
                    "target_doc_id": target.metadata["philo_id"].split()[0],
                    "target_context_before": target_context_before,
                    "target_passage": target_passage,
                    "target_context_after": target_context_after,
                    "target_passage_with_matches": target_passage_with_matches,
                    "similarity": str(best_score),
                    **{f"source_{field}": value for field, value in source.metadata.items()},
                    **{f"target_{field}": value for field, value in target.metadata.items()},
                }
            )
            print(result_object, file=output)
    print(f"Found {len(matches)}...")
    os.system(f"rm -rf {TEMP_DIR}/tmp_docs")


if __name__ == "__main__":
    configuration: Dict[str, Any] = {
        "source_path": "/var/www/html/philologic/contrat_social/data/words_and_philo_ids",
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
        "min_text_obj_length": 3,
        "minimum_word_length": 3,
        "lemmatizer": "/home/clovis/french_lemmas",
        "stopwords": "/shared/PhiloLogic4/extras/FrenchStopwords.txt",
        "workers": 32,
        "pos_to_keep": ["NOUN", "PROPN", "ADJ"],
        "n_chunk": 15,
        "min_similarity": 0.2,
        "similarity_metric": "cosine",
        "max_iter": "inf",
    }
    run_vsm(configuration)
