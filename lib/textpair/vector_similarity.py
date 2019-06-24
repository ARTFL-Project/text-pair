#! /usr/bin/env python3
"""Passage similarity detection"""

import json
import os
import re
import sys
from collections import deque
from html import unescape as unescape_html
from itertools import chain
from typing import Any, Deque, Dict, Iterable, Iterator, List, Optional, Tuple, Set, Union
from xml.sax.saxutils import unescape as unescape_xml

from dill import load, dump
import numpy as np
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.textcorpus import TextCorpus
from gensim.models import TfidfModel
from gensim.models.phrases import Phraser, Phrases
from gensim.similarities import LevenshteinSimilarityIndex
from gensim.similarities.docsim import SparseMatrixSimilarity, MatrixSimilarity
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
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


def fast_cosine(X, Y):
    return np.inner(X, Y) / np.sqrt(np.dot(X, X) * np.dot(Y, Y))


class CorpusLoader(TextCorpus):
    """Base class for linking gensim's TextCorpus and the text-preprocessing output"""

    def __init__(
        self,
        texts: Iterator[Tokens],
        text_object_definition: str = "n_token",
        min_text_obj_length: int = 15,
        n_chunk: int = 3,
        text_object_level_split: str = "doc",
        phrase_model: Phrases = None,
    ):
        """Intialize CorpusVectorizer"""
        self.texts: Iterable[Tokens] = texts
        self.min_text_obj_length: int = min_text_obj_length
        self.n_chunk: int = n_chunk
        self.length: int = 0
        self.metadata: list = []
        self.text_object_level_split = text_object_level_split
        self.phrase_model: Phraser = phrase_model
        self.text_object_definition: str = text_object_definition
        os.system(f"mkdir -p {TEMP_DIR}/tmp_docs")
        self.tmp_dir = os.path.abspath(f"{TEMP_DIR}/tmp_docs/")

    def __getitem__(self, item: int) -> List[Tuple[int, int]]:
        return self.vectors[item]

    def __len__(self) -> int:
        return self.length

    def get_text_chunks(self) -> Iterator[List[str]]:
        """Load all texts and create gensim dictionary"""
        chunk_group: Deque[Tokens] = deque(maxlen=self.n_chunk)
        min_chunk_length: int = self.n_chunk * self.min_text_obj_length
        current_text_level_id: str = "0"
        full_doc = Tokens([], {})
        current_doc_id = None
        text = None
        done = 0
        for text in self.texts:
            print(f"\rProcessing texts... {done} done...", end="", flush=True)
            done += 1
            doc_id = text.metadata["philo_id"].split()[0]
            if doc_id != current_doc_id and current_doc_id is not None:
                self.__save_doc(full_doc)
                full_doc = Tokens([], text.metadata)
            full_doc.extend(text)
            text.purge()
            text_level_id: str = " ".join(
                text.metadata["philo_id"].split()[: PHILO_TEXT_OBJECT_LEVELS[self.text_object_level_split]]
            )
            if text_level_id != current_text_level_id:
                chunk_group_length: int = sum([len(t) for t in chunk_group])
                if chunk_group_length >= min_chunk_length:
                    self.__store_metadata(chunk_group)
                    self.length += 1
                    yield [t.text for chunk in chunk_group for t in chunk]
                chunk_group.clear()
            current_text_level_id = text_level_id
            if self.phrase_model is not None:
                text = Tokens(self.phrase_model[text], text.metadata)
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
                        self.__store_metadata(chunk_group)
                        self.length += 1
                        yield [t.text for chunk in chunk_group for t in chunk]
            else:
                chunks_to_return: List[List[Tokens]] = []
                for chunk in text.split_tokens(self.min_text_obj_length):
                    if not chunk:
                        continue
                    if len(chunk) != self.min_text_obj_length:  # We've reached the end of our text object
                        try:
                            chunk_group[-1].extend(chunk)
                            self.metadata.pop()
                            self.__store_metadata(chunk_group)
                            chunks_to_return[-1] = [t.text for chunk in chunk_group for t in chunk]
                            break
                        except IndexError:
                            pass
                    else:
                        chunk_group.append(chunk)
                    if len(chunk_group) == self.n_chunk:
                        self.__store_metadata(chunk_group)
                        self.length += 1
                        chunks_to_return.append([t.text for chunk in chunk_group for t in chunk])
                for chunk in chunks_to_return:
                    yield chunk
            current_doc_id = doc_id
        self.__save_doc(full_doc)
        del self.texts
        print()

    def __save_doc(self, doc: Tokens):
        """Save doc to tmp dir for later retrieval"""
        start_bytes = {}
        end_bytes = {}
        for pos, token in enumerate(doc):
            start_bytes[token.ext["start_byte"]] = pos
            end_bytes[token.ext["end_byte"]] = pos
        cached_file_path = os.path.join(self.tmp_dir, str(hash(doc.metadata["parsed_filename"])))
        with open(cached_file_path, "wb") as output:
            dump((doc, start_bytes, end_bytes), output)

    def __store_metadata(self, chunk_group: Iterable[Tokens]):
        """Store Metadata for each chunk"""
        metadata: Dict[str, Any] = chunk_group[0].metadata
        metadata["start_byte"] = chunk_group[0][0].ext["start_byte"]
        metadata["end_byte"] = chunk_group[-1][-1].ext["end_byte"]
        self.metadata.append(metadata)

class CorpusVectorizer(CorpusLoader):
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
        """Intialize CorpusVectorizer"""
        super().__init__(
            texts,
            text_object_definition=text_object_definition,
            min_text_obj_length=min_text_obj_length,
            n_chunk=n_chunk,
            text_object_level_split=text_object_level_split,
            phrase_model=phrase_model,
        )
        if dictionary is None:
            self.dictionary = Dictionary()
        else:
            self.dictionary = dictionary
        os.system(f"mkdir -p {TEMP_DIR}/tmp_docs")
        self.tmp_dir = os.path.abspath(f"{TEMP_DIR}/tmp_docs/")
        self.vectors: List[List[Tuple[int, int]]] = [
            self.dictionary.doc2bow(text_chunk, allow_update=True) for text_chunk in self.get_text_chunks()
        ]

    def __iter__(self) -> Iterator[List[Tuple[int, int]]]:
        for vector in self.vectors:
            yield vector

    def __getitem__(self, item: int) -> List[Tuple[int, int]]:
        return self.vectors[item]

    def getstream(self) -> Iterator[List[Tuple[int, int]]]:
        """Yield vector when interating"""
        for vector in self.vectors:
            yield vector

    def update_with_tfidf(self, model) -> None:
        """Update vectors with TF-IDF score"""
        for pos, vector in enumerate(self.vectors):
            self.vectors[pos] = model[vector]


class Doc2VecCorpus(CorpusLoader):
    def __init__(
        self,
        texts: Iterator[Tokens] = None,
        text_object_definition: str = "n_token",
        min_text_obj_length: int = 15,
        n_chunk: int = 3,
        text_object_level_split: str = "doc",
        phrase_model: Phrases = None,
    ):
        """Intialize CorpusVectorizer"""
        super().__init__(
            texts,
            text_object_definition=text_object_definition,
            min_text_obj_length=min_text_obj_length,
            n_chunk=n_chunk,
            text_object_level_split=text_object_level_split,
            phrase_model=phrase_model,
        )
        self.text_chunks = list(self.get_text_chunks())
        self.vectors: List[Any] = []

    def build_vectors(self, model):
        for chunk in self.text_chunks:
            vector = model.infer_vector(chunk)
            self.vectors.append(list(enumerate(vector)))

    def __iter__(self) -> Iterator[List[Tuple[int, int]]]:
        for vector in self.vectors:
            yield vector

    def __getitem__(self, item: int) -> List[Tuple[int, int]]:
        return self.vectors[item]

    def getstream(self) -> Iterator[List[Tuple[int, int]]]:
        """Yield vector when interating"""
        for vector in self.vectors:
            yield vector


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


def clean_text(text: str) -> str:
    """Cleaning text function which removes tags and converts entities"""
    text = TAGS.sub("", text)
    text = unescape_xml(text)
    text = unescape_html(text)
    text = text.strip()
    return text


def get_text(start_byte: int, end_byte: int, filename: str, length: int = 300) -> str:
    """Grab all texts"""
    if start_byte < 0:
        start_byte = 0
    length = end_byte - start_byte
    with open(filename, "rb") as text_file:
        text_file.seek(start_byte)
        text: str = text_file.read(length).decode("utf8", "ignore")
    return clean_text(text)


def vectorize(
    model: Optional[Union[TfidfModel, Doc2Vec]], tokens: List[Token], dictionary: Optional[Dictionary] = None
) -> List[Tuple[int, int]]:
    """Vectorize list of tokens"""
    if isinstance(model, TfidfModel):
        try:
            return model[dictionary.doc2bow([w for w in tokens if w], allow_update=False)]
        except ValueError:
            return None
    else:
        return list(enumerate(model.infer_vector(tokens)))


def get_similarity(
    num_features: int, source_vector: List[Tuple[int, int]], target_vector: List[Tuple[int, int]]
) -> float:
    """Get similarity score"""
    index = SparseMatrixSimilarity([source_vector], num_features=num_features)
    similarity = index[target_vector][0]
    return similarity


def evaluate_score(start_score: float, new_score: float, min_score: float) -> bool:
    """Evaluate if new score is within 2/3 of start score"""
    # TODO: should we use Jaccard sim instead to control for sparsity?
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


def get_passage(doc: Tuple[Tokens, Dict[int, int], Dict[int, int]], start_byte: int, end_byte: int) -> Tokens:
    """Get passage within Tokens object"""
    text, start_bytes, end_bytes = doc
    start_index = start_bytes[start_byte]
    end_index = end_bytes[end_byte] + 1
    return text[start_index:end_index]


def merge_passages(
    matches: List[namedlist],
    model: SparseMatrixSimilarity,
    min_score: float,
    num_features: int,
    dictionary: Optional[Dictionary] = None,
) -> List[namedlist]:
    """Merge all passages into bigger passages"""
    # pylint: disable=E1101
    # TODO: should merging be done using Jaccard sim metric: to avoid sparsity
    last_count = len(matches)
    print(f"Merging matches: {last_count} matches before iteration 1", end="", flush=True)
    docs_with_matches: Dict[str, Tuple[Tokens, Dict[int, int], Dict[int, int]]] = get_docs_with_matches(matches)
    for iteration in range(sys.maxsize ** 10): # TODO: To replace with while loop
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
                    source_vector = vectorize(model, source_tokens, dictionary=dictionary)
                    new_score = get_similarity(num_features, source_vector, merged_group.target.vector)
                    if evaluate_score(start_score, new_score, min_score) is True:
                        merged_group.source.end_byte = source.end_byte
                        merged_group.source.metadata["end_byte"] = source.end_byte
                        merged_group.source.vector = source_vector
                        merged_group.similarity = new_score
                        merged_source = True
                elif source.end_byte == merged_group.source.end_byte:
                    merged_source = True
            if target.start_byte <= merged_group.target.end_byte:
                if target.end_byte > merged_group.target.end_byte:
                    target_tokens = get_passage(
                        docs_with_matches[target.metadata["parsed_filename"]],
                        merged_group.target.start_byte,
                        target.end_byte,
                    )
                    target_vector = vectorize(model, target_tokens, dictionary=dictionary)
                    new_score = get_similarity(num_features, merged_group.source.vector, target_vector)
                    if evaluate_score(start_score, new_score, min_score) is True:
                        merged_group.target.end_byte = target.end_byte
                        merged_group.target.metadata["end_byte"] = target.end_byte
                        merged_group.target.vector = target_vector
                        merged_group.similarity = new_score
                        merged_target = True
                elif target.end_byte == merged_group.target.end_byte:
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


def optimize_match(
    tokens: Tokens,
    intersection: Set[str],
    passage_group: namedlist,
    model: SparseMatrixSimilarity,
    dictionary: Optional[Dictionary] = None,
) -> namedlist:
    """Optimize a single match by trimming non-matching words on left and right side"""
    start = None
    end = None
    new_start_byte = None
    new_metadata_start_byte = None
    for pos, token in enumerate(tokens):
        if token.text in intersection:
            if start is None:
                start = pos
                new_start_byte = token.ext["start_byte"]
                new_metadata_start_byte = token.ext["start_byte"]
            end = pos
    new_vector = vectorize(model, tokens[start:end], dictionary=dictionary)
    if new_vector is not None:
        passage_group.start_byte = new_start_byte
        passage_group.metadata["start_byte"] = new_metadata_start_byte
        passage_group.vector = new_vector
        passage_group.end_byte = tokens[end].ext["end_byte"]
        passage_group.metadata["end_byte"] = tokens[end].ext["end_byte"]
    return passage_group


def optimize_matches(
    matches: List[Tuple[namedlist, namedlist, float]],
    model: SparseMatrixSimilarity,
    num_features: int,
    dictionary: Optional[Dictionary] = None,
) -> Tuple[List[Tuple[namedlist, namedlist, float]], Dict[str, Tokens]]:
    """Optimize matches to get highest sim score"""
    print("Optimizing matches...")
    docs_with_matches: Dict[str, Tuple[Tokens, Dict[int, int], Dict[int, int]]] = get_docs_with_matches(matches)
    optimized_matches: List[Tuple[namedlist, namedlist, float]] = []
    for source, target, best_score in matches:
        source_tokens = get_passage(
            docs_with_matches[source.metadata["parsed_filename"]], source.start_byte, source.end_byte
        )
        target_tokens = get_passage(
            docs_with_matches[target.metadata["parsed_filename"]], target.start_byte, target.end_byte
        )
        intersection = {t.text for t in source_tokens if t.text}.intersection(target_tokens)
        source = optimize_match(source_tokens, intersection, source, model, dictionary=dictionary)
        target = optimize_match(target_tokens, intersection, target, model, dictionary=dictionary)
        best_score = get_similarity(num_features, target.vector, source.vector)

        # Let's check the jaccard score to weed out matches with overinflated IDF weighting
        source_set = {token.text for token in source_tokens if token.text}
        target_set = {token.text for token in target_tokens if token.text}
        jaccard_sim = len(source_set.intersection(target_set)) / len(source_set.union(target_set))
        if jaccard_sim > 0.15:
            optimized_matches.append((source, target, best_score))
    return optimized_matches, docs_with_matches


def get_tokens(
    passage: namedlist, preproc: PreProcessor, doc: Tuple[Tokens, Dict[int, int], Dict[int, int]]
) -> List[Token]:
    """Get tokens while making sure we grab a full sentence for POS tagging"""
    sentence_boundaries: set = {".", "!", "?"}
    text: str = " "
    start_byte: int = passage.start_byte
    end_byte: int = passage.end_byte
    while text[0] not in sentence_boundaries:
        start_byte -= 1
        with open(passage.filename, "rb") as text_file:
            text_file.seek(start_byte)
            text = text_file.read(end_byte - start_byte).decode("utf8", "ignore")
        if start_byte == 0:
            break
    max_size = os.path.getsize(passage.filename)
    while text[-1] not in sentence_boundaries:
        end_byte += 1
        if end_byte > max_size:
            break
        with open(passage.filename, "rb") as text_file:
            text_file.seek(start_byte)
            text = text_file.read(end_byte - start_byte).decode("utf8", "ignore")
    tokens: List[Token] = []
    full_tokens, start_bytes, _ = doc
    for token in preproc.process_string(text):
        # print(token)
        end_byte = start_byte + len(token.surface_form.encode("utf8"))
        if start_byte >= passage.start_byte and end_byte <= passage.end_byte:
            try:
                surface_form = token.surface_form
                token = full_tokens[start_bytes[start_byte]]
                token.surface_form = surface_form
            except KeyError:  # Token was not indexed at parse time
                pass
            tokens.append(token)
            # print("missed", repr(token))
        if end_byte > passage.end_byte:
            break
        start_byte = end_byte
    # exit()
    return tokens


def post_process_passages(
    source: namedlist,
    target: namedlist,
    preproc: PreProcessor,
    source_doc: Tuple[Tokens, Dict[int, int], Dict[int, int]],
    target_doc: Tuple[Tokens, Dict[int, int], Dict[int, int]],
) -> Tuple[str, str]:
    """Post process function to highlight matching words in HTML tags"""
    source_tokens = get_tokens(source, preproc, source_doc)
    target_tokens = get_tokens(target, preproc, target_doc)
    source_set = {token.text for token in source_tokens if token.text}
    target_set = {token.text for token in target_tokens if token.text}
    source_passage_with_matches = []
    for token in source_tokens:
        if token.text and token.text in target_set:
            source_passage_with_matches.append(f'&lt;span class="token-match"&gt;{token.surface_form}&lt;/span&gt;')
        elif not token.text:
            source_passage_with_matches.append(f'&lt;span class="filtered-token"&gt;{token.surface_form}&lt;/span&gt;')
        else:
            source_passage_with_matches.append(token.surface_form)
    target_passage_with_matches = []
    for token in target_tokens:
        if token.text and token.text in source_set:
            target_passage_with_matches.append(f'&lt;span class="token-match"&gt;{token.surface_form}&lt;/span&gt;')
        elif not token.text:
            target_passage_with_matches.append(f'&lt;span class="filtered-token"&gt;{token.surface_form}&lt;/span&gt;')
        else:
            target_passage_with_matches.append(token.surface_form)

    return clean_text("".join(source_passage_with_matches)), clean_text("".join(target_passage_with_matches))


def simple_similarity(
    source_texts: Iterator[Tokens],
    config: Dict[str, Any],
    preproc: PreProcessor,
    phrase_model: Optional[Phraser],
    target_texts: Optional[Iterator[Tokens]] = None,
):
    count: int = 0
    matches: List[namedlist] = []
    # phrase_model = phrase_detection(config["source_path"], preproc, config["target_path"])
    source_corpus: CorpusVectorizer = CorpusVectorizer(
        source_texts,
        text_object_definition=config["text_object_definition"],
        min_text_obj_length=config["min_text_obj_length"],
        n_chunk=config["n_chunk"],
        text_object_level_split=config["text_object_level_split"],
        phrase_model=phrase_model,
    )

    target_corpus: CorpusVectorizer = CorpusVectorizer(
        target_texts,
        text_object_definition=config["text_object_definition"],
        dictionary=source_corpus.dictionary,
        min_text_obj_length=config["min_text_obj_length"],
        n_chunk=config["n_chunk"],
        text_object_level_split=config["text_object_level_split"],
        phrase_model=phrase_model,
    )
    source_corpus.dictionary = target_corpus.dictionary
    print("Vectorizing texts...", flush=True)
    model: TfidfModel = TfidfModel(chain(source_corpus.vectors, target_corpus.vectors), smartirs="atc")
    source_corpus.update_with_tfidf(model)
    target_corpus.update_with_tfidf(model)
    index: SparseMatrixSimilarity = SparseMatrixSimilarity(
        source_corpus, num_features=len(source_corpus.dictionary), num_docs=len(source_corpus)
    )
    results: np.array = index[target_corpus]
    with tqdm(total=source_corpus.length, leave=False) as pbar:
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
        matches, model, config["min_similarity"], len(source_corpus.dictionary), dictionary=source_corpus.dictionary
    )
    return matches, model, len(source_corpus.dictionary), source_corpus.dictionary


def doc2vec_similarity(
    source_texts: Iterator[Tokens],
    config: Dict[str, Any],
    preproc: PreProcessor,
    phrase_model: Optional[Phraser],
    target_texts: Optional[Iterator[Tokens]] = None,
):
    count: int = 0
    matches: List[namedlist] = []
    # phrase_model = phrase_detection(config["source_path"], preproc, config["target_path"])
    source_corpus: Doc2VecCorpus = Doc2VecCorpus(
        source_texts,
        text_object_definition=config["text_object_definition"],
        min_text_obj_length=config["min_text_obj_length"],
        n_chunk=config["n_chunk"],
        text_object_level_split=config["text_object_level_split"],
        phrase_model=phrase_model,
    )
    target_corpus: Doc2VecCorpus = Doc2VecCorpus(
        target_texts,
        text_object_definition=config["text_object_definition"],
        min_text_obj_length=config["min_text_obj_length"],
        n_chunk=config["n_chunk"],
        text_object_level_split=config["text_object_level_split"],
        phrase_model=phrase_model,
    )
    if not os.path.exists(config["doc2vec_model"]):
        docs = [
            TaggedDocument(doc, [i]) for i, doc in enumerate(chain(source_corpus.text_chunks, target_corpus.text_chunks))
        ]
        model = Doc2Vec(docs, vector_size=50, window=5, min_count=2, workers=config["workers"], epochs=20)
        model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        model.save(config["doc2vec_model"])
    else:
        model = Doc2Vec.load(config["doc2vec_model"])
    source_corpus.build_vectors(model)
    target_corpus.build_vectors(model)
    index = MatrixSimilarity(source_corpus, num_features=len(source_corpus.vectors[0]), corpus_len=len(source_corpus))
    results = index[target_corpus]
    with tqdm(total=source_corpus.length, leave=False) as pbar:
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
    matches = merge_passages(matches, model, config["min_similarity"], 50)
    return matches


def run_vsm(config: Dict[str, Any]):
    """Main function"""
    if config["text_object_definition"] not in ("n_token", "text_object"):
        print("Error: Only valid values for text object definition are 'n_token' and 'text_object'")
        exit()
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
    phrase_model: Optional[Phraser] = None
    if config["source_corpus"] is not None:
        with open(config["source_corpus"], "rb") as input_file:
            source_texts = load(input_file)
    else:
        source_texts: Iterator[Tokens] = preproc.process_texts(
            (file.path for file in os.scandir(config["source_path"])), keep_all=True, progress=False,
        )
    if config["target_corpus"] is not None:
        with open(config["target_corpus"], "rb") as input_file:
            target_texts = load(input_file)
    else:
        target_texts: Iterator[Tokens] = preproc.process_texts(
            (file.path for file in os.scandir(config["target_path"])), keep_all=True, progress=False
        )
    if config["similarity_metric"] == "cosine":
        matches, model, num_features, dictionary = simple_similarity(source_texts, config, preproc, phrase_model, target_texts=target_texts)
        matches, docs_with_matches = optimize_matches(matches, model, num_features, dictionary=dictionary)
    elif config["similarity_metric"] == "doc2vec":
        matches = doc2vec_similarity(source_texts, config, preproc, phrase_model, target_texts=target_texts)

    print("Writing out results...")
    with open("alignments.jsonl", "w") as output:
        for source, target, best_score in matches:
            source_context_before = get_text(source.start_byte - 300, source.start_byte, source.metadata["filename"])
            source_passage = get_text(source.start_byte, source.end_byte, source.metadata["filename"])
            source_context_after = get_text(source.end_byte, source.end_byte + 300, source.metadata["filename"])
            target_context_before = get_text(target.start_byte - 300, target.start_byte, target.metadata["filename"])
            target_passage = get_text(target.start_byte, target.end_byte, target.metadata["filename"])
            target_context_after = get_text(target.end_byte, target.end_byte + 300, target.metadata["filename"])
            if config["similarity_metric"] == "cosine":
                source_passage_with_matches, target_passage_with_matches = post_process_passages(
                    source,
                    target,
                    preproc,
                    docs_with_matches[source.metadata["parsed_filename"]],
                    docs_with_matches[target.metadata["parsed_filename"]],
                )
            else:
                source_passage_with_matches = source_passage
                target_passage_with_matches = target_passage
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
        "source_path": "/var/www/html/philologic/toutvoltaire/data/words_and_philo_ids",
        "target_path": "/var/www/html/philologic/AP_split/data/words_and_philo_ids",
        "source_corpus": None,
        "target_corpus": None,
        "language": "french",
        "text_object_type": "text_object",
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
        "min_text_obj_length": 10,
        "minimum_word_length": 3,
        "lemmatizer": "spacy",
        "stopwords": "/shared/PhiloLogic4/extras/FrenchStopwords.txt",
        "workers": 32,
        "pos_to_keep": ["NOUN", "ADJ", "VERB"],
        "n_chunk": 5,
        "min_similarity": 0.3,
        "similarity_metric": "cosine",
        "doc2vec_model": ""
    }
    run_vsm(configuration)
