#! /usr/bin/env python3
"""Passage similarity detection"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import deque
from html import unescape as unescape_html
from typing import Any, Deque, Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union
from xml.sax.saxutils import unescape as unescape_xml

import numpy as np
from dill import dump, load
from recordclass import dataobject
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from text_preprocessing import PreProcessor, Token, Tokens
from tqdm import tqdm

TAGS = re.compile(r"<[^>]+>")


class PassageGroup(dataobject, fast_new=True):
    """Text passage with all associated properties and vector representation"""

    vector: csr_matrix = csr_matrix([])
    start_byte: int = 0
    end_byte: int = 0
    filename: str = ""
    metadata: Dict = {}


class MergedGroup(dataobject, fast_new=True):
    """A source and target PassageGroup pair with similarity"""

    source: PassageGroup = PassageGroup()
    target: PassageGroup = PassageGroup()
    similarity: float = 0.0


PHILO_TEXT_OBJECT_LEVELS = {"doc": 1, "div1": 2, "div2": 3, "div3": 4, "para": 5, "sent": 6, "word": 7}

TEMP_DIR = os.getcwd()


class Corpus:
    """A Corpus of passages as preprocessed by the text-preprocessor"""

    def __init__(
        self,
        texts: Iterable[Tokens],
        text_object_definition: str = "n_token",
        min_text_obj_length: int = 15,
        n_chunk: int = 3,
        text_object_level_split: str = "doc",
        vectorizer: Optional[TfidfVectorizer] = None,
        min_freq: Union[int, float] = 1,
        max_freq: float = 1.0,
    ):
        """Intialize CorpusVectorizer"""
        self.texts: Iterable[Tokens] = texts
        self.min_text_obj_length: int = min_text_obj_length
        self.n_chunk: int = n_chunk
        self.metadata: List[Dict[str, Any]] = []
        self.text_object_level_split = text_object_level_split
        self.text_object_definition: str = text_object_definition
        self.tmp_dir = os.path.abspath(f"{TEMP_DIR}/output/")
        self.direction: str = "source"
        if vectorizer is None:
            os.system(f"rm -rf {self.tmp_dir}/*")
            os.system(f"mkdir {os.path.join(self.tmp_dir, self.direction)}")
            self.vectorizer = TfidfVectorizer(max_df=max_freq, min_df=min_freq, sublinear_tf=True)
            self.vectors: csr_matrix = self.vectorizer.fit_transform(self.__get_text_chunks())  # type: ignore
        else:
            self.direction = "target"
            os.system(f"mkdir {os.path.join(self.tmp_dir, self.direction)}")
            self.vectorizer = vectorizer
            self.vectors: csr_matrix = self.vectorizer.transform(self.__get_text_chunks())  # type: ignore
        self.length: int = self.vectors.shape[0]  # type: ignore
        self.dim: int = self.vectors.shape[1]  # type: ignore

    def __getitem__(self, item: int) -> csr_matrix:
        return self.vectors[item]

    def __len__(self) -> int:
        return self.length

    def __get_text_chunks(self) -> Iterator[str]:
        """Process all texts into smaller text chunks"""
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
            text.metadata["parsed_filename"] = os.path.join(
                self.tmp_dir, self.direction, os.path.basename(text.metadata["parsed_filename"])
            )
            doc_id = text.metadata["philo_id"].split()[0]
            if (
                doc_id != current_doc_id and current_doc_id is not None
            ):  # we save the current doc when doc_ids don't match
                self.__save_doc(full_doc)
                full_doc = Tokens([], text.metadata)
            full_doc.extend(text)
            text.purge()
            text_level_id: str = " ".join(
                text.metadata["philo_id"].split()[: PHILO_TEXT_OBJECT_LEVELS[self.text_object_level_split]]
            )
            if text_level_id != current_text_level_id:
                chunk_group_length: int = sum([len(t) for t in chunk_group])
                if chunk_group_length >= min_chunk_length and self.text_object_definition == "text_object":
                    chunk = [t for chunk in chunk_group for t in chunk]
                    self.__store_metadata(chunk_group[0].metadata, chunk)
                    yield " ".join(chunk)
                chunk_group.clear()
            current_text_level_id = text_level_id
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
                        chunk = [t for chunk in chunk_group for t in chunk]
                        self.__store_metadata(chunk_group[0].metadata, chunk)
                        yield " ".join(chunk)
            else:
                chunks_to_return: List[List[Token]] = []
                for chunk in text.split_tokens(self.min_text_obj_length):
                    if not chunk:
                        continue
                    if len(chunk) != self.min_text_obj_length:  # We've reached the end of our text object
                        try:
                            chunk_group[-1].extend(chunk)
                            chunk = [t for chunk in chunk_group for t in chunk if t.text]
                            self.__store_metadata(chunk_group[0].metadata, chunk)
                            yield " ".join(chunk)
                            break
                        except IndexError:
                            pass
                    else:
                        chunk_group.append(chunk)
                    if len(chunk_group) == self.n_chunk:
                        chunks_to_return.append([t for chunk in chunk_group for t in chunk if t.text])
                for chunk in chunks_to_return:
                    self.__store_metadata(chunk_group[0].metadata, chunk)
                    yield " ".join(chunk)
            current_doc_id = doc_id
        self.__save_doc(full_doc)
        del self.texts
        print()

    def __save_doc(self, doc: Tokens):
        """Save doc to tmp dir for later retrieval"""
        start_bytes: Dict[int, int] = {}
        end_bytes: Dict[int, int] = {}
        for pos, token in enumerate(doc):
            start_bytes[token.ext["start_byte"]] = pos
            end_bytes[token.ext["end_byte"]] = pos
        with open(doc.metadata["parsed_filename"], "wb") as output:
            dump((doc, start_bytes, end_bytes), output)

    def __store_metadata(self, metadata: Dict[str, Any], tokens: List[Token]):
        """Store Metadata for each chunk"""
        # metadata: Dict[str, Any] = chunk_group[0].metadata
        # metadata["start_byte"] = chunk_group[0][0].ext["start_byte"]
        # metadata["end_byte"] = chunk_group[-1][-1].ext["end_byte"]
        self.metadata.append(
            {**metadata, "start_byte": tokens[0].ext["start_byte"], "end_byte": tokens[-1].ext["end_byte"],}
        )
        # self.metadata.append(metadata)

    def inner_compare(self) -> csr_matrix:
        """Compare corpus with itself"""
        results: csr_matrix = linear_kernel(self.vectors)  # type: ignore
        scores: csr_matrix
        for doc_id, scores in enumerate(results):
            metadata = self.metadata[doc_id]
            for inner_doc, score in enumerate(scores):
                if self.metadata[inner_doc]["year"] >= metadata["year"]:
                    results[doc_id, inner_doc] = score
                else:
                    results[doc_id, inner_doc] = 0.0
        return results

    def outer_compare(self, target_corpus: Corpus):
        """Compare corpus with another corpus"""
        print("Comparing source collection to target collection...", flush=True)
        results: csr_matrix = linear_kernel(self.vectors, target_corpus.vectors)  # type: ignore
        scores: csr_matrix
        for doc_id, scores in enumerate(results):
            metadata = self.metadata[doc_id]
            for inner_doc, score in enumerate(scores):
                if target_corpus.metadata[inner_doc]["year"] >= metadata["year"]:
                    results[doc_id, inner_doc] = score
                else:
                    results[doc_id, inner_doc] = 0.0
        return results


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


def evaluate_score(start_score: float, new_score: float, min_score: float) -> bool:
    """Evaluate if new score is within 3/4 of start score"""
    # TODO: should we use Jaccard sim instead to control for sparsity?
    if new_score >= min_score or new_score / start_score > 0.75:
        return True
    return False


def get_docs_with_matches(matches: List[MergedGroup]) -> Dict[str, Tuple[Tokens, Dict[int, int], Dict[int, int]]]:
    """Fetch all documents with matches"""
    docs_with_matches: Dict[str, Tuple[Tokens, Dict[int, int], Dict[int, int]]] = {}
    for match in matches:
        if match.source.metadata["parsed_filename"] not in docs_with_matches:
            with open(match.source.metadata["parsed_filename"], "rb") as input_doc:
                docs_with_matches[match.source.metadata["parsed_filename"]] = load(input_doc)
        if match.target.metadata["parsed_filename"] not in docs_with_matches:
            with open(match.target.metadata["parsed_filename"], "rb") as input_doc:
                docs_with_matches[match.target.metadata["parsed_filename"]] = load(input_doc)
    return docs_with_matches


def get_passage(doc: Tuple[Tokens, Dict[int, int], Dict[int, int]], start_byte: int, end_byte: int) -> Iterable[Token]:
    """Get passage within Tokens object"""
    text, start_bytes, end_bytes = doc
    start_index = start_bytes[start_byte]
    end_index = end_bytes[end_byte] + 1
    return text[start_index:end_index]


def merge_passages(matches: List[MergedGroup], corpus: Corpus, min_score: float,) -> List[MergedGroup]:
    """Merge all passages into bigger passages"""
    # pylint: disable=E1101
    # TODO: should merging be done using Jaccard sim metric: to avoid sparsity
    docs_with_matches: Dict[str, Tuple[Tokens, Dict[int, int], Dict[int, int]]] = get_docs_with_matches(matches)
    last_count = len(matches)
    current_count = last_count + 1
    iteration = 1
    print(f"Merging matches: {last_count} matches before iteration 1", end="", flush=True)
    while last_count / current_count <= 1.0:  # we stop iterating if there are minimal change between iterations
        last_count = current_count
        matches.sort(
            key=lambda x: (
                x.source.filename,
                x.target.filename,
                x.source.start_byte,
                x.source.start_byte - x.source.end_byte,
                x.target.start_byte,
                x.target.start_byte - x.target.end_byte,
            )
        )  # sort by smaller start byte and bigger end_byte
        merged_group: MergedGroup = MergedGroup()
        saved_groups: List[MergedGroup] = []
        total_matches: int = len(matches)
        start_score: float = min_score

        for pos, match in enumerate(matches):
            merged_source: bool = False
            merged_target: bool = False
            if merged_group.source.filename == "":
                start_score = match.similarity
                merged_group = MergedGroup(match.source, match.target, start_score)
                continue
            if (
                match.source.filename != merged_group.source.filename
                or match.target.filename != merged_group.target.filename
            ):
                saved_groups.append(merged_group)
                start_score = match.similarity
                merged_group = MergedGroup(match.source, match.target, start_score)
                continue
            if match.source.start_byte <= merged_group.source.end_byte:
                if match.source.end_byte > merged_group.source.end_byte:
                    source_tokens = get_passage(
                        docs_with_matches[match.source.metadata["parsed_filename"]],
                        merged_group.source.start_byte,
                        match.source.end_byte,
                    )
                    source_vector: csr_matrix = corpus.vectorizer.transform([" ".join(source_tokens)])  # type: ignore
                    score_array: np.ndarray = linear_kernel(source_vector, merged_group.target.vector)  # type: ignore
                    new_score: float = score_array[0, 0]
                    if evaluate_score(start_score or min_score, new_score, min_score) is True:
                        merged_group.source.end_byte = match.source.end_byte
                        merged_group.source.metadata["end_byte"] = match.source.end_byte
                        merged_group.source.vector = source_vector
                        merged_group.similarity = new_score
                        merged_source = True
                elif match.source.end_byte == merged_group.source.end_byte:
                    merged_source = True
            if match.target.start_byte <= merged_group.target.end_byte:
                if match.target.end_byte > merged_group.target.end_byte:
                    target_tokens = get_passage(
                        docs_with_matches[match.target.metadata["parsed_filename"]],
                        merged_group.target.start_byte,
                        match.target.end_byte,
                    )
                    target_vector: csr_matrix = corpus.vectorizer.transform([" ".join(target_tokens)])  # type: ignore
                    score_array: np.ndarray = linear_kernel(merged_group.source.vector, target_vector)  # type: ignore
                    new_score: float = score_array[0, 0]
                    if evaluate_score(start_score or min_score, new_score, min_score) is True:
                        merged_group.target.end_byte = match.target.end_byte
                        merged_group.target.metadata["end_byte"] = match.target.end_byte
                        merged_group.target.vector = target_vector
                        merged_group.similarity = new_score
                        merged_target = True
                elif match.target.end_byte == merged_group.target.end_byte:
                    merged_target = True
            if merged_source is False and merged_target is False:
                saved_groups.append(merged_group)
                start_score = match.similarity
                merged_group = MergedGroup(match.source, match.target, start_score)
            if pos + 1 == total_matches:
                saved_groups.append(merged_group)
        matches = saved_groups
        iteration += 1
        current_count = len(saved_groups)
        print(f"\rMerging matches: {current_count} matches after iteration {iteration+1}...", end="", flush=True)
    print(flush=True)
    return matches


def optimize_match(
    tokens: Iterable[Token], intersection: Set[str], passage_group: PassageGroup, corpus: Corpus,
) -> PassageGroup:
    """Optimize a single match by trimming non-matching words on left and right side"""
    start = None
    end = None
    new_start_byte: int = 0
    new_metadata_start_byte = 0
    for pos, token in enumerate(tokens):
        if token.text in intersection:
            if start is None:
                start = pos
                new_start_byte = token.ext["start_byte"]
                new_metadata_start_byte = token.ext["start_byte"]
            end = pos
    new_vector: csr_matrix = corpus.vectorizer.transform([" ".join(tokens[start:end])])  # type: ignore
    if new_vector is not None:
        passage_group.start_byte = new_start_byte
        passage_group.metadata["start_byte"] = new_metadata_start_byte
        passage_group.vector = new_vector
        passage_group.end_byte = tokens[end].ext["end_byte"]  # type: ignore
        passage_group.metadata["end_byte"] = tokens[end].ext["end_byte"]  # type: ignore
    return passage_group


def optimize_matches(
    matches: List[MergedGroup], corpus: Corpus,
) -> Tuple[List[MergedGroup], Dict[str, Tuple[Tokens, Dict[int, int], Dict[int, int]]]]:
    """Optimize matches to get highest sim score"""
    print("Optimizing matches...")
    docs_with_matches: Dict[str, Tuple[Tokens, Dict[int, int], Dict[int, int]]] = get_docs_with_matches(matches)
    optimized_matches: List[MergedGroup] = []
    match_count = 0
    for match in tqdm(matches, total=len(matches), leave=False):
        source_tokens = get_passage(
            docs_with_matches[match.source.metadata["parsed_filename"]], match.source.start_byte, match.source.end_byte
        )
        target_tokens = get_passage(
            docs_with_matches[match.target.metadata["parsed_filename"]], match.target.start_byte, match.target.end_byte
        )
        intersection = {t.text for t in source_tokens if t.text}.intersection({t.text for t in target_tokens})
        source = optimize_match(source_tokens, intersection, match.source, corpus)
        target = optimize_match(target_tokens, intersection, match.target, corpus)
        best_score_matrix: np.ndarray = linear_kernel(target.vector, source.vector)  # type: ignore
        best_score: float = best_score_matrix[0, 0]

        # Let's check the jaccard score to weed out matches with overinflated IDF weighting
        # source_set = {token.text for token in source_tokens if token.text}
        # target_set = {token.text for token in target_tokens if token.text}
        # jaccard_sim = len(source_set.intersection(target_set)) / len(source_set.union(target_set))
        # print(jaccard_sim)
        # if jaccard_sim > 0.15:
        optimized_matches.append(MergedGroup(source, target, best_score))
        match_count += 1
    return optimized_matches, docs_with_matches


def get_tokens(
    passage: PassageGroup, preproc: PreProcessor, doc: Tuple[Tokens, Dict[int, int], Dict[int, int]]
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
    source: PassageGroup,
    target: PassageGroup,
    source_preproc: PreProcessor,
    target_preproc: PreProcessor,
    source_doc: Tuple[Tokens, Dict[int, int], Dict[int, int]],
    target_doc: Tuple[Tokens, Dict[int, int], Dict[int, int]],
) -> Tuple[str, str]:
    """Post process function to highlight matching words in HTML tags"""
    source_tokens = get_tokens(source, source_preproc, source_doc)
    target_tokens = get_tokens(target, target_preproc, target_doc)
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
    source_texts: Iterable[Tokens],
    source_config: Dict[str, Any],
    target_config: Dict[str, Any],
    min_similarity: float,
    target_texts: Optional[Iterable[Tokens]] = None,
) -> Tuple[List[MergedGroup], Corpus]:
    """Annoy cosine similarity"""
    count: int = 0
    matches: List[MergedGroup] = []
    source_corpus: Corpus = Corpus(
        source_texts,
        text_object_definition=source_config["text_object_definition"],
        min_text_obj_length=source_config["min_text_object_length"],
        n_chunk=source_config["n_chunk"],
        text_object_level_split=source_config["text_object_level_split"],
        min_freq=source_config["min_freq"],
        max_freq=source_config["max_freq"],
    )
    if target_texts is not None:
        target_corpus: Corpus = Corpus(
            target_texts,
            text_object_definition=target_config["text_object_definition"],
            vectorizer=source_corpus.vectorizer,
            min_text_obj_length=target_config["min_text_object_length"],
            n_chunk=target_config["n_chunk"],
            text_object_level_split=target_config["text_object_level_split"],
        )

        matching_docs = source_corpus.outer_compare(target_corpus)
    else:
        matching_docs = source_corpus.inner_compare()
        target_corpus = source_corpus

    for source_doc, score_array in enumerate(matching_docs):
        filtered_results: np.array = np.where(score_array > min_similarity)[0]  # type: ignore
        count += len(filtered_results)
        for target_doc in filtered_results:
            matches.append(
                MergedGroup(
                    PassageGroup(
                        source_corpus[source_doc],
                        source_corpus.metadata[source_doc]["start_byte"],
                        source_corpus.metadata[source_doc]["end_byte"],
                        source_corpus.metadata[source_doc]["filename"],
                        source_corpus.metadata[source_doc],
                    ),
                    PassageGroup(
                        target_corpus[target_doc],
                        target_corpus.metadata[target_doc]["start_byte"],
                        target_corpus.metadata[target_doc]["end_byte"],
                        target_corpus.metadata[target_doc]["filename"],
                        target_corpus.metadata[target_doc],
                    ),
                    score_array[target_doc],  # type: ignore
                )
            )
    print(f"{count} matches found.")
    print("\n### Post-processing results ###", flush=True)
    matches = merge_passages(matches, source_corpus, min_similarity,)
    return matches, source_corpus


def run_vsa(source_path: str, target_path: str, workers: int, config: Dict[str, Any]):
    """Main function"""
    if config["source"]["text_object_definition"] not in ("n_token", "text_object"):
        print("Error: Only valid values for text object definition are 'n_token' and 'text_object'")
        exit()
    if config["target"]["text_object_definition"] not in ("n_token", "text_object"):
        print("Error: Only valid values for text object definition are 'n_token' and 'text_object'")
        exit()
    if config["source"]["text_object_definition"] == "n_token":
        config["source"]["text_object_type"] = config["source"]["text_object_level_split"]
    if config["target"]["text_object_definition"] == "n_token":
        config["target"]["text_object_type"] = config["target"]["text_object_level_split"]
    source_preproc: PreProcessor = PreProcessor(is_philo_db=True, workers=workers, **config["source"])
    source_texts: Iterable[Tokens] = source_preproc.process_texts(
        (file.path for file in os.scandir(source_path)), keep_all=True, progress=False
    )
    target_preproc: PreProcessor = PreProcessor(is_philo_db=True, workers=workers, **config["target"])
    target_texts: Iterable[Tokens] = target_preproc.process_texts(
        (file.path for file in os.scandir(target_path)), keep_all=True, progress=False
    )

    matches, source_corpus = simple_similarity(
        source_texts, config["source"], config["target"], config["min_similarity"], target_texts=target_texts,
    )
    matches, docs_with_matches = optimize_matches(matches, source_corpus)

    print("Writing out processed results...")
    os.system("mkdir -p output/results")
    with open("output/results/alignments.jsonl", "w") as output:
        for match in tqdm(matches, total=len(matches), leave=False):
            source_context_before = get_text(
                match.source.start_byte - 300, match.source.start_byte, match.source.metadata["filename"]
            )
            source_passage = get_text(match.source.start_byte, match.source.end_byte, match.source.metadata["filename"])
            source_context_after = get_text(
                match.source.end_byte, match.source.end_byte + 300, match.source.metadata["filename"]
            )
            target_context_before = get_text(
                match.target.start_byte - 300, match.target.start_byte, match.target.metadata["filename"]
            )
            target_passage = get_text(match.target.start_byte, match.target.end_byte, match.target.metadata["filename"])
            target_context_after = get_text(
                match.target.end_byte, match.target.end_byte + 300, match.target.metadata["filename"]
            )
            if config["similarity_metric"] == "cosine":
                source_passage_with_matches, target_passage_with_matches = post_process_passages(
                    match.source,
                    match.target,
                    source_preproc,
                    target_preproc,
                    docs_with_matches[match.source.metadata["parsed_filename"]],
                    docs_with_matches[match.target.metadata["parsed_filename"]],
                )
            else:
                source_passage_with_matches = source_passage
                target_passage_with_matches = target_passage
            result_object: str = json.dumps(
                {
                    "source_doc_id": match.source.metadata["philo_id"].split()[0],
                    "source_context_before": source_context_before,
                    "source_passage": source_passage,
                    "source_context_after": source_context_after,
                    "source_passage_with_matches": source_passage_with_matches,
                    "target_doc_id": match.target.metadata["philo_id"].split()[0],
                    "target_context_before": target_context_before,
                    "target_passage": target_passage,
                    "target_context_after": target_context_after,
                    "target_passage_with_matches": target_passage_with_matches,
                    "similarity": str(match.similarity),
                    **{f"source_{field}": value for field, value in match.source.metadata.items()},
                    **{f"target_{field}": value for field, value in match.target.metadata.items()},
                }
            )
            print(result_object, file=output)
