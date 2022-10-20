#! /usr/bin/env python3
"""Passage similarity detection"""

from __future__ import annotations

import json
import os
import re
from abc import ABC
from collections import deque
from html import unescape as unescape_html
from math import floor
from shutil import rmtree
from typing import Any, Deque, Dict, Generator, Iterable, List, Optional, Set, Tuple, Union
from xml.sax.saxutils import unescape as unescape_xml

import dill as pickle
import numpy as np
import spacy
import torch
from dill import dump, load
from gensim.models import KeyedVectors
from gensim.similarities import WmdSimilarity
from numpy.lib.utils import source
from recordclass import dataobject
from scipy.sparse import csr_matrix
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from text_preprocessing import PreProcessor, Token, Tokens
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

TAGS = re.compile(r"<[^>]+>")
PHILO_TEXT_OBJECT_LEVELS = {"doc": 1, "div1": 2, "div2": 3, "div3": 4, "para": 5, "sent": 6, "word": 7}
TEMP_DIR = os.getcwd()


class PassageGroup(dataobject, fast_new=True):
    """Text passage with all associated properties and vector representation"""

    vector: Any = []
    start_byte: int = 0
    end_byte: int = 0
    filename: str = ""
    metadata: Dict = {}


class MergedGroup(dataobject, fast_new=True):
    """A source and target PassageGroup pair with similarity"""

    source: PassageGroup = PassageGroup()
    target: PassageGroup = PassageGroup()
    similarity: float = 0.0


class Vector(dataobject, fast_new=True):
    """Object continaing a vector and its L2 norm"""

    vector: np.ndarray
    vector_norm: np.ndarray


class DocumentChunks:
    """A generator with caching"""

    def __init__(
        self, docs: Generator[List[str], None, None], save_path: str, return_type="list", transform_function=None
    ):
        self.docs = docs
        self.doc_list: List[List[str]] = []
        self.doc_count = 0
        self.generator_exhausted = False
        self.return_type = return_type
        self.transform_function = transform_function
        self.corpus_type = self.transform_function.__qualname__.split(".")[0]
        self.path = os.path.join(TEMP_DIR, "output/chunks/", save_path)
        if os.path.exists(self.path):
            rmtree(self.path)
        os.makedirs(self.path, exist_ok=True)

    def __iter__(self) -> Generator[Union[str, List[str]], None, None]:
        if self.generator_exhausted is False:
            if self.doc_count == 0:
                for doc in self.docs:
                    doc = self.__format_doc(doc)
                    self.__save(doc)
                    self.doc_count += 1
                    yield doc
            else:
                for doc_name in range(self.doc_count):
                    yield self.__load(doc_name)
                for doc in self.docs:
                    doc = self.__format_doc(doc)
                    self.__save(doc)
                    self.doc_count += 1
                    yield doc
            self.generator_exhausted = True
        else:
            for doc_name in self.doc_list:
                yield self.__load(doc_name)

    def __save(self, doc: Union[List[str], str]):
        filename = os.path.join(self.path, str(self.doc_count))
        if self.transform_function is None:
            with open(filename, "wb") as output_file:
                pickle.dump(doc, output_file)
        elif self.corpus_type == "SentenceEmbeddingsCorpus":
            transformed_doc = self.transform_function([doc])
            torch.save(transformed_doc, f"{filename}.pt")
        else:
            transformed_doc = self.transform_function([doc])
            np.save(f"{filename}.npy", transformed_doc.vector)
            np.save(f"{filename}_norm.npy", transformed_doc.vector_norm)

    def __load(self, doc_name) -> Union[List[str], torch.Tensor, Vector]:
        filename = os.path.join(self.path, str(doc_name))
        if self.transform_function is None:
            with open(filename, "rb") as input_file:
                doc = pickle.load(input_file)
            return doc
        elif self.corpus_type == "SentenceEmbeddingsCorpus":
            return torch.load(f"{filename}.pt")
        return Vector(np.load(f"{filename}.npy"), np.load(f"{filename}_norm.npy"))

    def __get_doc(self, index: int) -> Union[List[str], str]:
        doc = None
        while index > self.doc_count:
            try:
                doc = next(self.docs)
                self.__format_doc(doc)
                self.__save(doc)
                self.doc_count += 1
            except StopIteration:
                raise IndexError
        if doc is None:
            return self.__load(index)
        return doc

    def __getitem__(
        self, item: Union[int, slice]
    ) -> Union[Union[List[str], str], Union[List[Union[List[str], str]], torch.Tensor]]:
        if isinstance(item, slice):
            end = item.stop
            if item.stop > len(self):  # avoid index out of range
                end = len(self)
            if self.transform_function is None or self.corpus_type == "Word2VecEmbeddingCorpus":
                return [self.__get_doc(index) for index in range(item.start, end)]
            return torch.cat([self.__get_doc(index) for index in range(item.start, end)])  # type:ignore
        return self.__get_doc(item)

    def __format_doc(self, doc: List[str]) -> Union[str, List[str]]:
        if self.return_type == "str":
            return " ".join(doc)
        return doc

    def __len__(self):
        if self.generator_exhausted is False:
            for _ in self:
                pass
        return self.doc_count


class Matches:
    """Matches cached to disk"""

    def __init__(self, matches: Iterable[MergedGroup]):
        self.path = os.path.join(TEMP_DIR, "output/results/matches")
        os.system(f"mkdir -p {self.path}")
        self.matches = matches
        if isinstance(self.matches, list):
            self.is_cached = False
            self.count = len(self.matches)
        else:
            self.is_cached = True
            self.count = self.__save()  # save generator to disk

    def __save(self):
        count = 0
        for count, match in enumerate(self.matches):
            with open(os.path.join(self.path, f"{count}"), "wb") as output:
                pickle.dump(match, output)
        return count + 1

    @classmethod
    def load(cls):
        """Load instance of class by reading previously cached matches"""
        matches = []
        for file in os.scandir(os.path.join(TEMP_DIR, "output/results/matches")):
            with open(file.path, "rb") as input_file:
                matches.append(pickle.load(input_file))
        return cls(matches)

    def __len__(self):
        return self.count

    def __get_file(self, index):
        if self.is_cached is False:
            return self.matches[index]  # type: ignore
        else:
            with open(os.path.join(self.path, f"{index}"), "rb") as input_file:
                return pickle.load(input_file)

    def __iter__(self):
        for file in range(self.count):
            yield self.__get_file(file)


class Corpus(ABC):
    """A Corpus of passages as preprocessed by the text-preprocessor"""

    def __init__(
        self,
        texts: Iterable[Tokens],
        text_object_definition: str = "n_token",
        min_text_obj_length: int = 15,
        n_chunk: int = 3,
        text_object_level_split: str = "doc",
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
        self.length = 0

    def __len__(self) -> int:
        return self.length

    def __getitem__(self):
        pass

    def get_text_chunks(self) -> Generator[List[str], None, None]:
        """Process all texts into smaller text chunks"""
        chunk_group: Deque[Tokens] = deque(maxlen=self.n_chunk)
        min_chunk_length: int = self.n_chunk * self.min_text_obj_length
        current_text_level_id: str = "0"
        full_doc = Tokens([], {})
        current_doc_id = None
        chunks_done = 0
        for text in self.texts:
            print(f"\rProcessing {self.direction} texts... {chunks_done} text chunks extracted...", end="", flush=True)
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
                    chunk_group.popleft()
                    if chunk_group:
                        chunk = [t for chunk in chunk_group for t in chunk]
                        self.__store_metadata(chunk_group[0].metadata, chunk)
                        chunks_done += 1
                        yield [t.text for t in chunk]
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
                        chunk = [t for c in chunk_group for t in c]
                        self.__store_metadata(chunk_group[0].metadata, chunk)
                        chunks_done += 1
                        yield [t.text for t in chunk]
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
                            chunks_done += 1
                            yield [t.text for t in chunk]
                            break
                        except IndexError:
                            pass
                    else:
                        chunk_group.append(chunk)
                    if len(chunk_group) == self.n_chunk:
                        chunks_to_return.append([t for chunk in chunk_group for t in chunk if t.text])
                for chunk in chunks_to_return:
                    self.__store_metadata(chunk_group[0].metadata, chunk)
                    chunks_done += 1
                    yield [t.text for t in chunk]
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
        self.metadata.append(
            {
                **metadata,
                "start_byte": tokens[0].ext["start_byte"],
                "end_byte": tokens[-1].ext["end_byte"],
            }
        )

    def process_inner_compare(self, results, min_similarity: float) -> Iterable[MergedGroup]:
        """Compare corpus with itself"""
        for outer_doc_id, inner_doc_id in np.argwhere(results >= min_similarity):
            if (
                self.metadata[outer_doc_id]["year"] <= self.metadata[inner_doc_id]["year"]
                and inner_doc_id != outer_doc_id
            ):
                yield MergedGroup(
                    PassageGroup(
                        self[outer_doc_id],
                        self.metadata[outer_doc_id]["start_byte"],
                        self.metadata[outer_doc_id]["end_byte"],
                        self.metadata[outer_doc_id]["filename"],
                        self.metadata[outer_doc_id],
                    ),
                    PassageGroup(
                        self[inner_doc_id],
                        self.metadata[inner_doc_id]["start_byte"],
                        self.metadata[inner_doc_id]["end_byte"],
                        self.metadata[inner_doc_id]["filename"],
                        self.metadata[inner_doc_id],
                    ),
                    results[outer_doc_id, inner_doc_id],  # type: ignore
                )

    def process_outer_compare(
        self, results: np.ndarray, target_corpus: Corpus, min_similarity
    ) -> Iterable[MergedGroup]:
        """Compare corpus with another corpus"""
        for outer_doc_id, inner_doc_id in np.argwhere(results >= min_similarity):
            if (
                self.metadata[outer_doc_id]["year"] < target_corpus.metadata[inner_doc_id]["year"]
            ):  # TODO: makes this <= instead?
                yield MergedGroup(
                    PassageGroup(
                        self[outer_doc_id],  # type: ignore
                        self.metadata[outer_doc_id]["start_byte"],
                        self.metadata[outer_doc_id]["end_byte"],
                        self.metadata[outer_doc_id]["filename"],
                        self.metadata[outer_doc_id],
                    ),
                    PassageGroup(
                        target_corpus[inner_doc_id],  # type:ignore
                        target_corpus.metadata[inner_doc_id]["start_byte"],
                        target_corpus.metadata[inner_doc_id]["end_byte"],
                        target_corpus.metadata[inner_doc_id]["filename"],
                        target_corpus.metadata[inner_doc_id],
                    ),
                    results[outer_doc_id, inner_doc_id],  # type: ignore
                )


class TfIdfCorpus(Corpus):
    """Corpus object which builds TF-IDF vectors"""

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
        super().__init__(
            texts,
            text_object_definition=text_object_definition,
            min_text_obj_length=min_text_obj_length,
            n_chunk=n_chunk,
            text_object_level_split=text_object_level_split,
        )
        if vectorizer is None:
            os.system(f"rm -rf {self.tmp_dir}/*")
            os.system(f"mkdir {os.path.join(self.tmp_dir, self.direction)}")
            self.vectorizer = TfidfVectorizer(max_df=max_freq, min_df=min_freq, ngram_range=(1, 2))
            self.vectors: csr_matrix = self.vectorizer.fit_transform(" ".join(d) for d in self.get_text_chunks())  # type: ignore
        else:
            self.direction = "target"
            os.system(f"mkdir {os.path.join(self.tmp_dir, self.direction)}")
            self.vectorizer = vectorizer
            self.vectors: csr_matrix = self.vectorizer.transform(" ".join(d) for d in self.get_text_chunks())  # type: ignore

        self.length: int = self.vectors.shape[0]  # type: ignore
        self.dim: int = self.vectors.shape[1]  # type: ignore

    def __getitem__(self, item: int) -> csr_matrix:
        return self.vectors[item]

    def inner_compare(self, min_similarity: float) -> Matches:
        """Compare corpus with itself"""
        results: np.ndarray = linear_kernel(self.vectors, dense_output=False)  # type: ignore
        return Matches(self.process_inner_compare(results, min_similarity))

    def outer_compare(self, target_corpus: Corpus, min_similarity: float) -> Matches:
        """Compare corpus with another corpus"""
        print("Comparing source collection to target collection...", flush=True)
        results: np.ndarray = linear_kernel(self.vectors, target_corpus.vectors, dense_output=False)  # type: ignore
        return Matches(self.process_outer_compare(results, target_corpus, min_similarity))


class WmdCorpus(Corpus):
    """Corpus object which builds doc embeddings from word2vec for Word Mover's Distance"""

    def __init__(
        self,
        texts: Iterable[Tokens],
        w2v_model: Optional[KeyedVectors] = None,
        text_object_definition: str = "n_token",
        min_text_obj_length: int = 15,
        n_chunk: int = 3,
        text_object_level_split: str = "doc",
    ):
        super().__init__(
            texts,
            text_object_definition=text_object_definition,
            min_text_obj_length=min_text_obj_length,
            n_chunk=n_chunk,
            text_object_level_split=text_object_level_split,
        )
        text_chunks = list(self.get_text_chunks())
        if w2v_model is not None:
            self.wmd_index: WmdSimilarity = WmdSimilarity(text_chunks, w2v_model)
        else:
            self.text_chunks = text_chunks

    def __getitem__(self, item: int) -> np.ndarray:
        return self.wmd_index.index[item]

    def inner_compare(self, min_similarity: float) -> Matches:
        """Compare corpus with itself"""
        results: csr_matrix = csr_matrix(self.wmd_index[self.wmd_index.index])  # type: ignore
        return Matches(self.process_inner_compare(results, min_similarity))

    def outer_compare(self, target_corpus: WmdCorpus, min_similarity: float) -> Matches:
        """Compare corpus with another corpus"""
        print("Comparing source collection to target collection...", flush=True)
        results: np.ndarray = self.wmd_index[target_corpus.text_chunks]  # type: ignore
        return Matches(self.process_outer_compare(results, target_corpus, min_similarity))


class Word2VecEmbeddingCorpus(Corpus):
    """Corpus object which builds doc embeddings using the average of token word2vec vectors"""

    def __init__(
        self,
        texts: Iterable[Tokens],
        model: Union[str, spacy.Language],
        batch_size: int,
        text_object_definition: str = "n_token",
        min_text_obj_length: int = 15,
        n_chunk: int = 3,
        text_object_level_split: str = "doc",
        direction: str = "source",
    ):
        super().__init__(
            texts,
            text_object_definition=text_object_definition,
            min_text_obj_length=min_text_obj_length,
            n_chunk=n_chunk,
            text_object_level_split=text_object_level_split,
        )
        if isinstance(model, str):
            self.model = spacy.load(model)
        else:
            self.model = model
        self.docs = DocumentChunks(
            self.get_text_chunks(), save_path=direction, return_type="str", transform_function=self.create_embeddings
        )
        self.length = len(self.docs)
        self.batch_size = batch_size
        self.chunk_size = floor((self.length + self.batch_size - 1) / self.batch_size)

    def create_embeddings(self, text_chunk):
        """Create document embeddings"""
        return self.model(" ".join(text_chunk))

    def __compare(self, target_corpus: Optional[Word2VecEmbeddingCorpus] = None) -> np.ndarray:
        results: np.ndarray
        if target_corpus is None:
            target_corpus = self
        results = np.zeros((self.length, len(target_corpus)))
        target_len = target_corpus.length
        start_pos = 0
        with tqdm(total=self.length * target_len, leave=False) as pbar:
            for source_embeddings in self.create_batch():
                end_pos = start_pos + len(source_embeddings)
                target_start_pos = 0
                target_batches = target_corpus.create_batch()
                partial_results = list(
                    map(
                        lambda doc: [
                            np.dot(doc.vector, inner_doc.vector) / (doc.vector_norm * inner_doc.vector_norm)
                            for target_embeddings in target_batches
                            for inner_doc in target_embeddings
                        ],
                        source_embeddings,
                    )
                )
                for partial_result in partial_results:
                    target_end_pos = target_start_pos + len(partial_result)
                    results[start_pos:end_pos, target_start_pos:target_end_pos] = partial_result
                    target_start_pos = target_end_pos
                    pbar.update(self.length * len(partial_result))
                start_pos = end_pos
        return results

    def create_batch(self) -> Generator[spacy.tokens.doc.Doc, None, None]:  # type:ignore
        for i in range(0, self.length, self.chunk_size):
            yield self.docs[i : i + self.chunk_size]  # type: ignore

    def __getitem__(self, item: int) -> List[str]:
        return self.docs[item]  # type: ignore

    def __len__(self):
        return self.length

    def inner_compare(self, min_similarity: float) -> Iterable[MergedGroup]:
        """Compare corpus with itself"""
        print("Comparing source collection to itself...", flush=True)
        results = self.__compare()
        return Matches(self.process_inner_compare(results, min_similarity))

    def outer_compare(self, target_corpus: Word2VecEmbeddingCorpus, min_similarity: float) -> Matches:
        """Compare corpus with another corpus"""
        print("Comparing source collection to target collection...", flush=True)
        results = self.__compare(target_corpus=target_corpus)
        return Matches(self.process_outer_compare(results, target_corpus, min_similarity))


class SentenceEmbeddingsCorpus(Corpus):
    """Corpus object which builds doc embeddings using sentence transformers for similarity"""

    def __init__(
        self,
        texts: Iterable[Tokens],
        batch_size: int,
        text_object_definition: str = "n_token",
        min_text_obj_length: int = 15,
        n_chunk: int = 3,
        text_object_level_split: str = "doc",
        direction="source",
        model=None,
    ):
        super().__init__(
            texts,
            text_object_definition=text_object_definition,
            min_text_obj_length=min_text_obj_length,
            n_chunk=n_chunk,
            text_object_level_split=text_object_level_split,
        )

        if model is None:
            self.model = SentenceTransformer("inokufu/flaubert-base-uncased-xnli-sts")
        else:
            self.model = model

        self.docs = DocumentChunks(
            self.get_text_chunks(), save_path=direction, return_type="str", transform_function=self.create_embeddings
        )
        self.length = len(self.docs)
        self.batch_size = batch_size
        self.chunk_size = floor((self.length + self.batch_size - 1) / self.batch_size)
        with open(f"{direction}_metadata.json", "w") as output:
            json.dump(self.metadata, output)

    def create_batch(self) -> Generator[torch.Tensor, None, None]:
        for i in range(0, self.length, self.chunk_size):
            yield self.docs[i : i + self.chunk_size]  # type: ignore

    def __getitem__(self, item: int) -> List[str]:
        return self.docs[item]  # type: ignore

    def __len__(self):
        return self.length

    def create_embeddings(self, text_chunks) -> torch.Tensor:
        """Create document embedding"""
        embeddings: torch.Tensor = self.model.encode(list(text_chunks), convert_to_tensor=True)  # type: ignore
        return embeddings

    def similarity(self, embeddings_1, embeddings_2) -> np.ndarray:
        """Compute similarity between embeddings"""
        results = util.cos_sim(embeddings_1, embeddings_2)
        return results.cpu().numpy()

    def __compare(self, target_corpus: Optional[SentenceEmbeddingsCorpus] = None) -> np.ndarray:
        results: np.ndarray
        if target_corpus is None:
            target_corpus = self
        if self.batch_size == 1 and target_corpus.batch_size:
            results = self.similarity(self.docs[0 : self.length], target_corpus.docs[0 : target_corpus.length])
        else:
            results = np.zeros((self.length, len(target_corpus)))
            target_len = target_corpus.length
            target_chunk_size = target_corpus.chunk_size
            start_pos = 0
            count = 0
            with tqdm(total=self.length * target_len, leave=False) as pbar:
                for source_embeddings in self.create_batch():
                    end_pos = start_pos + source_embeddings.shape[0]
                    target_start_pos = 0
                    source_torch_embeddings = source_embeddings
                    for target_embeddings in target_corpus.create_batch():
                        target_end_pos = target_start_pos + target_embeddings.shape[0]
                        partial_results: np.ndarray = self.similarity(source_torch_embeddings, target_embeddings)  # type: ignore
                        results[start_pos:end_pos, target_start_pos:target_end_pos] = partial_results
                        target_start_pos = target_end_pos
                        count += target_chunk_size
                        pbar.update(self.length * target_embeddings.shape[0])  # type: ignore
                    start_pos = end_pos
        return results

    def inner_compare(self, min_similarity: float) -> Matches:
        """Compare corpus with itself"""
        print("Comparing source collection to itself...", flush=True)
        results = self.__compare()
        return Matches(self.process_inner_compare(results, min_similarity))

    def outer_compare(self, target_corpus: SentenceEmbeddingsCorpus, min_similarity: float) -> Matches:
        """Compare corpus with another corpus"""
        print("Comparing source collection to target collection...", flush=True)
        results = self.__compare(target_corpus=target_corpus)
        return Matches(self.process_outer_compare(results, target_corpus, min_similarity))


def clean_text(text: str) -> str:
    """Cleaning text function which removes tags and converts entities"""
    text = TAGS.sub("", text)
    text = unescape_xml(text)
    text = unescape_html(text)
    text = text.replace("\n", " ")
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


def jaccard_sim(X, Y):
    """Jaccard Similarity"""
    assert X.shape[1] == Y.shape[1]

    X = X.astype(bool).astype(int)
    Y = Y.astype(bool).astype(int)

    intersect = X.dot(Y.T)

    x_sum = X.sum(axis=1).A1
    y_sum = Y.sum(axis=1).A1
    xx, yy = np.meshgrid(x_sum, y_sum)
    union = (xx + yy).T - intersect
    return (intersect / union).A


def evaluate_score(start_score: float, new_score: float, min_score: float) -> bool:
    """Evaluate if new score is within 3/4 of start score"""
    # TODO: evaluate whether Jaccard sim should have same score as Cosine sim
    if new_score >= min_score or new_score / start_score > 0.75:
        return True
    return False


def get_docs_with_matches(matches: Matches) -> Dict[str, Tuple[Tokens, Dict[int, int], Dict[int, int]]]:
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


def merge_passages(
    matches: Matches,
    corpus: Union[TfIdfCorpus, WmdCorpus, SentenceEmbeddingsCorpus],
    min_score: float,
) -> List[MergedGroup]:
    """Merge all passages into bigger passages"""
    # pylint: disable=E1101
    # TODO: should merging be done using Jaccard sim metric: to avoid sparsity
    docs_with_matches: Dict[str, Tuple[Tokens, Dict[int, int], Dict[int, int]]] = get_docs_with_matches(matches)
    last_count = len(matches)
    current_count = last_count + 1
    iteration = 1
    merged_matches: List[MergedGroup] = Matches.load().matches  # type:ignore
    print(f"Merging matches: {last_count} matches before iteration 1", end="", flush=True)
    while last_count / current_count <= 1.0:  # we stop iterating if there are minimal change between iterations
        last_count = current_count
        merged_matches = sorted(
            merged_matches,
            key=lambda x: (
                x.source.filename,
                x.target.filename,
                x.source.start_byte,
                x.source.start_byte - x.source.end_byte,
                x.target.start_byte,
                x.target.start_byte - x.target.end_byte,
            ),
        )  # sort by smaller start byte and bigger end_byte
        merged_group: MergedGroup = MergedGroup()
        saved_groups: List[MergedGroup] = []
        total_matches: int = len(matches)
        start_score: float = min_score

        for pos, match in enumerate(merged_matches):
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
                    if isinstance(corpus, TfIdfCorpus):
                        source_vector: csr_matrix = corpus.vectorizer.transform([" ".join(source_tokens)])  # type: ignore
                        score_array: np.ndarray = jaccard_sim(source_vector, merged_group.target.vector)  # type: ignore
                        new_score: float = score_array[0, 0]
                    elif isinstance(corpus, WmdCorpus):
                        source_vector = WmdSimilarity([t.text for t in source_tokens], corpus.wmd_index.wv)  # type: ignore
                        new_score = source_vector[[t.text for t in target_tokens]]  # type: ignore
                    else:
                        source_vector = corpus.create_embeddings([" ".join(source_tokens)])
                        new_score = util.cos_sim(source_vector, merged_group.target.vector)
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
                    if isinstance(corpus, TfIdfCorpus):
                        target_vector: csr_matrix = corpus.vectorizer.transform([" ".join(target_tokens)])  # type: ignore
                        score_array: np.ndarray = jaccard_sim(merged_group.source.vector, target_vector)  # type: ignore
                        new_score: float = score_array[0, 0]
                    elif isinstance(corpus, WmdCorpus):
                        new_score = merged_group.source.vector[[t.text for t in target_tokens]]  # type: ignore
                    else:
                        target_vector = corpus.create_embeddings([" ".join(target_tokens)])
                        new_score = util.cos_sim(target_vector, merged_group.source.vector)
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
        merged_matches = saved_groups
        iteration += 1
        print(len(saved_groups), iteration)
        current_count = len(saved_groups)
        print(f"\rMerging matches: {current_count} matches after iteration {iteration+1}...", end="", flush=True)
    print(flush=True)
    return merged_matches


def optimize_match(
    tokens: Iterable[Token],
    intersection: Set[str],
    passage_group: PassageGroup,
    corpus: Union[TfIdfCorpus, WmdCorpus, SentenceEmbeddingsCorpus],
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
    if isinstance(corpus, TfIdfCorpus):
        new_vector: csr_matrix = corpus.vectorizer.transform([" ".join(tokens[start:end])])  # type: ignore
    elif isinstance(corpus, WmdCorpus):
        new_vector = WmdSimilarity([t.text for t in tokens], corpus.wmd_index.wv)  # type: ignore
    else:
        new_vector = corpus.create_embeddings([" ".join(tokens)])
    if new_vector is not None:
        passage_group.start_byte = new_start_byte
        passage_group.metadata["start_byte"] = new_metadata_start_byte
        passage_group.vector = new_vector
        passage_group.end_byte = tokens[end].ext["end_byte"]  # type: ignore
        passage_group.metadata["end_byte"] = tokens[end].ext["end_byte"]  # type: ignore
    return passage_group


def optimize_matches(
    matches: List[MergedGroup], corpus: Union[TfIdfCorpus, WmdCorpus, SentenceEmbeddingsCorpus], min_matching_words: int
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
        if len(intersection) >= min_matching_words or isinstance(corpus, (WmdCorpus, SentenceEmbeddingsCorpus)):
            source = optimize_match(source_tokens, intersection, match.source, corpus)
            target = optimize_match(target_tokens, intersection, match.target, corpus)
            best_score_matrix: np.ndarray = linear_kernel(target.vector, source.vector)  # type: ignore
            best_score: float = best_score_matrix[0, 0]
            optimized_matches.append(MergedGroup(source, target, best_score))
            match_count += 1
    if len(matches) != match_count:
        print(
            f"{match_count} matches remaining: {len(matches)-match_count} matches were dropped due to low number of unique matching words."
        )
    return optimized_matches, docs_with_matches


def get_tokens(
    passage: PassageGroup, preproc: PreProcessor, doc: Tuple[Tokens, Dict[int, int], Dict[int, int]]
) -> List[Token]:
    """Get tokens"""
    text: str = " "
    start_byte: int = passage.start_byte
    end_byte: int = passage.end_byte
    with open(passage.filename, "rb") as text_file:
        text_file.seek(start_byte)
        text = text_file.read(end_byte - start_byte).decode("utf8", "ignore")
    tokens: List[Token] = []
    full_tokens, start_bytes, _ = doc
    pos = 0
    for token in preproc.process_string(text):
        pos += 1
        end_byte = start_byte + len(token.surface_form.encode("utf8"))
        # if start_byte >= passage.start_byte and end_byte <= passage.end_byte:
        surface_form = token.surface_form.replace("\n", " ")
        try:
            token = full_tokens[start_bytes[start_byte]]
        except KeyError:  # Token was not indexed at parse time
            pass
        token.surface_form = surface_form
        tokens.append(token)
        # if end_byte > passage.end_byte:
        #     break
        start_byte = end_byte
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
            source_passage_with_matches.append(
                f'&lt;span class="filtered-token"&gt;{token.surface_form or " "}&lt;/span&gt;'
            )
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
) -> Tuple[TfIdfCorpus, Matches]:
    """Cosine similarity of TF-IDF vectors"""
    source_corpus: TfIdfCorpus = TfIdfCorpus(
        source_texts,
        text_object_definition=source_config["text_object_definition"],
        min_text_obj_length=source_config["min_text_object_length"],
        n_chunk=source_config["n_chunk"],
        text_object_level_split=source_config["text_object_level_split"],
        min_freq=source_config["min_freq"],
        max_freq=source_config["max_freq"],
    )
    if target_texts is not None:
        target_corpus: TfIdfCorpus = TfIdfCorpus(
            target_texts,
            text_object_definition=target_config["text_object_definition"],
            vectorizer=source_corpus.vectorizer,
            min_text_obj_length=target_config["min_text_object_length"],
            n_chunk=target_config["n_chunk"],
            text_object_level_split=target_config["text_object_level_split"],
        )

        matching_docs = source_corpus.outer_compare(target_corpus, min_similarity)
    else:
        matching_docs = source_corpus.inner_compare(min_similarity)
        target_corpus = source_corpus
    return source_corpus, matching_docs


def word_movers_similarity(
    source_texts: Iterable[Tokens],
    source_config: Dict[str, Any],
    target_config: Dict[str, Any],
    min_similarity: float,
    target_texts: Optional[Iterable[Tokens]] = None,
) -> Tuple[WmdCorpus, Iterable[MergedGroup]]:
    """Word-movers distance"""
    w2v_model = KeyedVectors.load(source_config["w2v_model_path"], mmap="r")
    source_corpus: WmdCorpus = WmdCorpus(
        source_texts,
        text_object_definition=source_config["text_object_definition"],
        min_text_obj_length=source_config["min_text_object_length"],
        n_chunk=source_config["n_chunk"],
        text_object_level_split=source_config["text_object_level_split"],
        w2v_model=w2v_model,
    )
    if target_texts is not None:
        target_corpus: WmdCorpus = WmdCorpus(
            target_texts,
            text_object_definition=target_config["text_object_definition"],
            min_text_obj_length=target_config["min_text_object_length"],
            n_chunk=target_config["n_chunk"],
            text_object_level_split=target_config["text_object_level_split"],
        )
        matching_docs = source_corpus.outer_compare(
            target_corpus,
            config["min_similarity"],
        )
    else:
        matching_docs = source_corpus.inner_compare(
            config["min_similarity"],
        )
        target_corpus = source_corpus
    return source_corpus, matching_docs


def sentence_embed_similarity(
    source_texts: Iterable[Tokens],
    source_config: Dict[str, Any],
    target_config: Dict[str, Any],
    min_similarity: float,
    source_batch: int,
    target_texts: Optional[Iterable[Tokens]] = None,
    target_batch: int = 1,
) -> Tuple[SentenceEmbeddingsCorpus, Matches]:
    """Cosine similarity of sentence embeddings from transformer models"""
    source_corpus: SentenceEmbeddingsCorpus = SentenceEmbeddingsCorpus(
        source_texts,
        source_batch,
        text_object_definition=source_config["text_object_definition"],
        min_text_obj_length=source_config["min_text_object_length"],
        n_chunk=source_config["n_chunk"],
        text_object_level_split=source_config["text_object_level_split"],
    )
    if target_texts is not None:
        target_corpus: SentenceEmbeddingsCorpus = SentenceEmbeddingsCorpus(
            target_texts,
            target_batch,
            direction="target",
            text_object_definition=target_config["text_object_definition"],
            min_text_obj_length=target_config["min_text_object_length"],
            n_chunk=target_config["n_chunk"],
            text_object_level_split=target_config["text_object_level_split"],
            model=source_corpus.model,
        )
        matching_docs = source_corpus.outer_compare(target_corpus, min_similarity)
    else:
        matching_docs = source_corpus.inner_compare(min_similarity)
        target_corpus = source_corpus
    return source_corpus, matching_docs


def word2vec_embed_similarity(
    source_texts: Iterable[Tokens],
    source_config: Dict[str, Any],
    target_config: Dict[str, Any],
    min_similarity: float,
    source_batch: int,
    target_texts: Optional[Iterable[Tokens]] = None,
    target_batch: int = 1,
) -> Tuple[Word2VecEmbeddingCorpus, Iterable[MergedGroup]]:
    """Cosine similarity of sentence embeddings using mean w2v vectors"""
    source_corpus: Word2VecEmbeddingCorpus = Word2VecEmbeddingCorpus(
        source_texts,
        source_config["spacy_model_name"],
        source_batch,
        text_object_definition=source_config["text_object_definition"],
        min_text_obj_length=source_config["min_text_object_length"],
        n_chunk=source_config["n_chunk"],
        text_object_level_split=source_config["text_object_level_split"],
    )
    if target_texts is not None:
        target_corpus: Word2VecEmbeddingCorpus = Word2VecEmbeddingCorpus(
            target_texts,
            source_corpus.model,
            target_batch,
            direction="target",
            text_object_definition=target_config["text_object_definition"],
            min_text_obj_length=target_config["min_text_object_length"],
            n_chunk=target_config["n_chunk"],
            text_object_level_split=target_config["text_object_level_split"],
        )
        matching_docs = source_corpus.outer_compare(target_corpus, min_similarity)
    else:
        matching_docs = source_corpus.inner_compare(min_similarity)
        target_corpus = source_corpus
    return source_corpus, matching_docs


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
    config["source"]["strip_tags"] = True  # this is useful for post-processing passages where we have tags included.
    config["target"]["strip_tags"] = True
    source_preproc: PreProcessor = PreProcessor(is_philo_db=True, workers=workers, **config["source"])
    source_texts: Iterable[Tokens] = source_preproc.process_texts(
        (file.path for file in os.scandir(source_path)), keep_all=True, progress=False
    )
    target_preproc: PreProcessor = PreProcessor(is_philo_db=True, workers=workers, **config["target"])
    target_texts: Iterable[Tokens] = target_preproc.process_texts(
        (file.path for file in os.scandir(target_path)), keep_all=True, progress=False
    )

    if config["source"]["vectorization"] == "tfidf":
        source_corpus, matches = simple_similarity(
            source_texts,
            config["source"],
            config["target"],
            config["min_similarity"],
            target_texts=target_texts,
        )
    elif config["source"]["vectorization"] == "wmd":
        source_corpus, matches = word_movers_similarity(
            source_texts,
            config["source"],
            config["target"],
            config["min_similarity"],
            target_texts=target_texts,
        )
    elif config["source"]["vectorization"] == "sent_embed":
        source_corpus, matches = sentence_embed_similarity(
            source_texts,
            config["source"],
            config["target"],
            config["min_similarity"],
            config["source_batch"],
            target_texts=target_texts,
            target_batch=config["target_batch"],
        )
    else:
        source_corpus, matches = word2vec_embed_similarity(
            source_texts,
            config["source"],
            config["target"],
            config["min_similarity"],
            config["source_batch"],
            target_texts=target_texts,
            target_batch=config["target_batch"],
        )
    print(f"{len(matches)} matches found.")
    if config["source"]["vectorization"] not in ("embed", "sent_embed"):
        print("\n### Post-processing results ###", flush=True)
        merged_matches = merge_passages(
            matches,
            source_corpus,
            config["min_similarity"],
        )
        matches, docs_with_matches = optimize_matches(merged_matches, source_corpus, config["min_matching_words"])
    else:
        docs_with_matches = get_docs_with_matches(matches)

    print("Formatting and writing out processed results...(this may take some time)")
    os.system("mkdir -p output/results")
    source_preproc.options = {
        **source_preproc.options,
        "strip_tags": False,
        "with_pos": False,
        "spacy_lemmatizer": False,
    }
    source_preproc.pos_to_keep = set()
    target_preproc.options = {
        **target_preproc.options,
        "strip_tags": False,
        "with_pos": False,
        "spacy_lemmatizer": False,
    }
    target_preproc.pos_to_keep = set()
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
            if config["source"]["vectorization"] == "tfidf":
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
