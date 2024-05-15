"""Passage similarity detection"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from abc import ABC
from collections import deque
from html import unescape as unescape_html
from shutil import rmtree
from typing import Any, Callable, Iterable, Optional
from xml.sax.saxutils import unescape as unescape_xml

import dill as pickle
import lz4.frame
import msgspec
import numpy as np
import spacy
import torch
from msgspec import field
from scipy.sparse import csr_matrix
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from spacy.tokens import Doc
from text_preprocessing import PreProcessor, Token, Tokens
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TAGS = re.compile(r"<[^>]+>")
PHILO_TEXT_OBJECT_LEVELS = {"doc": 1, "div1": 2, "div2": 3, "div3": 4, "para": 5, "sent": 6, "word": 7}
TEMP_DIR = os.getcwd()

class PassageGroup(msgspec.Struct, array_like=True):
    """Text passage with all associated properties and vector representation"""

    start_byte: int = 0
    end_byte: int = 0
    filename: str = ""
    metadata: dict = {}


class MergedGroup(msgspec.Struct, array_like=True):
    """A source and target PassageGroup pair with similarity"""

    source: PassageGroup = field(default_factory=PassageGroup)
    target: PassageGroup = field(default_factory=PassageGroup)
    similarity: float = 0.0

ENCODER = msgspec.msgpack.Encoder()
DECODER = msgspec.msgpack.Decoder(type=MergedGroup)

class DocumentChunks:
    """A generator with caching"""

    def __init__(self, docs: Iterable[list[str]], save_path: str, transform_function: Callable):
        self.docs = docs
        self.doc_list: list[list[str]] = []
        self.doc_count = 0
        self.generator_exhausted = False
        self.transform_function = transform_function
        self.corpus_type = self.transform_function.__qualname__.split(".")[0]
        self.path = os.path.join(TEMP_DIR, "output/chunks/", save_path)
        if os.path.exists(self.path):
            rmtree(self.path)
        os.makedirs(self.path, exist_ok=True)

    def __iter__(self) -> Iterable[str | list[str] | torch.Tensor | np.ndarray]:
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

    def __save(self, doc: list[str] | str):
        filename = os.path.join(self.path, str(self.doc_count))
        if self.transform_function is None:
            with open(filename, "wb") as output_file:
                pickle.dump(doc, output_file)
        transformed_doc = self.transform_function([doc])
        if self.corpus_type == "TransformerCorpus":
            torch.save(transformed_doc, f"{filename}.pt")
        else:
            np.save(f"{filename}.npy", transformed_doc)

    def __load(self, doc_name) -> list[str] | torch.Tensor | np.ndarray:
        filename = os.path.join(self.path, str(doc_name))
        if self.transform_function is None:
            with open(filename, "rb") as input_file:
                doc = pickle.load(input_file)
            return doc
        elif self.corpus_type == "TransformerCorpus":
            return torch.load(f"{filename}.pt")
        return np.load(f"{filename}.npy")[0]

    def __get_doc(self, index: int) -> list[str] | torch.Tensor | np.ndarray:
        doc = None
        while index > self.doc_count:
            try:
                doc = next(self.docs)
                self.__format_doc(doc)
                self.__save(doc)
                self.doc_count += 1
            except StopIteration as e:
                raise IndexError from e
        if doc is None:
            return self.__load(index)
        return doc

    def __getitem__(self, item: int | slice) -> list[str] | str | list[list[str] | str] | np.ndarray | torch.Tensor:
        if isinstance(item, slice):
            end = item.stop
            if item.stop > len(self):  # avoid index out of range
                end = len(self)
            if self.transform_function is None or self.corpus_type == "Word2VecEmbeddingCorpus":
                return np.array([self.__get_doc(index) for index in range(item.start, end)])
            return torch.cat([self.__get_doc(index) for index in range(item.start, end)])  # type:ignore
        return self.__get_doc(item)

    def __format_doc(self, doc: list[str]) -> str:
        return " ".join(doc)

    def __len__(self):
        if self.generator_exhausted is False:
            for _ in self:
                pass
        return self.doc_count


class Matches:
    """Matches cached to disk"""

    def __init__(self, matches: Iterable[MergedGroup]):
        self.path = os.path.join(TEMP_DIR, "output/results/matches")
        os.makedirs(self.path, exist_ok=True)
        self.count = 0
        if isinstance(matches, list) and matches:
            self.matches = matches
            self.is_cached = False
            self.count = len(self.matches)
        else:
            self.conn = sqlite3.connect(os.path.join(self.path, "matches.db"))
            self.cursor = self.conn.cursor()
            self.cursor.execute("DROP TABLE IF EXISTS matches")
            self.cursor.execute("CREATE TABLE matches (match_id INTEGER, match blob)")
            self.cursor.execute("CREATE INDEX match_id_index ON matches (match_id)")
            self.matches = None
            self.is_cached = True
            self.count = self.__save(matches)  # save generator to disk

    def match_generator(self, new_matches):
        for match in new_matches:
            dump = ENCODER.encode(match)
            yield (self.count, dump)
            self.count += 1

    def extend(self, new_matches: Iterable[MergedGroup]):
        """Add new matches to existing matches"""
        encoded_matches = self.match_generator(new_matches)
        self.cursor.executemany("INSERT INTO matches VALUES (?, ?)", encoded_matches)

    def __save(self, matches):
        count = 0
        for count, match in enumerate(matches):
            dump = ENCODER.encode(match)
            self.cursor.execute("INSERT INTO matches VALUES (?, ?)", (self.count, dump))
        if count == 0:
            return 0
        self.conn.commit()
        return count + 1

    def done(self):
        """Commit changes to database"""
        self.conn.commit()
        self.conn.close()

    @classmethod
    def load(cls):
        """Load instance of class by reading previously cached matches"""
        matches = []
        conn = sqlite3.connect(os.path.join(TEMP_DIR, "output/results/matches/matches.db"))
        cursor = conn.cursor()
        cursor.execute("SELECT match from matches ORDER BY match_id")
        for match in cursor:
            matches.append(DECODER.decode(match[0]))
        conn.close()
        return cls(matches)

    def __len__(self):
        return self.count

    def __iter__(self):
        if self.is_cached is False:
            for index in range(self.count):
                yield self.matches[index] # type: ignore
        else:
            self.cursor.execute("SELECT match FROM matches ORDER BY match_id")
            for match in self.cursor:
                yield DECODER.decode(match[0])


class Corpus(ABC):
    """A Corpus of passages as preprocessed by the text-preprocessor"""

    def __init__(
        self,
        texts: Iterable[Tokens],
        output_path: str,
        similarity_function: Callable,
        min_text_obj_length: int = 15,
        n_chunk: int = 3,
        text_object_type_split: str = "doc",
        direction="source",
        n_batches=1,
    ):
        """Initialize Corpus Object"""
        self.texts: Iterable[Tokens] = texts
        self.min_text_obj_length: int = min_text_obj_length
        self.n_chunk: int = n_chunk
        self.metadata: list[dict[str, Any]] = []
        self.text_object_type_split = text_object_type_split
        self.output_dir = os.path.abspath(output_path)
        self.direction: str = direction
        os.makedirs(os.path.join(self.output_dir, self.direction, "saved_docs"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, self.direction, "doc_chunks"), exist_ok=True)
        self.length = 0
        self.n_batches = n_batches
        self.similarity = similarity_function
        self.docs: DocumentChunks
        self.max_tokens = float("inf")
        self.device = "cpu"

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, _):
        pass

    def get_text_chunks(self) -> Iterable[list[str]]:
        """Process all texts into smaller text chunks"""
        chunk_group: deque[Tokens] = deque(maxlen=self.n_chunk)
        min_chunk_length: int = self.n_chunk * self.min_text_obj_length
        current_text_level_id: str = "0"
        full_doc = Tokens([], {})
        current_doc_id = None
        chunks_done = 0
        docs = {}
        current_chunk_group_length = 0
        for text in self.texts:
            docs[text.metadata["philo_id"]] = " ".join([t.text for t in text])
            print(f"\rProcessing {self.direction} texts... {chunks_done} text chunks extracted...", end="", flush=True)
            text.metadata["parsed_filename"] = os.path.join(
                self.output_dir,
                self.direction,
                "saved_docs",
                os.path.basename(text.metadata["parsed_filename"].replace(".lz4", "")),
            )
            doc_id = text.metadata["philo_id"].split()[0]
            if (
                doc_id != current_doc_id and current_doc_id is not None
            ):  # we save the current doc when doc_ids don't match
                full_doc.save(full_doc.metadata["parsed_filename"])
                full_doc = Tokens([], text.metadata)
            full_doc.extend(text)
            text.purge()
            text_level_id: str = " ".join(
                text.metadata["philo_id"].split()[: PHILO_TEXT_OBJECT_LEVELS[self.text_object_type_split]]
            )
            if text_level_id != current_text_level_id:
                if current_chunk_group_length >= min_chunk_length:
                    text_chunk = self.__build_text_chunk(chunk_group)
                    if text_chunk:  # make sure this chunk is not empty
                        chunks_done += 1
                        yield text_chunk
                chunk_group.clear()
            current_text_level_id = text_level_id
            current_chunk_group_length = sum([len(t.tokens) for t in chunk_group])
            text_length = len(text)
            if current_chunk_group_length + text_length > self.max_tokens and current_chunk_group_length:
                chunks_done += 1
                yield self.__build_text_chunk(chunk_group)
            if text_length < self.min_text_obj_length:
                try:
                    chunk_group[-1].extend(text)
                    continue
                except IndexError:
                    pass
            chunk_group.append(text)
            if len(chunk_group) == self.n_chunk:
                current_chunk_group_length = sum([len(t) for t in chunk_group])
                if current_chunk_group_length >= min_chunk_length:
                    chunks_done += 1
                    yield self.__build_text_chunk(chunk_group)
            current_doc_id = doc_id
        full_doc.save(full_doc.metadata["parsed_filename"])
        print()

    def __build_text_chunk(self, chunk_group: deque[Tokens]) -> list[str]:
        """Build chunks from a group of text objects"""
        chunk = [t for c in chunk_group for t in c]
        self.metadata.append(
            {
                **chunk_group[0].metadata,
                "start_byte": chunk[0].ext["start_byte"],
                "end_byte": chunk[-1].ext["end_byte"],
            }
        )
        return [t.text for t in chunk]

    def __compare(self, target_corpus=None) -> np.ndarray:
        """Compare the corpus to another corpus"""
        results: np.ndarray
        if target_corpus is None:
            target_corpus = self
        results = self.similarity(self.docs[0 : self.length], target_corpus.docs[0 : target_corpus.length])  # type: ignore
        return results

    def __batched_compare(self, min_similarity: float, target_corpus: Corpus | None = None) -> Matches:
        """Compare the corpus to another corpus"""
        inner_compare = False
        if target_corpus is None:
            target_corpus = self
            inner_compare = True
        source_batch_size = int(np.ceil(self.length / self.n_batches))
        target_batch_size = int(np.ceil(target_corpus.length / target_corpus.n_batches))
        matches: Matches = Matches([])
        with tqdm(total=self.length * target_corpus.length, leave=False) as pbar:
            for outer_start_index in range(0, self.length, source_batch_size):
                outer_end_index = outer_start_index + source_batch_size
                source_embeddings = self.docs[outer_start_index:outer_end_index]
                for inner_start_index in range(0, target_corpus.length, target_batch_size):
                    inner_end_index = inner_start_index + target_batch_size
                    target_embeddings = target_corpus.docs[inner_start_index:inner_end_index]
                    partial_results: np.ndarray = self.similarity(source_embeddings, target_embeddings)
                    if inner_compare is False:
                        processed_results = self.process_outer_compare(partial_results, target_corpus, min_similarity, outer_start_index=outer_start_index, inner_start_index=inner_start_index)
                    else:
                        processed_results = self.process_inner_compare(partial_results, min_similarity, outer_start_index=outer_start_index, inner_start_index=inner_start_index)
                    matches.extend(processed_results)
                    pbar.update(source_batch_size*target_batch_size)
        matches.done()
        return matches

    def inner_compare(self, min_similarity: float) -> Matches:
        """Compare corpus with itself"""
        print("Comparing source collection to itself...", flush=True)
        if self.n_batches == 1:
            results = self.__compare()
            return Matches(self.process_inner_compare(results, min_similarity))
        return self.__batched_compare(min_similarity)

    def outer_compare(self, target_corpus, min_similarity: float) -> Matches:
        """Compare corpus with another corpus"""
        print("Comparing source collection to target collection...", flush=True)
        if self.n_batches == 1 and target_corpus.n_batches == 1:
            results = self.__compare(target_corpus=target_corpus)
            return Matches(self.process_outer_compare(results, target_corpus, min_similarity))
        return self.__batched_compare(min_similarity, target_corpus=target_corpus)

    def process_inner_compare(self, results, min_similarity: float, outer_start_index=0, inner_start_index=0) -> Iterable[MergedGroup]:
        """Compare corpus with itself"""
        print("Processing similarity results...", flush=True, end=" ")
        for outer_doc_id, inner_doc_id in np.argwhere(results >= min_similarity):
            outer_doc_id += outer_start_index
            inner_doc_id += inner_start_index
            if (
                self.metadata[outer_doc_id]["year"] <= self.metadata[inner_doc_id]["year"]
                and inner_doc_id != outer_doc_id
            ):
                yield MergedGroup(
                    PassageGroup(
                        self.metadata[outer_doc_id]["start_byte"],
                        self.metadata[outer_doc_id]["end_byte"],
                        self.metadata[outer_doc_id]["filename"],
                        self.metadata[outer_doc_id],
                    ),
                    PassageGroup(
                        self.metadata[inner_doc_id]["start_byte"],
                        self.metadata[inner_doc_id]["end_byte"],
                        self.metadata[inner_doc_id]["filename"],
                        self.metadata[inner_doc_id],
                    ),
                    float(results[outer_doc_id, inner_doc_id]),  # type: ignore
                )

    def process_outer_compare(
        self, results: np.ndarray, target_corpus: Corpus, min_similarity, outer_start_index=0, inner_start_index=0
    ) -> Iterable[MergedGroup]:
        """Compare corpus with another corpus"""
        print("Processing similarity results...", flush=True, end=" ")
        for outer_doc_id, inner_doc_id in np.argwhere(results >= min_similarity):
            outer_index = outer_doc_id + outer_start_index
            inner_index = inner_doc_id + inner_start_index
            yield MergedGroup(
                PassageGroup(
                    self.metadata[outer_index]["start_byte"],
                    self.metadata[outer_index]["end_byte"],
                    self.metadata[outer_index]["filename"],
                    self.metadata[outer_index],
                ),
                PassageGroup(
                    target_corpus.metadata[inner_index]["start_byte"],
                    target_corpus.metadata[inner_index]["end_byte"],
                    target_corpus.metadata[inner_index]["filename"],
                    target_corpus.metadata[inner_index],
                ),
                float(results[outer_doc_id, inner_doc_id]),  # type: ignore
            )


class TfIdfCorpus(Corpus):
    """Corpus object which builds TF-IDF vectors"""

    def __init__(
        self,
        texts: Iterable[Tokens],
        output_path: str,
        min_text_obj_length: int = 15,
        n_chunk: int = 3,
        text_object_type_split: str = "doc",
        vectorizer: Optional[TfidfVectorizer] = None,
        min_freq: int | float = 1,
        max_freq: float = 1.0,
        direction="source",
    ):
        super().__init__(
            texts,
            output_path,
            lambda x: None,  #
            min_text_obj_length=min_text_obj_length,
            n_chunk=n_chunk,
            text_object_type_split=text_object_type_split,
            direction=direction,
        )
        if vectorizer is None:
            self.vectorizer = TfidfVectorizer(max_df=max_freq, min_df=min_freq)  # type: ignore
            self.vectors: csr_matrix = self.vectorizer.fit_transform(" ".join(d) for d in self.get_text_chunks())  # type: ignore
        else:
            self.direction = "target"
            self.vectorizer = vectorizer
            self.vectors: csr_matrix = self.vectorizer.transform(" ".join(d) for d in self.get_text_chunks())  # type: ignore

        self.length: int = self.vectors.shape[0]  # type: ignore
        self.dim: int = self.vectors.shape[1]  # type: ignore

    def __getitem__(self, item: int) -> csr_matrix:
        return self.vectors[item]  # type: ignore

    def __filter_by_jaccard_sim(
        self, similarity_matrix: np.ndarray, min_similarity: float, other_vectors: csr_matrix | None = None
    ) -> np.ndarray:
        """Give a score of 0 for all matches where the Jaccard similarity score is under 75% of the min score"""
        for outer_doc_id, inner_doc_id in np.argwhere(similarity_matrix >= min_similarity):
            outer_vector = self[outer_doc_id]
            if other_vectors is not None:
                inner_vector = other_vectors[inner_doc_id]
            else:
                inner_vector = self[inner_doc_id]
            jaccard_similarity = jaccard_sim(outer_vector, inner_vector)
            if jaccard_similarity < 0.5 * min_similarity:
                similarity_matrix[outer_doc_id, inner_doc_id] = 0.0
        return similarity_matrix

    def inner_compare(self, min_similarity: float) -> Matches:
        """Compare corpus with itself"""
        results: np.ndarray = linear_kernel(self.vectors, dense_output=False)  # type: ignore
        results = self.__filter_by_jaccard_sim(results, min_similarity)
        return Matches(self.process_inner_compare(results, min_similarity))

    def outer_compare(self, target_corpus: TfIdfCorpus, min_similarity: float) -> Matches:
        """Compare corpus with another corpus"""
        print("Comparing source collection to target collection...", flush=True)
        results: np.ndarray = linear_kernel(self.vectors, target_corpus.vectors, dense_output=False)  # type: ignore
        results = self.__filter_by_jaccard_sim(results, min_similarity, target_corpus.vectors)
        return Matches(self.process_outer_compare(results, target_corpus, min_similarity))


class Word2VecEmbeddingCorpus(Corpus):
    """Corpus object which builds doc embeddings using the average of token word2vec vectors"""

    def __init__(
        self,
        texts: Iterable[Tokens],
        output_path: str,
        model: str | spacy.Language,
        n_batches: int,
        min_text_obj_length: int = 15,
        n_chunk: int = 3,
        text_object_type_split: str = "doc",
        direction: str = "source",
    ):
        super().__init__(
            texts,
            output_path,
            similarity_function=linear_kernel,
            min_text_obj_length=min_text_obj_length,
            n_chunk=n_chunk,
            text_object_type_split=text_object_type_split,
            direction=direction,
            n_batches=n_batches,
        )
        if isinstance(model, str):
            self.model = spacy.load(model)
        else:
            self.model = model
        self.docs = DocumentChunks(
            self.get_text_chunks(),
            self.direction,
            self.create_embeddings,
        )
        self.length = len(self.docs)

    def create_embeddings(self, text_chunk) -> np.ndarray:
        """Create document embeddings"""
        doc: Doc = self.model(" ".join(text_chunk))
        return (doc.vector / doc.vector_norm).reshape(1, -1)  # type: ignore

    def __getitem__(self, item: int) -> list[str]:
        return self.docs[item]  # type: ignore

    def __len__(self):
        return self.length


class TransformerCorpus(Corpus):
    """Corpus object which builds doc embeddings using sentence-transformers for similarity"""

    def __init__(
        self,
        texts: Iterable[Tokens],
        output_path: str,
        model_name: str,
        n_batches: int,
        min_text_obj_length: int = 15,
        n_chunk: int = 3,
        text_object_type_split: str = "sent",
        model=None,
        direction="source",
    ):
        def sim_function(x, y):
            sim = util.cos_sim(x, y).cpu().numpy()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return sim

        super().__init__(
            texts,
            output_path,
            similarity_function=sim_function,
            min_text_obj_length=min_text_obj_length,
            n_chunk=n_chunk,
            text_object_type_split=text_object_type_split,
            direction=direction,
            n_batches=n_batches,
        )

        if model is None:
            self.model = SentenceTransformer(model_name)
        else:
            self.model = model

        self.model.max_seq_length = self.model.get_max_seq_length() - 2 # needed to enable truncating long sequences
        self.max_tokens: int = int(self.model.max_seq_length / 2)

        self.docs = DocumentChunks(
            self.get_text_chunks(),
            self.direction,
            self.create_embeddings,
        )
        self.length = len(self.docs)

        if torch.cuda.is_available():
            torch.cuda.empty_cache() # clear GPU cache after creating embeddings
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

    def __getitem__(self, item: int) -> list[str]:
        return self.docs[item]  # type: ignore

    def __len__(self):
        return self.length

    def create_embeddings(self, text_chunks) -> torch.Tensor:
        """Create document embedding"""
        tensor = self.model.encode(list(text_chunks), convert_to_tensor=True)
        return tensor  # type: ignore


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


def get_passage(doc: Tokens, start_byte: int, end_byte: int) -> list[Token]:
    """Get passage within Tokens object"""
    tokens = []
    for token in doc:
        if token.ext["start_byte"] >= start_byte and token.ext["end_byte"] <= end_byte:
            tokens.append(token)
        elif token.ext["end_byte"] > end_byte:
            break
    return tokens


def merge_passages(
    matches: Matches,
    min_score: float,
) -> list[MergedGroup]:
    """Merge all passages into bigger passages. Similarity is computed as the mean similarity of all passages in the group."""
    # TODO: should merging be done using Jaccard sim metric: to avoid sparsity
    last_count = len(matches)
    current_count = last_count + 1
    iteration = 1
    merged_matches: list[MergedGroup] = Matches.load().matches  # type:ignore
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
        saved_groups: list[MergedGroup] = []
        total_matches: int = len(matches)
        start_score: float = min_score
        merged_pairs: list[float] = []

        for pos, match in enumerate(merged_matches):
            merged_source: bool = False
            merged_target: bool = False
            if merged_group.source.filename == "":
                merged_pairs.append(match.similarity)
                merged_group = MergedGroup(match.source, match.target, start_score)
                continue
            if (
                match.source.filename != merged_group.source.filename
                or match.target.filename != merged_group.target.filename
            ):
                merged_group.similarity = sum(merged_pairs) / len(merged_pairs)
                saved_groups.append(merged_group)
                merged_pairs = [match.similarity]
                merged_group = MergedGroup(match.source, match.target, start_score)
                continue
            if match.source.start_byte <= merged_group.source.end_byte:
                if match.source.end_byte > merged_group.source.end_byte:
                    merged_group.source.end_byte = match.source.end_byte
                    merged_group.source.metadata["end_byte"] = match.source.end_byte
                    merged_source = True
                elif match.source.end_byte == merged_group.source.end_byte:
                    merged_source = True
            if match.target.start_byte <= merged_group.target.end_byte:
                if match.target.end_byte > merged_group.target.end_byte:
                    merged_group.target.end_byte = match.target.end_byte
                    merged_group.target.metadata["end_byte"] = match.target.end_byte
                    merged_target = True
                elif match.target.end_byte == merged_group.target.end_byte:
                    merged_target = True
            if any((merged_source, merged_target)):
                merged_pairs.append(match.similarity)
            if merged_source is False and merged_target is False:
                merged_group.similarity = sum(merged_pairs) / len(merged_pairs)
                saved_groups.append(merged_group)
                merged_pairs = [match.similarity]
                merged_group = MergedGroup(match.source, match.target, match.similarity)
            if pos + 1 == total_matches:
                merged_group.similarity = sum(merged_pairs) / len(merged_pairs)
                saved_groups.append(merged_group)
        merged_matches = saved_groups
        iteration += 1
        current_count = len(saved_groups)
        print(f"\rMerging matches: {current_count} matches after iteration {iteration+1}...", end="", flush=True)
    print(flush=True)
    return merged_matches


def get_tokens(passage: PassageGroup, preproc: PreProcessor) -> list[tuple[str, str]]:
    """Get tokens"""
    text: str = " "
    start_byte: int = passage.start_byte
    end_byte: int = passage.end_byte
    with open(passage.filename, "rb") as text_file:
        text_file.seek(start_byte)
        text = text_file.read(end_byte - start_byte).decode("utf8", "ignore")
    tokens: list[tuple[str, str]] = []
    pos = 0
    for token in preproc.process_string(text):
        pos += 1
        surface_form = token.surface_form.replace("\n", " ")
        token.surface_form = surface_form
        tokens.append((token.text, token.surface_form))
    return tokens


def post_process_passages(
    source: PassageGroup,
    target: PassageGroup,
    source_preproc: PreProcessor,
    target_preproc: PreProcessor,
) -> tuple[str, str]:
    """Post process function to highlight matching words in HTML tags"""
    # print(source.start_byte, source.end_byte, source.filename)
    source_tokens = get_tokens(source, source_preproc)
    target_tokens = get_tokens(target, target_preproc)
    source_set = {word for word, _ in source_tokens if word}
    target_set = {word for word, _ in target_tokens if word}
    source_passage_with_matches = []
    for word, surface_form in source_tokens:
        if word and word in target_set:
            source_passage_with_matches.append(f'&lt;span class="token-match"&gt;{surface_form}&lt;/span&gt;')
        elif not word:
            source_passage_with_matches.append(f'&lt;span class="filtered-token"&gt;{surface_form or " "}&lt;/span&gt;')
        else:
            source_passage_with_matches.append(surface_form)
    target_passage_with_matches = []
    for word, surface_form in target_tokens:
        if word and word in source_set:
            target_passage_with_matches.append(f'&lt;span class="token-match"&gt;{surface_form}&lt;/span&gt;')
        elif not word:
            target_passage_with_matches.append(f'&lt;span class="filtered-token"&gt;{surface_form}&lt;/span&gt;')
        else:
            target_passage_with_matches.append(surface_form)
    return clean_text("".join(source_passage_with_matches)), clean_text("".join(target_passage_with_matches))


def text_object_upper_bound(config) -> str:
    """Find the text object level above the one specified in the config"""
    object_type_to_level = {v: k for k, v in PHILO_TEXT_OBJECT_LEVELS.items()}
    text_object_level = PHILO_TEXT_OBJECT_LEVELS[config["text_object_type"]]
    if text_object_level == 1:
        return "doc"
    return object_type_to_level[text_object_level - 1]


def simple_similarity(
    source_texts: Iterable[Tokens],
    source_config: dict[str, Any],
    target_config: dict[str, Any],
    min_similarity: float,
    output_path: str,
    target_texts: Optional[Iterable[Tokens]] = None,
) -> tuple[TfIdfCorpus, Matches, list[dict[str, Any]], list[dict[str, Any]]]:
    """Cosine similarity of TF-IDF vectors"""
    source_corpus: TfIdfCorpus = TfIdfCorpus(
        source_texts,
        output_path,
        min_text_obj_length=source_config["min_text_object_length"],
        n_chunk=source_config["n_chunk"],
        text_object_type_split=text_object_upper_bound(source_config),
        min_freq=source_config["min_freq"],
        max_freq=source_config["max_freq"],
    )
    if target_texts is not None:
        target_corpus: TfIdfCorpus = TfIdfCorpus(
            target_texts,
            output_path,
            vectorizer=source_corpus.vectorizer,
            min_text_obj_length=target_config["min_text_object_length"],
            n_chunk=target_config["n_chunk"],
            text_object_type_split=text_object_upper_bound(target_config),
            direction="target",
        )

        matching_docs = source_corpus.outer_compare(target_corpus, min_similarity)
    else:
        matching_docs = source_corpus.inner_compare(min_similarity)
        target_corpus = source_corpus
    return source_corpus, matching_docs, source_corpus.metadata, target_corpus.metadata


def transformer_similarity(
    source_texts: Iterable[Tokens],
    source_config: dict[str, Any],
    target_config: dict[str, Any],
    min_similarity: float,
    source_batch: int,
    output_path: str,
    target_texts: Optional[Iterable[Tokens]] = None,
    target_batch: int = 1,
) -> tuple[Matches, list[dict[str, Any]], list[dict[str, Any]]]:
    """Cosine similarity of sentence embeddings from transformer models"""
    source_corpus: TransformerCorpus = TransformerCorpus(
        source_texts,
        output_path,
        source_config["model_name"],
        source_batch,
        min_text_obj_length=source_config["min_text_object_length"],
        n_chunk=source_config["n_chunk"],
        text_object_type_split=text_object_upper_bound(source_config),
    )
    if target_texts is not None:
        target_corpus: TransformerCorpus = TransformerCorpus(
            target_texts,
            output_path,
            source_config["model_name"],
            target_batch,
            direction="target",
            min_text_obj_length=target_config["min_text_object_length"],
            n_chunk=target_config["n_chunk"],
            text_object_type_split=text_object_upper_bound(target_config),
            model=source_corpus.model,
        )
        matching_docs = source_corpus.outer_compare(target_corpus, min_similarity)
    else:
        matching_docs = source_corpus.inner_compare(min_similarity)
        target_corpus = source_corpus
    return matching_docs, source_corpus.metadata, target_corpus.metadata


def word2vec_embed_similarity(
    source_texts: Iterable[Tokens],
    source_config: dict[str, Any],
    target_config: dict[str, Any],
    min_similarity: float,
    source_batch: int,
    output_path: str,
    target_texts: Optional[Iterable[Tokens]] = None,
    target_batch: int = 1,
) -> tuple[Word2VecEmbeddingCorpus, Matches, list[dict[str, Any]], list[dict[str, Any]]]:
    """Cosine similarity of sentence embeddings using mean w2v vectors"""
    source_corpus: Word2VecEmbeddingCorpus = Word2VecEmbeddingCorpus(
        source_texts,
        output_path,
        source_config["model_name"],
        source_batch,
        min_text_obj_length=source_config["min_text_object_length"],
        n_chunk=source_config["n_chunk"],
        text_object_type_split=text_object_upper_bound(source_config),
    )
    if target_texts is not None:
        target_corpus: Word2VecEmbeddingCorpus = Word2VecEmbeddingCorpus(
            target_texts,
            output_path,
            source_corpus.model,
            target_batch,
            direction="target",
            min_text_obj_length=target_config["min_text_object_length"],
            n_chunk=target_config["n_chunk"],
            text_object_type_split=text_object_upper_bound(target_config),
        )
        matching_docs = source_corpus.outer_compare(target_corpus, min_similarity)
    else:
        matching_docs = source_corpus.inner_compare(min_similarity)
        target_corpus = source_corpus
    return source_corpus, matching_docs, source_corpus.metadata, target_corpus.metadata


def run_vsa(source_path: str, target_path: str, workers: int, config: dict[str, Any], output_path: str):
    """Main function"""
    config["source"]["strip_tags"] = True  # this is useful for post-processing passages where we have tags included.
    config["target"]["strip_tags"] = True
    source_preproc: PreProcessor | None = None
    target_preproc: PreProcessor | None = None
    if config["source"]["vectorization"] == "transformer":
        config["source"]["strip_punctuation"] = False
        config["target"]["strip_punctuation"] = False
    source_preproc = PreProcessor(is_philo_db=True, workers=workers, **config["source"])
    target_preproc = PreProcessor(is_philo_db=True, workers=workers, nlp_model=source_preproc.nlp, using_gpu=source_preproc.using_gpu, **config["target"])
    source_texts: Iterable[Tokens] = source_preproc.process_texts(
        (file.path for file in os.scandir(source_path)), keep_all=True, progress=False
    )
    target_texts: Iterable[Tokens] = target_preproc.process_texts(
        (file.path for file in os.scandir(target_path)), keep_all=True, progress=False
    )

    if config["source"]["vectorization"] == "tfidf":
        source_corpus, matches, source_metadata, target_metadata = simple_similarity(
            source_texts,
            config["source"],
            config["target"],
            config["min_similarity"],
            output_path,
            target_texts=target_texts,
        )
    elif config["source"]["vectorization"] == "transformer":
        matches, source_metadata, target_metadata = transformer_similarity(
            source_texts,
            config["source"],
            config["target"],
            config["min_similarity"],
            config["source_batch"],
            output_path,
            target_texts=target_texts,
            target_batch=config["target_batch"],
        )
    else:
        source_corpus, matches, source_metadata, target_metadata = word2vec_embed_similarity(
            source_texts,
            config["source"],
            config["target"],
            config["min_similarity"],
            config["source_batch"],
            output_path,
            target_texts=target_texts,
            target_batch=config["target_batch"],
        )
    if len(matches) == 0:
        print("No matches found. Exiting...")
        exit()
    print(f"{len(matches)} matches found.")

    matches = merge_passages(
        matches,
        config["min_similarity"],
    )

    print("Formatting and writing out processed results...(this may take some time)")
    os.system("mkdir -p output/results")

    if source_preproc is None:
        source_preproc = PreProcessor(is_philo_db=True, workers=workers, **config["source"])
        target_preproc = PreProcessor(
            is_philo_db=True, workers=workers, nlp_model=source_preproc.nlp, **config["target"]
        )

    with lz4.frame.open(f"{output_path}/results/alignments.jsonl.lz4", mode="wb", compression_level=3) as output:
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
                source_preproc.normalize_options = {
                    **source_preproc.normalize_options,
                    "strip_tags": False,
                }
                source_preproc.pos_to_keep = []
                target_preproc.normalize_options = {
                    **target_preproc.normalize_options,
                    "strip_tags": False,
                }
                target_preproc.pos_to_keep = []
                source_passage_with_matches, target_passage_with_matches = post_process_passages(
                    match.source,
                    match.target,
                    source_preproc,
                    target_preproc,
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
            output.write(f"{result_object}\n".encode("utf8"))  # type: ignore
    with open("output/results/count.txt", "w", encoding="utf8") as output_file:
        output_file.write(f"{len(matches)}")

    # Generating metadata files to mimic output of generate_ngrams
    os.makedirs("output/source/metadata/", exist_ok=True)
    with open("output/source/metadata/metadata.json", "w", encoding="utf8") as output_file:
        output_file.write(json.dumps(dict(enumerate(source_metadata))))

    os.makedirs("output/target/metadata/", exist_ok=True)
    with open("output/target/metadata/metadata.json", "w", encoding="utf8") as output_file:
        output_file.write(json.dumps(dict(enumerate(target_metadata))))
