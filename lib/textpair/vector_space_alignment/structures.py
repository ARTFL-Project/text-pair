"""Data structures for vector space alignment"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterable
from shutil import rmtree
from typing import Callable

import dill as pickle
import msgspec
import numpy as np
import torch
from msgspec import field
from text_preprocessing import Tokens

# Global constants for serialization and path management
TEMP_DIR = os.getcwd()
PHILO_TEXT_OBJECT_LEVELS = {"doc": 1, "div1": 2, "div2": 3, "div3": 4, "para": 5, "sent": 6, "word": 7}


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


# Msgpack encoders/decoders for serialization
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


# Lightweight, serializable data structure for efficient sentence searching.
class TokenSearchData(msgspec.Struct):
    """A lightweight container for token data needed for sentence searching."""
    start_bytes: list[int]
    end_bytes: list[int]
    surface_forms: list[str]
    sentence_ids: list[str]


def save_tokens(tokens: Tokens, parsed_filename: str):
    """
    Saves token search data to a cache file using msgpack serialization.
    """
    start_bytes = [token.ext['start_byte'] for token in tokens.tokens]
    end_bytes = [token.ext['end_byte'] for token in tokens.tokens]
    surface_forms = [token.surface_form for token in tokens.tokens]
    sentence_ids = [get_sentence_id(token) for token in tokens.tokens]

    search_data = TokenSearchData(
        start_bytes=start_bytes,
        end_bytes=end_bytes,
        surface_forms=surface_forms,
        sentence_ids=sentence_ids,
    )

    # Save the data to the cache file
    encoder = msgspec.msgpack.Encoder()
    with open(parsed_filename, "wb") as f:
        f.write(encoder.encode(search_data))

def load_token_search_data(parsed_filename: str) -> TokenSearchData:
    """
    Loads token search data from a cache if available, otherwise creates it
    from the full Tokens object and caches it.
    """
    decoder = msgspec.msgpack.Decoder(TokenSearchData)

    with open(parsed_filename, "rb") as f:
        return decoder.decode(f.read())


def find_token_index_by_byte(bytes: list[int], byte_offset: int) -> int:
    """
    Finds the index of the token at a given byte offset using binary search
    on a pre-computed list of start_bytes.
    """
    import bisect
    if not bytes:
        return -1

    # bisect_left finds the insertion point for the byte_offset.
    index = bisect.bisect_left(bytes, byte_offset)

    # If the offset is exactly a token's start, we found it.
    if index < len(bytes) and bytes[index] == byte_offset:
        return index

    # If the insertion point is 0, it must be the first token.
    if index == 0:
        return 0

    # Otherwise, the correct token is the one *before* the insertion point.
    return index - 1


def get_sentence_id(token) -> str:
    """Extracts the sentence ID from a token's position string."""
    try:
        # The sentence ID is composed of the first 6 integers of the position string.
        return " ".join(token.ext['position'].split()[:6])
    except (AttributeError, KeyError, IndexError):
        return ""