"""Corpus classes for vector space alignment"""

from __future__ import annotations

import os
from abc import ABC
from collections import deque
from collections.abc import Iterable
from typing import Any, Callable, Optional

import faiss
import numpy as np
import spacy
import torch
from scipy.sparse import csr_matrix
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from spacy.tokens import Doc
from text_preprocessing import Tokens
from tqdm import tqdm

from .structures import (
    PHILO_TEXT_OBJECT_LEVELS,
    DocumentChunks,
    Matches,
    MergedGroup,
    PassageGroup,
    save_tokens,
)


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
    return (intersect / union).toarray()


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
        print(f"Processing {self.direction} texts... ", end="", flush=True)
        for text in self.texts:
            docs[text.metadata["philo_id"]] = " ".join([t.text for t in text])
            text.metadata["parsed_filename"] = os.path.join(
                self.output_dir,
                self.direction,
                "saved_docs",
                os.path.basename(text.metadata["parsed_filename"].replace(".lz4", ".token_cache")),
            )
            doc_id = text.metadata["philo_id"].split()[0]
            if (
                doc_id != current_doc_id and current_doc_id is not None
            ):  # we save the current doc when doc_ids don't match
                save_tokens(full_doc, full_doc.metadata["parsed_filename"])
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
                        print(f"\rProcessing {self.direction} texts... {chunks_done} text chunks extracted...", end="", flush=True)
                        yield text_chunk
                chunk_group.clear()
            current_text_level_id = text_level_id
            current_chunk_group_length = sum([len(t.tokens) for t in chunk_group])
            text_length = len(text)
            if current_chunk_group_length + text_length > self.max_tokens and current_chunk_group_length:
                chunks_done += 1
                print(f"\rProcessing {self.direction} texts... {chunks_done} text chunks extracted...", end="", flush=True)
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
                    print(f"\rProcessing {self.direction} texts... {chunks_done} text chunks extracted...", end="", flush=True)
                    yield self.__build_text_chunk(chunk_group)
            current_doc_id = doc_id
        save_tokens(full_doc, full_doc.metadata["parsed_filename"])
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
        use_llm_evaluation: bool = False,
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
        self.use_llm_evaluation = use_llm_evaluation
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
        if not self.use_llm_evaluation:
            results = self.__filter_by_jaccard_sim(results, min_similarity)
        return Matches(self.process_inner_compare(results, min_similarity))

    def outer_compare(self, target_corpus: TfIdfCorpus, min_similarity: float) -> Matches:
        """Compare corpus with another corpus"""
        print("Comparing source collection to target collection...", flush=True)
        results: np.ndarray = linear_kernel(self.vectors, target_corpus.vectors, dense_output=False)  # type: ignore
        if not self.use_llm_evaluation:
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
            # Handle BFloat16 conversion issues
            if hasattr(x, 'dtype') and x.dtype == torch.bfloat16:
                x = x.to(torch.float32)
            if hasattr(y, 'dtype') and y.dtype == torch.bfloat16:
                y = y.to(torch.float32)

            sim = util.cos_sim(x, y).cpu()
            if sim.dtype == torch.bfloat16:
                sim = sim.to(torch.float32)
            sim = sim.numpy()
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
            self.model = SentenceTransformer(model_name, trust_remote_code=False)
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

    def vectordb_compare(self, target_corpus, min_similarity: float) -> Matches:
        """Compare using FAISS vector database for efficient similarity search"""
        print(f"Building FAISS index for {target_corpus.length} target embeddings...", flush=True)

        # Get all target embeddings as a batch (2D tensor)
        target_embeddings_tensor = target_corpus.docs[0:target_corpus.length]

        # Handle tensor conversion
        if hasattr(target_embeddings_tensor, 'cpu'):  # PyTorch tensor
            target_embeddings = target_embeddings_tensor.cpu().numpy()
        else:  # Already numpy array
            target_embeddings = np.array(target_embeddings_tensor, dtype=np.float32)

        # Normalize for cosine similarity (each row is an embedding)
        norms = np.linalg.norm(target_embeddings, axis=1, keepdims=True)
        target_embeddings = target_embeddings / norms

        # Create FAISS index for cosine similarity
        dimension = target_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner Product for normalized vectors = cosine similarity
        index.add(target_embeddings)

        print(f"Searching for matches above {min_similarity} similarity...", flush=True)

        # Get all source embeddings as a batch (2D tensor)
        source_embeddings_tensor = self.docs[0:self.length]

        # Handle tensor conversion
        if hasattr(source_embeddings_tensor, 'cpu'):  # PyTorch tensor
            source_embeddings = source_embeddings_tensor.cpu().numpy()
        else:  # Already numpy array
            source_embeddings = np.array(source_embeddings_tensor, dtype=np.float32)

        # Normalize for cosine similarity (each row is an embedding)
        norms = np.linalg.norm(source_embeddings, axis=1, keepdims=True)
        source_embeddings = source_embeddings / norms

        # Search for matches above threshold
        def process_vectordb_matches():
            """Generator that yields matches from FAISS search results"""
            # Use range_search to find all matches above threshold - much more efficient!
            lims, distances, target_indices = index.range_search(source_embeddings, min_similarity)

            # Check if this is inner comparison (corpus comparing to itself)
            is_inner_compare = target_corpus is self

            with tqdm(total=self.length, desc="Processing matches", leave=False) as pbar:
                for source_idx in range(self.length):
                    # Get the range of results for this source embedding
                    start_idx = lims[source_idx]
                    end_idx = lims[source_idx + 1]

                    # Process all matches for this source
                    for i in range(start_idx, end_idx):
                        similarity = float(distances[i])
                        target_idx = int(target_indices[i])

                        if is_inner_compare:
                            # Skip self-matches
                            if source_idx == target_idx:
                                continue
                            # Respect chronological order constraint
                            if self.metadata[source_idx]["year"] > self.metadata[target_idx]["year"]:
                                continue

                        # Yield match using metadata
                        yield MergedGroup(
                            PassageGroup(
                                self.metadata[source_idx]["start_byte"],
                                self.metadata[source_idx]["end_byte"],
                                self.metadata[source_idx]["filename"],
                                    self.metadata[source_idx],
                                ),
                                PassageGroup(
                                    target_corpus.metadata[target_idx]["start_byte"],
                                    target_corpus.metadata[target_idx]["end_byte"],
                                    target_corpus.metadata[target_idx]["filename"],
                                    target_corpus.metadata[target_idx],
                                ),
                                similarity
                            )

                    pbar.update(1)

        return Matches(process_vectordb_matches())

    def inner_compare(self, min_similarity: float) -> Matches:
        """Compare corpus with itself using vector database"""
        return self.vectordb_compare(self, min_similarity)

    def outer_compare(self, target_corpus, min_similarity: float) -> Matches:
        """Compare corpus with another corpus using vector database"""
        return self.vectordb_compare(target_corpus, min_similarity)