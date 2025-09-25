"""Vector Space Alignment Module

This module provides text passage alignment using vector space methods.
It includes corpus management, LLM evaluation, passage expansion, and various
similarity calculation methods.

Main entry point: run_vsa()
"""

import json
import os
from typing import Any, Iterable, Optional

import lz4.frame
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import linear_kernel
from text_preprocessing import PreProcessor, Token, Tokens
from tqdm import tqdm

from textpair.utils import clean_text, get_text, jaccard_sim, text_object_upper_bound

from .corpus import Corpus, TfIdfCorpus, TransformerCorpus, Word2VecEmbeddingCorpus
from .expansion import expand_validated_matches
from .llm_evaluation import AsyncLLMEvaluator, evaluate_passages_with_llm
from .structures import DocumentChunks, Matches, MergedGroup, PassageGroup

# ============================================================================
# Helper Functions for Text Processing and Post-processing
# ============================================================================

def get_passage(doc: Tokens, start_byte: int, end_byte: int) -> list[Token]:
    """Get passage within Tokens object"""
    tokens = []
    try:
        for token in doc:  # type: ignore
            if token.ext["start_byte"] >= start_byte and token.ext["end_byte"] <= end_byte:
                tokens.append(token)
            elif token.ext["end_byte"] > end_byte:
                break
    except (TypeError, AttributeError):
        # Handle cases where iteration might not work as expected
        pass
    return tokens


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
    try:
        for token in preproc.process_string(text):  # type: ignore
            pos += 1
            surface_form = token.surface_form.replace("\n", " ")
            token.surface_form = surface_form
            tokens.append((token.text, token.surface_form))
    except (TypeError, AttributeError):
        # Handle cases where process_string might not work as expected
        pass
    return tokens


def post_process_passages(
    source: PassageGroup,
    target: PassageGroup,
    source_preproc: PreProcessor,
    target_preproc: PreProcessor,
) -> tuple[str, str]:
    """Post process function to highlight matching words in HTML tags"""
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


# ============================================================================
# Passage Merging and Similarity Computation
# ============================================================================

def merge_passages(
    matches: Matches,
    min_score: float,
    vectorization: str,
    model: Optional[Any] = None,
    use_llm_evaluation: bool = False,
) -> list[MergedGroup]:
    """Merge all passages into bigger passages. Similarity is recomputed for merged text content."""
    last_count = len(matches)
    current_count = last_count + 1
    iteration = 1
    merged_matches: list[MergedGroup] = Matches.load().matches  # type:ignore

    print(f"Merging matches: {last_count} matches before iteration 1", end="", flush=True)

    def compute_merged_similarity(merged_group: MergedGroup, constituent_similarities: list[float]) -> float:
        """Compute similarity for merged passages by encoding the actual text content"""
        # If LLM evaluation will be used, use fast placeholder similarity
        if use_llm_evaluation and constituent_similarities:
            # Use average of constituent similarities as fast, meaningful placeholder
            return sum(constituent_similarities) / len(constituent_similarities)

        if model is None:
            # Fallback to min_score if no model provided
            return min_score

        try:
            # Get the actual text content for both merged passages
            source_text = get_text(
                merged_group.source.start_byte,
                merged_group.source.end_byte,
                merged_group.source.filename
            )
            target_text = get_text(
                merged_group.target.start_byte,
                merged_group.target.end_byte,
                merged_group.target.filename
            )

            # Compute the baseline similarity using embeddings (no LLM evaluation here)
            computed_similarity = min_score
            if vectorization == "transformer" or vectorization == "transformer_vectordb":
                source_embedding = model.encode([source_text], convert_to_tensor=True)
                target_embedding = model.encode([target_text], convert_to_tensor=True)
                computed_similarity = util.cos_sim(source_embedding, target_embedding).cpu().numpy()[0][0]
            elif vectorization == "word2vec":
                try:
                    source_doc = model.model(source_text)  # type: ignore
                    source_vector = (source_doc.vector / source_doc.vector_norm).reshape(1, -1)
                    target_doc = model.model(" ".join(target_text.split()))  # type: ignore
                    target_vector = (target_doc.vector / target_doc.vector_norm).reshape(1, -1)
                    similarity_result = linear_kernel(source_vector, target_vector, dense_output=False)
                    if hasattr(similarity_result, 'toarray'):
                        computed_similarity = similarity_result.toarray()[0][0]  # type: ignore
                    else:
                        computed_similarity = similarity_result[0][0]  # type: ignore
                except Exception:
                    computed_similarity = min_score
            elif vectorization == "tfidf":
                try:
                    source_vector = model.vectorizer.transform([source_text])  # type: ignore
                    target_vector = model.vectorizer.transform([target_text])  # type: ignore
                    similarity_result = linear_kernel(source_vector, target_vector, dense_output=False)
                    if hasattr(similarity_result, 'toarray'):
                        computed_similarity = similarity_result.toarray()[0][0]  # type: ignore
                    else:
                        computed_similarity = similarity_result[0][0]  # type: ignore
                except Exception:
                    computed_similarity = min_score

            return float(computed_similarity)
        except Exception as e:
            # Fallback to min_score if computation fails
            return min_score

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
        total_matches: int = len(matches)  # Use original matches count like the working version
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
                # Recompute similarity for the merged group
                merged_group.similarity = compute_merged_similarity(merged_group, merged_pairs)
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
                # Recompute similarity for the current merged group
                merged_group.similarity = compute_merged_similarity(merged_group, merged_pairs)
                saved_groups.append(merged_group)
                merged_pairs = [match.similarity]
                merged_group = MergedGroup(match.source, match.target, match.similarity)
            if pos + 1 == total_matches:
                # Recompute similarity for the final merged group
                merged_group.similarity = compute_merged_similarity(merged_group, merged_pairs)
                saved_groups.append(merged_group)
        merged_matches = saved_groups
        iteration += 1
        current_count = len(saved_groups)
        print(f"\rMerging matches: {current_count} matches after iteration {iteration+1}...", end="", flush=True)
    print(flush=True)
    return merged_matches


# ============================================================================
# Similarity Calculation Functions
# ============================================================================

def simple_similarity(
    source_texts: Iterable[Tokens],
    source_config: dict[str, Any],
    target_config: dict[str, Any],
    min_similarity: float,
    output_path: str,
    target_texts: Optional[Iterable[Tokens]] = None,
    use_llm_evaluation: bool = False,
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
        use_llm_evaluation=use_llm_evaluation,  # Skip Jaccard filtering when using LLM
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
            use_llm_evaluation=use_llm_evaluation,  # Skip Jaccard filtering when using LLM
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
) -> tuple[Matches, list[dict[str, Any]], list[dict[str, Any]], Optional[SentenceTransformer]]:
    """Cosine similarity of sentence embeddings from transformer models"""
    source_corpus: TransformerCorpus = TransformerCorpus(
        source_texts,
        output_path,
        source_config["embedding_model"],
        source_batch,
        min_text_obj_length=source_config["min_text_object_length"],
        n_chunk=source_config["n_chunk"],
        text_object_type_split=text_object_upper_bound(source_config),
    )
    if target_texts is not None:
        target_corpus: TransformerCorpus = TransformerCorpus(
            target_texts,
            output_path,
            source_config["embedding_model"],
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
    return matching_docs, source_corpus.metadata, target_corpus.metadata, source_corpus.model


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
        source_config["embedding_model"],
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


# ============================================================================
# Main Orchestrator Function
# ============================================================================

async def run_vsa(
    source_path: str,
    target_path: str,
    workers: int,
    config: dict[str, Any],
    output_path: str,
    debug_llm: bool = True
):
    """Main function for vector space alignment workflow"""
    config["source"]["strip_tags"] = True  # this is useful for post-processing passages where we have tags included.
    config["target"]["strip_tags"] = True
    source_preproc: PreProcessor | None = None
    target_preproc: PreProcessor | None = None
    if config["source"]["vectorization"] == "transformer":
        config["source"]["strip_punctuation"] = False
        config["target"]["strip_punctuation"] = False
    source_preproc = PreProcessor(is_philo_db=True, workers=workers, **config["source"])
    target_preproc = PreProcessor(is_philo_db=True, workers=workers, **config["target"])
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
            use_llm_evaluation=bool(config["llm_model"]),  # Skip Jaccard if LLM will be used
        )
        model = None
    elif config["source"]["vectorization"] == "transformer":
        matches, source_metadata, target_metadata, model = transformer_similarity(
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
        model = None
    if len(matches) == 0:
        print("No matches found. Exiting...")
        exit()
    print(f"{len(matches)} matches found.")

    # First merge passages (now synchronous)
    matches = merge_passages(
        matches,
        config["min_similarity"],
        config["source"]["vectorization"],
        model,
        use_llm_evaluation=bool(config["llm_model"]),  # Use placeholders if LLM will evaluate
    )

    # Then evaluate with LLM if model path is provided
    if config["llm_model"]:
        matches, llm_evaluator = await evaluate_passages_with_llm(
            matches,
            config["min_similarity"],
            config["llm_model"],
            config["llm_context_window"],
            config["llm_similarity_threshold"],
            config["llm_debug"],
            output_path,
            get_text_func=get_text,
        )
        # Iteratively expand the boundaries of the validated matches
        try:
            if matches: # Only run expansion if there are matches left
                matches = await expand_validated_matches(
                    matches,
                    evaluator=llm_evaluator,
                    get_text_func=get_text,
                )
        finally:
            # Ensure we always stop the server
            llm_evaluator.stop_server()

    print("Formatting and writing out processed results...(this may take some time)")
    os.system("mkdir -p output/results")

    if source_preproc is None:
        source_preproc = PreProcessor(is_philo_db=True, workers=workers, **config["source"])
        target_preproc = PreProcessor(
            is_philo_db=True, workers=workers, **config["target"]
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
                source_preproc.strip_tags = False  # type: ignore
                source_preproc.pos_to_keep = []  # type: ignore
                target_preproc.strip_tags = False  # type: ignore
                target_preproc.pos_to_keep = []  # type: ignore
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


# ============================================================================
# Public API
# ============================================================================

__all__ = ["run_vsa"]