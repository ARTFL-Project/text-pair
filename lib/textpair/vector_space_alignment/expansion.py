"""
Passage expansion module for extending matched passages using LLM evaluation.

This module provides functions for expanding passage boundaries to find
optimal matches while maintaining similarity thresholds through LLM evaluation.
"""

import asyncio
import copy
import itertools

import regex as re
from tqdm import tqdm

from textpair.utils import get_text

from .structures import MergedGroup

# Sentence boundary regex for detecting sentence splits
SENTENCE_BOUNDARY_REGEX = re.compile(r'(?<=.{10,}[.?!])\s+(?=\p{Lu}|\p{L})')


def count_sentences(text: str) -> int:
    """Count the number of sentences in a text using regex."""
    sentences = [s for s in SENTENCE_BOUNDARY_REGEX.split(text) if s.strip()]
    return len(sentences)


def get_previous_sentences(filepath: str, start_byte: int, count: int) -> tuple[list[str], int]:
    """Gets previous N sentences and the new start_byte for the entire chunk."""
    start_read = max(0, start_byte - 4000)
    with open(filepath, "rb") as f:
        f.seek(start_read)
        buffer_bytes = f.read(start_byte - start_read)

    # Decode the raw bytes into a string, ignoring any broken characters
    buffer = buffer_bytes.decode("utf-8", errors="ignore")

    sentences = [s for s in SENTENCE_BOUNDARY_REGEX.split(buffer) if s.strip()]
    chunk = sentences[-count:] if len(sentences) >= count else sentences

    if not chunk:
        return [], start_byte # Return empty list and original byte if no sentences found

    # Find the character position of the start of our chunk within the buffer
    start_char_pos = buffer.find(chunk[0])

    # Convert that char position to a byte offset and calculate the new absolute start_byte
    new_start_byte = start_read + len(buffer[:start_char_pos].encode('utf-8'))
    return chunk, new_start_byte


def get_next_sentences(filepath: str, end_byte: int, count: int) -> tuple[list[str], int]:
    """Gets next N sentences and the new end_byte for the entire chunk."""
    with open(filepath, "rb") as f:
        f.seek(end_byte)
        buffer_bytes = f.read(4000)

    # Decode the raw bytes into a string, ignoring any broken characters
    buffer = buffer_bytes.decode("utf-8", errors="ignore")

    sentences = [s for s in SENTENCE_BOUNDARY_REGEX.split(buffer) if s.strip()]
    chunk = sentences[:count] if len(sentences) >= count else sentences

    if not chunk:
        return [], end_byte # Return empty list and original byte if no sentences found

    # Find the character position of the end of our chunk within the buffer
    end_char_pos = buffer.find(chunk[-1]) + len(chunk[-1])

    # Convert to a byte offset and calculate the new absolute end_byte
    new_end_byte = end_byte + len(buffer[:end_char_pos].encode('utf-8'))
    return chunk, new_end_byte


def _build_expanded_text(
    expand_backward: bool,
    expand_forward: bool,
    prev_sents: list[str],
    original_text: str,
    next_sents: list[str]
) -> str:
    """Builds a combined text string based on expansion flags."""
    parts = []
    if expand_backward:
        parts.extend(prev_sents)
    parts.append(original_text)
    if expand_forward:
        parts.extend(next_sents)
    return " ".join(s for s in parts if s)


async def _expand_single_match(
    match: MergedGroup, evaluator, threshold: float, expand_source: bool, expand_target: bool, semaphore: asyncio.Semaphore
) -> tuple[MergedGroup, bool]:
    """
    Reads all potential text components once, then brute-forces the 16 combinations
    to find the best expansion.
    """
    async with semaphore:
        # Read all necessary text and boundary components exactly ONCE
        original_source_text = get_text(match.source.start_byte, match.source.end_byte, match.source.filename)
        original_target_text = get_text(match.target.start_byte, match.target.end_byte, match.target.filename)

        # Initialize sentence buffers for source
        source_prev_sents, s_start_expanded = ([], match.source.start_byte)
        source_next_sents, s_end_expanded = ([], match.source.end_byte)
        if expand_source:
            source_prev_sents, s_start_expanded = get_previous_sentences(match.source.filename, match.source.start_byte, 2)
            source_next_sents, s_end_expanded = get_next_sentences(match.source.filename, match.source.end_byte, 2)

        # Initialize sentence buffers for target
        target_prev_sents, t_start_expanded = ([], match.target.start_byte)
        target_next_sents, t_end_expanded = ([], match.target.end_byte)
        if expand_target:
            target_prev_sents, t_start_expanded = get_previous_sentences(match.target.filename, match.target.start_byte, 2)
            target_next_sents, t_end_expanded = get_next_sentences(match.target.filename, match.target.end_byte, 2)


        # Early exit: Check the max expansion case first
        source_max_text = _build_expanded_text(expand_source, expand_source, source_prev_sents, original_source_text, source_next_sents)
        target_max_text = _build_expanded_text(expand_target, expand_target, target_prev_sents, original_target_text, target_next_sents)

        score, _ = await evaluator.evaluate_similarity(source_max_text, target_max_text)
        if score >= threshold:
            expanded_match = copy.deepcopy(match)
            if expand_source:
                expanded_match.source.start_byte, expanded_match.source.end_byte = s_start_expanded, s_end_expanded
            if expand_target:
                expanded_match.target.start_byte, expanded_match.target.end_byte = t_start_expanded, t_end_expanded
            expanded_match.similarity = score
            return expanded_match, True

        # Dynamically generate combinations based on eligibility
        s_choices = [False, True] if expand_source else [False]
        t_choices = [False, True] if expand_target else [False]
        # Creates 1, 4, or 16 combinations as needed
        combinations = list(itertools.product(s_choices, s_choices, t_choices, t_choices))

        tasks = []
        passage_data_list = []
        for expand_s_back, expand_s_fwd, expand_t_back, expand_t_fwd in combinations:
            current_source_text = _build_expanded_text(
                expand_s_back, expand_s_fwd, source_prev_sents, original_source_text, source_next_sents)
            current_target_text = _build_expanded_text(
                expand_t_back, expand_t_fwd, target_prev_sents, original_target_text, target_next_sents)

            # This part for tracking byte offsets remains the same
            config = copy.deepcopy(match)
            if expand_s_back:
                config.source.start_byte = s_start_expanded
            if expand_s_fwd:
                config.source.end_byte = s_end_expanded
            if expand_t_back:
                config.target.start_byte = t_start_expanded
            if expand_t_fwd:
                config.target.end_byte = t_end_expanded

            passage_data_list.append({'match_config': config, 'source_text': current_source_text, 'target_text': current_target_text})
            tasks.append(evaluator.evaluate_similarity(current_source_text, current_target_text))

        results = await asyncio.gather(*tasks)

        valid_candidates = []
        for i, (score, _) in enumerate(results):
            if score >= threshold:
                data = passage_data_list[i]
                s_sents = count_sentences(data['source_text'])
                t_sents = count_sentences(data['target_text'])
                valid_candidates.append({'score': score, 'size': s_sents + t_sents, 'match': data['match_config']})

        if not valid_candidates:
            return match, False

        valid_candidates.sort(key=lambda x: (x['size'], x['score']), reverse=True)
        winner = valid_candidates[0]['match']
        winner.similarity = valid_candidates[0]['score']

        was_expanded = (winner.source.start_byte != match.source.start_byte or winner.source.end_byte != match.source.end_byte or
                    winner.target.start_byte != match.target.start_byte or winner.target.end_byte != match.target.end_byte)

        return winner, was_expanded


async def expand_validated_matches(
    matches: list[MergedGroup],
    evaluator,
) -> list[MergedGroup]:
    """Orchestrator for the async, LLM-based expansion."""
    if not matches:
        return []

    # Limit concurrency to avoid running out of file descriptors
    semaphore = asyncio.Semaphore(100)

    expansion_tasks = []
    final_matches = [] # This will hold all matches, expanded or not

    # 1. Triage: Separate matches into those that need expansion and those that don't.
    expansion_count = 0
    for match in matches:
        source_sents = count_sentences(get_text(match.source.start_byte, match.source.end_byte, match.source.filename))
        target_sents = count_sentences(get_text(match.target.start_byte, match.target.end_byte, match.target.filename))

        should_expand_source = source_sents < 10
        should_expand_target = target_sents < 10

        # If at least one side is shorter than 10 sentences, it's eligible for expansion.
        if should_expand_source or should_expand_target:
            # The threshold for expansion is the match's current score
            task = _expand_single_match(
                match, evaluator, match.similarity,
                should_expand_source,
                should_expand_target,
                semaphore
            )
            expansion_tasks.append(task)
        else:
            # This match is already long enough, so add it directly to the final list.
            final_matches.append(match)

    # 2. Process & Combine: Run the expansion tasks and add the results to the final list.
    if expansion_tasks:
        for future in tqdm(asyncio.as_completed(expansion_tasks), total=len(expansion_tasks), desc="Looking for potential passage expansions", leave=False):
            expanded_match, was_expanded = await future
            final_matches.append(expanded_match)
            if was_expanded:
                expansion_count += 1

    print(f"Looking for potential passage expansions: expanded {expansion_count} passages.", flush=True)

    return final_matches