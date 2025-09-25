"""
Passage expansion module for extending matched passages using LLM evaluation.

This module provides functions for expanding passage boundaries to find
optimal matches while maintaining similarity thresholds through LLM evaluation.
"""

import asyncio
import copy
import itertools
from typing import Optional

import regex as re
from tqdm import tqdm

from .structures import MergedGroup

# Sentence boundary regex for detecting sentence splits
SENTENCE_BOUNDARY_REGEX = re.compile(r'(?<=.{10,}[.?!])\s+(?=\p{Lu}|\p{L})')


def count_sentences(text: str) -> int:
    """Count the number of sentences in a text using regex."""
    sentences = [s for s in SENTENCE_BOUNDARY_REGEX.split(text) if s.strip()]
    return len(sentences)


def get_previous_sentences_with_boundary(filepath: str, start_byte: int, count: int) -> tuple[list[str], int]:
    """Gets previous N sentences and the new start_byte for the entire chunk."""
    start_read = max(0, start_byte - 4000)
    with open(filepath, "r", encoding="utf-8") as f:
        f.seek(start_read)
        buffer = f.read(start_byte - start_read)

    sentences = [s for s in SENTENCE_BOUNDARY_REGEX.split(buffer) if s.strip()]
    chunk = sentences[-count:] if len(sentences) >= count else sentences

    # Find the character position of the start of our chunk
    if not chunk:
        return [], start_byte
    start_char_pos = buffer.find(chunk[0])

    # Convert that to a byte offset and calculate the new absolute start_byte
    new_start_byte = start_read + len(buffer[:start_char_pos].encode('utf-8'))
    return chunk, new_start_byte


def get_next_sentences_with_boundary(filepath: str, end_byte: int, count: int) -> tuple[list[str], int]:
    """Gets next N sentences and the new end_byte for the entire chunk."""
    with open(filepath, "r", encoding="utf-8") as f:
        f.seek(end_byte)
        buffer = f.read(4000)

    sentences = [s for s in SENTENCE_BOUNDARY_REGEX.split(buffer) if s.strip()]
    chunk = sentences[:count] if len(sentences) >= count else sentences

    # Find the character position of the end of our chunk
    if not chunk:
        return [], end_byte
    # Find where the last sentence in our chunk ends within the buffer
    end_char_pos = buffer.find(chunk[-1]) + len(chunk[-1])

    # Convert to a byte offset and calculate the new absolute end_byte
    new_end_byte = end_byte + len(buffer[:end_char_pos].encode('utf-8'))
    return chunk, new_end_byte


def get_adjacent_sentence(
    filename: str, byte_pos: int, direction: str, buffer_size: int = 2048
) -> Optional[tuple[int, int]]:
    """
    Finds the start and end bytes of an adjacent sentence using a fast regex-based approach.
    """
    try:
        with open(filename, "rb") as f:
            if direction == "forward":
                f.seek(byte_pos)
                buffer = f.read(buffer_size).decode("utf-8", "ignore").lstrip()

                # Find the first sentence boundary in the buffer
                match = SENTENCE_BOUNDARY_REGEX.search(buffer)
                if not match:
                    return None

                end_char = match.start()
                # Return the byte offsets for the found sentence
                return (byte_pos, byte_pos + buffer[:end_char].encode("utf-8").__len__())

            elif direction == "backward":
                start_read = max(0, byte_pos - buffer_size)
                f.seek(start_read)
                buffer = f.read(byte_pos - start_read).decode("utf-8", "ignore").rstrip()

                # Find all sentence boundaries and take the one just before the end
                matches = list(SENTENCE_BOUNDARY_REGEX.finditer(buffer))
                if not matches:
                    return None

                # The start of the last sentence is the end of the second-to-last sentence
                start_char = matches[-2].end() if len(matches) > 1 else 0
                end_char = matches[-1].start()

                # Return byte offsets for the found sentence
                return (start_read + buffer[:start_char].encode("utf-8").__len__(),
                        start_read + buffer[:end_char].encode("utf-8").__len__())

    except (IOError, IndexError):
        return None
    return None


def _get_sentence_chunk(filename: str, start_byte: int, direction: str, count: int) -> list[tuple[int, int]]:
    """Gathers a chunk of N consecutive sentences."""
    sentences = []
    current_pos = start_byte
    for _ in range(count):
        next_sent_bounds = get_adjacent_sentence(filename, current_pos, direction)
        if next_sent_bounds:
            sentences.append(next_sent_bounds)
            current_pos = next_sent_bounds[1] if direction == 'forward' else next_sent_bounds[0]
        else:
            break
    return sentences

async def _expand_single_match(
    match: MergedGroup, evaluator, threshold: float, get_text_func
) -> tuple[MergedGroup, bool]:
    """
    Reads all potential text components once, then brute-forces the 16 combinations
    to find the best expansion.
    """
    # Read all necessary text and boundary components exactly ONCE. (6 reads)
    original_source_text = get_text_func(match.source.start_byte, match.source.end_byte, match.source.filename)
    source_prev_sents, s_start_expanded = get_previous_sentences_with_boundary(match.source.filename, match.source.start_byte, 2)
    source_next_sents, s_end_expanded = get_next_sentences_with_boundary(match.source.filename, match.source.end_byte, 2)

    original_target_text = get_text_func(match.target.start_byte, match.target.end_byte, match.target.filename)
    target_prev_sents, t_start_expanded = get_previous_sentences_with_boundary(match.target.filename, match.target.start_byte, 2)
    target_next_sents, t_end_expanded = get_next_sentences_with_boundary(match.target.filename, match.target.end_byte, 2)

    # Early exit: Check the max expansion case first
    source_max_text = " ".join(source_prev_sents + [original_source_text] + source_next_sents)
    target_max_text = " ".join(target_prev_sents + [original_target_text] + target_next_sents)
    score, _ = await evaluator.evaluate_similarity(source_max_text, target_max_text)
    if score >= threshold:
        expanded_match = copy.deepcopy(match)
        expanded_match.source.start_byte, expanded_match.source.end_byte = s_start_expanded, s_end_expanded
        expanded_match.target.start_byte, expanded_match.target.end_byte = t_start_expanded, t_end_expanded
        expanded_match.similarity = score
        return expanded_match, True

    # Generate all 16 combinations from in-memory strings.
    combinations = list(itertools.product([False, True], repeat=4))
    tasks = []

    passage_data_list = []

    for combo in combinations:
        # Build text strings and track byte offsets simultaneously
        source_parts = []
        s_start, s_end = match.source.start_byte, match.source.end_byte
        if combo[0]:
            source_parts.extend(source_prev_sents)
            s_start = s_start_expanded
        source_parts.append(original_source_text)
        if combo[1]:
            source_parts.extend(source_next_sents)
            s_end = s_end_expanded

        target_parts = []
        t_start, t_end = match.target.start_byte, match.target.end_byte
        if combo[2]:
            target_parts.extend(target_prev_sents)
            t_start = t_start_expanded
        target_parts.append(original_target_text)
        if combo[3]:
            target_parts.extend(target_next_sents)
            t_end = t_end_expanded

        current_source_text = " ".join(s for s in source_parts if s)
        current_target_text = " ".join(s for s in target_parts if s)

        # Store the config and the generated text together
        config = copy.deepcopy(match)
        config.source.start_byte, config.source.end_byte = s_start, s_end
        config.target.start_byte, config.target.end_byte = t_start, t_end

        passage_data_list.append({
            'match_config': config,
            'source_text': current_source_text,
            'target_text': current_target_text
        })
        tasks.append(evaluator.evaluate_similarity(current_source_text, current_target_text))

    results = await asyncio.gather(*tasks)

    # Filter, sort, and find the winner
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
    get_text_func,
) -> list[MergedGroup]:
    """Orchestrator for the async, LLM-based expansion."""
    if not matches:
        return []

    expansion_tasks = []
    final_matches = [] # This will hold all matches, expanded or not

    # 1. Triage: Separate matches into those that need expansion and those that don't.
    expansion_count = 0
    for match in matches:
        source_sents = count_sentences(get_text_func(match.source.start_byte, match.source.end_byte, match.source.filename))
        target_sents = count_sentences(get_text_func(match.target.start_byte, match.target.end_byte, match.target.filename))

        # If at least one side is shorter than 10 sentences, it's eligible for expansion.
        if source_sents < 10 or target_sents < 10:
            # The threshold for expansion is the match's current score
            task = _expand_single_match(match, evaluator, match.similarity, get_text_func)
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