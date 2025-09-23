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
    Tests the largest combination first as a short-circuit. If that fails, it
    tests all 16 combinations and returns the longest valid passage pair.
    Returns: (expanded_match, was_expanded)
    """
    # 1. Get the potential boundaries for the "+2 sentence" expansions
    s_back_chunk = _get_sentence_chunk(match.source.filename, match.source.start_byte, "backward", 2)
    s_fwd_chunk = _get_sentence_chunk(match.source.filename, match.source.end_byte, "forward", 2)
    t_back_chunk = _get_sentence_chunk(match.target.filename, match.target.start_byte, "backward", 2)
    t_fwd_chunk = _get_sentence_chunk(match.target.filename, match.target.end_byte, "forward", 2)

    s_start_expanded = s_back_chunk[0][0] if len(s_back_chunk) == 2 else match.source.start_byte
    s_end_expanded = s_fwd_chunk[-1][1] if len(s_fwd_chunk) == 2 else match.source.end_byte
    t_start_expanded = t_back_chunk[0][0] if len(t_back_chunk) == 2 else match.target.start_byte
    t_end_expanded = t_fwd_chunk[-1][1] if len(t_fwd_chunk) == 2 else match.target.end_byte

    # --- The "Early Exit" Check ---
    # 2. First, test only the single largest possible combination.
    max_expansion_match = copy.deepcopy(match)
    max_expansion_match.source.start_byte = s_start_expanded
    max_expansion_match.source.end_byte = s_end_expanded
    max_expansion_match.target.start_byte = t_start_expanded
    max_expansion_match.target.end_byte = t_end_expanded

    source_text = get_text_func(max_expansion_match.source.start_byte, max_expansion_match.source.end_byte, max_expansion_match.source.filename)
    target_text = get_text_func(max_expansion_match.target.start_byte, max_expansion_match.target.end_byte, max_expansion_match.target.filename)

    if source_text and target_text:
        score, _ = await evaluator.evaluate_similarity(source_text, target_text)
        # 3. If it's valid, it must be the best one, so we can return it immediately.
        if score >= threshold:
            max_expansion_match.similarity = score
            return max_expansion_match, True  # Successfully expanded
    # --- End of Early Exit Check ---

    # 4. If the early exit failed, proceed with the full 16-combination check.
    combinations = list(itertools.product([False, True], repeat=4))
    tasks = []
    passage_configs = []

    for combo in combinations:
        # We already tested the all-True combo, but re-evaluating it is simpler
        # than filtering it out and handling the results separately.
        temp_match = copy.deepcopy(match)
        if combo[0]: temp_match.source.start_byte = s_start_expanded
        if combo[1]: temp_match.source.end_byte = s_end_expanded
        if combo[2]: temp_match.target.start_byte = t_start_expanded
        if combo[3]: temp_match.target.end_byte = t_end_expanded

        source_text = get_text_func(temp_match.source.start_byte, temp_match.source.end_byte, temp_match.source.filename)
        target_text = get_text_func(temp_match.target.start_byte, temp_match.target.end_byte, temp_match.target.filename)
        if source_text and target_text:
            tasks.append(evaluator.evaluate_similarity(source_text, target_text))
            passage_configs.append(temp_match)

    if not tasks:
        return match, False  # No expansion attempted

    results = await asyncio.gather(*tasks)

    # 5. Filter for valid candidates and find the best one based on the new rule.
    valid_candidates = []
    for i, (score, _) in enumerate(results):
        if score >= threshold:
            config = passage_configs[i]
            s_sents = count_sentences(get_text_func(config.source.start_byte, config.source.end_byte, config.source.filename))
            t_sents = count_sentences(get_text_func(config.target.start_byte, config.target.end_byte, config.target.filename))
            valid_candidates.append({'score': score, 'size': s_sents + t_sents, 'match': config})

    if not valid_candidates:
        return match, False  # No valid expansion found

    # Sort to find the best: prioritize size first, then score as a tie-breaker.
    valid_candidates.sort(key=lambda x: (x['size'], x['score']), reverse=True)

    winner = valid_candidates[0]['match']
    winner.similarity = valid_candidates[0]['score']

    # Check if this is actually an expansion (different from original)
    was_expanded = (winner.source.start_byte != match.source.start_byte or
                   winner.source.end_byte != match.source.end_byte or
                   winner.target.start_byte != match.target.start_byte or
                   winner.target.end_byte != match.target.end_byte)

    return winner, was_expanded


async def expand_validated_matches(
    matches: list[MergedGroup],
    llm_model_path: str,
    llm_context_window: int,
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