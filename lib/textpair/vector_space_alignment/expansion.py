"""
Passage expansion module for extending matched passages using LLM evaluation.
"""

import copy
import itertools

from tqdm import tqdm

from textpair.utils import get_text

from .structures import MergedGroup, find_token_index_by_byte, load_token_search_data


def count_sentences_from_tokens(filepath: str, start_byte: int, end_byte: int) -> int:
    """Counts the number of unique sentences within a byte range using token data."""
    token_data = load_token_search_data(filepath)
    if not token_data.start_bytes:
        return 0

    start_index = find_token_index_by_byte(token_data.start_bytes, start_byte)
    end_index = find_token_index_by_byte(token_data.start_bytes, end_byte)

    if start_index == -1 or end_index == -1:
        return 0

    # Ensure start_index is not after end_index
    if start_index > end_index:
        start_index, end_index = end_index, start_index

    # Slice the sentence IDs and count the unique ones
    seen_sentence_ids = set(token_data.sentence_ids[start_index : end_index + 1])
    return len(seen_sentence_ids)


def get_previous_sentences_from_tokens(filepath: str, start_byte: int, count: int) -> list[tuple[str, int]]:
    """
    Gets previous N sentences.
    """
    token_data = load_token_search_data(filepath)
    if not token_data.start_bytes or count == 0:
        return []

    def get_previous_sentence(current_match_start_index):
        """Helper to get the previous sentence before the match."""
        sentence = []
        start_byte_of_sentence = start_byte
        first_token_index = current_match_start_index

        if current_match_start_index - 1 < 0:
            return ([], start_byte, first_token_index)

        current_sentence_id = token_data.sentence_ids[current_match_start_index - 1]
        for i in range(current_match_start_index - 1, -1, -1):  # Iterate backwards
            if token_data.sentence_ids[i] == current_sentence_id:
                sentence.append(token_data.surface_forms[i])
                start_byte_of_sentence = token_data.start_bytes[i]
                first_token_index = i
            else:
                break
        return (sentence, start_byte_of_sentence, first_token_index)

    sentences = []
    match_start_index = find_token_index_by_byte(token_data.start_bytes, start_byte)
    if match_start_index == -1:
        return sentences
    for _ in range(count):
        prev_sentence, start_byte_of_sentence, first_token_index = get_previous_sentence(match_start_index)
        if not prev_sentence:
            break
        sentences.append(("".join(reversed(prev_sentence)), start_byte_of_sentence))
        match_start_index = first_token_index

    return sentences


def get_next_sentences_from_tokens(filepath: str, end_byte: int, count: int) -> list[tuple[str, int]]:
    """
    Gets next N sentences.
    """
    token_data = load_token_search_data(filepath)
    if not token_data.start_bytes or count == 0:
        return []

    def get_next_sentence(current_match_end_index):
        """Helper to get the next sentence after the match."""
        sentence = []
        end_byte_of_sentence = end_byte
        last_token_index = current_match_end_index

        if current_match_end_index + 1 >= len(token_data.sentence_ids):
            return ([], end_byte, last_token_index)
        current_sentence_id = token_data.sentence_ids[current_match_end_index + 1]
        for i in range(current_match_end_index + 1, len(token_data.end_bytes)):
            if token_data.sentence_ids[i] == current_sentence_id:
                sentence.append(token_data.surface_forms[i])
                # if token_data.end_bytes[i] > end_byte_of_sentence: # works around bug in PhiloLogic where sentence end bytes are incorrect
                #     end_byte_of_sentence = token_data.end_bytes[i]
                last_token_index = i
            else:
                end_byte_of_sentence = token_data.start_bytes[
                    i
                ]  # Assume sentence ends at start of next token: works around bug in PhiloLogic where punct start/end bytes are incorrect when sentence terminator
                break
        return (sentence, end_byte_of_sentence, last_token_index)

    sentences = []
    match_end_index = find_token_index_by_byte(token_data.end_bytes, end_byte)
    if match_end_index == -1:
        return sentences
    for _ in range(count):
        next_sentence, end_byte_of_sentence, last_token_index = get_next_sentence(match_end_index)
        if not next_sentence:
            break
        sentences.append(("".join(next_sentence), end_byte_of_sentence))
        match_end_index = last_token_index

    return sentences


def _build_expanded_text(
    original_text: str,
    prev_sents: list[str] | None = None,
    next_sents: list[str] | None = None,
) -> str:
    """Builds a combined text string from optional components."""
    parts = []
    if prev_sents:
        parts.extend(prev_sents)
    parts.append(original_text)
    if next_sents:
        parts.extend(next_sents)
    return " ".join(s for s in parts if s)


def _prepare_expansion_step(match: MergedGroup, step: int, direction: str) -> dict:
    """
    Prepare a single expansion for progressive evaluation.

    Args:
        match: The match to expand
        step: How many sentences to add (1 or 2)
        direction: 'prev', 'next', or 'both'
    """
    original_source_text = get_text(match.source.start_byte, match.source.end_byte, match.source.filename)
    original_target_text = get_text(match.target.start_byte, match.target.end_byte, match.target.filename)

    # Get contextual sentences
    source_prev = (
        get_previous_sentences_from_tokens(match.source.metadata["parsed_filename"], match.source.start_byte, step)
        if direction in ["prev", "both"]
        else []
    )
    source_next = (
        get_next_sentences_from_tokens(match.source.metadata["parsed_filename"], match.source.end_byte, step)
        if direction in ["next", "both"]
        else []
    )
    target_prev = (
        get_previous_sentences_from_tokens(match.target.metadata["parsed_filename"], match.target.start_byte, step)
        if direction in ["prev", "both"]
        else []
    )
    target_next = (
        get_next_sentences_from_tokens(match.target.metadata["parsed_filename"], match.target.end_byte, step)
        if direction in ["next", "both"]
        else []
    )

    # Build expanded text
    expanded_source = _build_expanded_text(
        original_source_text,
        prev_sents=[text for text, _ in source_prev],
        next_sents=[text for text, _ in source_next],
    )
    expanded_target = _build_expanded_text(
        original_target_text,
        prev_sents=[text for text, _ in target_prev],
        next_sents=[text for text, _ in target_next],
    )

    # Update byte boundaries
    config = copy.deepcopy(match)
    if source_prev:
        config.source.start_byte = source_prev[-1][1]
    if source_next:
        config.source.end_byte = source_next[-1][1]
    if target_prev:
        config.target.start_byte = target_prev[-1][1]
    if target_next:
        config.target.end_byte = target_next[-1][1]

    return {
        "match_config": config,
        "source_text": expanded_source,
        "target_text": expanded_target,
    }


def _prepare_expansion_combinations(
    match: MergedGroup,
    source_sents_count: int,
    target_sents_count: int,
    strategy: str = "all",
) -> list[dict]:
    """
    DEPRECATED: Old expansion logic, kept for compatibility.
    Use progressive expansion via expand_validated_matches instead.
    """
    # Read all necessary text and boundary components exactly ONCE
    original_source_text = get_text(match.source.start_byte, match.source.end_byte, match.source.filename)
    original_target_text = get_text(match.target.start_byte, match.target.end_byte, match.target.filename)

    # Determine how many sentences to add based on the dynamic strategy (max 1 for progressive)
    s_to_add = 1 if source_sents_count < 5 else 0
    t_to_add = 1 if target_sents_count < 5 else 0

    # Fetch the maximum number of contextual sentences we might need
    source_prev_sents_data = get_previous_sentences_from_tokens(
        match.source.metadata["parsed_filename"], match.source.start_byte, s_to_add
    )
    source_next_sents_data = get_next_sentences_from_tokens(
        match.source.metadata["parsed_filename"], match.source.end_byte, s_to_add
    )
    target_prev_sents_data = get_previous_sentences_from_tokens(
        match.target.metadata["parsed_filename"], match.target.start_byte, t_to_add
    )
    target_next_sents_data = get_next_sentences_from_tokens(
        match.target.metadata["parsed_filename"], match.target.end_byte, t_to_add
    )

    def get_distribution_choices(num_to_add: int, prev_avail: int, next_avail: int) -> list[tuple[int, int]]:
        """Get all (prev, next) combinations that sum to num_to_add."""
        if num_to_add == 0:
            return [(0, 0)]

        choices = []
        for i in range(num_to_add + 1):
            prev_needed = i
            next_needed = num_to_add - i
            if prev_needed <= prev_avail and next_needed <= next_avail:
                choices.append((prev_needed, next_needed))
        return choices

    # Generate all valid ways to distribute the needed sentences for source and target
    s_distributions = get_distribution_choices(s_to_add, len(source_prev_sents_data), len(source_next_sents_data))
    t_distributions = get_distribution_choices(t_to_add, len(target_prev_sents_data), len(target_next_sents_data))

    # Include smaller expansions as well
    s_choices = [(0, 0)] + s_distributions
    t_choices = [(0, 0)] + t_distributions
    if s_to_add > 1:
        s_choices.extend(
            get_distribution_choices(s_to_add - 1, len(source_prev_sents_data), len(source_next_sents_data))
        )
    if t_to_add > 1:
        t_choices.extend(
            get_distribution_choices(t_to_add - 1, len(target_prev_sents_data), len(target_next_sents_data))
        )

    # Remove duplicates and sort
    s_choices = sorted(list(set(s_choices)))
    t_choices = sorted(list(set(t_choices)))

    all_combinations = list(itertools.product(s_choices, t_choices))

    # The maximal combination is the one that adds the most sentences.
    # In a sorted list of tuples, this will be the last element.
    maximal_combo = all_combinations[-1] if len(all_combinations) > 1 else None

    if strategy == "maximal":
        combinations = [maximal_combo] if maximal_combo else []
    elif strategy == "fallback":
        # Exclude original (0,0) and maximal
        combinations = [c for c in all_combinations if c != ((0, 0), (0, 0)) and c != maximal_combo]
    else:  # 'all'
        combinations = all_combinations

    passage_data_list = []
    for (s_prev_count, s_next_count), (t_prev_count, t_next_count) in combinations:
        s_prev = source_prev_sents_data[:s_prev_count]
        s_next = source_next_sents_data[:s_next_count]
        t_prev = target_prev_sents_data[:t_prev_count]
        t_next = target_next_sents_data[:t_next_count]

        current_source_text = _build_expanded_text(
            original_source_text,
            prev_sents=[text for text, _ in s_prev],
            next_sents=[text for text, _ in s_next],
        )
        current_target_text = _build_expanded_text(
            original_target_text,
            prev_sents=[text for text, _ in t_prev],
            next_sents=[text for text, _ in t_next],
        )

        config = copy.deepcopy(match)
        if s_prev:
            config.source.start_byte = s_prev[-1][1]
        if s_next:
            config.source.end_byte = s_next[-1][1]
        if t_prev:
            config.target.start_byte = t_prev[-1][1]
        if t_next:
            config.target.end_byte = t_next[-1][1]

        passage_data_list.append(
            {
                "match_config": config,
                "source_text": current_source_text,
                "target_text": current_target_text,
            }
        )

    return passage_data_list


async def expand_validated_matches(
    matches: list[MergedGroup],
    evaluator,
    chunk_size: int = 50,
) -> list[MergedGroup]:
    """Orchestrator for the async, LLM-based expansion using chunked batch evaluation."""
    if not matches:
        return []

    # 1. Triage matches into expansion candidates and final matches
    expansion_candidates = []
    final_matches = []

    for match in matches:
        source_sents = count_sentences_from_tokens(
            match.source.metadata["parsed_filename"],
            match.source.start_byte,
            match.source.end_byte,
        )
        target_sents = count_sentences_from_tokens(
            match.target.metadata["parsed_filename"],
            match.target.start_byte,
            match.target.end_byte,
        )

        # If at least one side is shorter than 5 sentences, it's eligible for expansion.
        if source_sents < 5 or target_sents < 5:
            expansion_candidates.append((match, source_sents, target_sents))
        else:
            # This match is already long enough, so add it directly to the final list.
            final_matches.append(match)

    # 2. Process expansion candidates in chunks to limit memory usage
    expansion_count = 0
    total_candidates = len(expansion_candidates)

    if expansion_candidates:
        with tqdm(total=total_candidates, desc="Expanding short passages", unit="passage") as pbar:
            for i in range(0, total_candidates, chunk_size):
                chunk = expansion_candidates[i : i + chunk_size]
                chunk_expansion_count = await _process_expansion_chunk(chunk, evaluator)
                expansion_count += chunk_expansion_count
                pbar.update(len(chunk))

                for match, _, _ in chunk:
                    final_matches.append(match)

        print(
            f"Expansion complete: {expansion_count}/{total_candidates} passages expanded.",
            flush=True,
        )

    return final_matches


async def _process_expansion_chunk(chunk: list[tuple[MergedGroup, int, int]], evaluator) -> int:
    """Process a chunk of expansion candidates using progressive expansion strategy."""
    expansion_count = 0

    # --- Step 1: Try +1 sentence in both directions ---
    step1_prev_expansions = []
    step1_next_expansions = []
    match_map = []

    for match, source_sents_count, target_sents_count in chunk:
        # Only expand if at least one side is short
        if source_sents_count >= 5 and target_sents_count >= 5:
            continue

        step1_prev_expansions.append(_prepare_expansion_step(match, step=1, direction="prev"))
        step1_next_expansions.append(_prepare_expansion_step(match, step=1, direction="next"))
        match_map.append(match)

    if not match_map:
        return 0

    # Evaluate both directions
    prev_pairs = [(exp["source_text"], exp["target_text"]) for exp in step1_prev_expansions]
    next_pairs = [(exp["source_text"], exp["target_text"]) for exp in step1_next_expansions]

    prev_results = await evaluator.evaluate_batch(prev_pairs, batch_size=8, show_progress=False)
    next_results = await evaluator.evaluate_batch(next_pairs, batch_size=8, show_progress=False)

    # --- Step 2: Determine winners and prepare next step ---
    step2_candidates = []
    step2_directions = []
    step2_match_map = []

    for i, original_match in enumerate(match_map):
        prev_score, _, _ = prev_results[i]
        next_score, _, _ = next_results[i]
        original_score = original_match.similarity

        # Check if either direction improves
        prev_improves = prev_score >= original_score
        next_improves = next_score >= original_score

        if not prev_improves and not next_improves:
            # No improvement, keep original
            continue

        # Determine best direction(s)
        if prev_improves and next_improves:
            if abs(prev_score - next_score) < 0.001:  # Tied (within floating point tolerance)
                # Both help equally - add both directions
                step1_winner = _prepare_expansion_step(original_match, step=1, direction="both")
                step2_direction = "both"
            elif prev_score > next_score:
                step1_winner = step1_prev_expansions[i]
                step2_direction = "prev"
            else:
                step1_winner = step1_next_expansions[i]
                step2_direction = "next"
        elif prev_improves:
            step1_winner = step1_prev_expansions[i]
            step2_direction = "prev"
        else:
            step1_winner = step1_next_expansions[i]
            step2_direction = "next"

        # Update match with step 1 result
        step1_config = step1_winner["match_config"]
        original_match.source = step1_config.source
        original_match.target = step1_config.target
        original_match.similarity = (
            max(prev_score, next_score)
            if prev_improves and next_improves
            else (prev_score if prev_improves else next_score)
        )
        expansion_count += 1

        # Prepare step 2 expansion (+1 more sentence beyond what step 1 already added)
        step2_candidates.append(_prepare_expansion_step(original_match, step=1, direction=step2_direction))
        step2_directions.append(step2_direction)
        step2_match_map.append(original_match)

    if not step2_candidates:
        return expansion_count

    # --- Step 3: Evaluate step 2 expansions ---
    step2_pairs = [(exp["source_text"], exp["target_text"]) for exp in step2_candidates]
    step2_results = await evaluator.evaluate_batch(step2_pairs, batch_size=8, show_progress=False)

    for i, (step2_score, _, _) in enumerate(step2_results):
        original_match = step2_match_map[i]
        step1_score = original_match.similarity

        # Accept step 2 if it improves over step 1
        if step2_score >= step1_score:
            step2_config = step2_candidates[i]["match_config"]
            original_match.source = step2_config.source
            original_match.target = step2_config.target
            original_match.similarity = step2_score

    return expansion_count
