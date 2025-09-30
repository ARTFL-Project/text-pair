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
    seen_sentence_ids = set(token_data.sentence_ids[start_index:end_index + 1])
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
        for i in range(current_match_start_index - 1, -1, -1): # Iterate backwards
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

    print("PREVIOUS sentences:", sentences)
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
        current_sentence_id = token_data.sentence_ids[current_match_end_index +1]
        for i in range(current_match_end_index + 1, len(token_data.end_bytes)):
            if token_data.sentence_ids[i] == current_sentence_id:
                sentence.append(token_data.surface_forms[i])
                # if token_data.end_bytes[i] > end_byte_of_sentence: # works around bug in PhiloLogic where sentence end bytes are incorrect
                #     end_byte_of_sentence = token_data.end_bytes[i]
                last_token_index = i
            else:
                end_byte_of_sentence = token_data.start_bytes[i] # Assume sentence ends at start of next token: works around bug in PhiloLogic where punct start/end bytes are incorrect when sentence terminator
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

    print("NEXT sentences:", sentences)
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


def _prepare_expansion_combinations(
    match: MergedGroup,
    source_sents_count: int,
    target_sents_count: int,
    strategy: str = 'all'
) -> list[dict]:
    """
    Prepare expansion combinations for a single match based on a dynamic strategy.
    """
    # Read all necessary text and boundary components exactly ONCE
    original_source_text = get_text(match.source.start_byte, match.source.end_byte, match.source.filename)
    original_target_text = get_text(match.target.start_byte, match.target.end_byte, match.target.filename)

    # Determine how many sentences to add based on the dynamic strategy (max 2)
    s_to_add = 1 if source_sents_count == 4 else (2 if source_sents_count < 4 else 0)
    t_to_add = 1 if target_sents_count == 4 else (2 if target_sents_count < 4 else 0)

    # Fetch the maximum number of contextual sentences we might need
    source_prev_sents_data = get_previous_sentences_from_tokens(match.source.metadata["parsed_filename"], match.source.start_byte, s_to_add)
    source_next_sents_data = get_next_sentences_from_tokens(match.source.metadata["parsed_filename"], match.source.end_byte, s_to_add)
    target_prev_sents_data = get_previous_sentences_from_tokens(match.target.metadata["parsed_filename"], match.target.start_byte, t_to_add)
    target_next_sents_data = get_next_sentences_from_tokens(match.target.metadata["parsed_filename"], match.target.end_byte, t_to_add)

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
    s_choices = [(0,0)] + s_distributions
    t_choices = [(0,0)] + t_distributions
    if s_to_add > 1:
        s_choices.extend(get_distribution_choices(s_to_add - 1, len(source_prev_sents_data), len(source_next_sents_data)))
    if t_to_add > 1:
        t_choices.extend(get_distribution_choices(t_to_add - 1, len(target_prev_sents_data), len(target_next_sents_data)))

    # Remove duplicates and sort
    s_choices = sorted(list(set(s_choices)))
    t_choices = sorted(list(set(t_choices)))

    all_combinations = list(itertools.product(s_choices, t_choices))

    # The maximal combination is the one that adds the most sentences.
    # In a sorted list of tuples, this will be the last element.
    maximal_combo = all_combinations[-1] if len(all_combinations) > 1 else None

    if strategy == 'maximal':
        combinations = [maximal_combo] if maximal_combo else []
    elif strategy == 'fallback':
        # Exclude original (0,0) and maximal
        combinations = [c for c in all_combinations if c != ((0,0),(0,0)) and c != maximal_combo]
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
            next_sents=[text for text, _ in s_next]
        )
        current_target_text = _build_expanded_text(
            original_target_text,
            prev_sents=[text for text, _ in t_prev],
            next_sents=[text for text, _ in t_next]
        )

        config = copy.deepcopy(match)
        if s_prev: config.source.start_byte = s_prev[-1][1]
        if s_next: config.source.end_byte = s_next[-1][1]
        if t_prev: config.target.start_byte = t_prev[-1][1]
        if t_next: config.target.end_byte = t_next[-1][1]

        passage_data_list.append({
            'match_config': config,
            'source_text': current_source_text,
            'target_text': current_target_text,
        })

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

    print("Identifying expansion candidates...", flush=True)

    for match in matches:
        source_sents = count_sentences_from_tokens(match.source.metadata["parsed_filename"], match.source.start_byte, match.source.end_byte)
        target_sents = count_sentences_from_tokens(match.target.metadata["parsed_filename"], match.target.start_byte, match.target.end_byte)

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
        print(f"Processing {total_candidates} expansion candidates in chunks of {chunk_size}...", flush=True)

        with tqdm(total=total_candidates, desc="Looking for potential passage expansions", leave=False) as pbar:
            for i in range(0, total_candidates, chunk_size):
                chunk = expansion_candidates[i:i + chunk_size]
                chunk_expansion_count = await _process_expansion_chunk(chunk, evaluator)
                expansion_count += chunk_expansion_count

                # Add processed matches to final results
                for match, _, _ in chunk:
                    final_matches.append(match)

                pbar.update(len(chunk))

    print(f"Looking for potential passage expansions: expanded {expansion_count} passages.", flush=True)

    return final_matches


async def _process_expansion_chunk(
    chunk: list[tuple[MergedGroup, int, int]],
    evaluator
) -> int:
    """Process a chunk of expansion candidates using the dynamic two-step greedy strategy."""
    expansion_count = 0

    # --- Step 1: Greedy evaluation of maximal expansions ---
    maximal_expansions = []
    match_map = []  # Use a list to keep order
    for match, source_sents_count, target_sents_count in chunk:
        maximal_combo = _prepare_expansion_combinations(
            match, source_sents_count, target_sents_count, strategy='maximal'
        )
        if maximal_combo:
            maximal_expansions.append(maximal_combo[0])
            match_map.append(match)

    if not maximal_expansions:
        return 0

    maximal_pairs = [(combo['source_text'], combo['target_text']) for combo in maximal_expansions]
    maximal_results = await evaluator.evaluate_batch(maximal_pairs, batch_size=8)

    # --- Step 2: Process results and prepare fallback evaluation ---
    fallback_candidates = []
    for i, (score, _) in enumerate(maximal_results):
        original_match = match_map[i]
        if score >= original_match.similarity:
            winner = maximal_expansions[i]['match_config']
            winner.similarity = score
            original_match.source = winner.source
            original_match.target = winner.target
            original_match.similarity = winner.similarity
            expansion_count += 1
        else:
            fallback_candidates.append(original_match)

    if not fallback_candidates:
        return expansion_count

    # --- Step 3: Evaluate fallback combinations for failed matches ---
    fallback_batches = []
    fallback_batch_info = {}
    for original_match in fallback_candidates:
        _, source_sents_count, target_sents_count = next(m for m in chunk if m[0] is original_match)
        fallback_combos = _prepare_expansion_combinations(
            original_match, source_sents_count, target_sents_count, strategy='fallback'
        )
        if fallback_combos:
            start_index = len(fallback_batches)
            fallback_batches.extend([(c['source_text'], c['target_text']) for c in fallback_combos])
            end_index = len(fallback_batches)
            fallback_batch_info[id(original_match)] = {
                'combinations': fallback_combos,
                'slice': slice(start_index, end_index)
            }

    if not fallback_batches:
        return expansion_count

    fallback_results = await evaluator.evaluate_batch(fallback_batches, batch_size=8)

    # --- Step 4: Process fallback results and select the best ---
    for original_match in fallback_candidates:
        match_info = fallback_batch_info.get(id(original_match))
        if not match_info:
            continue

        results_slice = match_info['slice']
        match_results = fallback_results[results_slice]

        valid_candidates = [{
            'score': original_match.similarity,
            'size': count_sentences_from_tokens(original_match.source.metadata["parsed_filename"], original_match.source.start_byte, original_match.source.end_byte) + \
                    count_sentences_from_tokens(original_match.target.metadata["parsed_filename"], original_match.target.start_byte, original_match.target.end_byte),
            'match': original_match
        }]

        for i, (score, _) in enumerate(match_results):
            if score >= original_match.similarity:
                combo_data = match_info['combinations'][i]
                s_sents = count_sentences_from_tokens(
                    combo_data['match_config'].source.metadata['parsed_filename'],
                    combo_data['match_config'].source.start_byte,
                    combo_data['match_config'].source.end_byte
                )
                t_sents = count_sentences_from_tokens(
                    combo_data['match_config'].target.metadata['parsed_filename'],
                    combo_data['match_config'].target.start_byte,
                    combo_data['match_config'].target.end_byte
                )
                valid_candidates.append({
                    'score': score,
                    'size': s_sents + t_sents,
                    'match': combo_data['match_config']
                })

        if valid_candidates:
            valid_candidates.sort(key=lambda x: (x['size'], x['score']), reverse=True)
            winner_data = valid_candidates[0]
            winner = winner_data['match']

            was_expanded = (winner.source.start_byte != original_match.source.start_byte or
                          winner.source.end_byte != original_match.source.end_byte or
                          winner.target.start_byte != original_match.target.start_byte or
                          winner.target.end_byte != original_match.target.end_byte)

            if was_expanded:
                expansion_count += 1

            original_match.source = winner.source
            original_match.target = winner.target
            original_match.similarity = winner_data['score']

    return expansion_count