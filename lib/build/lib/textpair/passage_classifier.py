"""Passage classification for thematic categorization of alignments"""

import html
import re

import lz4.frame
import orjson
from tqdm import tqdm
from transformers import pipeline


def get_expanded_passage(alignment: dict, context_bytes: int = 1000) -> str:
    """
    Pulls context around a target passage using byte offsets.

    Args:
        alignment: Alignment dict with target_passage, target_filename,
                   target_start_byte, target_end_byte
        context_bytes: Number of bytes to read before and after (default: 1000)

    Returns:
        Expanded passage string with context before and after
    """
    target_passage = alignment.get("target_passage", "")
    filename = alignment.get("target_filename", "")
    start_byte = alignment.get("target_start_byte", 0)
    end_byte = alignment.get("target_end_byte", 0)

    # If we don't have the required fields, just return the original passage
    if not filename or not start_byte or not end_byte:
        return target_passage

    context_before = ""
    context_after = ""

    try:
        with open(filename, "rb") as f:
            # Get context before
            seek_pos_before = max(0, start_byte - context_bytes)
            read_len_before = start_byte - seek_pos_before
            if read_len_before > 0:
                f.seek(seek_pos_before)
                bytes_before = f.read(read_len_before)
                context_before = bytes_before.decode("utf-8", errors="ignore").strip()
                context_before = re.sub(r"\s+", " ", context_before)
                context_before = re.sub(r"^\w+>", "", context_before)
                context_before = re.sub(r"<.*?>", "", context_before)
                context_before = html.unescape(context_before)

            # Get context after
            f.seek(end_byte)
            bytes_after = f.read(context_bytes)
            context_after = bytes_after.decode("utf-8", errors="ignore").strip()
            context_after = re.sub(r"\s+", " ", context_after)
            context_after = re.sub(r"<[^>]$", "", context_after)
            context_after = re.sub(r"<.*?>", "", context_after)
            context_after = html.unescape(context_after)
    except Exception:
        # If file reading fails, fall back to original passage
        return target_passage

    # Return the expanded passage
    return f"{context_before} {target_passage} {context_after}".strip()


async def classify_passages(
    input_path: str,
    zero_shot_model: str,
    classification_classes: dict[str, str],
    min_confidence: float = 0.7,
    top_k: int = 3,
    batch_size: int = 32
) -> int:
    """
    Classify passages into thematic categories using zero-shot classification.

    This performs multi-label classification where each passage can receive multiple
    category labels based on confidence thresholds.

    Args:
        input_path: Path to alignments file (jsonl.lz4 format)
        zero_shot_model: Hugging Face model for zero-shot classification
        classification_classes: Dict mapping class names to their definitions/criteria
        min_confidence: Minimum confidence score (0-1) to assign a label (default: 0.3)
        top_k: Maximum number of labels to assign per passage (default: 3)
        batch_size: Number of passages to process at once (default: 32)

    Returns:
        Number of passages classified
    """
    if not classification_classes:
        print("No classification classes defined. Skipping passage classification.")
        return 0

    print(f"Loading passage classifier: {zero_shot_model}")
    classifier = pipeline(
        "zero-shot-classification",
        model=zero_shot_model,
        device=0  # Use GPU if available
    )

    # Extract class labels and their descriptions
    candidate_labels = list(classification_classes.keys())

    # Prepare output
    temp_output_path = input_path.replace(".jsonl.lz4", ".jsonl_temp.lz4")

    # Count lines for progress
    with lz4.frame.open(input_path, "rb") as f_count:
        num_lines = sum(1 for _ in f_count)

    if num_lines == 0:
        print("Input file is empty.")
        return 0

    classified_count = 0
    with (lz4.frame.open(temp_output_path, "wb") as output_file,
          lz4.frame.open(input_path, "rb") as f_in,
          tqdm(total=num_lines, desc="Passage classification") as pbar):

        batch = []
        batch_alignments = []

        for line_b in f_in:
            alignment = orjson.loads(line_b)

            # Expand passage with surrounding context for better classification
            expanded_passage = get_expanded_passage(alignment, context_bytes=1000)

            batch.append(expanded_passage)
            batch_alignments.append(alignment)

            # Process batch
            if len(batch) >= batch_size:
                results = classifier(
                    batch,
                    candidate_labels,
                    multi_label=True,  # Allow multiple labels per passage
                    batch_size=batch_size
                )

                for alignment, result in zip(batch_alignments, results):
                    # Filter labels by confidence threshold and take top_k
                    labels_and_scores = list(zip(result["labels"], result["scores"]))

                    # Filter by minimum confidence
                    filtered = [(label, score) for label, score in labels_and_scores if score >= min_confidence]

                    # Take top_k
                    top_labels = filtered[:top_k]

                    # Store results
                    alignment["passage_categories"] = [label for label, _ in top_labels]
                    alignment["passage_categories_scores"] = [round(score, 3) for _, score in top_labels]

                    if top_labels:
                        classified_count += 1

                    output_file.write(orjson.dumps(alignment) + b"\n")
                    pbar.update(1)

                batch = []
                batch_alignments = []

        # Process remaining batch
        if batch:
            results = classifier(
                batch,
                candidate_labels,
                multi_label=True,
                batch_size=len(batch)
            )

            for alignment, result in zip(batch_alignments, results):
                labels_and_scores = list(zip(result["labels"], result["scores"]))
                filtered = [(label, score) for label, score in labels_and_scores if score >= min_confidence]
                top_labels = filtered[:top_k]

                alignment["passage_categories"] = [label for label, _ in top_labels]
                alignment["passage_categories_scores"] = [round(score, 3) for _, score in top_labels]

                if top_labels:
                    classified_count += 1

                output_file.write(orjson.dumps(alignment) + b"\n")
                pbar.update(1)

    # Replace original with classified version
    import os
    os.replace(temp_output_path, input_path)

    print(f"Classification complete: {classified_count}/{num_lines} passages received category labels")
    print(f"(Passages with no labels had all scores below {min_confidence} threshold)")

    return num_lines


if __name__ == "__main__":
    import asyncio
    import sys

    if len(sys.argv) < 2:
        print("Usage: python passage_classifier.py <path_to_alignments.jsonl.lz4>")
        sys.exit(1)

    file_path = sys.argv[1]

    # Test with example categories
    test_classes = {
        "Satire & Humor": "Passages using irony, satire, humor, parody, or comical situations",
        "Religion & Spirituality": "Speech about faith, God, theology, scripture, church",
        "Philosophy": "Speech about morality, ethics, virtue, reason, metaphysics",
    }

    total = asyncio.run(classify_passages(
        file_path,
        "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        test_classes,
        min_confidence=0.3,
        top_k=3
    ))
    print(f"Total passages processed: {total}")
