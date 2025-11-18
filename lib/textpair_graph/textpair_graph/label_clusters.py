"""
Label clusters using LLM after graph building.

This module reads the graph_data output from build_graph_model.py and generates
thematic labels for each cluster using an LLM. It then updates the graph JSON files
with the labels.
"""

import asyncio
import json
import os

import lz4.frame
import numpy as np
import orjson
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from textpair_llm import AsyncLLMEvaluator


def truncate_passage(text: str, max_chars: int = 600) -> str:
    """Truncate passage at sentence boundaries, not mid-word."""
    if len(text) <= max_chars:
        return text
    # Find last sentence ending before max_chars
    truncated = text[:max_chars]
    for delimiter in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
        last_pos = truncated.rfind(delimiter)
        if last_pos > max_chars * 0.5:  # Keep at least 50% of text
            return truncated[:last_pos + 1]
    # No sentence boundary found, truncate at word
    return truncated.rsplit(' ', 1)[0] + '...'


def create_batch_summary_prompt(passages: list[str]) -> str:
    """
    Stage 1: Create prompt for summarizing a batch of passages.
    Returns a 2-3 sentence thematic summary.
    """
    passages_text = "\n\n".join([f"[{i+1}] {truncate_passage(p)}"
                                  for i, p in enumerate(passages)])

    prompt = f"""Read these text passages that were grouped together by algorithmic similarity and summarize their common themes.

TASK:
Write a 2-3 sentence summary that captures:
1. The main topic or subject matter shared across these passages
2. Key themes, concepts, or patterns that appear multiple times
3. Any notable subtopics or variations within the main theme

REQUIREMENTS:
- Focus on WHAT the passages discuss, not HOW they're written
- Be specific and concrete (avoid vague terms like "various topics")
- Capture both the main theme AND any important variations
- Write in clear, complete sentences

PASSAGES:
{passages_text}

SUMMARY:"""

    return prompt


def create_final_label_prompt(batch_summaries: list[str]) -> str:
    """
    Stage 2: Create prompt for generating final label from batch summaries.
    Takes multiple thematic summaries and produces a concise label.
    """
    summaries_text = "\n\n".join([f"[Batch {i+1}] {summary}"
                                   for i, summary in enumerate(batch_summaries)])

    prompt = f"""Below are thematic summaries from different batches of text passages that belong to the same cluster. Your task is to create a single unified label.

THEMATIC SUMMARIES:
{summaries_text}

TASK:
1. Identify the overarching theme that connects ALL these summaries
2. Provide a brief one-sentence rationale explaining the connection
3. Generate a precise 1-3 word label (noun phrase preferred)

REQUIREMENTS:
- Be specific, not generic (avoid "Various Topics", "Text", "Content")
- Use concrete concepts when identifiable (e.g., "Military Strategy" not "Actions")
- The label should encompass the main themes from ALL summaries
- Do not be overly focused on format or medium
- Keep label under 4 words

OUTPUT FORMAT:
RATIONALE: [One sentence explaining the overarching theme]
LABEL: [1-3 word label]

Answer ONLY in the specified OUTPUT FORMAT. Do not include any additional text."""

    return prompt


async def generate_cluster_labels_async(
    alignments_file: str,
    cluster_labels_modified: np.ndarray,
    cluster_centroids: np.ndarray,
    embeddings: np.ndarray,
    n_clusters: int,
    evaluator,
    num_batches: int = 5,
    batch_size: int = 20,
) -> dict:
    """
    Generate thematic labels for each cluster using LLM with two-stage approach.

    Stage 1: Select diverse passages using MMR, divide into batches, generate summaries
    Stage 2: Combine summaries to create final label

    Args:
        alignments_file: Path to alignments file
        cluster_labels_modified: Cluster assignments (including singletons)
        cluster_centroids: Cluster centroids in SBERT space
        embeddings: All passage embeddings in SBERT space
        n_clusters: Number of real clusters (excluding singletons)
        evaluator: AsyncLLMEvaluator instance
        num_batches: Number of batches to create (default: 5)
        batch_size: Number of passages per batch (default: 20, so 5*20=100 total)

    Returns:
        Dictionary mapping cluster_id -> label
    """
    total_passages = num_batches * batch_size
    print(f"\nGenerating labels for {n_clusters} clusters using {num_batches} batches of {batch_size} passages ({total_passages} total per cluster)...")

    cluster_label_map = {}
    used_labels = set()  # Track labels to ensure uniqueness

    # Process only real clusters (not singletons)
    for cluster_id in tqdm(range(n_clusters), desc="Processing clusters", leave=False):
        # Find all passages in this cluster
        cluster_mask = cluster_labels_modified == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) == 0:
            raise ValueError(f"Cluster {cluster_id} has no passages - data integrity issue!")

        # Get embeddings for this cluster
        cluster_embeddings = embeddings[cluster_indices]
        centroid = cluster_centroids[cluster_id].reshape(1, -1)

        # Calculate similarity to centroid for initial ranking
        similarities = cosine_similarity(cluster_embeddings, centroid).flatten()

        # Use Maximal Marginal Relevance (MMR) to select diverse, representative passages
        # Select total_passages = num_batches * batch_size passages
        total_passages = num_batches * batch_size
        lambda_param = 0.5  # Balance between relevance and diversity
        n_samples = min(total_passages, len(cluster_indices))

        # Start with the most relevant passage (highest similarity to centroid)
        selected_indices = [np.argmax(similarities)]
        selected_embeddings = [cluster_embeddings[selected_indices[0]]]

        # Iteratively select passages that maximize MMR score
        while len(selected_indices) < n_samples:
            relevance_scores = similarities.copy()

            # Calculate diversity: maximum similarity to already selected passages
            sim_to_selected = cosine_similarity(cluster_embeddings, np.array(selected_embeddings))
            max_sim_to_selected = np.max(sim_to_selected, axis=1)

            # MMR score: relevance - diversity penalty
            mmr_scores = lambda_param * relevance_scores - (1 - lambda_param) * max_sim_to_selected
            mmr_scores[selected_indices] = -np.inf  # Don't re-select

            # Select passage with highest MMR score
            next_idx = np.argmax(mmr_scores)
            selected_indices.append(next_idx)
            selected_embeddings.append(cluster_embeddings[next_idx])

        # Map back to original alignment indices
        selected_alignment_indices = cluster_indices[np.array(selected_indices)]

        # Extract passage texts from alignments file
        all_passages = []
        indices_to_fetch = set(int(i) for i in selected_alignment_indices)
        max_index = max(indices_to_fetch)
        min_index = min(indices_to_fetch)
        with lz4.frame.open(alignments_file, "rb") as f:
            for i, line in enumerate(f):
                if i < min_index:
                    continue
                if i in indices_to_fetch:
                    alignment = orjson.loads(line)
                    passage = alignment["source_passage"]
                    all_passages.append(passage)
                    if i == max_index:
                        break

        # Stage 1: Generate summaries for each batch (concurrently)
        print(f"\n  Cluster {cluster_id}: Stage 1 - Generating {num_batches} batch summaries (concurrent)...")

        # Create all batch tasks
        async def process_batch(batch_idx: int):
            """Process a single batch and return (batch_idx, summary or None)"""
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(all_passages))
            batch_passages = all_passages[start_idx:end_idx]

            if len(batch_passages) == 0:
                return (batch_idx, None)

            prompt = create_batch_summary_prompt(batch_passages)

            try:
                llm_params = {
                    "temperature": 0.0,
                    "max_tokens": 150,
                }

                result = await evaluator._make_completion_request(prompt, llm_params)

                if result.get("choices") and len(result["choices"]) > 0:
                    summary = result["choices"][0].get("text", "").strip()
                    if summary:
                        return (batch_idx, summary)
                    else:
                        print(f"    Batch {batch_idx + 1}/{num_batches}: ❌ Empty response")
                        return (batch_idx, None)
                else:
                    return (batch_idx, None)

            except Exception as e:
                print(f"    Batch {batch_idx + 1}/{num_batches}: ❌ Error: {e}")
                return (batch_idx, None)

        # Execute all batches concurrently
        batch_tasks = [process_batch(i) for i in range(num_batches)]
        results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        # Collect summaries in order
        batch_summaries = []
        for result in results:
            if isinstance(result, Exception):
                print(f"    ❌ Batch error: {result}")
                continue
            if result is not None and isinstance(result, tuple):
                batch_idx, summary = result
                if summary:
                    batch_summaries.append(summary)
                    print(f"    Batch {batch_idx + 1}/{num_batches}: ✓")

        print(f"    Generated {len(batch_summaries)}/{num_batches} summaries")

        if len(batch_summaries) == 0:
            print(f"    ❌ No summaries generated, skipping cluster")
            label = ""
        else:
            # Stage 2: Generate final label from summaries
            print(f"  Cluster {cluster_id}: Stage 2 - Generating final label from {len(batch_summaries)} summaries...")
            prompt = create_final_label_prompt(batch_summaries)

            try:
                llm_params = {
                    "temperature": 0.0,
                    "max_tokens": 200,
                }

                result = await evaluator._make_completion_request(prompt, llm_params)

                if result.get("choices") and len(result["choices"]) > 0:
                    response_text = result["choices"][0].get("text", "").strip()

                    # Parse the LABEL line from the response
                    label = ""
                    for line in response_text.split('\n'):
                        line = line.strip()
                        if line.startswith("LABEL:"):
                            label = line.replace("LABEL:", "").strip()
                            break

                    # Clean up the label
                    label = label.replace('"', '').replace("'", "").strip()

                    print(f"    Final label: '{label}'")
                else:
                    print(f"    ❌ No response from LLM")
                    label = ""

            except Exception as e:
                print(f"    ❌ Error generating final label: {e}")
                label = ""        # Ensure uniqueness
        if label:
            original_label = label
            counter = 2
            while label in used_labels:
                label = f"{original_label} ({counter})"
                counter += 1
            used_labels.add(label)
            cluster_label_map[cluster_id] = label
        else:
            cluster_label_map[cluster_id] = ""

    return cluster_label_map


def update_graph_json_files(graph_data_path: str, cluster_label_map: dict):
    """
    Update the precomputed graph JSON file with cluster labels.

    Args:
        graph_data_path: Path to graph_data directory
        cluster_label_map: Dictionary mapping cluster_id -> label
    """
    # Update precomputed_graph_api.json (for fast API loading)
    precomputed_api_path = os.path.join(graph_data_path, 'precomputed_graph_api.json')
    if os.path.exists(precomputed_api_path):
        with open(precomputed_api_path, 'rb') as f:
            graph_data = orjson.loads(f.read())

        # Add cluster_label to nodes
        for node in graph_data['nodes']:
            cluster_id = node.get('cluster_id', 0)
            node['cluster_label'] = cluster_label_map.get(cluster_id, '')

        with open(precomputed_api_path, 'wb') as f:
            f.write(orjson.dumps(graph_data))
        print(f"✓ Updated precomputed_graph_api.json")


async def generate_and_update_cluster_labels(
    alignments_file: str,
    graph_data_path: str,
    model_path: str = "unsloth/gemma-3-4b-it-qat-GGUF",
    context_window: int = 8192,
    num_batches: int = 5,
    batch_size: int = 20,
    port: int = 8080,
) -> dict:
    """
    Generate thematic labels for each cluster using LLM and update graph JSON files.

    Args:
        alignments_file: Path to alignments file
        graph_data_path: Path to graph_data directory (output from build_graph_model.py)
        model_path: LLM model path or HuggingFace model ID
        context_window: Context window size for LLM
        num_batches: Number of batches to create per cluster (default: 5)
        batch_size: Number of passages per batch (default: 20, so 5*20=100 total)
        port: Port for llama-server

    Returns:
        Dictionary mapping cluster_id -> label
    """
    if not os.path.exists(graph_data_path):
        print(f"Error: Graph data path does not exist: {graph_data_path}")
        raise FileNotFoundError(f"Graph data path not found: {graph_data_path}")

    # Find alignments file (go up one directory from graph_data)
    parent_dir = os.path.dirname(graph_data_path)

    print(f"Using alignments file: {alignments_file}")

    # Load necessary data
    print("\rLoading graph data...", end="", flush=True)
    cluster_labels_modified = np.load(os.path.join(graph_data_path, 'cluster_labels_modified.npy'))
    cluster_centroids = np.load(os.path.join(graph_data_path, 'cluster_centroids.npy'))

    with open(os.path.join(graph_data_path, 'cluster_metadata.json'), 'rb') as f:
        cluster_metadata = orjson.loads(f.read())
    n_clusters = cluster_metadata['n_clusters']

    # Load embeddings (check for memmap first, then regular npy)
    import glob
    embeddings_dirs = glob.glob(os.path.join(parent_dir, "*_embeddings"))
    print(f"\rFound embeddings directories: {embeddings_dirs}")

    if embeddings_dirs:
        embeddings_cache_path = os.path.join(embeddings_dirs[0], "passage_embeddings.dat")
        embeddings_meta_path = os.path.join(embeddings_dirs[0], "metadata.json")

        with open(embeddings_meta_path, 'rb') as f:
            metadata = orjson.loads(f.read())

        sbert_dim = metadata['sbert_dim']
        alignment_counts = metadata['alignment_counts']

        print(f"\rLoading embeddings from memmap...", end="", flush=True)
        embeddings_memmap = np.memmap(
            embeddings_cache_path,
            dtype='float32',
            mode='r',
            shape=(alignment_counts, sbert_dim)
        )
        all_embeddings = embeddings_memmap[:]
    else:
        print("Error: Could not find embeddings directory")
        raise FileNotFoundError("Embeddings directory not found")

    print(f"\r✓ Loaded {n_clusters} clusters with {len(all_embeddings)} passages" + " " * 20)

    # Initialize LLM evaluator
    print(f"\nStarting LLM server on port {port}...", end="", flush=True)
    evaluator = AsyncLLMEvaluator(
        model_path=model_path,
        port=port,
        context_window=context_window,
        concurrency_limit=4
    )

    try:
        evaluator.start_server()
        print("\r✓ LLM server started successfully" + " " * 20)

        # Generate cluster labels - await the async function
        cluster_label_map = await generate_cluster_labels_async(
            alignments_file=alignments_file,
            cluster_labels_modified=cluster_labels_modified,
            cluster_centroids=cluster_centroids,
            embeddings=all_embeddings,
            n_clusters=n_clusters,
            evaluator=evaluator,
            num_batches=num_batches,
            batch_size=batch_size,
        )

        # Save cluster labels
        labels_path = os.path.join(graph_data_path, 'cluster_labels.json')
        # Convert integer keys to strings for JSON
        cluster_label_map_str = {str(k): v for k, v in cluster_label_map.items()}
        with open(labels_path, 'wb') as f:
            f.write(orjson.dumps(cluster_label_map_str))
        print(f"\n✓ Saved cluster labels to {labels_path}")

        # Print some examples
        print("\nSample cluster labels:")
        for cluster_id in list(cluster_label_map.keys())[:10]:
            label = cluster_label_map.get(cluster_id, "")
            if label:
                print(f"  Cluster {cluster_id}: {label}")

        # Update graph JSON files
        update_graph_json_files(graph_data_path, cluster_label_map)

        print("\n✓ Cluster labeling complete!")
        return cluster_label_map

    except Exception as e:
        print(f"Error during cluster labeling: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        print("\nStopping LLM server...")
        evaluator.stop_server()
        if evaluator._session and not evaluator._session.closed:
            await evaluator._session.close()
        print("Server stopped.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate and update cluster labels using LLM.")
    parser.add_argument("--alignments_file", type=str, required=True, help="Path to alignments file")
    parser.add_argument("--graph_data_path", type=str, required=True, help="Path to graph_data directory")
    parser.add_argument("--model_path", type=str, default="unsloth/gemma-3-4b-it-qat-GGUF", help="LLM model path or HuggingFace model ID")
    parser.add_argument("--context_window", type=int, default=8192, help="Context window size for LLM")
    parser.add_argument("--num_batches", type=int, default=25, help="Number of batches per cluster")
    parser.add_argument("--batch_size", type=int, default=20, help="Number of passages per batch")
    parser.add_argument("--port", type=int, default=8080, help="Port for llama-server")

    args = parser.parse_args()

    asyncio.run(generate_and_update_cluster_labels(
        alignments_file=args.alignments_file,
        graph_data_path=args.graph_data_path,
        model_path=args.model_path,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        port=args.port,
        context_window=args.context_window,
    ))