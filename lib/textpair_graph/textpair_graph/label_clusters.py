"""
Label clusters using LLM after graph building.

This module reads the graph_data output from build_graph_model.py and generates
thematic labels for each cluster using an LLM. It then updates the graph JSON files
with the labels.
"""

import asyncio
import os

import lz4.frame
import numpy as np
import orjson
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from textpair_llm import AsyncLLMEvaluator


def create_cluster_labeling_prompt(top_passages: list[str]) -> str:
    """
    Create prompt for cluster labeling using LLM.
    Passages are the most representative texts in the cluster (closest to centroid).
    """
    passages_text = "\n---\n".join([f"Passage {i+1}: {p[:500]}"
                                     for i, p in enumerate(top_passages)])

    prompt = f"""You are analyzing a thematic cluster of historical text reuse passages.
Below are the most representative passages from this cluster.

Your task is two-fold:
    1. **Rationale:** Briefly (in one sentence) explain the common action or subject matter shared across the passages.
    2. **Label:** Generate a concise thematic label (2-5 words) based on your rationale.

Guidelines:
- Focus on the specific topic, not generic terms
- Use concrete concepts when possible
- Prefer noun phrases
- Be precise but concise

Examples of good labels for reference only:
- "Politics"
- "Geography"
- "Philosophy"

Examples of bad labels:
- "Various Topics" (too generic)
- "The complex interrelation..." (too verbose)
- "Text" (not descriptive)

Representative Passages:
{passages_text}

Output Format:
    RATIONALE: [Your one-sentence explanation.]
    LABEL: [Your 2-5 word final label.]

    Answer with ONLY the RATIONALE and LABEL lines in the specified format, nothing else:"""

    return prompt


async def generate_cluster_labels_async(
    alignments_file: str,
    cluster_labels_modified: np.ndarray,
    cluster_centroids: np.ndarray,
    embeddings: np.ndarray,
    n_clusters: int,
    evaluator,
    top_k: int = 25,
) -> dict:
    """
    Generate thematic labels for each cluster using LLM.

    Args:
        alignments_file: Path to alignments file
        cluster_labels_modified: Cluster assignments (including singletons)
        cluster_centroids: Cluster centroids in SBERT space
        embeddings: All passage embeddings in SBERT space
        n_clusters: Number of real clusters (excluding singletons)
        evaluator: AsyncLLMEvaluator instance
        top_k: Number of top passages to use per cluster

    Returns:
        Dictionary mapping cluster_id -> label
    """
    print(f"\nGenerating labels for {n_clusters} clusters using top {top_k} passages per cluster...")

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

        # Calculate similarity to centroid
        similarities = cosine_similarity(cluster_embeddings, centroid).flatten()

        # Get top-k most similar passages
        top_indices_in_cluster = np.argsort(similarities)[-top_k:][::-1]
        top_alignment_indices = cluster_indices[top_indices_in_cluster]

        # Extract passage texts from alignments file (fetch all at once)
        all_passages = []
        indices_to_fetch = set(int(i) for i in top_alignment_indices)
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

        # Try with progressively fewer passages if context limit exceeded
        passages_to_use = len(all_passages)
        min_passages = 5
        label = ""

        while passages_to_use >= min_passages:
            # Generate prompt with current number of passages
            prompt = create_cluster_labeling_prompt(all_passages[:passages_to_use])

            try:
                llm_params = {
                    "temperature": 0.0,
                    "max_tokens": 200,
                }

                result = await evaluator._make_completion_request(prompt, llm_params)

                if result.get("choices") and len(result["choices"]) > 0:
                    response_text = result["choices"][0].get("text", "").strip()

                    if not response_text:
                        break

                    # Parse the LABEL line from the response
                    for line in response_text.split('\n'):
                        line = line.strip()
                        if line.startswith("LABEL:"):
                            label = line.replace("LABEL:", "").strip()
                            break

                    # If no LABEL: line found, try to extract from full response
                    if not label and len(response_text) < 50 and '\n' not in response_text:
                        label = response_text

                    # Clean up the label
                    label = label.replace('"', '').replace("'", "").strip()

                    # Validate: reject template/placeholder responses
                    invalid_patterns = [
                        'your',
                        'word final label',
                        '[',
                        ']',
                        'insert',
                        'replace',
                        'fill in',
                    ]

                    if label and any(pattern in label.lower() for pattern in invalid_patterns):
                        print(f"\n  Cluster {cluster_id}: Invalid label detected (template response), retrying...")
                        label = ""
                        passages_to_use = max(min_passages, passages_to_use - 5)  # Try with fewer passages
                        continue

                break  # Success, exit retry loop

            except Exception as e:
                error_str = str(e)

                # Check for context overflow error
                if "exceed_context_size" in error_str:
                    # Try to parse token counts for proportional reduction
                    import json
                    try:
                        json_part = error_str.split("HTTP 400:", 1)[1].strip() if "HTTP 400:" in error_str else error_str
                        error_data = json.loads(json_part)
                        n_prompt = error_data.get("error", {}).get("n_prompt_tokens", 0)
                        n_ctx = error_data.get("error", {}).get("n_ctx", 1)

                        if n_prompt and n_ctx:
                            # Reduce proportionally with safety margin
                            passages_to_use = int(passages_to_use * (n_ctx / n_prompt) * 0.85)
                        else:
                            # Couldn't parse, reduce by half
                            passages_to_use = passages_to_use // 2
                    except:
                        # JSON parsing failed, reduce by half
                        passages_to_use = passages_to_use // 2

                    passages_to_use = max(min_passages, passages_to_use)
                    print(f"\n  Cluster {cluster_id}: Context error, retrying with {passages_to_use} passages")
                    continue
                else:
                    # Different error, don't retry
                    print(f"\n  Error generating label for cluster {cluster_id}: {e}")
                    break

        # Ensure uniqueness
        if label:
            original_label = label
            counter = 1
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
    Update the graph JSON files with cluster labels.

    Args:
        graph_data_path: Path to graph_data directory
        cluster_label_map: Dictionary mapping cluster_id -> label
    """
    # Update full_graph.json
    full_graph_path = os.path.join(graph_data_path, 'full_graph.json')
    if os.path.exists(full_graph_path):
        with open(full_graph_path, 'rb') as f:
            graph_data = orjson.loads(f.read())

        # Add cluster_label to nodes
        for node in graph_data['nodes']:
            cluster_id = node.get('cluster_id', 0)
            node['cluster_label'] = cluster_label_map.get(cluster_id, '')

        # Add to metadata (convert keys to strings for JSON)
        if 'metadata' not in graph_data:
            graph_data['metadata'] = {}
        graph_data['metadata']['cluster_labels'] = {str(k): v for k, v in cluster_label_map.items()}

        with open(full_graph_path, 'wb') as f:
            f.write(orjson.dumps(graph_data))
        print(f"✓ Updated full_graph.json")

    # Update full_graph_graphology.json
    graphology_path = os.path.join(graph_data_path, 'full_graph_graphology.json')
    if os.path.exists(graphology_path):
        with open(graphology_path, 'rb') as f:
            graph_data = orjson.loads(f.read())

        # Add cluster_label to node attributes
        for node in graph_data['nodes']:
            cluster_id = node.get('attributes', {}).get('cluster_id', 0)
            if 'attributes' not in node:
                node['attributes'] = {}
            node['attributes']['cluster_label'] = cluster_label_map.get(cluster_id, '')

        with open(graphology_path, 'wb') as f:
            f.write(orjson.dumps(graph_data))
        print(f"✓ Updated full_graph_graphology.json")


async def generate_and_update_cluster_labels(
    alignments_file: str,
    graph_data_path: str,
    model_path: str = "unsloth/gemma-3-4b-it-qat-GGUF",
    context_window: int = 8192,
    top_k: int = 25,
    port: int = 8080,
) -> dict:
    """
    Generate thematic labels for each cluster using LLM and update graph JSON files.

    Args:
        graph_data_path: Path to graph_data directory (output from build_graph_model.py)
        model_path: LLM model path or HuggingFace model ID
        top_k: Number of top passages to use per cluster
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
            top_k=top_k,
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
    parser.add_argument("--top_k", type=int, default=50, help="Number of top passages to use per cluster")
    parser.add_argument("--port", type=int, default=8080, help="Port for llama-server")

    args = parser.parse_args()

    asyncio.run(generate_and_update_cluster_labels(
        alignments_file=args.alignments_file,
        graph_data_path=args.graph_data_path,
        model_path=args.model_path,
        top_k=args.top_k,
        port=args.port,
    ))