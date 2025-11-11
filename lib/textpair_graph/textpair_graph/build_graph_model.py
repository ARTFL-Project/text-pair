""" Building a Thematic Identity Graph Model combining cluster similarity and author contributions."""

import os
import sys

import lz4.frame
import networkx as nx
import numpy as np
import orjson
import torch
from numba import jit
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Check for GPU acceleration libraries and assign implementations
try:
    import cuml.cluster
    import cuml.manifold
    import cupy as cp
    import nx_cugraph as nxcg

    # Use GPU implementations
    UMAP = cuml.manifold.UMAP
    HDBSCAN = cuml.cluster.HDBSCAN
    array_lib = cp  # cupy for GPU arrays
    # Register nx-cugraph as backend (it auto-dispatches for supported algorithms)
    USE_GPU = True
    print("✓ cuML and nx-cugraph detected - GPU acceleration enabled")
except ImportError:
    import hdbscan
    import umap

    # Use CPU implementations
    UMAP = umap.UMAP
    HDBSCAN = hdbscan.HDBSCAN
    array_lib = np  # numpy for CPU arrays
    USE_GPU = False
    print("✓ cuML not available - using CPU implementations")

# Model Hyperparameters
BATCH_SIZE = 4096  # For SBERT encoding


def build_alignment_data(alignments_file: str, alignment_counts: int, sbert_model_name: str):
    """Preprocess alignments and encode passages."""

    print("Building author mapping and encoding passages...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create embeddings cache directory based on model name
    safe_model_name = sbert_model_name.replace('/', '_').replace('\\', '_')
    embeddings_cache_dir = os.path.join(os.path.dirname(alignments_file), f"{safe_model_name}_embeddings")
    os.makedirs(embeddings_cache_dir, exist_ok=True)
    embeddings_cache_path = os.path.join(embeddings_cache_dir, "passage_embeddings.dat")
    embeddings_meta_path = os.path.join(embeddings_cache_dir, "metadata.json")

    # Build author mapping (for metadata/analysis purposes only)
    author_to_id = {}
    current_id = 0

    print("Building author mapping...")
    with lz4.frame.open(alignments_file, "rb") as f:
        for line in tqdm(f, total=alignment_counts, desc="Mapping authors", leave=False):
            alignment = orjson.loads(line)

            # Map authors to IDs
            if alignment["source_author"] not in author_to_id:
                author_to_id[alignment["source_author"]] = current_id
                current_id += 1
            if alignment["target_author"] not in author_to_id:
                author_to_id[alignment["target_author"]] = current_id
                current_id += 1

    # Check if embeddings are cached
    if os.path.exists(embeddings_cache_path) and os.path.exists(embeddings_meta_path):
        print(f"Loading cached embeddings from {embeddings_cache_path}...")
        with open(embeddings_meta_path, 'rb') as f:
            metadata = orjson.loads(f.read())

        cached_count = metadata['alignment_counts']
        sbert_dim = metadata['sbert_dim']

        if cached_count != alignment_counts:
            print(f"⚠ Warning: Cached count ({cached_count}) doesn't match current ({alignment_counts}). Recomputing...")
            os.remove(embeddings_cache_path)
            os.remove(embeddings_meta_path)
            passage_embeddings_memmap = None
        else:
            # Use memmap to read cached embeddings without loading into RAM
            passage_embeddings_memmap = np.memmap(embeddings_cache_path, dtype='float32', mode='r',
                                                  shape=(alignment_counts, sbert_dim))
            print(f"✓ Loaded cached embeddings as memmap: shape={passage_embeddings_memmap.shape}")
    else:
        passage_embeddings_memmap = None

    # Encode passages if not cached
    if passage_embeddings_memmap is None:
        print(f"Computing embeddings (will be cached to {embeddings_cache_path})...")
        sbert_model = SentenceTransformer(sbert_model_name, device=device, model_kwargs={"dtype": torch.float32})
        sbert_dim = sbert_model.get_sentence_embedding_dimension()

        # Create memmap for cache
        passage_embeddings_memmap = np.memmap(embeddings_cache_path, dtype='float32', mode='w+',
                                             shape=(alignment_counts, sbert_dim))

        # Encode passages in batches (streaming approach - never load all passages into memory)
        print("Encoding passages with SBERT...")
        passages_batch = []
        batch_idx = 0

        with lz4.frame.open(alignments_file, "rb") as f:
            for i in tqdm(range(alignment_counts), desc="Encoding passages", total=alignment_counts, leave=False):
                line = f.readline()
                if not line:
                    break

                alignment = orjson.loads(line)
                passage = alignment["source_passage"]
                passages_batch.append(passage)

                # Encode when batch is full
                if len(passages_batch) >= BATCH_SIZE:
                    embeddings = sbert_model.encode(passages_batch, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
                    start = batch_idx * BATCH_SIZE
                    end = start + len(passages_batch)
                    passage_embeddings_memmap[start:end] = embeddings.cpu().numpy()
                    passages_batch = []
                    batch_idx += 1

        # Encode remaining passages
        if len(passages_batch) > 0:
            embeddings = sbert_model.encode(passages_batch, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
            start = batch_idx * BATCH_SIZE
            end = start + len(passages_batch)
            passage_embeddings_memmap[start:end] = embeddings.cpu().numpy()

        passage_embeddings_memmap.flush()

        # Save metadata
        metadata = {
            'sbert_dim': sbert_dim,
            'alignment_counts': alignment_counts,
            'model_name': sbert_model_name
        }
        with open(embeddings_meta_path, 'wb') as f:
            f.write(orjson.dumps(metadata))
        print(f"✓ Embeddings cached to {embeddings_cache_path}")

        del sbert_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        'passage_embeddings_memmap': passage_embeddings_memmap,  # Keep as memmap
        'author_to_id': author_to_id,
        'sbert_dim': sbert_dim,
        'num_authors': len(author_to_id)
    }


# @jit(nopython=True, cache=True)
def compute_single_author_contributions(alignment_indices, cluster_labels_modified, embeddings_umap, cluster_centroids_umap, total_clusters):
    """
    Numba-optimized computation of a single author's cluster contributions.
    For each cluster the author contributes to, compute the inverse distance
    from the author's average embedding (in that cluster) to the cluster centroid.
    """
    contributions = np.zeros(total_clusters, dtype=np.float32)

    if len(alignment_indices) == 0:
        return contributions

    # Get cluster labels for this author's alignments
    author_cluster_labels = cluster_labels_modified[alignment_indices]

    # Find unique clusters (manual since np.unique not supported in nopython mode)
    unique_clusters = []
    for cluster_id in author_cluster_labels:
        if cluster_id not in unique_clusters:
            unique_clusters.append(cluster_id)

    # For each cluster, compute contribution
    for cluster_id in unique_clusters:
        # Count alignments in this cluster and collect their embeddings
        cluster_count = 0
        embedding_dim = embeddings_umap.shape[1]
        cluster_sum = np.zeros(embedding_dim, dtype=np.float32)

        for i in range(len(alignment_indices)):
            if author_cluster_labels[i] == cluster_id:
                alignment_idx = alignment_indices[i]
                cluster_sum += embeddings_umap[alignment_idx]
                cluster_count += 1

        if cluster_count == 0:
            continue

        # Average embedding for this author in this cluster
        cluster_mean = cluster_sum / cluster_count

        # Distance to centroid
        centroid = cluster_centroids_umap[cluster_id]
        distance = 0.0
        for d in range(embedding_dim):
            diff = cluster_mean[d] - centroid[d]
            distance += diff * diff
        distance = np.sqrt(distance)

        # Inverse distance (quality measure)
        inv_distance = 1.0 / (distance + 1e-6)
        contributions[cluster_id] = inv_distance

    return contributions


def cluster_alignments(data, output_path, alignments_file, alignment_counts):
    """
    Cluster all alignments using HDBSCAN on SBERT embeddings.
    Uses passage length filtering: fit UMAP/HDBSCAN on long passages (≥50 chars),
    then transform/predict short passages.
    Returns cluster labels for direct lookup at runtime.
    """
    print("\n=== Clustering alignments ===")
    print(f"Using {'GPU-accelerated cuML' if USE_GPU else 'CPU-based'} implementations")

    # Extract passage lengths (character count excluding whitespace)
    print("Extracting passage lengths...")
    passage_lengths = []

    with lz4.frame.open(alignments_file, "rb") as f:
        for line in tqdm(f, total=alignment_counts, desc="Reading passage lengths", leave=False):
            alignment = orjson.loads(line)
            passage = alignment["source_passage"]
            # Count characters excluding whitespace
            char_count = len(passage.replace(' ', '').replace('\n', '').replace('\t', ''))
            passage_lengths.append(char_count)

    passage_lengths = np.array(passage_lengths)

    # Filter by length threshold (≥50 chars without whitespace)
    length_threshold = 50
    long_passage_mask = passage_lengths >= length_threshold
    n_long = long_passage_mask.sum()
    n_short = (~long_passage_mask).sum()

    print(f"Passage length filtering (threshold: {length_threshold} chars):")
    print(f"  Long passages (≥{length_threshold} chars): {n_long} ({100*n_long/len(passage_lengths):.1f}%)")
    print(f"  Short passages (<{length_threshold} chars): {n_short} ({100*n_short/len(passage_lengths):.1f}%)")
    print(f"  Compute savings: ~{100*n_short/len(passage_lengths):.0f}% fewer passages for clustering")

    # Load all SBERT embeddings (from memmap, so no RAM issue)
    print("Loading SBERT embeddings...")
    all_embeddings = data['passage_embeddings_memmap'][:]

    print(f"Embeddings shape: {all_embeddings.shape}")
    sbert_dim = all_embeddings.shape[1]

    # Split embeddings into long and short passages
    long_embeddings = all_embeddings[long_passage_mask]
    short_embeddings = all_embeddings[~long_passage_mask]

    print(f"Long embeddings shape: {long_embeddings.shape}")
    print(f"Short embeddings shape: {short_embeddings.shape}")

    # Convert to GPU array if using GPU (only long passages for fitting)
    long_embeddings_array = array_lib.asarray(long_embeddings, dtype=array_lib.float32 if USE_GPU else np.float32)

    # UMAP dimensionality reduction #1: 768 → 100 dims (for clustering)
    # FIT ONLY ON LONG PASSAGES
    umap_dim_100 = 100
    # We want the neighbors to be between 15 and 100, but based on dataset size: a cluster should cover ~0.5% of data
    n_neighbors = max(15, min(100, int(0.005 * len(long_embeddings_array))))

    print(f"Fitting UMAP for clustering on long passages ({sbert_dim} → {umap_dim_100} dims)...")
    reducer_100d = UMAP(
        n_components=umap_dim_100,
        n_neighbors=15,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    long_embeddings_reduced_100d = reducer_100d.fit_transform(long_embeddings_array)

    # Convert back to numpy if using GPU
    if USE_GPU:
        long_embeddings_reduced_100d = array_lib.asnumpy(long_embeddings_reduced_100d)

    print(f"UMAP 100D fit done! Reduced long embeddings shape: {long_embeddings_reduced_100d.shape}")

    # TRANSFORM short passages into the fitted 100D space
    print(f"Transforming short passages to 100D space...")
    if n_short > 0:
        short_embeddings_array = array_lib.asarray(short_embeddings, dtype=array_lib.float32 if USE_GPU else np.float32)
        short_embeddings_reduced_100d = reducer_100d.transform(short_embeddings_array)

        if USE_GPU:
            short_embeddings_reduced_100d = array_lib.asnumpy(short_embeddings_reduced_100d)

        print(f"✓ Transformed short embeddings shape: {short_embeddings_reduced_100d.shape}")
    else:
        short_embeddings_reduced_100d = np.zeros((0, umap_dim_100), dtype=np.float32)

    # HDBSCAN clustering (FIT ONLY ON LONG PASSAGES)
    print("Running HDBSCAN clustering on long passages...")

    # Convert to appropriate array type for clustering
    cluster_input = array_lib.asarray(long_embeddings_reduced_100d, dtype=array_lib.float32 if USE_GPU else np.float32)
    print(f"Using min_cluster_size={n_neighbors}")

    # Set leaf size based on dataset size to avoid memory issues
    leaf_size = max(40, min(250, int(len(cluster_input) / 10000))) # between 40 and 250
    print(f"Using leaf_size={leaf_size}")

    clusterer = HDBSCAN(
        min_cluster_size=n_neighbors, # we correlate min_cluster_size with n_neighbors for better results
        min_samples=2,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )
    long_cluster_labels = clusterer.fit_predict(cluster_input)

    # Convert to numpy if using GPU
    if USE_GPU:
        long_cluster_labels = array_lib.asnumpy(long_cluster_labels).astype(np.int32)

    n_clusters = len(set(long_cluster_labels)) - (1 if -1 in long_cluster_labels else 0)
    n_noise_long = (long_cluster_labels == -1).sum()
    print(f"Found {n_clusters} clusters with {n_noise_long} noise points in long passages")

    # PREDICT cluster assignments for short passages using approximate_predict
    if n_short > 0:
        print(f"Predicting cluster assignments for {n_short} short passages...")

        # Convert short embeddings to appropriate array type
        short_embeddings_for_predict = array_lib.asarray(short_embeddings_reduced_100d, dtype=array_lib.float32 if USE_GPU else np.float32)

        if USE_GPU:
            # cuML: approximate_predict is a standalone function
            short_cluster_labels, _ = cuml.cluster.hdbscan.approximate_predict(clusterer, short_embeddings_for_predict)
            short_cluster_labels = array_lib.asnumpy(short_cluster_labels).astype(np.int32)
        else:
            # CPU hdbscan: approximate_predict is a module-level function
            short_cluster_labels, _ = hdbscan.approximate_predict(clusterer, short_embeddings_for_predict)

        n_noise_short = (short_cluster_labels == -1).sum()
        print(f"  Predicted {n_short - n_noise_short} short passages to clusters, {n_noise_short} marked as noise")
    else:
        short_cluster_labels = np.array([], dtype=np.int32)
        n_noise_short = 0

    # Combine cluster labels (long first, then short) to match original order
    cluster_labels = np.zeros(len(passage_lengths), dtype=np.int32)
    cluster_labels[long_passage_mask] = long_cluster_labels
    cluster_labels[~long_passage_mask] = short_cluster_labels

    n_noise = n_noise_long + n_noise_short
    print(f"Total: {n_clusters} clusters with {n_noise} noise points ({n_noise_long} long + {n_noise_short} short)")

    # Combine 100D embeddings (long first, then short) to match original order
    embeddings_reduced_100d = np.zeros((len(passage_lengths), umap_dim_100), dtype=np.float32)
    embeddings_reduced_100d[long_passage_mask] = long_embeddings_reduced_100d
    embeddings_reduced_100d[~long_passage_mask] = short_embeddings_reduced_100d

    print(f"Combined 100D embeddings shape: {embeddings_reduced_100d.shape}")

    # Handle noise cluster: treat each noise point as its own singleton cluster
    print("\nHandling noise cluster (-1)...")
    noise_indices = np.where(cluster_labels == -1)[0]

    # Reassign noise points to singleton clusters starting after the real clusters
    modified_cluster_labels = cluster_labels.copy()
    next_cluster_id = n_clusters

    for noise_idx in noise_indices:
        modified_cluster_labels[noise_idx] = next_cluster_id
        next_cluster_id += 1

    # Update cluster count to include singleton clusters
    total_clusters = n_clusters + n_noise
    print(f"  Created {n_noise} singleton clusters from noise points")
    print(f"  Total clusters (including singletons): {total_clusters}")

    # Compute cluster centroids (mean of all embeddings in each cluster)
    print("\nComputing cluster centroids...")
    cluster_centroids = {}
    cluster_centroids_umap = {}  # Also compute centroids in UMAP space

    # Real clusters (0 to n_clusters-1)
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        if mask.sum() > 0:
            # Use original SBERT embeddings for centroids
            cluster_embeddings = all_embeddings[mask]
            centroid = cluster_embeddings.mean(axis=0)
            cluster_centroids[cluster_id] = centroid

            # Also compute centroid in UMAP space (100D)
            cluster_embeddings_umap = embeddings_reduced_100d[mask]
            centroid_umap = cluster_embeddings_umap.mean(axis=0)
            cluster_centroids_umap[cluster_id] = centroid_umap

    # Singleton clusters (noise points - each is its own centroid)
    for i, noise_idx in enumerate(noise_indices):
        singleton_cluster_id = n_clusters + i
        cluster_centroids[singleton_cluster_id] = all_embeddings[noise_idx]
        cluster_centroids_umap[singleton_cluster_id] = embeddings_reduced_100d[noise_idx]

    # Convert to arrays for easy saving
    centroids_array = np.array([cluster_centroids[i] for i in range(total_clusters)])
    centroids_umap_array = np.array([cluster_centroids_umap[i] for i in range(total_clusters)])

    # Save files needed by API and label_clusters
    np.save(os.path.join(output_path, 'cluster_labels_modified.npy'), modified_cluster_labels)  # API needs this
    np.save(os.path.join(output_path, 'cluster_centroids.npy'), centroids_array)  # label_clusters needs this
    # Note: cluster_centroids_umap.npy, embeddings_umap_100d.npy are NOT saved - not needed

    # UMAP dimensionality reduction #2: 100D → 2 dims (for visualization)
    # CRITICAL: Build from 100D space to match clustering structure, include ALL passages
    print(f"Reducing dimensionality with UMAP for visualization (100D → 2 dims, all passages)...")

    # Convert 100D embeddings to GPU array if needed
    embeddings_100d_array = array_lib.asarray(embeddings_reduced_100d, dtype=array_lib.float32 if USE_GPU else np.float32)

    reducer_2d = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.3,
        metric='euclidean',  # Use euclidean for 100D space (already in UMAP manifold)
        random_state=42
    )
    embeddings_reduced_2d = reducer_2d.fit_transform(embeddings_100d_array)

    # Convert back to numpy if using GPU
    if USE_GPU:
        embeddings_reduced_2d = array_lib.asnumpy(embeddings_reduced_2d)

    print(f"UMAP 2D done! Reduced shape: {embeddings_reduced_2d.shape}")
    np.save(os.path.join(output_path, 'embeddings_umap_2d.npy'), embeddings_reduced_2d)  # Save 2D UMAP embeddings


    # Compute cluster-to-cluster similarity matrix (cosine similarity) - only for real clusters
    print("\nComputing cluster similarity matrix (real clusters only, excluding singletons)...")
    from sklearn.metrics.pairwise import cosine_similarity

    # Only compute similarity for real clusters (not singletons)
    real_centroids = centroids_array[:n_clusters]
    similarity_matrix = cosine_similarity(real_centroids)
    np.save(os.path.join(output_path, 'cluster_similarity_matrix.npy'), similarity_matrix)
    print(f"✓ Saved cluster similarity matrix (shape: {similarity_matrix.shape}) - real clusters only")

    # Note: cluster_labels.npy NOT saved - API uses cluster_labels_modified.npy instead

    # Save metadata (including filtering statistics)
    metadata = {
        'n_clusters': int(n_clusters),
        'n_noise': int(n_noise),
        'total_clusters': int(total_clusters),
        'total_alignments': int(len(cluster_labels)),
        'metric': 'cosine',
        'centroid_dim': int(centroids_array.shape[1]),
        'filtering': {
            'length_threshold': int(length_threshold),
            'n_long_passages': int(n_long),
            'n_short_passages': int(n_short),
            'long_percentage': float(100 * n_long / len(passage_lengths)),
            'short_percentage': float(100 * n_short / len(passage_lengths)),
            'n_noise_long': int(n_noise_long),
            'n_noise_short': int(n_noise_short),
        }
    }
    with open(os.path.join(output_path, 'cluster_metadata.json'), 'wb') as f:
        f.write(orjson.dumps(metadata))

    return cluster_labels


def build_precomputed_api_graph(alignments_file: str, output_path: str):
    """
    Build precomputed graph for API in the same format as get_semantic_graph_data.

    Creates precomputed_graph_api.json with (author, cluster) pair nodes and edges.
    """
    print("\n" + "="*60)
    print("BUILDING PRECOMPUTED API GRAPH")
    print("="*60)

    # Load necessary data
    print("Loading data...")
    cluster_labels_modified = np.load(os.path.join(output_path, 'cluster_labels_modified.npy'))
    embeddings_umap_2d = np.load(os.path.join(output_path, 'embeddings_umap_2d.npy'))
    cluster_similarity = np.load(os.path.join(output_path, 'cluster_similarity_matrix.npy'))

    with open(os.path.join(output_path, 'author_to_id.json'), 'rb') as f:
        author_to_id = orjson.loads(f.read())

    with open(os.path.join(output_path, 'cluster_metadata.json'), 'rb') as f:
        metadata = orjson.loads(f.read())

    n_clusters = metadata['n_clusters']
    total_clusters = metadata['total_clusters']
    num_authors = len(author_to_id)
    alignment_counts = metadata['total_alignments']

    print(f"  {n_clusters} real clusters + {metadata['n_noise']} singletons = {total_clusters} total")
    print(f"  {num_authors} authors")
    print(f"  {alignment_counts} alignments")

    # Build (author, cluster) pair nodes directly from alignments
    print("\nEnumerating (author, cluster) pairs from alignments...")
    from collections import defaultdict

    # Track (author, cluster) pairs and their passage counts and 2D positions
    pair_passage_counts = defaultdict(int)
    pair_embeddings_2d = defaultdict(list)

    alignment_idx = 0
    with lz4.frame.open(alignments_file, "rb") as f:
        for line in tqdm(f, total=alignment_counts, desc="Processing alignments", leave=False):
            alignment = orjson.loads(line)
            source_author_id = author_to_id[alignment["source_author"]]
            target_author_id = author_to_id[alignment["target_author"]]

            cluster_id = int(cluster_labels_modified[alignment_idx])
            embedding_2d = embeddings_umap_2d[alignment_idx]

            # Track both source and target authors
            for author_id in [source_author_id, target_author_id]:
                pair_key = (author_id, cluster_id)
                pair_passage_counts[pair_key] += 1
                pair_embeddings_2d[pair_key].append(embedding_2d)

            alignment_idx += 1

    print(f"✓ Found {len(pair_passage_counts)} unique (author, cluster) pairs")

    # Calculate mean 2D positions for each pair
    print("\nComputing mean 2D positions for each pair...")
    pair_positions = {}
    for pair_key, embeddings_list in pair_embeddings_2d.items():
        mean_position = np.mean(embeddings_list, axis=0)
        pair_positions[pair_key] = mean_position

    # Build precomputed graph in API format (matches get_semantic_graph_data output)
    print("\nCreating precomputed graph for API...")

    id_to_author = {v: k for k, v in author_to_id.items()}

    # Build API-format graph with threshold filtering
    MIN_PASSAGES_THRESHOLD = 5
    api_nodes = []
    api_cluster_nodes = defaultdict(list)

    for (author_id, cluster_id), passage_count in pair_passage_counts.items():
        if passage_count < MIN_PASSAGES_THRESHOLD:
            continue

        node_id = f"author_{author_id}_cluster_{cluster_id}"
        position = pair_positions[(author_id, cluster_id)]

        api_nodes.append({
            'id': node_id,
            'label': id_to_author[author_id],
            'author_id': author_id,
            'author_name': id_to_author[author_id],
            'cluster_id': cluster_id,
            'cluster_label': '',
            'passages': int(passage_count),
            'size': int(passage_count),
            'x': float(position[0]),
            'y': float(position[1])
        })
        api_cluster_nodes[cluster_id].append(node_id)

    # Add cluster anchor nodes
    api_cluster_centroid_positions = {}
    for cluster_id in api_cluster_nodes.keys():
        cluster_2d_positions = []
        for pair_key, position in pair_positions.items():
            if pair_key[1] == cluster_id and pair_passage_counts[pair_key] >= MIN_PASSAGES_THRESHOLD:
                cluster_2d_positions.append(position)

        if cluster_2d_positions:
            mean_2d_position = np.mean(cluster_2d_positions, axis=0)
            api_cluster_centroid_positions[cluster_id] = mean_2d_position

    for cluster_id, position_2d in api_cluster_centroid_positions.items():
        anchor_node_id = f"anchor_cluster_{cluster_id}"
        api_nodes.append({
            'id': anchor_node_id,
            'label': '',
            'node_type': 'cluster_anchor',
            'cluster_id': cluster_id,
            'cluster_label': '',
            'size': 0.01,
            'x': float(position_2d[0]),
            'y': float(position_2d[1]),
            'hidden': True
        })

    # Build API-format edges
    api_edges = []

    # 1. Intra-cluster edges (star topology: nodes to anchors)
    for cluster_id, node_ids in api_cluster_nodes.items():
        if len(node_ids) > 1:
            anchor_id = f"anchor_cluster_{cluster_id}"
            for node_id in node_ids:
                api_edges.append({
                    'source': node_id,
                    'target': anchor_id,
                    'weight': 1.0,
                    'edge_type': 'intra_cluster',
                    'color': '#666666',
                    'size': 0.5
                })

    # 2. Centroid similarity edges
    filtered_cluster_ids = set(api_cluster_nodes.keys())
    for cluster_i in range(n_clusters):
        if cluster_i not in filtered_cluster_ids:
            continue

        for cluster_j in range(cluster_i + 1, n_clusters):
            if cluster_j not in filtered_cluster_ids:
                continue

            similarity = cluster_similarity[cluster_i, cluster_j]

            if similarity > 0:  # No threshold - include all edges
                anchor_i = f"anchor_cluster_{cluster_i}"
                anchor_j = f"anchor_cluster_{cluster_j}"

                api_edges.append({
                    'source': anchor_i,
                    'target': anchor_j,
                    'weight': float(similarity * 10),
                    'edge_type': 'centroid_similarity',
                    'color': '#999999',
                    'size': 1.0
                })

    api_graph = {
        'nodes': api_nodes,
        'edges': api_edges,
        'metadata': {
            'n_clusters': n_clusters,
            'total_nodes': len(api_nodes),
            'total_edges': len(api_edges),
            'min_passages_threshold': MIN_PASSAGES_THRESHOLD
        }
    }

    with open(os.path.join(output_path, 'precomputed_graph_api.json'), 'wb') as f:
        f.write(orjson.dumps(api_graph))

    print(f"✓ Saved precomputed API graph: {len(api_nodes)} nodes, {len(api_edges)} edges")

    print("\n" + "="*60)
    print("PRECOMPUTED API GRAPH STATISTICS:")
    print(f"  Total nodes: {len(api_nodes)}")
    print(f"  Clusters represented: {len(api_cluster_nodes)}")
    print(f"  Authors: {num_authors}")
    print(f"  Total edges: {len(api_edges)}")
    print(f"  Min passages threshold: {MIN_PASSAGES_THRESHOLD}")
    print("="*60)


def main():
    alignments_file = sys.argv[1]
    alignment_counts = int(sys.argv[2])
    sbert_model_name = sys.argv[3] if len(sys.argv) > 3 else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    output_path = os.path.join(os.path.dirname(alignments_file), "graph_data")
    os.makedirs(output_path, exist_ok=True)

    # Preprocess data - load SBERT embeddings
    data = build_alignment_data(alignments_file, alignment_counts, sbert_model_name)

    # Save author mapping for later use
    with open(os.path.join(output_path, 'author_to_id.json'), 'wb') as f:
        f.write(orjson.dumps(data['author_to_id']))

    # Cluster alignments by content similarity
    print("\n" + "="*60)
    print("CLUSTERING ALIGNMENTS")
    print("="*60)
    cluster_labels = cluster_alignments(data, output_path, alignments_file, alignment_counts)

    # Build precomputed graph for API
    build_precomputed_api_graph(alignments_file, output_path)

    print("\n✓ Graph preprocessing complete!")
    print(f"  All files saved to: {output_path}/")


if __name__ == "__main__":
    main()