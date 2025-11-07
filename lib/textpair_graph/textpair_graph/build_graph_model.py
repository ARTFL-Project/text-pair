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
        sbert_model = SentenceTransformer(sbert_model_name, device=device)
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


@jit(nopython=True, cache=True)
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


def cluster_alignments(data, output_path):
    """
    Cluster all alignments using HDBSCAN on SBERT embeddings.
    Returns cluster labels for direct lookup at runtime.
    """
    print("\n=== Clustering alignments ===")
    print(f"Using {'GPU-accelerated cuML' if USE_GPU else 'CPU-based'} implementations")

    # Load all SBERT embeddings (from memmap, so no RAM issue)
    print("Loading SBERT embeddings...")
    all_embeddings = data['passage_embeddings_memmap'][:]

    print(f"Embeddings shape: {all_embeddings.shape}")
    sbert_dim = all_embeddings.shape[1]

    # Convert to GPU array if using GPU
    embeddings_array = array_lib.asarray(all_embeddings, dtype=array_lib.float32 if USE_GPU else np.float32)

    # UMAP dimensionality reduction #1: 768 → 100 dims (for clustering)
    umap_dim_100 = 100
    print(f"Reducing dimensionality with UMAP for clustering ({sbert_dim} → {umap_dim_100} dims)...")
    reducer_100d = UMAP(
        n_components=umap_dim_100,
        n_neighbors=15,
        min_dist=0.0,
        metric='euclidean',
        random_state=42
    )
    embeddings_reduced_100d = reducer_100d.fit_transform(embeddings_array)

    # Convert back to numpy if using GPU
    if USE_GPU:
        embeddings_reduced_100d = array_lib.asnumpy(embeddings_reduced_100d)

    print(f"UMAP 100D done! Reduced shape: {embeddings_reduced_100d.shape}")

    # UMAP dimensionality reduction #2: 768 → 2 dims (for visualization)
    umap_dim_2 = 2
    print(f"Reducing dimensionality with UMAP for visualization ({sbert_dim} → {umap_dim_2} dims)...")
    reducer_2d = UMAP(
        n_components=umap_dim_2,
        n_neighbors=50,
        min_dist=0.3,
        metric='euclidean',
        random_state=42
    )
    embeddings_reduced_2d = reducer_2d.fit_transform(embeddings_array)

    # Convert back to numpy if using GPU
    if USE_GPU:
        embeddings_reduced_2d = array_lib.asnumpy(embeddings_reduced_2d)

    print(f"UMAP 2D done! Reduced shape: {embeddings_reduced_2d.shape}")

    # HDBSCAN clustering (using 100D embeddings)
    print("Running HDBSCAN clustering...")

    # Convert to appropriate array type for clustering
    cluster_input = array_lib.asarray(embeddings_reduced_100d, dtype=array_lib.float32 if USE_GPU else np.float32)

    density_fraction = 0.005 # a cluster should cover at least 0.5% of data
    min_cluster_size = max(15, int(density_fraction * len(cluster_input)))  # At least 0.5% of data or 15 points
    print(f"Using min_cluster_size={min_cluster_size} based on density fraction of {density_fraction*100}%")

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=2,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )
    cluster_labels = clusterer.fit_predict(cluster_input)

    # Convert to numpy if using GPU
    if USE_GPU:
        cluster_labels = array_lib.asnumpy(cluster_labels).astype(np.int32)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = (cluster_labels == -1).sum()
    print(f"Found {n_clusters} clusters with {n_noise} noise points")

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

    np.save(os.path.join(output_path, 'cluster_centroids.npy'), centroids_array)
    np.save(os.path.join(output_path, 'cluster_centroids_umap.npy'), centroids_umap_array)
    np.save(os.path.join(output_path, 'embeddings_umap_100d.npy'), embeddings_reduced_100d)  # Save 100D UMAP embeddings
    np.save(os.path.join(output_path, 'embeddings_umap_2d.npy'), embeddings_reduced_2d)  # Save 2D UMAP embeddings
    np.save(os.path.join(output_path, 'cluster_labels_modified.npy'), modified_cluster_labels)  # Save modified labels

    # Compute cluster-to-cluster similarity matrix (cosine similarity) - only for real clusters
    print("\nComputing cluster similarity matrix (real clusters only, excluding singletons)...")
    from sklearn.metrics.pairwise import cosine_similarity

    # Only compute similarity for real clusters (not singletons)
    real_centroids = centroids_array[:n_clusters]
    similarity_matrix = cosine_similarity(real_centroids)
    np.save(os.path.join(output_path, 'cluster_similarity_matrix.npy'), similarity_matrix)
    print(f"✓ Saved cluster similarity matrix (shape: {similarity_matrix.shape}) - real clusters only")

    # Save human-readable similarity stats
    if n_clusters > 1:
        print("\nCluster similarity statistics:")
        upper_tri = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        print(f"  Mean similarity: {upper_tri.mean():.4f}")
        print(f"  Min similarity: {upper_tri.min():.4f}")
        print(f"  Max similarity: {upper_tri.max():.4f}")
        print(f"  Std similarity: {upper_tri.std():.4f}")
        mean_similarity = float(upper_tri.mean())
    else:
        mean_similarity = 0.0

    # Save cluster labels for direct lookup
    np.save(os.path.join(output_path, 'cluster_labels.npy'), cluster_labels)

    # Save metadata
    metadata = {
        'n_clusters': int(n_clusters),
        'n_noise': int(n_noise),
        'total_clusters': int(total_clusters),
        'total_alignments': int(len(cluster_labels)),
        'metric': 'euclidean',
        'centroid_dim': int(centroids_array.shape[1]),
        'mean_cluster_similarity': mean_similarity,
    }
    with open(os.path.join(output_path, 'cluster_metadata.json'), 'wb') as f:
        f.write(orjson.dumps(metadata))

    return cluster_labels


def build_author_cluster_graph(alignments_file: str, output_path: str):
    """
    Build a force-directed graph combining cluster similarity and author contributions.

    Creates:
    1. Author-Cluster contribution vectors (inverse distance to centroids in UMAP space)
    2. Combined similarity matrix (clusters + authors)
    3. Force-directed graph visualization
    """
    print("\n" + "="*60)
    print("BUILDING AUTHOR-CLUSTER GRAPH")
    print("="*60)

    # Load necessary data
    print("Loading data...")
    cluster_labels = np.load(os.path.join(output_path, 'cluster_labels.npy'))
    cluster_labels_modified = np.load(os.path.join(output_path, 'cluster_labels_modified.npy'))  # With singletons
    cluster_centroids_umap = np.load(os.path.join(output_path, 'cluster_centroids_umap.npy'))
    embeddings_umap_100d = np.load(os.path.join(output_path, 'embeddings_umap_100d.npy'))  # 100D for clustering
    embeddings_umap_2d = np.load(os.path.join(output_path, 'embeddings_umap_2d.npy'))  # 2D for visualization
    cluster_similarity = np.load(os.path.join(output_path, 'cluster_similarity_matrix.npy'))

    with open(os.path.join(output_path, 'author_to_id.json'), 'rb') as f:
        author_to_id = orjson.loads(f.read())

    with open(os.path.join(output_path, 'cluster_metadata.json'), 'rb') as f:
        metadata = orjson.loads(f.read())

    n_clusters = metadata['n_clusters']  # Original clusters (excluding noise)
    total_clusters = metadata['total_clusters']  # Includes singletons
    num_authors = len(author_to_id)
    alignment_counts = metadata['total_alignments']

    print(f"  {n_clusters} real clusters + {metadata['n_noise']} singletons = {total_clusters} total")
    print(f"  {num_authors} authors")
    print(f"  {alignment_counts} alignments")



    # Build author to alignments mapping
    print("\nMapping authors to alignments...")
    author_alignments = {author_id: [] for author_id in range(num_authors)}

    alignment_idx = 0
    with lz4.frame.open(alignments_file, "rb") as f:
        for line in tqdm(f, total=alignment_counts, desc="Reading alignments", leave=False):
            alignment = orjson.loads(line)
            source_author_id = author_to_id[alignment["source_author"]]
            target_author_id = author_to_id[alignment["target_author"]]

            author_alignments[source_author_id].append(alignment_idx)
            author_alignments[target_author_id].append(alignment_idx)
            alignment_idx += 1

    # Compute author cluster contribution vectors
    print("\nComputing author-cluster contributions (inverse distance to centroids)...")
    print("  Computing per-cluster average embeddings for quality measurement...")

    # Prepare data as numpy arrays for Numba
    embeddings_umap_array = embeddings_umap_100d.astype(np.float32)  # Use 100D for contribution calculations
    cluster_labels_modified_array = cluster_labels_modified.astype(np.int32)
    cluster_centroids_umap_array = cluster_centroids_umap.astype(np.float32)

    # Compute contributions for each author using Numba-optimized function
    author_cluster_contributions = np.zeros((num_authors, total_clusters), dtype=np.float32)

    for author_id in tqdm(range(num_authors), desc="Processing authors"):
        alignment_indices = np.array(author_alignments[author_id], dtype=np.int32)

        if len(alignment_indices) == 0:
            continue

        # Use Numba-optimized function for the inner computation
        contributions = compute_single_author_contributions(
            alignment_indices,
            cluster_labels_modified_array,
            embeddings_umap_array,
            cluster_centroids_umap_array,
            total_clusters
        )

        author_cluster_contributions[author_id] = contributions

    # Normalize author contributions (softmax-like, per author)
    print("Normalizing author contributions...")
    row_sums = author_cluster_contributions.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero for authors with no alignments
    author_cluster_contributions = author_cluster_contributions / row_sums

    np.save(os.path.join(output_path, 'author_cluster_contributions.npy'), author_cluster_contributions)
    print(f"✓ Saved author-cluster contributions (shape: {author_cluster_contributions.shape})")

    # Build (author, cluster) pair nodes
    print("\n" + "="*60)
    print("BUILDING (AUTHOR, CLUSTER) PAIR GRAPH")
    print("="*60)

    print("\nEnumerating (author, cluster) pairs from alignments...")
    import itertools
    from collections import defaultdict

    # Track (author, cluster) pairs and their passage counts and 2D positions
    pair_passage_counts = defaultdict(int)
    pair_embeddings_2d = defaultdict(list)  # Collect all 2D embeddings for each pair

    alignment_idx = 0
    with lz4.frame.open(alignments_file, "rb") as f:
        for line in tqdm(f, total=alignment_counts, desc="Processing alignments", leave=False):
            alignment = orjson.loads(line)
            source_author_id = author_to_id[alignment["source_author"]]
            target_author_id = author_to_id[alignment["target_author"]]

            # Get cluster for this alignment
            cluster_id = int(cluster_labels_modified[alignment_idx])

            # Get 2D position for this alignment
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

    # Build NetworkX graph
    print("\nBuilding graph with (author, cluster) pair nodes...")
    if USE_GPU:
        print("  Using GPU-accelerated graph operations")

    G = nx.Graph()

    # Add (author, cluster) pair nodes
    id_to_author = {v: k for k, v in author_to_id.items()}

    for (author_id, cluster_id), passage_count in pair_passage_counts.items():
        node_id = f"author_{author_id}_cluster_{cluster_id}"
        position = pair_positions[(author_id, cluster_id)]

        G.add_node(node_id,
                   node_type='author_cluster_pair',
                   author_id=author_id,
                   author_name=id_to_author[author_id],
                   cluster_id=cluster_id,
                   cluster_label="",
                   size=int(passage_count),
                   x=float(position[0]),
                   y=float(position[1]))

    print(f"✓ Added {G.number_of_nodes()} (author, cluster) pair nodes")

    # Add invisible cluster anchor nodes (centroids)

    # Load 2D UMAP embeddings to compute centroid positions
    cluster_centroid_positions_2d = {}

    # Compute 2D centroids for each cluster from the pair embeddings
    for cluster_id in range(total_clusters):
        # Get all 2D positions for nodes in this cluster
        cluster_2d_positions = []
        for pair_key, position in pair_positions.items():
            if pair_key[1] == cluster_id:
                cluster_2d_positions.append(position)

        if cluster_2d_positions:
            # Mean of all 2D positions in this cluster
            mean_2d_position = np.mean(cluster_2d_positions, axis=0)
            cluster_centroid_positions_2d[cluster_id] = mean_2d_position

    # Add cluster anchor nodes (invisible, at centroid positions)
    for cluster_id, position_2d in cluster_centroid_positions_2d.items():
        anchor_node_id = f"anchor_cluster_{cluster_id}"

        G.add_node(anchor_node_id,
                   node_type='cluster_anchor',
                   cluster_id=cluster_id,
                   cluster_label="",
                   size=0.01,  # Very small, nearly invisible
                   x=float(position_2d[0]),
                   y=float(position_2d[1]),
                   hidden=True)


    # Group nodes by cluster
    cluster_nodes = defaultdict(list)
    for node_id in G.nodes():
        if G.nodes[node_id]['node_type'] == 'author_cluster_pair':
            cluster_id = G.nodes[node_id]['cluster_id']
            cluster_nodes[cluster_id].append(node_id)

    # Add intra-cluster edges: complete subgraphs within each cluster
    intra_edges = 0
    intra_weight = 100.0  # High constant weight for intra-cluster cohesion

    # Create complete subgraphs for each cluster
    for cluster_id, nodes in cluster_nodes.items():
        if len(nodes) > 1:
            # Add edges between all pairs of nodes in this cluster
            for node1, node2 in itertools.combinations(nodes, 2):
                G.add_edge(node1, node2, weight=intra_weight, edge_type='intra_cluster')
                intra_edges += 1


    # Add inter-cluster edges: centroid-to-centroid based on cluster similarity
    centroid_edges = 0

    # Use similarity threshold to avoid too many edges
    similarity_threshold = np.percentile(cluster_similarity[np.triu_indices_from(cluster_similarity, k=1)], 75)

    # For each pair of real clusters with high similarity
    for cluster_i in range(n_clusters):
        for cluster_j in range(cluster_i + 1, n_clusters):
            similarity = cluster_similarity[cluster_i, cluster_j]

            if similarity > similarity_threshold:
                anchor_i = f"anchor_cluster_{cluster_i}"
                anchor_j = f"anchor_cluster_{cluster_j}"

                # Only add edge if both anchors exist
                if anchor_i in G.nodes() and anchor_j in G.nodes():
                    G.add_edge(anchor_i, anchor_j, weight=float(similarity * 10), edge_type='centroid_similarity')
                    centroid_edges += 1


    # Save graph as pickle
    import pickle
    with open(os.path.join(output_path, 'author_cluster_pair_graph.pkl'), 'wb') as f:
        pickle.dump(G, f)

    # Save graph as JSON for API consumption
    print("\nConverting graph to JSON format...")
    nodes_json = []
    edges_json = []

    for node_id in G.nodes():
        attrs = G.nodes[node_id]
        # Skip invisible cluster anchor nodes in JSON output
        if attrs.get('node_type') == 'cluster_anchor':
            continue

        node_data = {
            'id': node_id,
            'label': attrs.get('author_name', ''),
            'cluster_label': attrs.get('cluster_label', ''),
            'x': float(attrs.get('x', 0)),
            'y': float(attrs.get('y', 0)),
            'size': int(attrs.get('size', 1)),
            'cluster_id': int(attrs.get('cluster_id', 0)),
            'author_id': int(attrs.get('author_id', 0)),
        }
        nodes_json.append(node_data)

    for source, target, attrs in G.edges(data=True):
        # Skip edges connected to anchor nodes (they're invisible)
        if G.nodes[source].get('node_type') == 'cluster_anchor' or G.nodes[target].get('node_type') == 'cluster_anchor':
            continue

        edge_data = {
            'source': source,
            'target': target,
            'weight': float(attrs.get('weight', 1.0)),
            'edge_type': attrs.get('edge_type', 'default'),
        }
        edges_json.append(edge_data)

    graph_json = {
        'nodes': nodes_json,
        'edges': edges_json,
        'metadata': metadata  # Include cluster metadata (n_clusters, n_noise, etc)
    }

    with open(os.path.join(output_path, 'full_graph.json'), 'wb') as f:
        f.write(orjson.dumps(graph_json))

    print(f"✓ Saved full graph as JSON: {len(nodes_json)} nodes, {len(edges_json)} edges")

    # Save in Graphology format for efficient Sigma.js loading
    print("\nConverting to Graphology format for Sigma.js...")
    graphology_format = {
        'attributes': {
            'n_clusters': int(n_clusters)  # For client-side mini-cluster detection
        },
        'options': {
            'type': 'undirected',
            'multi': False,
            'allowSelfLoops': False
        },
        'nodes': [],
        'edges': []
    }

    # Add all nodes (including anchors for proper graph structure)
    for node_id in G.nodes():
        attrs = G.nodes[node_id]
        node_type = attrs.get('node_type', 'default')

        # Generate color based on cluster
        cluster_id = attrs.get('cluster_id', 0)
        hue = (cluster_id * 360 / total_clusters) % 360
        color = f'hsl({hue}, 70%, 60%)'

        node_entry = {
            'key': node_id,
            'attributes': {
                'label': attrs.get('author_name', ''),
                'cluster_label': attrs.get('cluster_label', ''),
                'x': float(attrs.get('x', 0)),
                'y': float(attrs.get('y', 0)),
                'size': float(attrs.get('size', 1)),
                'mass': 1.0,  # Uniform mass for all nodes
                'color': color,
                'cluster_id': int(cluster_id),
                'node_type': node_type,
            }
        }

        # Make anchor nodes nearly invisible
        if node_type == 'cluster_anchor':
            node_entry['attributes']['size'] = 0.01
            node_entry['attributes']['hidden'] = True
            node_entry['attributes']['color'] = '#00000000'  # Transparent

        graphology_format['nodes'].append(node_entry)

    # Add edges with visual attributes (EXCLUDE anchor_connection edges to reduce redundancy)
    # Keep intra_cluster (complete subgraphs) and centroid_similarity (inter-cluster) edges
    edge_id = 0
    for source, target, attrs in G.edges(data=True):
        edge_type = attrs.get('edge_type', 'default')
        weight = attrs.get('weight', 1.0)

        # Skip anchor_connection edges - they're redundant with intra_cluster complete subgraphs
        if edge_type == 'anchor_connection':
            continue

        # Different colors and sizes for different edge types
        if edge_type == 'intra_cluster':
            color = '#666666'
            size = 2.0
        elif edge_type == 'centroid_similarity':
            color = '#999999'
            size = 1.5
        else:
            color = '#cccccc'
            size = 1.0

        edge_entry = {
            'key': f'edge_{edge_id}',
            'source': source,
            'target': target,
            'attributes': {
                'weight': float(weight),
                'edge_type': edge_type,
                'size': size,
                'color': color,
            }
        }
        graphology_format['edges'].append(edge_entry)
        edge_id += 1

    with open(os.path.join(output_path, 'full_graph_graphology.json'), 'wb') as f:
        f.write(orjson.dumps(graphology_format))

    print(f"✓ Saved Graphology format: {len(graphology_format['nodes'])} nodes, {len(graphology_format['edges'])} edges")

    # Compute graph statistics
    total_degree = sum(G.degree[node] for node in G.nodes())
    avg_degree = total_degree / G.number_of_nodes() if G.number_of_nodes() > 0 else 0

    # Count nodes per cluster
    nodes_per_cluster = {cluster_id: len(nodes) for cluster_id, nodes in cluster_nodes.items()}
    avg_nodes_per_cluster = np.mean(list(nodes_per_cluster.values()))

    graph_stats = {
        'num_nodes': G.number_of_nodes(),
        'num_author_cluster_nodes': len([n for n in G.nodes() if G.nodes[n]['node_type'] == 'author_cluster_pair']),
        'num_anchor_nodes': len([n for n in G.nodes() if G.nodes[n]['node_type'] == 'cluster_anchor']),
        'num_real_clusters': n_clusters,
        'num_singleton_clusters': total_clusters - n_clusters,
        'total_clusters': total_clusters,
        'num_authors': num_authors,
        'num_edges': G.number_of_edges(),
        'num_intra_cluster_edges': intra_edges,
        'num_centroid_edges': centroid_edges,
        'avg_degree': avg_degree,
        'avg_nodes_per_cluster': float(avg_nodes_per_cluster),
        'graph_type': 'author_cluster_pairs_with_centroids',
    }

    with open(os.path.join(output_path, 'graph_stats.json'), 'wb') as f:
        f.write(orjson.dumps(graph_stats))

    print("\n" + "="*60)
    print("(AUTHOR, CLUSTER) PAIR GRAPH STATISTICS:")
    print(f"  Total nodes: {G.number_of_nodes()}")
    print(f"    - Author-cluster pairs: {len([n for n in G.nodes() if G.nodes[n]['node_type'] == 'author_cluster_pair'])}")
    print(f"    - Cluster anchors (invisible): {len([n for n in G.nodes() if G.nodes[n]['node_type'] == 'cluster_anchor'])}")
    print(f"  Clusters: {total_clusters} ({n_clusters} real + {total_clusters - n_clusters} singletons)")
    print(f"  Authors: {num_authors}")
    print(f"  Edges:")
    print(f"    - Intra-cluster (complete subgraphs): {intra_edges}")
    print(f"    - Centroid similarity: {centroid_edges}")
    print(f"    - Total: {G.number_of_edges()}")
    print(f"  Avg degree: {avg_degree:.2f}")
    print(f"  Avg author-cluster nodes per cluster: {avg_nodes_per_cluster:.2f}")
    print("="*60)

    return G


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
    cluster_labels = cluster_alignments(data, output_path)

    # Build full corpus graph
    print("\n" + "="*60)
    print("BUILDING FULL CORPUS GRAPH")
    print("="*60)
    G = build_author_cluster_graph(alignments_file, output_path)

    print("\n✓ Full graph preprocessing complete!")
    print(f"  All files saved to: {output_path}/")


if __name__ == "__main__":
    main()