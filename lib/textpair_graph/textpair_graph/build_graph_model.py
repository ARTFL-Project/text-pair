""" Building a Thematic Identity Graph Model combining cluster similarity and author contributions."""

import gc
import os
import sys

import faiss
import lz4.frame
import networkx as nx
import numpy as np
import orjson
import torch
from numba import jit
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
threads = os.cpu_count()-1
faiss.omp_set_num_threads(threads)

# Check for GPU acceleration libraries and assign implementations
try:
    import cuml
    import cuml.cluster
    import cuml.manifold
    cuml.set_global_output_type('numpy')
    UMAP = cuml.manifold.UMAP
    HDBSCAN = cuml.cluster.HDBSCAN
    USE_GPU = True
except ImportError:
    import hdbscan
    import umap
    UMAP = umap.UMAP
    HDBSCAN = hdbscan.HDBSCAN
    USE_GPU = False

# Model Hyperparameters
BATCH_SIZE = 4096  # For SBERT encoding


def build_alignment_data(alignments_file: str, alignment_counts: int, sbert_model_name: str):
    """Preprocess alignments and encode passages."""

    print("Building author mapping...", end=' ')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create embeddings cache directory based on model name
    safe_model_name = sbert_model_name.replace('/', '_').replace('\\', '_')
    embeddings_cache_dir = os.path.join(os.path.dirname(alignments_file), f"{safe_model_name}_embeddings")
    os.makedirs(embeddings_cache_dir, exist_ok=True)
    embeddings_cache_path = os.path.join(embeddings_cache_dir, "passage_embeddings.dat")
    embeddings_meta_path = os.path.join(embeddings_cache_dir, "metadata.json")

    author_to_id = {}
    current_id = 0

    with lz4.frame.open(alignments_file, "rb") as f:
        for line in tqdm(f, total=alignment_counts, desc="Mapping authors", leave=False):
            alignment = orjson.loads(line)

            if alignment["source_author"] not in author_to_id:
                author_to_id[alignment["source_author"]] = current_id
                current_id += 1
            if alignment["target_author"] not in author_to_id:
                author_to_id[alignment["target_author"]] = current_id
                current_id += 1
    print(f"done. Total authors: {len(author_to_id)}")

    # Check if embeddings are cached
    if os.path.exists(embeddings_cache_path) and os.path.exists(embeddings_meta_path):
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
            passage_embeddings_memmap = np.memmap(embeddings_cache_path, dtype='float32', mode='r',
                                                  shape=(alignment_counts, sbert_dim))
    else:
        passage_embeddings_memmap = None

    if passage_embeddings_memmap is None:
        sbert_model = SentenceTransformer(sbert_model_name, device=device, model_kwargs={"dtype": torch.float32})
        sbert_dim = sbert_model.get_sentence_embedding_dimension()

        passage_embeddings_memmap = np.memmap(embeddings_cache_path, dtype='float32', mode='w+',
                                             shape=(alignment_counts, sbert_dim))

        print("Embedding passages...", end=' ')
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

                if len(passages_batch) >= BATCH_SIZE:
                    embeddings = sbert_model.encode(passages_batch, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
                    start = batch_idx * BATCH_SIZE
                    end = start + len(passages_batch)
                    passage_embeddings_memmap[start:end] = embeddings.cpu().numpy()
                    passages_batch = []
                    batch_idx += 1

        if len(passages_batch) > 0:
            embeddings = sbert_model.encode(passages_batch, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
            start = batch_idx * BATCH_SIZE
            end = start + len(passages_batch)
            passage_embeddings_memmap[start:end] = embeddings.cpu().numpy()

        passage_embeddings_memmap.flush()

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

    print("done.")

    return {
        'passage_embeddings_memmap': passage_embeddings_memmap,
        'author_to_id': author_to_id,
        'sbert_dim': sbert_dim,
        'num_authors': len(author_to_id)
    }

def cluster_alignments(data, output_path, alignments_file, alignment_counts):
    """
    Cluster all alignments using HDBSCAN on SBERT embeddings.
    Returns cluster labels for direct lookup at runtime.
    """
    all_embeddings = data['passage_embeddings_memmap']
    sbert_dim = all_embeddings.shape[1]

    low_dim = 32
    n_neighbors = min(100, max(15, alignment_counts // 10000))
    print(f"Reducing embeddings dimensionality ({sbert_dim} to {low_dim}D)...", end=' ')
    reducer_low_dim = UMAP(
        n_components=low_dim,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    low_dim_embeddings = reducer_low_dim.fit_transform(all_embeddings)
    print("done.")
    gc.collect()

    n_total = len(low_dim_embeddings)

    # Adaptive train size: cap training set at 1M for large datasets
    if n_total >= 1_000_000:
        train_size = min(1_000_000, n_total)  # Cap at 1M for training
        train_ratio = train_size / n_total
        print(f"Splitting data: {train_size:,} train ({train_ratio*100:.1f}%), {n_total - train_size:,} test")
        train_indices, test_indices = train_test_split(
            np.arange(n_total),
            train_size=train_size,
            shuffle=True,
            random_state=42
        )

        train_embeddings = low_dim_embeddings[train_indices]
        test_embeddings = low_dim_embeddings[test_indices]

        print(f"Clustering with HDBSCAN on {len(train_embeddings):,} training passages...", end=' ')
        min_cluster_size = max(15, int(0.005 * len(train_embeddings)))
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=2,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        train_labels = clusterer.fit_predict(train_embeddings)
        train_labels = train_labels.astype(np.int32)

        n_clusters = len(set(train_labels)) - (1 if -1 in train_labels else 0)
        n_noise_train = (train_labels == -1).sum()
        print(f"done. Found {n_clusters} clusters with {n_noise_train} noise points in training set")

        # Predict test set in batches to avoid OOM
        print(f"Predicting labels for {len(test_embeddings):,} test passages in batches...")
        predict_batch_size = 100000
        test_labels = np.zeros(len(test_embeddings), dtype=np.int32)
        n_test_batches = (len(test_embeddings) + predict_batch_size - 1) // predict_batch_size

        for i in tqdm(range(0, len(test_embeddings), predict_batch_size),
                      desc="Predicting test batches", total=n_test_batches, leave=False):
            end_idx = min(i + predict_batch_size, len(test_embeddings))
            batch = test_embeddings[i:end_idx]

            if USE_GPU:
                batch_labels, _ = cuml.cluster.hdbscan.approximate_predict(clusterer, batch)
            else:
                batch_labels, _ = hdbscan.approximate_predict(clusterer, batch)

            test_labels[i:end_idx] = batch_labels.astype(np.int32)

        n_noise_test = (test_labels == -1).sum()
        print(f"done. {n_noise_test:,} noise points in test set")

        # Combine labels in original order
        cluster_labels = np.zeros(n_total, dtype=np.int32)
        cluster_labels[train_indices] = train_labels
        cluster_labels[test_indices] = test_labels

        del train_embeddings, test_embeddings, train_labels, test_labels
        gc.collect()
    else:
        print(f"Clustering all {n_total:,} passages with HDBSCAN...", end=' ')
        min_cluster_size = max(15, int(0.005 * n_total))
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=2,
            metric='euclidean',
            cluster_selection_method='eom',
            cluster_selection_epsilon=0.3,
            prediction_data=True
        )
        cluster_labels = clusterer.fit_predict(low_dim_embeddings)
        cluster_labels = cluster_labels.astype(np.int32)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        print(f"done. Found {n_clusters} clusters")
        gc.collect()

    del clusterer, reducer_low_dim
    gc.collect()

    n_noise = (cluster_labels == -1).sum()
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"Final: {n_clusters} clusters with {n_noise} total outliers ({100*n_noise/len(cluster_labels):.1f}%)")




    # Handle noise cluster: treat each noise point as its own singleton cluster
    noise_indices = np.where(cluster_labels == -1)[0]

    # Reassign noise points to singleton clusters starting after the real clusters
    modified_cluster_labels = cluster_labels.copy()
    next_cluster_id = n_clusters

    for noise_idx in noise_indices:
        modified_cluster_labels[noise_idx] = next_cluster_id
        next_cluster_id += 1
    total_clusters = n_clusters + n_noise

    print("Computing cluster centroids...", end=" ")
    cluster_centroids = {}
    cluster_centroids_low_dim = {}

    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        if mask.sum() > 0:
            cluster_embeddings = all_embeddings[mask]
            centroid = cluster_embeddings.mean(axis=0)
            cluster_centroids[cluster_id] = centroid

            cluster_low_dim_embeddings = low_dim_embeddings[mask]
            centroid_low_dim = cluster_low_dim_embeddings.mean(axis=0)
            cluster_centroids_low_dim[cluster_id] = centroid_low_dim

    for i, noise_idx in enumerate(noise_indices):
        singleton_cluster_id = n_clusters + i
        cluster_centroids[singleton_cluster_id] = all_embeddings[noise_idx]
        cluster_centroids_low_dim[singleton_cluster_id] = low_dim_embeddings[noise_idx]

    centroids_array = np.array([cluster_centroids[i] for i in range(total_clusters)])
    centroids_low_dim_array = np.array([cluster_centroids_low_dim[i] for i in range(total_clusters)])

    np.save(os.path.join(output_path, 'cluster_labels_modified.npy'), modified_cluster_labels)
    np.save(os.path.join(output_path, 'cluster_centroids.npy'), centroids_array)
    print("done.")

    print(f"Reducing dimensionality with UMAP for visualization ({low_dim}D → 2D)...", end=' ')

    reducer_2d = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.0,
        metric='euclidean',
        random_state=42
    )
    embeddings_2d = reducer_2d.fit_transform(low_dim_embeddings)

    del reducer_2d
    gc.collect()

    np.save(os.path.join(output_path, 'embeddings_umap_2d.npy'), embeddings_2d)

    print("done.")

    print("Computing cluster similarity matrix...", end=' ')
    real_centroids_low_dim = centroids_low_dim_array[:n_clusters]
    similarity_matrix = cosine_similarity(real_centroids_low_dim)
    np.save(os.path.join(output_path, 'cluster_similarity_matrix.npy'), similarity_matrix)
    print("done.")

    metadata = {
        'n_clusters': int(n_clusters),
        'n_noise': int(n_noise),
        'total_clusters': int(total_clusters),
        'total_alignments': int(len(cluster_labels)),
        'metric': 'cosine',
        'centroid_dim': int(centroids_array.shape[1]),
        'n_outliers': int(n_noise),
        'outlier_percentage': float(100 * n_noise / len(cluster_labels)),
    }
    with open(os.path.join(output_path, 'cluster_metadata.json'), 'wb') as f:
        f.write(orjson.dumps(metadata))

    return cluster_labels


def build_precomputed_api_graph(alignments_file: str, output_path: str):
    """
    Build precomputed graph for API in the same format as get_semantic_graph_data.

    Creates precomputed_graph_api.json with (author, cluster) pair nodes and edges.
    """
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

    print("\nEnumerating (author, cluster) pairs from alignments...")
    from collections import defaultdict

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

    print(f"✓ Saved precomputed full graph: {len(api_nodes)} nodes, {len(api_edges)} edges")


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
    cluster_labels = cluster_alignments(data, output_path, alignments_file, alignment_counts)

    # Build precomputed graph for API
    build_precomputed_api_graph(alignments_file, output_path)

    print("\nThematic Identify Graph done.")


if __name__ == "__main__":
    main()