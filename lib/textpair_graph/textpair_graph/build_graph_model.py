"""Building a Thematic Identity Graph Model combining cluster similarity and author contributions."""

import gc
import hashlib
import os
import sys
from collections import defaultdict

import faiss
import lz4.frame
import numpy as np
import orjson
import pacmap
import torch
from numba import jit
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .passage_expansion import build_expansion_map

os.environ["TOKENIZERS_PARALLELISM"] = "false"
threads = os.cpu_count() - 1
faiss.omp_set_num_threads(threads)

# Check for GPU acceleration libraries and assign implementations
try:
    import cuml
    import cuml.cluster
    import cuml.manifold

    cuml.set_global_output_type("numpy")
    UMAP = cuml.manifold.UMAP
    HDBSCAN = cuml.cluster.HDBSCAN
    USE_GPU = True
except Exception as _cuml_error:  # noqa: BLE001
    # Broad on purpose: a cuML/scikit-learn version skew raises AttributeError
    # mid-import, and an optional accelerator should not break this module.
    if not isinstance(_cuml_error, ImportError):
        print(f"cuML present but unusable ({_cuml_error}); falling back to CPU UMAP/HDBSCAN.")
    import hdbscan
    import umap

    UMAP = umap.UMAP
    HDBSCAN = hdbscan.HDBSCAN
    USE_GPU = False

# Model Hyperparameters
BATCH_SIZE = 4096  # For SBERT encoding

# Decoupled from min_cluster_size: letting hdbscan default it there coarsens
# the density estimate and collapses the partition.
MIN_SAMPLES = 15

# "eom" prefers a few large clusters; "leaf" more and smaller ones.
CLUSTER_SELECTION_METHOD = "eom"

# Clustering happens in 5D; 2D is display only.
DEFAULT_UMAP_COMPONENTS = 5
# Not scaled with corpus size: a property of the embedding space, not row count.
DEFAULT_UMAP_NEIGHBORS = 15
# Token cap for encoding; bounds peak memory on the long tail of passage lengths.
DEFAULT_MAX_SEQ_LENGTH = 512
# Every cluster pair has positive centroid cosine, so keeping all edges gives a
# near-uniform complete graph.
DEFAULT_SIMILARITY_NEIGHBORS = 4
# Configurable: some corpora record "no attribution" as a name.
DEFAULT_SOURCE_AUTHOR_FIELD = "source_author"
DEFAULT_TARGET_AUTHOR_FIELD = "target_author"

# Neighbours for the 2D projection (LocalMAP's own default is 10).
DEFAULT_LAYOUT_NEIGHBORS = 30

# Share of cross-theme neighbour pairs rewired onto same-theme ones. Higher
# tightens themes but flattens the geometry that makes position readable.
LAYOUT_LABEL_WEIGHT = 0.3


def build_alignment_data(
    alignments_file: str,
    alignment_counts: int,
    sbert_model_name: str,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    source_author_field: str = DEFAULT_SOURCE_AUTHOR_FIELD,
    target_author_field: str = DEFAULT_TARGET_AUTHOR_FIELD,
) -> dict:
    """Preprocess alignments and encode passages."""

    print("Building author mapping...", end=" ")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create embeddings cache directory based on model name
    safe_model_name = sbert_model_name.replace("/", "_").replace("\\", "_")
    embeddings_cache_dir = os.path.join(os.path.dirname(alignments_file), f"{safe_model_name}_embeddings")
    os.makedirs(embeddings_cache_dir, exist_ok=True)
    embeddings_cache_path = os.path.join(embeddings_cache_dir, "passage_embeddings.dat")
    embeddings_meta_path = os.path.join(embeddings_cache_dir, "metadata.json")

    author_to_id = {}
    current_id = 0

    with lz4.frame.open(alignments_file, "rb") as f:
        for line in tqdm(f, total=alignment_counts, desc="Mapping authors", leave=False):
            alignment = orjson.loads(line)

            # .get, not []: below the doc object type some text units have no
            # author key at all. Blank is the no-author case downstream.
            for field in (source_author_field, target_author_field):
                name = alignment.get(field) or ""
                if name not in author_to_id:
                    author_to_id[name] = current_id
                    current_id += 1
    print(f"done. Total authors: {len(author_to_id)}")

    # Check if embeddings are cached
    if os.path.exists(embeddings_cache_path) and os.path.exists(embeddings_meta_path):
        with open(embeddings_meta_path, "rb") as f:
            metadata = orjson.loads(f.read())

        cached_count = metadata["alignment_counts"]
        sbert_dim = metadata["sbert_dim"]

        if cached_count != alignment_counts:
            print(
                f"⚠ Warning: Cached count ({cached_count}) doesn't match current ({alignment_counts}). Recomputing..."
            )
            os.remove(embeddings_cache_path)
            os.remove(embeddings_meta_path)
            passage_embeddings_memmap = None
        else:
            passage_embeddings_memmap = np.memmap(
                embeddings_cache_path,
                dtype="float32",
                mode="r",
                shape=(alignment_counts, sbert_dim),
            )
    else:
        passage_embeddings_memmap = None

    if passage_embeddings_memmap is None:
        sbert_model = SentenceTransformer(sbert_model_name, device=device, model_kwargs={"dtype": torch.float32})
        # Cap sequence length: attention is quadratic and the passage-length
        # tail is long, so a few outliers otherwise set peak memory for the run.
        if max_seq_length and sbert_model.max_seq_length > max_seq_length:
            print(f"(capping max_seq_length {sbert_model.max_seq_length} -> {max_seq_length}) ", end="")
            sbert_model.max_seq_length = max_seq_length
        sbert_dim = sbert_model.get_sentence_embedding_dimension()

        passage_embeddings_memmap = np.memmap(
            embeddings_cache_path,
            dtype="float32",
            mode="w+",
            shape=(alignment_counts, sbert_dim),
        )

        # Short passages get their enclosing and neighbouring sentences first:
        # bare fragments are word lists an embedder cannot place.
        expanded_passages = build_expansion_map(alignments_file, alignment_counts)

        print("Embedding passages...", end=" ")
        passages_batch = []
        batch_idx = 0

        with lz4.frame.open(alignments_file, "rb") as f:
            for i in tqdm(
                range(alignment_counts),
                desc="Encoding passages",
                total=alignment_counts,
                leave=False,
            ):
                line = f.readline()
                if not line:
                    break

                alignment = orjson.loads(line)
                passage = expanded_passages.get(i) or alignment["source_passage"]
                passages_batch.append(passage)

                if len(passages_batch) >= BATCH_SIZE:
                    embeddings = sbert_model.encode(
                        passages_batch,
                        normalize_embeddings=True,
                        convert_to_tensor=True,
                        show_progress_bar=False,
                    )
                    start = batch_idx * BATCH_SIZE
                    end = start + len(passages_batch)
                    passage_embeddings_memmap[start:end] = embeddings.cpu().numpy()
                    passages_batch = []
                    batch_idx += 1

        if len(passages_batch) > 0:
            embeddings = sbert_model.encode(
                passages_batch,
                normalize_embeddings=True,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            start = batch_idx * BATCH_SIZE
            end = start + len(passages_batch)
            passage_embeddings_memmap[start:end] = embeddings.cpu().numpy()

        passage_embeddings_memmap.flush()

        metadata = {
            "sbert_dim": sbert_dim,
            "alignment_counts": alignment_counts,
            "model_name": sbert_model_name,
        }
        with open(embeddings_meta_path, "wb") as f:
            f.write(orjson.dumps(metadata))
        print(f"✓ Embeddings cached to {embeddings_cache_path}")

        del sbert_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("done.")

    return {
        "passage_embeddings_memmap": passage_embeddings_memmap,
        "author_to_id": author_to_id,
        "sbert_dim": sbert_dim,
        "num_authors": len(author_to_id),
        # So the UMAP reduction can be cached alongside the embeddings.
        "embeddings_cache_dir": embeddings_cache_dir,
    }


def _cached_projection(cache_dir: str | None, name: str, key: dict, shape: tuple, compute):
    """Memoize a UMAP projection on disk.

    HDBSCAN costs about 1.5% of the reduction it runs on (measured: 0.3s against
    19s for 1024->5D over 16k rows), so caching the projection is what makes
    re-clustering with a different min_cluster_size interactive rather than a
    fresh twenty-second wait.
    """
    if not cache_dir:
        return compute()
    data_path = os.path.join(cache_dir, f"{name}.dat")
    meta_path = os.path.join(cache_dir, f"{name}.json")
    if os.path.exists(data_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, "rb") as f:
                cached = orjson.loads(f.read())
            if cached == key:
                print(f"(reusing cached {name}) ", end="", flush=True)
                return np.array(np.memmap(data_path, dtype="float32", mode="r", shape=shape))
        except Exception:
            pass  # unreadable cache is a cache miss, not an error
    result = np.asarray(compute(), dtype=np.float32)
    try:
        mm = np.memmap(data_path, dtype="float32", mode="w+", shape=result.shape)
        mm[:] = result
        mm.flush()
        del mm
        with open(meta_path, "wb") as f:
            f.write(orjson.dumps(key))
    except Exception as e:
        print(f"(could not cache {name}: {e}) ", end="")
    return result


def _balance_stats(labels: np.ndarray) -> dict:
    """Shape of a partition, by measures that do not depend on cluster count.

    A largest-cluster share threshold cannot be used here: 10% is twice the mean
    at k=20 but thirty times the mean at k=300. Ratio-to-mean inverts outright --
    measured, an 8-cluster partition with a 41.5% largest cluster scores *more*
    balanced than a 287-cluster one with a 2.3% largest. Gini and normalized
    entropy both separate usable from degenerate partitions and are scale-free.
    """
    k = len(set(int(x) for x in labels)) - (1 if -1 in labels else 0)
    if k <= 0:
        return {"k": 0, "noise": 1.0, "largest": 0.0, "gini": 1.0, "entropy": 0.0}
    sizes = np.array([(labels == i).sum() for i in range(k)], dtype=float)
    sizes = sizes[sizes > 0]
    k = len(sizes)
    p = sizes / sizes.sum()
    entropy = float(-(p * np.log(p)).sum() / np.log(k)) if k > 1 else 0.0
    cum = np.cumsum(np.sort(p))
    gini = float(1 - 2 * np.trapezoid(cum, dx=1 / k) + 1 / k)
    return {
        "k": int(k),
        "noise": float((labels == -1).sum() / len(labels)),
        "largest": float(p.max()),
        "top5": float(np.sort(p)[::-1][:5].sum()),
        "gini": gini,
        "entropy": entropy,
    }


def _candidate_min_cluster_sizes(n_rows: int, steps: int = 9) -> list[int]:
    """Candidate min_cluster_size values, 10 up to n/80.

    Geometric rather than linear: the interesting range is at the bottom. The
    ceiling is where candidates stop yielding a usable number of clusters.
    """
    ceiling = max(40, n_rows // 80)
    floor = 10
    if ceiling <= floor:
        return [floor]

    def _round(value: float) -> int:
        if value < 20:
            return int(round(value))
        if value <= 100:
            return int(round(value / 5) * 5)
        if value <= 1000:
            return int(round(value / 10) * 10)
        return int(round(value / 50) * 50)

    ratio = (ceiling / floor) ** (1 / (steps - 1))
    out: list[int] = []
    for i in range(steps):
        value = _round(floor * ratio**i)
        if value not in out:
            out.append(value)
    return out


SELECTION_METHODS = ("eom", "leaf")


def sweep_min_cluster_size(
    low_dim_embeddings: np.ndarray,
    candidates: list[int] | None = None,
    methods: tuple[str, ...] | None = None,
) -> list[dict]:
    """Cluster at each (min_cluster_size, selection method) and report the shape.

    Both dimensions, because which one matters is corpus-dependent: sweeping
    min_cluster_size alone can miss the only balanced partition. Cheap, since
    the reduction is computed once and each cell is one HDBSCAN fit.
    """
    candidates = candidates or _candidate_min_cluster_sizes(len(low_dim_embeddings))
    methods = methods or SELECTION_METHODS
    rows = []
    for method in methods:
        for mcs in candidates:
            clusterer = HDBSCAN(
                min_cluster_size=mcs,
                min_samples=MIN_SAMPLES,
                metric="euclidean",
                cluster_selection_method=method,
                prediction_data=True,
            )
            stats = _balance_stats(clusterer.fit_predict(low_dim_embeddings).astype(np.int32))
            stats["min_cluster_size"] = mcs
            stats["cluster_selection_method"] = method
            rows.append(stats)
    return rows


def choose_clustering(rows: list[dict]) -> tuple[int, str]:
    """Pick the (min_cluster_size, selection method) with the most balanced partition.

    Balance is judged by Gini over cluster sizes, which is scale-free, unlike a
    largest-cluster share threshold. Not selected on noise, which is
    non-monotonic: large clusters absorb points that met no size threshold
    before, so the lowest-noise settings are often the worst partitions. Ties
    break toward fewer clusters.

    Known weakness: Gini alone favours fewer, more even clusters even when that
    means discarding much more of the corpus as noise.
    """
    usable = [r for r in rows if r["k"] > 1]
    if not usable:
        first = rows[0]
        return first["min_cluster_size"], first.get("cluster_selection_method", CLUSTER_SELECTION_METHOD)
    best = min(usable, key=lambda r: (round(r["gini"], 3), r["k"]))
    return best["min_cluster_size"], best.get("cluster_selection_method", CLUSTER_SELECTION_METHOD)


def format_sweep(rows: list[dict], n_rows: int) -> str:
    lines = [
        f"{'mcs':>6} {'method':>7} {'clusters':>9} {'noise':>7} {'largest':>8} {'top5':>7} {'gini':>6} {'entropy':>8}",
    ]
    for r in rows:
        lines.append(
            f"{r['min_cluster_size']:>6} {r.get('cluster_selection_method', ''):>7} {r['k']:>9} "
            f"{100 * r['noise']:>6.1f}% {100 * r['largest']:>7.1f}% {100 * r.get('top5', 0):>6.1f}% "
            f"{r['gini']:>6.3f} {r['entropy']:>8.3f}"
        )
    lines.append("")
    lines.append("Shares are of themed alignments. Judge by gini/entropy (scale-free), not by")
    lines.append("noise -- noise is non-monotonic because large clusters absorb former outliers.")
    return "\n".join(lines)


def cluster_alignments(
    data: dict,
    output_path: str,
    alignment_counts: int,
    merge_unthemed: bool = True,
    min_cluster_size: int | str | None = "auto",
    umap_components: int = DEFAULT_UMAP_COMPONENTS,
    umap_neighbors: int = DEFAULT_UMAP_NEIGHBORS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cluster all alignments using HDBSCAN on SBERT embeddings.

    Writes two partitions. `cluster_labels_core.npy` is what HDBSCAN assigned
    directly, keeping -1 for noise; labeling reads it so that merged-in points
    cannot dilute a cluster's characteristic terms. `cluster_labels_modified.npy`
    is what the graph is built from: identical to the core unless
    `merge_unthemed` is set, in which case noise is assigned to its nearest
    cluster centroid.

    `min_cluster_size="auto"` sweeps candidate values and takes the largest one
    whose partition is still balanced (see `choose_min_cluster_size`). This is
    affordable because the UMAP reduction is cached and each candidate costs one
    HDBSCAN fit.
    """
    all_embeddings = data["passage_embeddings_memmap"]
    sbert_dim = all_embeddings.shape[1]

    low_dim = umap_components
    n_neighbors = umap_neighbors
    cache_dir = data.get("embeddings_cache_dir")
    n_total_embeddings = all_embeddings.shape[0]
    embeddings_digest = _embeddings_fingerprint(all_embeddings, n_total_embeddings)
    reduction_key = {
        "n_rows": int(n_total_embeddings),
        "n_components": int(low_dim),
        "n_neighbors": int(n_neighbors),
        "metric": "cosine",
        "backend": "cuml" if USE_GPU else "cpu",
        "embeddings": embeddings_digest,
    }

    print(f"Reducing embeddings dimensionality ({sbert_dim} to {low_dim}D)...", end=" ")
    low_dim_embeddings = _cached_projection(
        cache_dir, f"low_dim_{low_dim}d", reduction_key,
        (n_total_embeddings, low_dim),
        lambda: _reduce_to_low_dim(all_embeddings, n_total_embeddings, low_dim, n_neighbors, sbert_dim),
    )
    print("done.")
    gc.collect()

    n_total = len(low_dim_embeddings)

    selection_method = CLUSTER_SELECTION_METHOD
    if min_cluster_size in (None, "auto", "AUTO"):
        print("Selecting clustering parameters (auto)...", flush=True)
        rows = sweep_min_cluster_size(low_dim_embeddings)
        print(format_sweep(rows, n_total))
        min_cluster_size, selection_method = choose_clustering(rows)
        print(f"→ chose min_cluster_size={min_cluster_size}, selection={selection_method}\n")
    min_cluster_size = int(min_cluster_size)

    cluster_labels = _fit_hdbscan(
        low_dim_embeddings, n_total, min_cluster_size, selection_method=selection_method
    )
    n_noise = int((cluster_labels == -1).sum())
    n_clusters = len(set(int(x) for x in cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"Final: {n_clusters} clusters with {n_noise} total outliers ({100 * n_noise / len(cluster_labels):.1f}%)")

    return _finish_clustering(
        all_embeddings,
        low_dim_embeddings,
        cluster_labels,
        n_clusters,
        n_noise,
        n_total,
        output_path,
        cache_dir,
        low_dim,
        min_cluster_size,
        merge_unthemed,
        selection_method,
        n_neighbors,
    )

def _label_aware_pairs(pairs, labels, embeddings, weight, n_neighbors, seed=0):
    """Rewire a fraction of cross-theme neighbour pairs onto same-theme ones.

    LocalMAP takes no labels but does accept the neighbour pairs it would build
    itself, and a pair is an attractive force. Replacements come from the
    point's nearest same-theme neighbours, so a theme tightens without
    collapsing onto its centre.
    """
    rng = np.random.default_rng(seed)
    out = pairs.copy()
    cross = np.where(labels[out[:, 0]] != labels[out[:, 1]])[0]
    chosen = cross[rng.random(len(cross)) < weight]
    if not len(chosen):
        return out

    # Once per theme, via faiss: a per-theme sklearn fit is quadratic in the
    # theme's size.
    by_theme = defaultdict(list)
    for row in chosen:
        by_theme[int(labels[out[row, 0]])].append(row)

    for theme, rows in by_theme.items():
        member_idx = np.where(labels == theme)[0]
        if len(member_idx) < 2:
            continue
        vectors = np.ascontiguousarray(embeddings[member_idx], dtype=np.float32)
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)
        k = min(n_neighbors + 1, len(member_idx))
        _, neighbour_slots = index.search(vectors, k)
        position = {int(v): i for i, v in enumerate(member_idx)}
        for row in rows:
            slot = position.get(int(out[row, 0]))
            if slot is None:
                continue
            candidates = neighbour_slots[slot][1:]
            candidates = candidates[candidates >= 0]
            if len(candidates):
                out[row, 1] = member_idx[candidates[rng.integers(len(candidates))]]
        del index
    return out


def _embeddings_fingerprint(all_embeddings, n_rows: int) -> str:
    """Content hash of the embedding matrix, for cache keys.

    Without it the reduction and layout caches key only on shape and
    parameters, so new embeddings hit the cache and cluster stale coordinates
    silently. Strided to bound the cost on large corpora.
    """
    budget_rows = 4096
    stride = max(1, n_rows // budget_rows)
    digest = hashlib.sha1()
    digest.update(str((n_rows, int(all_embeddings.shape[1]))).encode())
    for start in range(0, n_rows, stride):
        digest.update(np.asarray(all_embeddings[start], dtype=np.float32).tobytes())
    return digest.hexdigest()


def _reduce_to_low_dim(all_embeddings, n_total_embeddings, low_dim, n_neighbors, sbert_dim):
    """The UMAP reduction feeding HDBSCAN. Cached by the caller."""
    umap_train_limit = 1_000_000

    if n_total_embeddings > umap_train_limit:
        print(f"\nLarge dataset ({n_total_embeddings:,} items). Training UMAP on random {umap_train_limit:,} subset...")

        # Random sample for training
        rng = np.random.default_rng(42)
        train_indices = rng.choice(n_total_embeddings, size=umap_train_limit, replace=False)
        train_indices.sort()

        # Load training data into memory
        X_train = all_embeddings[train_indices]

        reducer_low_dim = UMAP(
            n_components=low_dim,
            n_neighbors=n_neighbors,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )
        reducer_low_dim.fit(X_train)

        del X_train
        gc.collect()

        print(f"Transforming all {n_total_embeddings:,} embeddings in batches...")
        low_dim_embeddings = np.zeros((n_total_embeddings, low_dim), dtype=np.float32)

        batch_size = 1000000
        n_batches = (n_total_embeddings + batch_size - 1) // batch_size

        for i in tqdm(
            range(0, n_total_embeddings, batch_size),
            desc="UMAP Transform",
            total=n_batches,
        ):
            end_idx = min(i + batch_size, n_total_embeddings)
            batch = all_embeddings[i:end_idx]
            low_dim_embeddings[i:end_idx] = reducer_low_dim.transform(batch)

    else:
        reducer_low_dim = UMAP(
            n_components=low_dim,
            n_neighbors=n_neighbors,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )
        low_dim_embeddings = reducer_low_dim.fit_transform(all_embeddings)

    del reducer_low_dim
    gc.collect()
    return np.asarray(low_dim_embeddings, dtype=np.float32)


def _fit_hdbscan(low_dim_embeddings, n_total, min_cluster_size, selection_method=None):
    """HDBSCAN over the reduced space, returning core labels with -1 for noise.

    Above a million rows the model is fit on a capped subset and the remainder
    assigned with approximate_predict, which is a memory accommodation rather
    than a modelling choice.
    """
    selection_method = selection_method or CLUSTER_SELECTION_METHOD
    if n_total >= 1_000_000:
        train_size = min(1_000_000, n_total)  # Cap at 1M for training
        train_ratio = train_size / n_total
        print(f"Splitting data: {train_size:,} train ({train_ratio * 100:.1f}%), {n_total - train_size:,} test")
        train_indices, test_indices = train_test_split(
            np.arange(n_total), train_size=train_size, shuffle=True, random_state=42
        )

        train_embeddings = low_dim_embeddings[train_indices]
        test_embeddings = low_dim_embeddings[test_indices]

        print(
            f"Clustering with HDBSCAN on {len(train_embeddings):,} training passages...",
            end=" ",
        )
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=MIN_SAMPLES,
            metric="euclidean",
            cluster_selection_method=selection_method,
            prediction_data=True,
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

        for i in tqdm(
            range(0, len(test_embeddings), predict_batch_size),
            desc="Predicting test batches",
            total=n_test_batches,
            leave=False,
        ):
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
        print(f"Clustering all {n_total:,} passages with HDBSCAN...", end=" ")
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=MIN_SAMPLES,
            metric="euclidean",
            cluster_selection_method=selection_method,
            cluster_selection_epsilon=0.0,
            prediction_data=True,
        )
        cluster_labels = clusterer.fit_predict(low_dim_embeddings)
        cluster_labels = cluster_labels.astype(np.int32)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        print(f"done. Found {n_clusters} clusters")
        gc.collect()

    del clusterer
    gc.collect()
    return cluster_labels


def _finish_clustering(
    all_embeddings,
    low_dim_embeddings,
    cluster_labels,
    n_clusters,
    n_noise,
    n_total,
    output_path,
    cache_dir,
    low_dim,
    min_cluster_size,
    merge_unthemed,
    selection_method,
    n_neighbors,
):
    """Centroids, coherence, optional merge, layout, similarity, metadata."""
    # Noise kept as -1, and saved before any merging: the labeling stage must
    # describe a cluster's own members, not what was swept in afterwards.
    np.save(os.path.join(output_path, "cluster_labels_core.npy"), cluster_labels)

    print("Computing cluster centroids...", end=" ")
    # Batched: masking the memmap materializes a whole cluster at once.
    sbert_dim = all_embeddings.shape[1]
    centroid_sums = np.zeros((n_clusters, sbert_dim), dtype=np.float64)
    centroid_counts = np.zeros(n_clusters, dtype=np.int64)
    centroid_batch = 200_000
    for start_idx in range(0, n_total, centroid_batch):
        end_idx = min(start_idx + centroid_batch, n_total)
        batch_embeddings = np.asarray(all_embeddings[start_idx:end_idx], dtype=np.float64)
        batch_labels = cluster_labels[start_idx:end_idx]
        for cluster_id in np.unique(batch_labels):
            if cluster_id < 0:
                continue
            mask = batch_labels == cluster_id
            centroid_sums[cluster_id] += batch_embeddings[mask].sum(axis=0)
            centroid_counts[cluster_id] += int(mask.sum())
    safe_counts = np.clip(centroid_counts, 1, None)[:, None]
    centroids_array = (centroid_sums / safe_counts).astype(np.float32)

    # Mean cosine of members to their own centroid. In embedding space, not from
    # term statistics, which score a residual cluster higher than a real one.
    unit_centroids_for_coherence = centroids_array / np.clip(
        np.linalg.norm(centroids_array, axis=1, keepdims=True), 1e-12, None
    )
    coherence_sums = np.zeros(n_clusters, dtype=np.float64)
    for start_idx in range(0, n_total, centroid_batch):
        end_idx = min(start_idx + centroid_batch, n_total)
        batch_labels = cluster_labels[start_idx:end_idx]
        assigned = batch_labels >= 0
        if not assigned.any():
            continue
        emb = np.asarray(all_embeddings[start_idx:end_idx][assigned], dtype=np.float32)
        emb /= np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
        labels_here = batch_labels[assigned]
        sims = np.einsum("ij,ij->i", emb, unit_centroids_for_coherence[labels_here])
        np.add.at(coherence_sums, labels_here, sims)
    cluster_coherence = (coherence_sums / np.clip(centroid_counts, 1, None)).astype(np.float32)
    np.save(os.path.join(output_path, "cluster_coherence.npy"), cluster_coherence)

    del centroid_sums
    gc.collect()
    print("done.")

    modified_cluster_labels = cluster_labels.copy()
    if merge_unthemed and n_noise and n_clusters:
        # Nearest centroid by cosine, not HDBSCAN's membership probabilities,
        # which send every outlier to the densest cluster.
        print(f"Merging {n_noise:,} unthemed alignments into their nearest cluster...", end=" ")
        norms = np.clip(np.linalg.norm(centroids_array, axis=1, keepdims=True), 1e-12, None)
        unit_centroids = (centroids_array / norms).astype(np.float32)
        noise_indices = np.where(cluster_labels == -1)[0]
        merge_similarity = np.zeros(len(noise_indices), dtype=np.float32)
        merge_batch = 100_000
        for start_idx in range(0, len(noise_indices), merge_batch):
            chunk = noise_indices[start_idx : start_idx + merge_batch]
            emb = np.asarray(all_embeddings[chunk], dtype=np.float32)
            emb /= np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
            similarity = emb @ unit_centroids.T
            best = similarity.argmax(axis=1)
            modified_cluster_labels[chunk] = best.astype(np.int32)
            merge_similarity[start_idx : start_idx + len(chunk)] = similarity[
                np.arange(len(chunk)), best
            ]
        # Kept as membership strength: it separates a theme's core from its
        # periphery and lets a consumer reject a weak assignment.
        np.save(os.path.join(output_path, "merge_similarity.npy"), merge_similarity)
        np.save(os.path.join(output_path, "merge_indices.npy"), noise_indices)
        print(f"done. Mean similarity to assigned centroid: {merge_similarity.mean():.3f}")
    elif n_noise:
        print(
            f"{n_noise:,} alignments left unthemed (-1). "
            "Pass --merge-unthemed to assign them to their nearest cluster."
        )

    total_clusters = n_clusters

    np.save(
        os.path.join(output_path, "cluster_labels_modified.npy"),
        modified_cluster_labels,
    )
    np.save(os.path.join(output_path, "cluster_centroids.npy"), centroids_array)

    # Cluster ids are reassigned every run, so stale labels would name unrelated
    # clusters. Discard them only when the partition itself changed.
    fingerprint = hashlib.sha1(
        b"|".join(
            [
                str(min_cluster_size).encode(),
                str(selection_method).encode(),
                str(MIN_SAMPLES).encode(),
                str(low_dim).encode(),
                str(n_clusters).encode(),
                cluster_labels.tobytes(),
            ]
        )
    ).hexdigest()
    fingerprint_path = os.path.join(output_path, "clustering_fingerprint.txt")
    previous = None
    if os.path.exists(fingerprint_path):
        with open(fingerprint_path) as f:
            previous = f.read().strip()
    if previous and previous != fingerprint:
        for stale in ("cluster_labels.json", "topic_words.json"):
            stale_path = os.path.join(output_path, stale)
            if os.path.exists(stale_path):
                os.remove(stale_path)
                print(f"  removed stale {stale} (the clustering changed)")
    elif previous == fingerprint and os.path.exists(os.path.join(output_path, "cluster_labels.json")):
        print("  clustering unchanged; keeping existing labels")
    with open(fingerprint_path, "w") as f:
        f.write(fingerprint)

    # Readable as theme membership, size and adjacency -- not as inter-theme
    # distance, nor as position within a theme.
    print("Projecting to 2D for display (LocalMAP)...", end=" ", flush=True)

    def _compute_2d():
        # LocalMAP's defaults: it reduces to 100 components itself, and euclidean
        # is correct because the embeddings are unit-norm.
        vectors = np.asarray(all_embeddings, dtype=np.float32)
        base = pacmap.LocalMAP(
            n_components=2,
            n_neighbors=DEFAULT_LAYOUT_NEIGHBORS,
            random_state=42,
        )
        base.fit_transform(vectors.copy(), save_pairs=True)
        pairs = _label_aware_pairs(
            base.pair_neighbors,
            modified_cluster_labels,
            vectors,
            LAYOUT_LABEL_WEIGHT,
            DEFAULT_LAYOUT_NEIGHBORS,
        )
        reducer_2d = pacmap.LocalMAP(
            n_components=2,
            n_neighbors=DEFAULT_LAYOUT_NEIGHBORS,
            random_state=42,
            pair_neighbors=pairs,
            pair_MN=base.pair_MN,
            pair_FP=base.pair_FP,
        )
        out = reducer_2d.fit_transform(vectors.copy())
        del reducer_2d, base, pairs, vectors
        gc.collect()
        return np.asarray(out, dtype=np.float32)

    embeddings_2d = _cached_projection(
        cache_dir,
        "layout_2d",
        {
            "n_rows": int(n_total),
            "method": "localmap",
            "n_neighbors": int(DEFAULT_LAYOUT_NEIGHBORS),
            "label_weight": LAYOUT_LABEL_WEIGHT,
            "embeddings": _embeddings_fingerprint(all_embeddings, int(n_total)),
            # The rewiring reads the partition, so a different one is a
            # different layout.
            "partition": hashlib.sha1(modified_cluster_labels.tobytes()).hexdigest(),
        },
        (n_total, 2),
        _compute_2d,
    )
    print("done.")

    np.save(os.path.join(output_path, "embeddings_umap_2d.npy"), embeddings_2d)

    print("done.")

    print("Computing cluster similarity matrix...", end=" ")
    # From the full-dimension centroids, not the projection: cross-cluster
    # similarity is global structure, which the projection does not preserve.
    similarity_matrix = cosine_similarity(centroids_array)
    np.save(os.path.join(output_path, "cluster_similarity_matrix.npy"), similarity_matrix)
    print("done.")

    metadata = {
        "n_clusters": int(n_clusters),
        "n_noise": int(n_noise),
        "total_clusters": int(total_clusters),
        "total_alignments": int(len(cluster_labels)),
        # What HDBSCAN actually ran on, rather than the cosine used for the
        # initial UMAP reduction -- the two were previously conflated here.
        "metric": "euclidean_on_umap",
        "umap_components": int(low_dim),
        "umap_neighbors": int(DEFAULT_UMAP_NEIGHBORS),
        "cluster_similarity_space": "full_embedding",
        "cluster_selection_method": selection_method,
        "min_samples": MIN_SAMPLES,
        "min_cluster_size": int(min_cluster_size) if min_cluster_size else None,
        "centroid_dim": int(centroids_array.shape[1]),
        "merge_unthemed": bool(merge_unthemed),
        "n_outliers": int(n_noise),
        "outlier_percentage": float(100 * n_noise / len(cluster_labels)),
        "backend": "cuml" if USE_GPU else "cpu",
        "mean_cluster_coherence": float(cluster_coherence.mean()) if n_clusters else 0.0,
        "min_cluster_coherence": float(cluster_coherence.min()) if n_clusters else 0.0,
    }
    with open(os.path.join(output_path, "cluster_metadata.json"), "wb") as f:
        f.write(orjson.dumps(metadata))

    return modified_cluster_labels, embeddings_2d


def _project_pair_positions(
    pair_passage_counts,
    pair_embedding_sums,
    pair_position_sums,
    output_path,
    n_clusters,
):
    """2D coordinates for each (author, cluster) node.

    Mean-pools each node's alignments in the original embedding space, then fits
    one UMAP over those node centroids together with the cluster centroids, so
    nodes and cluster anchors share a single coherent projection and the
    projection is optimized for the points actually being displayed.

    Falls back to averaging the precomputed 2D coordinates when the embeddings
    are unavailable, which is the older and less sound behaviour: averaging UMAP
    coordinates has no geometric meaning, and for a node whose passages span
    several regions the mean lands between them.
    """
    if pair_embedding_sums is None:
        print("\nComputing mean 2D positions for each pair (no embeddings available)...")
        return {k: pair_position_sums[k] / c for k, c in pair_passage_counts.items()}

    # Nothing to project when the corpus carries no author metadata: every
    # alignment has a blank author on both sides, so there are no pairs.
    if not pair_passage_counts:
        return {}

    keys = sorted(pair_passage_counts)
    print(f"\nProjecting {len(keys)} (author, cluster) centroids to 2D...", end=" ", flush=True)
    node_centroids = np.array(
        [pair_embedding_sums[k] / pair_passage_counts[k] for k in keys], dtype=np.float32
    )

    cluster_centroids_path = os.path.join(output_path, "cluster_centroids.npy")
    anchors = (
        np.load(cluster_centroids_path)[:n_clusters].astype(np.float32)
        if os.path.exists(cluster_centroids_path)
        else np.empty((0, node_centroids.shape[1]), dtype=np.float32)
    )
    combined = np.vstack([node_centroids, anchors])

    # n_neighbors cannot exceed the sample size; a few hundred points is a much
    # smaller problem than the per-alignment projection this replaces.
    neighbours = max(2, min(DEFAULT_UMAP_NEIGHBORS, len(combined) - 1))
    projected = UMAP(
        n_components=2,
        n_neighbors=neighbours,
        min_dist=0.1,  # for display: 0.0 packs points together, which is a clustering setting
        metric="cosine",
        random_state=42,
    ).fit_transform(combined)
    projected = np.asarray(projected, dtype=np.float32)

    node_xy = projected[: len(keys)]
    anchor_xy = projected[len(keys) :]

    if len(anchor_xy):
        np.save(os.path.join(output_path, "anchor_positions_2d.npy"), anchor_xy)
    print("done.")
    return {k: node_xy[i] for i, k in enumerate(keys)}


def build_precomputed_api_graph(
    alignments_file: str,
    output_path: str,
    author_to_id: dict,
    cluster_labels_modified: np.ndarray,
    embeddings_2d: np.ndarray,
    alignment_counts: int,
    all_embeddings=None,
    similarity_neighbors: int = DEFAULT_SIMILARITY_NEIGHBORS,
    source_author_field: str = DEFAULT_SOURCE_AUTHOR_FIELD,
    target_author_field: str = DEFAULT_TARGET_AUTHOR_FIELD,
) -> None:
    """
    Build precomputed graph for API in the same format as get_semantic_graph_data.

    Creates precomputed_graph_api.json with (author, cluster) pair nodes and edges.
    """
    print("Building precomputed graph...")

    # Load only cluster similarity matrix and metadata
    cluster_similarity = np.load(os.path.join(output_path, "cluster_similarity_matrix.npy"))

    with open(os.path.join(output_path, "cluster_metadata.json"), "rb") as f:
        metadata = orjson.loads(f.read())

    n_clusters = metadata["n_clusters"]
    total_clusters = metadata["total_clusters"]
    num_authors = len(author_to_id)
    alignment_counts = metadata["total_alignments"]

    merged = metadata.get("merge_unthemed", False)
    print(
        f"  {n_clusters} clusters, {metadata['n_noise']} alignments HDBSCAN left unthemed "
        + ("(merged into nearest cluster)" if merged else "(excluded from the graph)")
    )
    print(f"  {num_authors} authors")
    print(f"  {alignment_counts} alignments")

    print("\nEnumerating (author, cluster) pairs from alignments...")

    pair_passage_counts = defaultdict(int)
    pair_position_sums = defaultdict(lambda: np.zeros(2, dtype=np.float64))
    # Sums in the original embedding space: UMAP preserves local neighbourhoods,
    # so a mean of 2D coordinates lands where no passage is.
    embedding_dim = all_embeddings.shape[1] if all_embeddings is not None else 0
    pair_embedding_sums = (
        defaultdict(lambda: np.zeros(embedding_dim, dtype=np.float64)) if embedding_dim else None
    )

    blank_author_ids = {aid for name, aid in author_to_id.items() if not str(name).strip()}
    skipped_blank = 0

    alignment_idx = 0
    with lz4.frame.open(alignments_file, "rb") as f:
        for line in tqdm(f, total=alignment_counts, desc="Processing alignments", leave=False):
            alignment = orjson.loads(line)
            source_author_id = author_to_id.get(alignment.get(source_author_field) or "", -1)
            target_author_id = author_to_id.get(alignment.get(target_author_field) or "", -1)

            cluster_id = int(cluster_labels_modified[alignment_idx])
            if cluster_id < 0:
                # Unthemed: HDBSCAN called it noise and --merge-unthemed was off.
                alignment_idx += 1
                continue
            embedding_2d = embeddings_2d[alignment_idx]

            for author_id in [source_author_id, target_author_id]:
                # An empty author name is missing metadata, not a participant:
                # left in, it becomes a node in every cluster and dominates.
                if author_id in blank_author_ids:
                    skipped_blank += 1
                    continue
                pair_key = (author_id, cluster_id)
                pair_passage_counts[pair_key] += 1
                pair_position_sums[pair_key] += embedding_2d
                if pair_embedding_sums is not None:
                    pair_embedding_sums[pair_key] += all_embeddings[alignment_idx]

            alignment_idx += 1

    print(f"✓ Found {len(pair_passage_counts)} unique (author, cluster) pairs")
    if skipped_blank:
        print(f"  ({skipped_blank:,} author slots skipped for having no author name)")

    pair_positions = _project_pair_positions(
        pair_passage_counts,
        pair_embedding_sums,
        pair_position_sums,
        output_path,
        n_clusters,
    )
    del pair_position_sums, pair_embedding_sums
    gc.collect()

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

        api_nodes.append(
            {
                "id": node_id,
                "label": id_to_author[author_id],
                "author_id": author_id,
                "author_name": id_to_author[author_id],
                "cluster_id": cluster_id,
                "cluster_label": "",
                "passages": int(passage_count),
                "size": int(passage_count),
                "x": float(position[0]),
                "y": float(position[1]),
            }
        )
        api_cluster_nodes[cluster_id].append(node_id)

    # Add cluster anchor nodes. Positions come from the same UMAP fit as the
    # nodes, so both live in one space.
    anchor_positions_path = os.path.join(output_path, "anchor_positions_2d.npy")
    projected_anchors = np.load(anchor_positions_path) if os.path.exists(anchor_positions_path) else None

    api_cluster_centroid_positions = {}
    for cluster_id in api_cluster_nodes.keys():
        if projected_anchors is not None and cluster_id < len(projected_anchors):
            api_cluster_centroid_positions[cluster_id] = projected_anchors[cluster_id]
            continue
        member_positions = [
            position
            for pair_key, position in pair_positions.items()
            if pair_key[1] == cluster_id and pair_passage_counts[pair_key] >= MIN_PASSAGES_THRESHOLD
        ]
        if member_positions:
            api_cluster_centroid_positions[cluster_id] = np.mean(member_positions, axis=0)

    for cluster_id, position_2d in api_cluster_centroid_positions.items():
        anchor_node_id = f"anchor_cluster_{cluster_id}"
        api_nodes.append(
            {
                "id": anchor_node_id,
                "label": "",
                "node_type": "cluster_anchor",
                "cluster_id": cluster_id,
                "cluster_label": "",
                "size": 0.01,
                "x": float(position_2d[0]),
                "y": float(position_2d[1]),
                "hidden": True,
            }
        )

    # Build API-format edges
    api_edges = []

    # 1. Intra-cluster edges (star topology: nodes to anchors)
    for cluster_id, node_ids in api_cluster_nodes.items():
        # >= 1, not > 1: a cluster with a single node was previously given no
        # edge at all, leaving that node disconnected and flung to the margin.
        if len(node_ids) >= 1:
            anchor_id = f"anchor_cluster_{cluster_id}"
            for node_id in node_ids:
                api_edges.append(
                    {
                        "source": node_id,
                        "target": anchor_id,
                        "weight": 1.0,
                        "edge_type": "intra_cluster",
                        "color": "#666666",
                        "size": 0.5,
                    }
                )

    # 2. Centroid similarity edges. Each cluster keeps only its k most similar
    # neighbours: centroid cosine is positive for every pair, and a uniform
    # complete graph lays out as a ring.
    filtered_cluster_ids = sorted(api_cluster_nodes.keys())
    kept_pairs: set[tuple[int, int]] = set()
    for cluster_i in filtered_cluster_ids:
        if cluster_i >= n_clusters:
            continue
        neighbours = [
            (float(cluster_similarity[cluster_i, cluster_j]), cluster_j)
            for cluster_j in filtered_cluster_ids
            if cluster_j != cluster_i and cluster_j < n_clusters
        ]
        neighbours.sort(reverse=True)
        # Union of each cluster's top-k, so the result is symmetric and no
        # cluster is left without a link even if nobody else ranks it highly.
        for _, cluster_j in neighbours[:similarity_neighbors]:
            kept_pairs.add((min(cluster_i, cluster_j), max(cluster_i, cluster_j)))

    weights = [float(cluster_similarity[i, j]) for i, j in kept_pairs]
    lo, hi = (min(weights), max(weights)) if weights else (0.0, 1.0)
    span = max(hi - lo, 1e-6)
    for cluster_i, cluster_j in sorted(kept_pairs):
        similarity = float(cluster_similarity[cluster_i, cluster_j])
        # Rescaled across retained edges: raw cosine sits in a narrow band well
        # above zero, which reads as uniform.
        api_edges.append(
            {
                "source": f"anchor_cluster_{cluster_i}",
                "target": f"anchor_cluster_{cluster_j}",
                "weight": round(0.1 + 0.9 * (similarity - lo) / span, 4),
                "similarity": round(similarity, 4),
                "edge_type": "centroid_similarity",
                "color": "#999999",
                "size": 1.0,
            }
        )
    print(f"  similarity edges: {len(kept_pairs)} (top-{similarity_neighbors} per cluster, "
          f"was {len(filtered_cluster_ids) * (len(filtered_cluster_ids) - 1) // 2} unthresholded)")

    api_graph = {
        "nodes": api_nodes,
        "edges": api_edges,
        "metadata": {
            "n_clusters": n_clusters,
            "total_nodes": len(api_nodes),
            "total_edges": len(api_edges),
            "min_passages_threshold": MIN_PASSAGES_THRESHOLD,
        },
    }

    with open(os.path.join(output_path, "precomputed_graph_api.json"), "wb") as f:
        f.write(orjson.dumps(api_graph))

    print(f"✓ Saved precomputed full graph: {len(api_nodes)} nodes, {len(api_edges)} edges")


def main():
    alignments_file = sys.argv[1]
    alignment_counts = int(sys.argv[2])
    sbert_model_name = (
        sys.argv[3] if len(sys.argv) > 3 else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    output_path = os.path.join(os.path.dirname(alignments_file), "graph_data")
    os.makedirs(output_path, exist_ok=True)

    # Preprocess data - load SBERT embeddings
    data = build_alignment_data(alignments_file, alignment_counts, sbert_model_name)

    # Save author mapping for later use
    with open(os.path.join(output_path, "author_to_id.json"), "wb") as f:
        f.write(orjson.dumps(data["author_to_id"]))

    # Cluster alignments by content similarity
    merge_unthemed = "--merge-unthemed" in sys.argv
    modified_cluster_labels, embeddings_2d = cluster_alignments(
        data, output_path, alignment_counts, merge_unthemed=merge_unthemed
    )

    # Build precomputed graph for API
    build_precomputed_api_graph(
        alignments_file,
        output_path,
        data["author_to_id"],
        modified_cluster_labels,
        embeddings_2d,
        alignment_counts,
        all_embeddings=data["passage_embeddings_memmap"],
    )

    from .scatter_data import build_scatter_data

    build_scatter_data(alignments_file, output_path, data["author_to_id"])

    print("\nThematic identity graph done.")


if __name__ == "__main__":
    main()
