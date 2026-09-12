"""Distinctive-term extraction and LLM labeling for alignment clusters.

Two stages, deliberately separated:

1. c-TF-IDF over the clusters produces each cluster's characteristic terms.
   Terms come from the cluster *core* — the members HDBSCAN assigned directly —
   never from points merged in afterwards by `--merge-unthemed`, so a label
   describes what a theme is rather than what was swept into it.

2. Those terms are handed to `topic_labeler`, which turns them into short
   human-readable labels. That module is vendored from the TopoLogic project
   rather than reimplemented: it already declines to name incoherent clusters,
   relabels sibling clusters together so their labels stay distinguishable, and
   rejects degenerate "<word> and <word>" labels.

The interchange format is TopoLogic's `topic_words.json`: a list of
`{"name": <cluster id>, "top_words": [[word, weight], ...], "description": ...}`
entries, to which the labeler adds a `"label"` in place.
"""

import json
import os
import re
import subprocess
import tempfile
from collections import defaultdict

import lz4.frame
import numpy as np
import orjson
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

TOP_WORDS = 20  # what topic_labeler consumes per cluster
DEFAULT_MAX_PASSAGES_PER_CLUSTER = 2000
_SIMPLE_TOKEN = re.compile(r"[^\W\d_]{3,}", re.UNICODE)


LEMMATIZER_PYTHON = "/var/lib/text-pair/lemmatizer/bin/python"


def _regex_normalize(text: str) -> str:
    """Fallback tokenization: lowercase words of three or more letters.
    """
    return " ".join(m.group(0).lower() for m in _SIMPLE_TOKEN.finditer(text))


def _lemmatize_via_subprocess(
    passages: list[str], spacy_model: str, pos_to_keep: list[str]
) -> list[str] | None:
    """Normalize passages in the lemmatizer environment. None if unavailable.
    """
    if not os.path.exists(LEMMATIZER_PYTHON):
        print(
            f"  spacy_model is set but {LEMMATIZER_PYTHON} is missing; falling back to\n"
            "  regex tokenization. Re-run install.sh without -L to build it."
        )
        return None

    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lemmatize_worker.py")
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "passages.json")
        out_path = os.path.join(tmp, "normalized.json")
        with open(in_path, "w", encoding="utf-8") as f:
            json.dump(passages, f)
        command = [
            LEMMATIZER_PYTHON,
            worker,
            in_path,
            out_path,
            "--model",
            spacy_model,
            "--pos-to-keep",
            ",".join(pos_to_keep) if pos_to_keep else "",
        ]
        print(f"  lemmatizing {len(passages):,} passages with {spacy_model}...", flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode != 0 or not os.path.exists(out_path):
            print(f"  lemmatizer exited {result.returncode}; falling back to regex tokenization.")
            return None
        with open(out_path, "r", encoding="utf-8") as f:
            return json.load(f)


def collect_cluster_documents(
    alignments_file: str,
    core_labels: np.ndarray,
    n_clusters: int,
    max_passages_per_cluster: int = DEFAULT_MAX_PASSAGES_PER_CLUSTER,
) -> dict[int, list[str]]:
    """One pass over the alignments file, gathering passages per core cluster.

    Capped per cluster: c-TF-IDF is a frequency statistic and converges long
    before a large cluster is exhausted, while an uncapped pass would hold
    millions of passages in memory.
    """
    passages: dict[int, list[str]] = defaultdict(list)
    full = 0

    with lz4.frame.open(alignments_file, "rb") as f:
        for idx, line in tqdm(
            enumerate(f), total=len(core_labels), desc="Reading passages", leave=False
        ):
            if idx >= len(core_labels):
                break
            cluster_id = int(core_labels[idx])
            if cluster_id < 0 or cluster_id >= n_clusters:
                continue  # noise, or a merged-in point: never feeds the terms
            bucket = passages[cluster_id]
            if len(bucket) >= max_passages_per_cluster:
                continue
            alignment = orjson.loads(line)
            # Both sides: the theme of a relationship is not the property of
            # whichever passage happened to be indexed as the source.
            for field in ("source_passage", "target_passage"):
                text = alignment.get(field, "")
                if text:
                    bucket.append(text)
            if len(bucket) >= max_passages_per_cluster:
                full += 1

    print(f"✓ Collected passages for {len(passages)} clusters ({full} hit the per-cluster cap)")
    return passages


def compute_ctfidf(
    cluster_documents: dict[int, list[str]],
    preprocessing_params: dict | None = None,
    top_n: int = TOP_WORDS,
) -> dict[int, dict]:
    """Class-based TF-IDF: which terms are characteristic of each cluster.

        W(t,c) = tf(t,c) * log(1 + A / f(t))

    where tf(t,c) is term t's frequency within cluster c normalized by the
    cluster's length, f(t) is t's total frequency across every cluster, and A
    is the mean cluster length. Terms common to all clusters are damped; terms
    concentrated in one are promoted.
    """
    if not cluster_documents:
        return {}

    cluster_ids = sorted(cluster_documents)
    params = preprocessing_params or {}
    spacy_model = params.get("spacy_model") or params.get("language_model")

    print(f"Normalizing text for {len(cluster_ids)} clusters...")
    flat, owner = [], []
    for cluster_id in cluster_ids:
        for passage in cluster_documents[cluster_id]:
            flat.append(passage)
            owner.append(cluster_id)

    normalized = None
    if spacy_model:
        normalized = _lemmatize_via_subprocess(
            flat, spacy_model, params.get("pos_to_keep") or ["NOUN", "ADJ", "PROPN"]
        )
    if normalized is None:
        normalized = [_regex_normalize(p) for p in flat]

    grouped: dict[int, list[str]] = {cid: [] for cid in cluster_ids}
    for cluster_id, text in zip(owner, normalized):
        grouped[cluster_id].append(text)
    docs = [" ".join(grouped[cid]) for cid in cluster_ids]

    # No max_df: terms shared across every cluster are the evidence that a
    # cluster has no vocabulary of its own. log(1 + A/f) already damps them.
    vectorizer = CountVectorizer(min_df=1)
    try:
        counts = vectorizer.fit_transform(docs)
    except ValueError as e:  # empty vocabulary
        print(f"  c-TF-IDF vocabulary is empty ({e}); no terms extracted.")
        return {}

    vocab = np.array(vectorizer.get_feature_names_out())
    counts = np.asarray(counts.todense(), dtype=np.float64)

    cluster_totals = counts.sum(axis=1, keepdims=True)
    cluster_totals[cluster_totals == 0] = 1.0
    tf = counts / cluster_totals

    term_totals = counts.sum(axis=0)
    term_totals[term_totals == 0] = 1.0
    avg_cluster_len = counts.sum() / max(len(docs), 1)
    idf = np.log(1.0 + avg_cluster_len / term_totals)

    weights = tf * idf

    results: dict[int, dict] = {}
    for row, cluster_id in enumerate(cluster_ids):
        row_weights = weights[row]
        if not row_weights.any():
            results[cluster_id] = {"top_words": []}
            continue
        top_idx = [i for i in np.argsort(row_weights)[::-1][:top_n] if row_weights[i] > 0]
        results[cluster_id] = {
            "top_words": [(str(vocab[i]), float(row_weights[i])) for i in top_idx],
        }
    return results


def write_topic_words(
    path: str,
    terms_by_cluster: dict[int, dict],
    coherence: dict[int, float] | None = None,
) -> int:
    """Write topic_words.json, which the labeler updates in place.

    Extra keys beyond `name`/`top_words` are carried through untouched by the
    labeler, so coherence travels with the terms for later inspection.
    """
    coherence = coherence or {}
    entries = []
    for cluster_id in sorted(terms_by_cluster):
        top_words = terms_by_cluster[cluster_id]["top_words"]
        entry = {
            "name": int(cluster_id),
            "top_words": [[w, round(weight, 6)] for w, weight in top_words],
            # The labeler falls back to this when top_words is absent, and it
            # stays useful on its own if labeling is skipped entirely.
            "description": ", ".join(w for w, _ in top_words[:10]),
            "label": "",
        }
        if cluster_id in coherence:
            entry["coherence"] = round(float(coherence[cluster_id]), 4)
        entries.append(entry)
    with open(path, "wb") as f:
        f.write(orjson.dumps(entries))
    return len(entries)


def run_labeler(topic_words_path: str, model: str, language: str = "French") -> bool:
    """Label the clusters in place. Degrades rather than failing.

    The labeler is vendored (`topic_labeler`) and runs in this environment, so
    there is no external tool to locate.
    """
    from .topic_labeler import relabel_json

    print(f"Labeling clusters with {model}...", flush=True)
    try:
        written = relabel_json(topic_words_path, model, language)
    except Exception as e:
        print(
            f"Labeling failed ({e}).\n"
            "  Clusters keep their top-word descriptions. If the model was rejected as an\n"
            "  unknown architecture, the checkpoint needs a newer transformers than this\n"
            "  environment has; pick a model it recognises or upgrade it."
        )
        return False
    if not written:
        print("  No labels produced; clusters keep their top-word descriptions.")
        return False
    return True


def read_labels(topic_words_path: str) -> dict[int, str]:
    with open(topic_words_path, "rb") as f:
        entries = orjson.loads(f.read())
    return {int(e["name"]): (e.get("label") or "").strip() for e in entries}


def update_graph_json(graph_data_path: str, cluster_label_map: dict[int, str]) -> None:
    """Write labels onto the precomputed graph the API serves."""
    path = os.path.join(graph_data_path, "precomputed_graph_api.json")
    if not os.path.exists(path):
        print(f"⚠ {path} not found; skipping graph label update.")
        return
    with open(path, "rb") as f:
        graph_data = orjson.loads(f.read())
    for node in graph_data["nodes"]:
        if "cluster_id" in node:
            node["cluster_label"] = cluster_label_map.get(int(node["cluster_id"]), "")
    with open(path, "wb") as f:
        f.write(orjson.dumps(graph_data))
    print("✓ Updated precomputed_graph_api.json")


def generate_cluster_labels(
    alignments_file: str,
    graph_data_path: str,
    model: str,
    language: str = "French",
    max_passages_per_cluster: int = DEFAULT_MAX_PASSAGES_PER_CLUSTER,
    preprocessing_params: dict | None = None,
) -> dict[int, str]:
    """c-TF-IDF over cluster cores, then LLM labels, then update the graph."""
    core_path = os.path.join(graph_data_path, "cluster_labels_core.npy")
    if not os.path.exists(core_path):
        raise FileNotFoundError(
            f"{core_path} not found. Re-run `textpair-graph build` — labeling reads the "
            "pre-merge cluster assignment so that merged-in points cannot dilute the terms."
        )
    core_labels = np.load(core_path)

    with open(os.path.join(graph_data_path, "cluster_metadata.json"), "rb") as f:
        metadata = orjson.loads(f.read())
    n_clusters = metadata["n_clusters"]

    print(f"\nExtracting distinctive terms for {n_clusters} clusters (c-TF-IDF)...")
    cluster_documents = collect_cluster_documents(
        alignments_file, core_labels, n_clusters, max_passages_per_cluster
    )
    terms_by_cluster = compute_ctfidf(cluster_documents, preprocessing_params)

    if not terms_by_cluster:
        print("No terms extracted; skipping labeling.")
        return {}

    # Written by the build stage from the embeddings; see the note there on why
    # coherence is measured in embedding space rather than over terms.
    coherence: dict[int, float] = {}
    coherence_path = os.path.join(graph_data_path, "cluster_coherence.npy")
    if os.path.exists(coherence_path):
        values = np.load(coherence_path)
        coherence = {i: float(v) for i, v in enumerate(values)}

    topic_words_path = os.path.join(graph_data_path, "topic_words.json")
    count = write_topic_words(topic_words_path, terms_by_cluster, coherence)
    print(f"✓ Wrote {count} clusters to {topic_words_path}")

    # Reported, never used to gate labeling: coherence ranks formulaic material
    # above themes discussed in varied language.
    if coherence:
        sizes = {cid: int((core_labels == cid).sum()) for cid in terms_by_cluster}
        total = max(int((core_labels >= 0).sum()), 1)
        values = np.array([coherence[c] for c in sorted(coherence)])
        print(
            f"\nCluster coherence: mean {values.mean():.3f}, median {np.median(values):.3f}, "
            f"range {values.min():.3f}-{values.max():.3f}"
        )
        print(
            "  High coherence indicates repetition rather than quality — near 1.0 with short\n"
            "  repetitive terms is a boilerplate signature, not a good theme."
        )
        formulaic = sorted(
            (c for c in terms_by_cluster if coherence.get(c, 0.0) >= 0.97),
            key=lambda c: -sizes.get(c, 0),
        )
        if formulaic:
            print(f"  {len(formulaic)} cluster(s) at coherence >= 0.97, likely formulaic:")
            for cluster_id in formulaic[:5]:
                preview = ", ".join(w for w, _ in terms_by_cluster[cluster_id]["top_words"][:6])
                share = 100.0 * sizes.get(cluster_id, 0) / total
                print(
                    f"    cluster {cluster_id}: coherence {coherence[cluster_id]:.3f}, "
                    f"{share:.1f}% of themed alignments  [{preview}]"
                )

    run_labeler(topic_words_path, model, language)
    cluster_label_map = read_labels(topic_words_path)

    labels_path = os.path.join(graph_data_path, "cluster_labels.json")
    with open(labels_path, "wb") as f:
        f.write(orjson.dumps({str(k): v for k, v in cluster_label_map.items()}))
    print(f"✓ Saved cluster labels to {labels_path}")

    named = [(c, l) for c, l in sorted(cluster_label_map.items()) if l]
    print(f"\n{len(named)}/{len(cluster_label_map)} clusters labeled. Sample:")
    for cluster_id, label in named[:10]:
        print(f"  cluster {cluster_id}: {label}")

    update_graph_json(graph_data_path, cluster_label_map)
    return cluster_label_map
