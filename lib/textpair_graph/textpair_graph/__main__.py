#!/usr/bin/env python3
"""
TextPair Graph CLI - Build and label graph models for alignment data.
"""

import argparse
import os
import sys


def _build(args):
    import lz4.frame
    import orjson

    from .build_graph_model import (
        build_alignment_data,
        build_precomputed_api_graph,
        cluster_alignments,
    )

    print(f"Building graph model from {args.alignments_file}")
    print(f"SBERT model: {args.model}")

    print("\rCounting alignments...", end="", flush=True)
    alignment_counts = sum(1 for _ in lz4.frame.open(args.alignments_file, "rb"))
    print(f"\r✓ Counted {alignment_counts:,} alignments" + " " * 20)

    embed_kwargs = {}
    if args.max_seq_length is not None:
        embed_kwargs["max_seq_length"] = args.max_seq_length
    if args.source_author_field:
        embed_kwargs["source_author_field"] = args.source_author_field
    if args.target_author_field:
        embed_kwargs["target_author_field"] = args.target_author_field
    data = build_alignment_data(args.alignments_file, alignment_counts, args.model, **embed_kwargs)

    author_field_kwargs = {}
    if args.source_author_field:
        author_field_kwargs["source_author_field"] = args.source_author_field
    if args.target_author_field:
        author_field_kwargs["target_author_field"] = args.target_author_field

    if args.cluster_selection_method:
        # Module-level because it is read by the sweep helper too.
        import textpair_graph.build_graph_model as bgm

        bgm.CLUSTER_SELECTION_METHOD = args.cluster_selection_method

    if args.sweep:
        from .build_graph_model import (
            DEFAULT_UMAP_COMPONENTS,
            DEFAULT_UMAP_NEIGHBORS,
            _cached_projection,
            _reduce_to_low_dim,
            format_sweep,
            sweep_min_cluster_size,
        )

        low_dim = args.umap_components or DEFAULT_UMAP_COMPONENTS
        neighbors = args.umap_neighbors or DEFAULT_UMAP_NEIGHBORS
        n_rows = data["passage_embeddings_memmap"].shape[0]
        low = _cached_projection(
            data.get("embeddings_cache_dir"),
            f"low_dim_{low_dim}d",
            {
                "n_rows": int(n_rows),
                "n_components": int(low_dim),
                "n_neighbors": int(neighbors),
                "metric": "cosine",
                "backend": "cuml" if __import__("textpair_graph.build_graph_model", fromlist=["USE_GPU"]).USE_GPU else "cpu",
            },
            (n_rows, low_dim),
            lambda: _reduce_to_low_dim(
                data["passage_embeddings_memmap"], n_rows, low_dim, neighbors, data["sbert_dim"]
            ),
        )
        print()
        print(format_sweep(sweep_min_cluster_size(low), n_rows))
        print("\nNothing written (--sweep). Re-run without --sweep, optionally with")
        print("--min-cluster-size <n>, to build the graph.")
        return

    # Nothing is written before this point, so --sweep above leaves no artifacts.
    output_path = os.path.join(args.output_dir, "graph_data")
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, "author_to_id.json"), "wb") as f:
        f.write(orjson.dumps(data["author_to_id"]))

    # Persisted because `label` rebuilds the scatter payload: author_to_id is
    # keyed by these fields, and a mismatch drops authors to -1 silently.
    with open(os.path.join(output_path, "author_fields.json"), "wb") as f:
        f.write(
            orjson.dumps(
                {
                    "source_author_field": args.source_author_field or "source_author",
                    "target_author_field": args.target_author_field or "target_author",
                }
            )
        )

    print("\n" + "=" * 60)
    print("CLUSTERING ALIGNMENTS")
    print("=" * 60)
    min_cluster_size = args.min_cluster_size
    if isinstance(min_cluster_size, str) and min_cluster_size.lower() != "auto":
        min_cluster_size = int(min_cluster_size)
    cluster_kwargs = {
        "merge_unthemed": args.merge_unthemed,
        "min_cluster_size": min_cluster_size,
    }
    # Experiment-only overrides; no config equivalent by design.
    if args.umap_components is not None:
        cluster_kwargs["umap_components"] = args.umap_components
    if args.umap_neighbors is not None:
        cluster_kwargs["umap_neighbors"] = args.umap_neighbors

    modified_cluster_labels, embeddings_2d = cluster_alignments(
        data, output_path, alignment_counts, **cluster_kwargs
    )

    build_precomputed_api_graph(
        args.alignments_file,
        output_path,
        data["author_to_id"],
        modified_cluster_labels,
        embeddings_2d,
        alignment_counts,
        all_embeddings=data["passage_embeddings_memmap"],
        similarity_neighbors=args.similarity_neighbors,
        **author_field_kwargs,
    )

    from .scatter_data import build_scatter_data

    build_scatter_data(args.alignments_file, output_path, data["author_to_id"], **author_field_kwargs)

    print("\n✓ Graph model built successfully!")
    print(f"   Graph data saved to: {output_path}/")


def _label(args):
    from .cluster_labeling import generate_cluster_labels

    graph_data_path = args.graph_data_dir
    parent_dir = os.path.dirname(os.path.normpath(graph_data_path))

    alignments_file = args.alignments_file
    if alignments_file is None:
        for fname in sorted(os.listdir(parent_dir)):
            if fname.endswith(".jsonl.lz4"):
                alignments_file = os.path.join(parent_dir, fname)
                break
    if alignments_file is None:
        print(
            "Error: no alignments file (.jsonl.lz4) found next to the graph_data "
            "directory. Pass --alignments-file explicitly.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Labeling clusters in {graph_data_path}")
    print(f"Alignments: {alignments_file}")
    print(f"LLM model: {args.model}")

    preprocessing_params = {
        "language": args.text_language,
        "language_model": args.spacy_model,
        "modernize": not args.no_modernize,
        "lemmatizer": args.lemmatizer or False,
        "pos_to_keep": [p.strip() for p in args.pos_to_keep.split(",") if p.strip()],
    }

    cluster_labels = generate_cluster_labels(
        alignments_file=alignments_file,
        graph_data_path=graph_data_path,
        model=args.model,
        language=args.language,
        max_passages_per_cluster=args.max_passages,
        preprocessing_params=preprocessing_params,
    )
    print(f"\n✓ Generated labels for {len(cluster_labels)} clusters!")

    # Rebuild the scatter payload so it carries the names: `build` has to write
    # it before labels exist, since labeling reads the clustering.
    author_to_id_path = os.path.join(graph_data_path, "author_to_id.json")
    if os.path.exists(author_to_id_path):
        import orjson

        from .scatter_data import build_scatter_data

        with open(author_to_id_path, "rb") as f:
            author_to_id = orjson.loads(f.read())
        author_fields = {}
        fields_path = os.path.join(graph_data_path, "author_fields.json")
        if os.path.exists(fields_path):
            with open(fields_path, "rb") as f:
                author_fields = orjson.loads(f.read())
        build_scatter_data(alignments_file, graph_data_path, author_to_id, **author_fields)


def main():
    """Main entry point for textpair_graph CLI."""
    parser = argparse.ArgumentParser(
        description="TextPair Graph - Build and label graph models for alignment data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build graph model from alignments
  python -m textpair_graph build alignments.jsonl.lz4 ./output --model Qwen/Qwen3-Embedding-0.6B

  # Same, assigning HDBSCAN's noise points to their nearest cluster
  python -m textpair_graph build alignments.jsonl.lz4 ./output \\
      --model Qwen/Qwen3-Embedding-0.6B --merge-unthemed

  # Extract distinctive terms (c-TF-IDF) and label the clusters with an LLM
  python -m textpair_graph label ./output/graph_data --spacy-model fr_core_news_lg
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    build_parser = subparsers.add_parser("build", help="Build graph model from alignments")
    build_parser.add_argument("alignments_file", help="Path to alignments file (JSONL with lz4 compression)")
    build_parser.add_argument("output_dir", help="Output directory for graph data")
    build_parser.add_argument("--model", required=True, help="SentenceTransformer model name or path")
    build_parser.add_argument(
        "--min-cluster-size",
        default="auto",
        help="HDBSCAN min_cluster_size, or 'auto' (default) to sweep candidates and take "
        "the largest still-balanced partition. Re-running with a different value costs "
        "about a second once the UMAP reduction is cached.",
    )
    build_parser.add_argument(
        "--cluster-selection-method",
        choices=("eom", "leaf"),
        default=None,
        help="How HDBSCAN picks clusters from its tree (default: eom). 'leaf' gives more, "
        "smaller clusters and more noise.",
    )
    build_parser.add_argument(
        "--sweep",
        action="store_true",
        help="Print the min_cluster_size candidate table and exit without writing anything.",
    )
    build_parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Token cap when encoding passages (default: 512). Bounds peak memory: "
        "attention is quadratic and merged passages have a long length tail.",
    )
    build_parser.add_argument(
        "--umap-components",
        type=int,
        default=None,
        help="UMAP dimensions the clustering runs in (default: 5). Deliberately not a "
        "config setting: 5D and 32D measured indistinguishable. Here for experiments only.",
    )
    build_parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=None,
        help="UMAP n_neighbors (default: 15). Experiments only, as above.",
    )
    build_parser.add_argument(
        "--source-author-field",
        default=None,
        help="Alignment field holding the source author (default: source_author).",
    )
    build_parser.add_argument(
        "--target-author-field",
        default=None,
        help="Alignment field holding the target author (default: target_author). Worth "
        "setting when aligning against a sub-document object type: the default field may "
        "encode 'no attribution' as a name rather than leaving it blank, which turns it "
        "into a phantom author. On the ARTFL Encyclopedie at div1, target_kafauth is the "
        "authority-normalized alternative.",
    )
    build_parser.add_argument(
        "--similarity-neighbors",
        type=int,
        default=4,
        help="Cluster-to-cluster links kept per cluster (default: 4). Keeping all of them "
        "makes a complete graph, which any force layout renders as a ring.",
    )
    build_parser.add_argument(
        "--no-merge-unthemed",
        dest="merge_unthemed",
        action="store_false",
        help="Leave alignments HDBSCAN marked as noise unthemed and out of the graph. "
        "By default they are assigned to their nearest cluster centroid (cosine, full "
        "embedding space), since a quarter to a third of a corpus is otherwise "
        "unreachable. Labeling always runs on the pre-merge cores either way.",
    )
    build_parser.set_defaults(merge_unthemed=True)

    label_parser = subparsers.add_parser(
        "label", help="Extract distinctive terms (c-TF-IDF) and label clusters with an LLM"
    )
    label_parser.add_argument("graph_data_dir", help="Path to graph_data directory")
    label_parser.add_argument(
        "--alignments-file",
        default=None,
        help="Path to the alignments file. Default: the first .jsonl.lz4 beside graph_data.",
    )
    label_parser.add_argument(
        "--model",
        default="google/gemma-4-E2B-it",
        help="HuggingFace instruction-tuned model for topologic-labeler "
        "(default: google/gemma-4-E2B-it)",
    )
    label_parser.add_argument(
        "--language", default="French", help="Language the labels are written in (default: French)"
    )
    label_parser.add_argument(
        "--text-language",
        default="french",
        help="Language of the corpus, for tokenization (default: french)",
    )
    label_parser.add_argument(
        "--spacy-model",
        default=None,
        help="spaCy model for lemmatization and POS filtering. Without it, term "
        "extraction falls back to regex tokenization, which is markedly worse on "
        "archaic spelling.",
    )
    label_parser.add_argument(
        "--pos-to-keep",
        default="NOUN,ADJ,PROPN",
        help="Comma-separated POS tags to keep for term extraction (default: NOUN,ADJ,PROPN)",
    )
    label_parser.add_argument(
        "--lemmatizer", default=None, help="Lemmatizer to use, e.g. 'spacy'"
    )
    label_parser.add_argument(
        "--no-modernize", action="store_true", help="Skip archaic-spelling modernization"
    )
    label_parser.add_argument(
        "--max-passages",
        type=int,
        default=2000,
        help="Passages sampled per cluster for c-TF-IDF (default: 2000)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "build":
        _build(args)
    elif args.command == "label":
        _label(args)


if __name__ == "__main__":
    main()
