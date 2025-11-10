#!/usr/bin/env python3
"""
TextPair Graph CLI - Build and label graph models for alignment data.
"""

import argparse
import asyncio
import sys


def main():
    """Main entry point for textpair_graph CLI."""
    parser = argparse.ArgumentParser(
        description="TextPair Graph - Build and label graph models for alignment data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build graph model from alignments
  python -m textpair_graph build alignments.jsonl.lz4 ./output --model antoinelouis/french-mgte-base

  # Generate cluster labels using LLM
  python -m textpair_graph label ./output/graph_data --model unsloth/gemma-3-4b-it-qat-GGUF
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Build command
    build_parser = subparsers.add_parser('build', help='Build graph model from alignments')
    build_parser.add_argument('alignments_file', help='Path to alignments file (JSONL with lz4 compression)')
    build_parser.add_argument('output_dir', help='Output directory for graph data')
    build_parser.add_argument('--model', required=True, help='SentenceTransformer model name or path')
    build_parser.add_argument('--min-cluster-size', type=int, default=5,
                             help='Minimum cluster size for HDBSCAN (default: 5)')
    build_parser.add_argument('--use-cuda', action='store_true',
                             help='Use CUDA-accelerated UMAP/HDBSCAN if available')

    # Label command
    label_parser = subparsers.add_parser('label', help='Generate cluster labels using LLM')
    label_parser.add_argument('graph_data_dir', help='Path to graph_data directory')
    label_parser.add_argument('--model', default='unsloth/gemma-3-4b-it-qat-GGUF',
                             help='LLM model name or path (default: unsloth/gemma-3-4b-it-qat-GGUF)')
    label_parser.add_argument('--context-window', type=int, default=8192,
                             help='Context window size for LLM (default: 8192)')
    label_parser.add_argument('--top-k', type=int, default=25,
                             help='Number of top passages per cluster (default: 25)')
    label_parser.add_argument('--port', type=int, default=8080,
                             help='Port for llama-server (default: 8080)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == 'build':
        import os

        import lz4.frame
        import orjson

        from .build_graph_model import (
            build_alignment_data,
            build_author_cluster_graph,
            cluster_alignments,
        )

        print(f"Building graph model from {args.alignments_file}")
        print(f"SBERT model: {args.model}")

        # Count alignments
        print("\rCounting alignments...", end="", flush=True)
        alignment_counts = sum(1 for _ in lz4.frame.open(args.alignments_file, 'rb'))
        print(f"\r✓ Counted {alignment_counts:,} alignments" + " " * 20)

        output_path = os.path.join(args.output_dir, "graph_data")
        os.makedirs(output_path, exist_ok=True)

        # Build alignment data and embeddings
        data = build_alignment_data(args.alignments_file, alignment_counts, args.model)

        # Save author mapping
        with open(os.path.join(output_path, 'author_to_id.json'), 'wb') as f:
            f.write(orjson.dumps(data['author_to_id']))

        # Cluster alignments
        print("\n" + "="*60)
        print("CLUSTERING ALIGNMENTS")
        print("="*60)
        cluster_labels = cluster_alignments(data, output_path)

        # Build full corpus graph
        print("\n" + "="*60)
        print("BUILDING FULL CORPUS GRAPH")
        print("="*60)
        G = build_author_cluster_graph(args.alignments_file, output_path)

        print("\n✓ Graph model built successfully!")
        print(f"   Graph data saved to: {output_path}/")

    elif args.command == 'label':
        import os

        from .label_clusters import generate_and_update_cluster_labels

        # Find alignments file
        parent_dir = os.path.dirname(args.graph_data_dir)
        alignments_file = None
        for fname in os.listdir(parent_dir):
            if fname.endswith('.jsonl.lz4'):
                alignments_file = os.path.join(parent_dir, fname)
                break

        if not alignments_file:
            print("Error: Could not find alignments file (.jsonl.lz4) in parent directory")
            sys.exit(1)

        print(f"Generating cluster labels for {args.graph_data_dir}")
        print(f"LLM model: {args.model}")

        # Run async function
        cluster_labels = asyncio.run(
            generate_and_update_cluster_labels(
                alignments_file=alignments_file,
                graph_data_path=args.graph_data_dir,
                model_path=args.model,
                context_window=args.context_window,
                top_k=args.top_k,
                port=args.port
            )
        )

        print(f"\n✓ Generated labels for {len(cluster_labels)} clusters!")


if __name__ == "__main__":
    main()
