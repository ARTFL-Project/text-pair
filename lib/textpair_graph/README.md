# TextPair Graph

Graph building and clustering for TextPair alignment data.

## Installation

```bash
# Basic installation
uv pip install -e .

# With CUDA support (for faster UMAP/HDBSCAN)
uv pip install -e ".[cuda]"
```

## Usage

### Build Graph Model

```bash
python -m textpair_graph build <alignments_file> <output_dir> --model <sbert_model>
```

### Generate Cluster Labels

```bash
python -m textpair_graph label <graph_data_dir> --model <llm_model>
```

## Dependencies

- numpy, scipy, scikit-learn
- umap-learn, hdbscan (clustering)
- networkx (graph manipulation)
- sentence-transformers, torch (embeddings)
- numba (performance)
- lz4, orjson (data I/O)
- textpair (for LLM evaluation)

### Optional

- cuml-cu12 (CUDA-accelerated UMAP/HDBSCAN)
