---
title: Output files
nav_order: 6
---

# Output files

Each alignment writes its results to the `results/` directory of `--output_path`:

| File | Contents |
|:-----|:---------|
| `alignments.jsonl.lz4` | All alignments TextPAIR found, one JSON object per line. |
| `duplicate_files.csv` | Potential duplicate files between the source and target collections. |
| `banal_alignments.jsonl.lz4` | Alignments flagged as formulaic, set aside when `store_banalities = false`. |
| `filtered_passages.jsonl.lz4` | Alignments removed by the phrase filter. Only written when `phrase_filter` is set. |

The `.lz4` files are compressed; read them with `lz4 -dc alignments.jsonl.lz4`.

These files are meant for further inspection of the alignments, and for post-processing tasks such as filtering or clustering.
