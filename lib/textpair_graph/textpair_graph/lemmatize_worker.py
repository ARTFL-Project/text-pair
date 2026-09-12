"""Lemmatize and POS-filter passages with a spaCy model. Runs in its own venv.

Executed as a subprocess by `cluster_labeling`, not imported by it: a
transformer-based spaCy pipeline needs `spacy-transformers`, which pins
`transformers<4.53.3`, while the graph environment needs `>=5.5` for its
sentence-transformers and label model.

Reads a JSON array of passages, writes a JSON array of normalized strings in the
same order.
"""

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", help="JSON array of passages")
    parser.add_argument("output_path", help="JSON array of normalized strings")
    parser.add_argument("--model", required=True, help="spaCy model name or path")
    parser.add_argument(
        "--pos-to-keep",
        default="NOUN,ADJ,PROPN",
        help="Comma-separated POS tags to retain (default: NOUN,ADJ,PROPN)",
    )
    parser.add_argument("--min-length", type=int, default=3, help="Minimum lemma length")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    import spacy

    # Term extraction is far faster on GPU; not fatal if unavailable.
    try:
        spacy.require_gpu()
        print("lemmatizer: using GPU", file=sys.stderr)
    except Exception as e:
        print(f"lemmatizer: GPU unavailable ({e}); running on CPU", file=sys.stderr)

    keep = {p.strip().upper() for p in args.pos_to_keep.split(",") if p.strip()}

    with open(args.input_path, "r", encoding="utf-8") as f:
        passages = json.load(f)

    nlp = spacy.load(args.model)
    out = []
    for doc in nlp.pipe(passages, batch_size=args.batch_size):
        out.append(
            " ".join(
                token.lemma_.lower()
                for token in doc
                if (not keep or token.pos_ in keep) and len(token.lemma_) >= args.min_length
            )
        )

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"lemmatizer: normalized {len(out)} passages", file=sys.stderr)


if __name__ == "__main__":
    main()
