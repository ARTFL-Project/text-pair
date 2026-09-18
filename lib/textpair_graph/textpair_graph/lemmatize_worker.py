"""Normalize passages with TextPAIR's preprocessor. Runs in the textpair env.

Executed as a subprocess by `cluster_labeling`, not imported by it: the
preprocessor's spaCy pipeline needs `spacy-transformers`, which pins
`transformers<4.53.3`, while the graph environment needs `>=5.5` for its
sentence-transformers and label model. The textpair environment already
satisfies the first constraint, so the subprocess runs there rather than in an
environment of its own.

Reads {"params": ..., "passages": [...]} -- `params` being a [PREPROCESSING]
section as `PreprocessConfig.from_kwargs` reads one -- and writes a JSON array
of normalized strings in the same order.
"""

import json
import sys


def main() -> None:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <input.json> <output.json>", file=sys.stderr)
        sys.exit(2)
    input_path, output_path = sys.argv[1], sys.argv[2]

    from textpair.preprocessing import PreProcessor

    with open(input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    passages = payload["passages"]

    # One process: the spaCy stage runs in the parent anyway, and the pool would
    # only pay for itself if there were files to read.
    preprocessor = PreProcessor(workers=1, **payload["params"])
    print(
        f"lemmatizer: running on {'GPU' if preprocessor.using_gpu else 'CPU'}",
        file=sys.stderr,
    )
    normalized = [" ".join(forms) for forms in preprocessor.process_strings(passages)]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(normalized, f)
    print(f"lemmatizer: normalized {len(normalized)} passages", file=sys.stderr)


if __name__ == "__main__":
    main()
