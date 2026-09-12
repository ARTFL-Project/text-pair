"""Alignment-level scatter data for the semantic map.

One point per alignment -- the objects HDBSCAN actually clustered -- as parallel
arrays, with author names in a separate lookup to keep the payload small.
"""

import os
from collections import defaultdict

import lz4.frame
import numpy as np
import orjson
from tqdm import tqdm

# An author needs at least this many passages in a theme to get a marker.
DEFAULT_MIN_AUTHOR_PASSAGES = 5

# Radial rescale about the layout's centre, off by default: it buys canvas at
# the cost of inter-theme adjacency. "quantile" ranks radii, "power" applies
# r ** RADIAL_GAMMA; both preserve angles.
DEFAULT_RADIAL_MODE = "none"
DEFAULT_RADIAL_GAMMA = 0.55

def _medoid_index(positions: np.ndarray, indices: list[int]) -> int:
    """The member closest to the group's mean, so a marker sits on a real point.

    Not the centroid, which can land in empty space beside a group's passages.
    """
    points = positions[indices]
    centre = points.mean(axis=0)
    return int(indices[int(np.argmin(np.linalg.norm(points - centre, axis=1)))])


# Metadata fields, besides the author fields, the client can group points by.
# Each costs an int array per point, so keep the list short.
DEFAULT_FACET_FIELDS = ("title",)


def build_scatter_data(
    alignments_file: str,
    output_path: str,
    author_to_id: dict,
    min_author_passages: int = DEFAULT_MIN_AUTHOR_PASSAGES,
    radial_mode: str = DEFAULT_RADIAL_MODE,
    radial_gamma: float = DEFAULT_RADIAL_GAMMA,
    source_author_field: str = "source_author",
    target_author_field: str = "target_author",
    facet_fields: tuple[str, ...] = DEFAULT_FACET_FIELDS,
) -> None:
    core_labels = np.load(os.path.join(output_path, "cluster_labels_core.npy"))
    merged_labels_path = os.path.join(output_path, "cluster_labels_modified.npy")
    merged_labels = np.load(merged_labels_path) if os.path.exists(merged_labels_path) else core_labels
    positions = np.load(os.path.join(output_path, "embeddings_umap_2d.npy"))

    with open(os.path.join(output_path, "cluster_metadata.json"), "rb") as f:
        metadata = orjson.loads(f.read())

    labels_path = os.path.join(output_path, "cluster_labels.json")
    cluster_labels = {}
    if os.path.exists(labels_path):
        with open(labels_path, "rb") as f:
            cluster_labels = {int(k): v for k, v in orjson.loads(f.read()).items()}
    else:
        print("  no cluster labels yet — run `textpair_graph label` to name the themes")

    coherence_path = os.path.join(output_path, "cluster_coherence.npy")
    coherence = np.load(coherence_path) if os.path.exists(coherence_path) else None

    terms_path = os.path.join(output_path, "topic_words.json")
    cluster_terms = {}
    if os.path.exists(terms_path):
        with open(terms_path, "rb") as f:
            for entry in orjson.loads(f.read()):
                cluster_terms[int(entry["name"])] = [w for w, _ in entry.get("top_words", [])[:8]]

    blank_author_ids = {aid for name, aid in author_to_id.items() if not str(name).strip()}
    id_to_author = {v: k for k, v in author_to_id.items()}

    print("Building alignment scatter...")
    n = min(len(core_labels), len(positions))
    xs, ys, themes, merged_flags = [], [], [], []
    src_ids, tgt_ids, row_ids = [], [], []
    # One id stream per side per facet, interned as we go.
    facet_values: dict[str, list[str]] = {}
    facet_lookup: dict[str, dict[str, int]] = {}
    facet_ids: dict[str, list[int]] = {}
    for field in facet_fields:
        for side in ("source", "target"):
            facet_ids[f"{side}_{field}"] = []
        facet_values[field] = []
        facet_lookup[field] = {}

    def intern(field: str, value: str) -> int:
        value = (value or "").strip()
        if not value:
            return -1
        table = facet_lookup[field]
        if value not in table:
            table[value] = len(facet_values[field])
            facet_values[field].append(value)
        return table[value]
    author_groups: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    theme_members: defaultdict[int, list[int]] = defaultdict(list)

    with lz4.frame.open(alignments_file, "rb") as f:
        for idx, line in tqdm(enumerate(f), total=n, desc="Reading alignments", leave=False):
            if idx >= n:
                break
            theme = int(merged_labels[idx])
            if theme < 0:
                continue  # never placed in any theme
            alignment = orjson.loads(line)
            source_id = author_to_id.get(alignment.get(source_author_field) or "", -1)
            target_id = author_to_id.get(alignment.get(target_author_field) or "", -1)

            point_index = len(xs)
            xs.append(float(positions[idx][0]))
            ys.append(float(positions[idx][1]))
            themes.append(theme)
            # Flag rather than drop points merged in by nearest centroid, so the
            # client can de-emphasise them.
            merged_flags.append(0 if int(core_labels[idx]) >= 0 else 1)
            src_ids.append(source_id)
            tgt_ids.append(target_id)
            row_ids.append(idx + 1)  # DB rowid, for drill-down
            for field in facet_fields:
                for side in ("source", "target"):
                    facet_ids[f"{side}_{field}"].append(
                        intern(field, alignment.get(f"{side}_{field}") or "")
                    )

            theme_members[theme].append(point_index)
            for author_id in (source_id, target_id):
                if author_id >= 0 and author_id not in blank_author_ids:
                    author_groups[(author_id, theme)].append(point_index)

    P = np.column_stack([np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)])

    # Radial rescale, then normalize into [-1, 1] preserving aspect ratio: the
    # renderer plots the payload verbatim.
    if radial_mode and radial_mode != "none":
        # Median rather than mean: the mean is dragged toward whichever side the
        # outlying themes happen to be on, which would tilt the whole rescale.
        origin = np.median(P, axis=0)
        offset = P - origin
        radius = np.linalg.norm(offset, axis=1)

        if radial_mode == "quantile":
            # Clustered points only: noise is hidden by default and would size
            # the map around points nobody is looking at.
            reference = radius[np.array(merged_flags) == 0]
            if len(reference) < 2:
                reference = radius
            ordered = np.sort(reference)
            ranks = np.linspace(0.0, 1.0, len(ordered))
            new_radius = np.interp(radius, ordered, ranks)
        else:
            new_radius = np.power(radius, radial_gamma)

        scale = np.divide(new_radius, radius, out=np.ones_like(radius), where=radius > 0)
        P = origin + offset * scale[:, None]

    centre = (P.max(axis=0) + P.min(axis=0)) / 2.0
    half_span = float(np.max(P.max(axis=0) - P.min(axis=0))) / 2.0 or 1.0
    P = ((P - centre) / half_span).astype(np.float32)
    xs = [float(v) for v in P[:, 0]]
    ys = [float(v) for v in P[:, 1]]
    # No repositioning: the separation on screen is the separation the algorithm
    # found.
    print(f"✓ {len(xs):,} points across {len(theme_members)} themes")

    # Theme labels sit at the medoid of their own points.
    theme_records = []
    for theme, members in sorted(theme_members.items()):
        anchor = _medoid_index(P, members)
        theme_records.append(
            {
                "id": theme,
                "label": cluster_labels.get(theme, ""),
                "terms": cluster_terms.get(theme, []),
                "count": len(members),
                "coherence": round(float(coherence[theme]), 4)
                if coherence is not None and theme < len(coherence)
                else None,
                "x": round(float(P[anchor][0]), 4),
                "y": round(float(P[anchor][1]), 4),
            }
        )

    # Author markers: one per (author, theme), for the "in what contexts is this
    # author used" and "how important is each author here" readings.
    theme_totals = {theme: len(members) for theme, members in theme_members.items()}
    author_totals: defaultdict[int, int] = defaultdict(int)
    for (author_id, _), members in author_groups.items():
        author_totals[author_id] += len(members)

    author_records = []
    for (author_id, theme), members in sorted(author_groups.items()):
        if len(members) < min_author_passages:
            continue
        anchor = _medoid_index(P, members)
        author_records.append(
            {
                "author": author_id,
                "theme": theme,
                "count": len(members),
                # "Relative importance" is ambiguous: share_of_theme is who
                # dominates a theme, share_of_author what an author is used for.
                "share_of_theme": round(len(members) / max(theme_totals[theme], 1), 4),
                "share_of_author": round(len(members) / max(author_totals[author_id], 1), 4),
                "x": round(float(P[anchor][0]), 4),
                "y": round(float(P[anchor][1]), 4),
            }
        )

    # Every author referenced by a point, not just those with markers: the client
    # counts from the points themselves. Blanks excluded.
    used_authors = sorted(
        {a for a in src_ids if a >= 0 and a not in blank_author_ids}
        | {a for a in tgt_ids if a >= 0 and a not in blank_author_ids}
        | {r["author"] for r in author_records}
    )
    payload = {
        "points": {
            "x": [round(v, 4) for v in xs],
            "y": [round(v, 4) for v in ys],
            "theme": themes,
            "merged": merged_flags,
            "source_author": src_ids,
            "target_author": tgt_ids,
            "rowid": row_ids,
            **facet_ids,
        },
        "themes": theme_records,
        "authors": author_records,
        "author_names": {str(a): id_to_author.get(a, "") for a in used_authors},
        # What the client may group by. A facet that is the author field under
        # another name is skipped: corpora without author metadata point their
        # author field at the title, which would give two identical dimensions.
        "facets": [
            {"key": "author", "label": "Author", "names": None},
            *(
                {"key": field, "label": field.capitalize(), "names": facet_values[field]}
                for field in facet_fields
                if facet_values[field]
                and f"source_{field}" != source_author_field
                and f"target_{field}" != target_author_field
            ),
        ],
        "metadata": {
            "total_points": len(xs),
            "directly_themed": int(sum(1 for m in merged_flags if m == 0)),
            "merged_in": int(sum(merged_flags)),
            "n_themes": len(theme_records),
            "min_author_passages": min_author_passages,
            "cluster_selection_method": metadata.get("cluster_selection_method"),
            "min_cluster_size": metadata.get("min_cluster_size"),
        },
    }
    out = os.path.join(output_path, "scatter_data.json")
    with open(out, "wb") as f:
        f.write(orjson.dumps(payload))
    size_mb = os.path.getsize(out) / 1024 / 1024
    print(f"✓ Saved {out} ({size_mb:.1f} MB): {len(xs):,} points, "
          f"{len(theme_records)} themes, {len(author_records)} author markers")
