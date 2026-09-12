"""Turn each cluster's characteristic terms into a short human-readable label.

COPIED SOURCE -- NOT TRACKED UPSTREAM
=====================================
This is a manual copy of `labeler/topologic_labeler/__main__.py` from the
TopoLogic project (github.com/ARTFL-Project/TopoLogic), taken at commit
9a6dcbc9 (2026-09-01). Same authors, same licence (GPL-3).

It is a copy rather than a dependency so that TextPAIR does not require
TopoLogic to be installed alongside it. **Nothing propagates automatically in
either direction.** Improvements to the prompts here will not reach TopoLogic,
and improvements there will not reach here. To re-sync, diff against upstream:

    git -C /path/to/TopoLogic show 9a6dcbc9:labeler/topologic_labeler/__main__.py

Local changes from upstream are limited to this docstring and to user-facing
strings ("cluster labeler:" in place of "topologic-labeler:"); the prompts and
control flow are unmodified, so a diff should stay readable.

Runs in the graph environment, which carries a `transformers` new enough for
current instruction-tuned checkpoints.


WHAT IT DOES
============
Reads `topic_words.json` -- a list of `{"name": <cluster id>, "top_words":
[[word, weight], ...]}` entries -- and writes a `label` into each, in place.
Extra keys are preserved untouched.

Three properties matter for how it is used upstream of here:

1. It declines to invent a theme. When the terms are too generic or
   heterogeneous to support one, the prompt directs the model to return a
   residual category ("Vocabulaire abstrait") rather than a false specific label.
2. It relabels sibling clusters together. A first pass sees one cluster at a
   time and so cannot avoid giving two clusters the same name; a second pass
   labels similar clusters jointly, with the terms that separate them called
   out, and requires the model to cite the terms behind each label.
3. That second pass is gated. `labels_discriminate` asks the model to match
   clusters back to their own labels from distinctive terms alone; when it
   succeeds the labels already separate and the expensive pass is skipped.

Because term extraction upstream is c-TF-IDF rather than passage sampling, the
prompt carries roughly twenty weighted terms instead of a hundred passages, so a
small instruction-tuned model is sufficient. Note the checkpoint sets the real
floor on `transformers`, not this module: gemma-4 requires >= 5.5.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import unicodedata
from typing import Any, Dict, List, Sequence, Tuple

from tqdm import tqdm


PROMPT_INSTRUCTION = (
    "You are labeling topics from a topic model. Given the characteristic terms "
    "of one topic (ordered by weight, most important first, with weights), "
    "return a single {language} label that names the underlying theme.\n\n"
    "Rules — read carefully:\n"
    "1. Produce a noun phrase of 1 to 5 words. Prefer a single well-chosen word "
    "when a precise hypernym captures the theme (e.g., \"Fiscalité\", \"Religion\", "
    "\"Grammaire\") — a short label is usually better than a padded one.\n"
    "2. Consider ALL the terms in the list — including the lower-weighted ones. "
    "They are essential clues for disambiguating what the high-weight terms mean.\n"
    "3. NEVER output the pattern \"<top_word_1> <and/et/y/und/&/etc.> <top_word_2>\" "
    "or any trivial concatenation of the top two words with a conjunction. That "
    "is always considered a FAILED label — synthesize at a higher level of "
    "abstraction instead.\n"
    "4. Do not repeat a word and its adjective form (e.g., \"Religion religieuse\" "
    "is a tautology — write \"Religion\" or something more specific).\n"
    "5. If a word has multiple meanings (e.g., \"corps\" = human body OR political "
    "body), use the surrounding terms in the list to pick the right sense.\n"
    "6. If the terms are so generic or heterogeneous that no clear theme emerges, "
    "label it as a residual/abstract category (e.g., \"Vocabulaire abstrait\", "
    "\"Abstract vocabulary\") rather than forcing a false specific theme.\n"
    "7. Return only the label — no quotes, no explanation, no trailing punctuation.\n"
    "8. The label MUST be written in {language}. Even if the terms look like "
    "English or another language, your response is in {language}.\n\n"
    "Examples (apply the pattern in {language}):\n"
    "- terms [tax, owner, revenue, rate, value]\n"
    "    GOOD: \"Property taxation\"    BAD: \"Tax and owner\"\n"
    "- terms [science, art, genius, glory, talent, discovery, progress]\n"
    "    GOOD: \"Intellectual achievement\"    BAD: \"Science and art\"\n"
    "- terms [life, death, honor, parliament, protestant, magistrate, eulogy, friendship]\n"
    "    GOOD: \"Civic memorials\"    BAD: \"Life and death\"\n"
    "- terms [body, member, council, committee, session, deliberation]\n"
    "    GOOD: \"Legislative bodies\"    BAD: \"Body and member\" (AND never read \"body\" as anatomical when \"council\" appears)\n"
    "- terms [law, system, movement, legislation, execution, legislator]\n"
    "    GOOD: \"Lawmaking process\"    BAD: \"Law and movement\"\n"
    "- terms [religion, priest, god, clergy, worship, ritual]\n"
    "    GOOD: \"Religious practice\"    BAD: \"Religion religious\"  or  \"Religion and priest\"\n"
    "- terms [armée, troupe, ennemi, bataille, combat]\n"
    "    GOOD: \"Art militaire\"    BAD: \"Armée et troupe\"\n"
    "- terms [time, place, moment, motive, sort, gathering, side]  (generic)\n"
    "    GOOD: \"Abstract vocabulary\"    BAD: \"Time management\"\n"
)


# Second pass. The first pass sees one topic at a time, so it cannot avoid
# giving two topics the same label. This pass labels sibling topics together,
# with the terms that separate them called out. It never shows the existing
# label, which would only anchor the rewrite, and it requires the model to cite
# the terms behind each label so the claim can be checked.

SECOND_PASS_INSTRUCTION = (
    "Label each of the {n} topics below. They come from the same topic model and "
    "overlap, so a reader must be able to tell them apart from the labels alone.\n\n"
    "For each topic you get its main terms, which say what it is about, and the "
    "terms that set it apart from the others here.\n\n"
    "{blocks}"
    "Labels already used by other topics in this model — do not return any of "
    "these, or anything close:\n  {taken}\n\n"
    "Rules:\n"
    "1. Name the subject from the MAIN terms. The label should read as a natural "
    "name for the topic as a whole, not as a list of its rarest words.\n"
    "2. Then check it against the other topics here: if your label would fit more "
    "than one of them, use the SETS IT APART terms to narrow it until it fits "
    "only this one.\n"
    "3. Do not invent a subject the terms do not support, and do not drift off the "
    "subject the topics share.\n"
    "4. Each label: a {language} noun phrase of 1 to 5 words, grammatical on its "
    "own. Never name a topic by joining two of its terms with a conjunction "
    "(\"Finance et credit\" is always wrong) — find the word or phrase that covers "
    "both instead.\n"
    "5. All {n} labels must differ from each other.\n"
    "6. After each label, write || and then the 2 or 3 terms it comes from, copied "
    "exactly from that topic's lists.\n\n"
    "Return exactly {n} lines and nothing else:\n"
    "{example}"
)

# Gate for the second pass. Rather than asking whether labels are
# distinguishable, put them to work: show each topic's distinctive terms and ask
# the model to match topics to labels. Labels that discriminate make that easy.
# Keeps the pass from churning labels that were already fine.

DISCRIMINATION_INSTRUCTION = (
    "Each topic below is described by the terms that set it apart from the others. "
    "Match every topic to the one label that fits it best.\n\n"
    "Labels:\n{choices}\n\n"
    "{blocks}"
    "Each label is used exactly once. Return exactly {n} lines and nothing else, "
    "in this form:\n{example}"
)


def labels_discriminate(
    pipe,
    group: Sequence[int],
    labels: Dict[int, str],
    distinctive: Dict[int, List[str]],
) -> bool:
    """True when the model can match each topic back to its own label.

    Label order is shuffled (deterministically, so runs repeat) to keep position
    from giving the answer away.
    """
    letters = [chr(ord("A") + i) for i in range(len(group))]
    order = sorted(range(len(group)), key=lambda i: _fold(labels[group[i]]))
    choices = "\n".join(f"  {n + 1}. {labels[group[i]]}" for n, i in enumerate(order))
    blocks = "".join(
        f"{letter}. terms: {', '.join(distinctive[topic_id][:8])}\n"
        for letter, topic_id in zip(letters, group)
    )
    prompt = DISCRIMINATION_INSTRUCTION.format(
        choices=choices,
        blocks=blocks,
        n=len(group),
        example="\n".join(f"{letter}: <number>" for letter in letters),
    )
    try:
        reply = _generate(pipe, prompt, max_new_tokens=8 * len(group) + 16)
    except Exception:
        # If the test cannot run, assume the labels are fine and change nothing.
        return True

    picked: Dict[str, int] = {}
    for line in reply.splitlines():
        match = re.match(r"^\s*\(?([A-Z])\)?\s*[:.\)-]\s*(\d+)", line)
        if match and match.group(1) in letters:
            picked[match.group(1)] = int(match.group(2)) - 1
    if len(picked) != len(group) or len(set(picked.values())) != len(group):
        return False
    return all(order[picked[letter]] == i for i, letter in enumerate(letters))


REPAIR_INSTRUCTION = (
    "The label \"{label}\" was rejected: it names two things joined by a "
    "conjunction instead of naming one theme.\n\n"
    "Write a better {language} label for this topic. Find the single word or "
    "phrase that COVERS both halves — the category they are both instances of — "
    "rather than listing them. A one-word label is fine if it is the right word. "
    "The label must contain no conjunction.\n\n"
    "Return only the label.\n\n"
    "terms: {payload}"
)

_LABEL_STOPWORDS = {
    "de", "du", "des", "la", "le", "les", "of", "the", "a", "aux", "en", "dans",
    "sur", "for", "in", "der", "die", "das", "und", "el", "los", "las", "il",
    "lo", "di", "da",
}
def _fold(word: str) -> str:
    """Lowercase, strip accents, and truncate to a crude stem."""
    folded = unicodedata.normalize("NFKD", word.lower())
    return "".join(c for c in folded if not unicodedata.combining(c))



# Conjunctions to look for, resolved from --language. Several languages use a
# single letter that is a preposition or article elsewhere ("y", "e", "i", "a"),
# so those are only checked when that language is requested.
_CONJUNCTIONS_BY_LANGUAGE = {
    "english": {"and"},
    "french": {"et"}, "francais": {"et"}, "latin": {"et"},
    "german": {"und"}, "deutsch": {"und"},
    "spanish": {"y", "e"}, "espanol": {"y", "e"}, "castellano": {"y", "e"},
    "italian": {"e", "ed"}, "italiano": {"e", "ed"},
    "portuguese": {"e"}, "portugues": {"e"},
    "catalan": {"i"}, "polish": {"i"}, "polski": {"i"},
    "dutch": {"en"}, "nederlands": {"en"},
    "swedish": {"och"}, "svenska": {"och"},
    "norwegian": {"og"}, "danish": {"og"}, "dansk": {"og"},
    "finnish": {"ja"}, "suomi": {"ja"},
    "czech": {"a"}, "cestina": {"a"},
    "romanian": {"si"}, "romana": {"si"},
    "turkish": {"ve"}, "turkce": {"ve"},
    "hungarian": {"es"}, "magyar": {"es"},
    "russian": {"и"}, "ukrainian": {"и", "та"}, "greek": {"και"},
}
# For an unrecognized language: multi-letter conjunctions only, since those
# cannot be mistaken for a preposition.
_DEFAULT_CONJUNCTIONS = {"and", "et", "und", "och", "og", "ed", "ja", "ve", "и", "και", "та"}

# A conjunction is never a content word when comparing two labels.
_LABEL_STOPWORDS |= set().union(*_CONJUNCTIONS_BY_LANGUAGE.values()) | _DEFAULT_CONJUNCTIONS

_JUXTAPOSITION_RE = re.compile(r"^(\w+)\s+(\w+)\s+(\w+)$", re.UNICODE)


def conjunctions_for(language: str) -> set:
    """Conjunctions worth checking for labels written in `language`."""
    key = re.sub(r"[^a-z]", "", _fold(language or ""))
    return _CONJUNCTIONS_BY_LANGUAGE.get(key, _DEFAULT_CONJUNCTIONS)


def _label_tokens(label: str) -> List[str]:
    r"""Content words of a label, folded and truncated so financiere ~ finance.

    \w rather than [a-z] so non-Latin scripts are not dropped.
    """
    return [w[:6] for w in re.findall(r"\w+", _fold(label), re.UNICODE)
            if w not in _LABEL_STOPWORDS and len(w) > 1]


def candidate_groups(labels: Dict[int, str]) -> List[List[int]]:
    """Topics whose labels may be too alike, grouped transitively.

    Deliberately high recall: no lexical measure can tell "chretienne vs
    catholique" (the same thing) from "penal vs de propriete" (not), so the
    decision is left to the discrimination gate. A group that turns out to be
    fine costs one generation.
    """
    pairs = []
    for a, b in itertools.combinations(sorted(labels), 2):
        ta, tb = _label_tokens(labels[a]), _label_tokens(labels[b])
        sa, sb = set(ta), set(tb)
        if not sa or not sb:
            continue
        jaccard = len(sa & sb) / len(sa | sb)
        if labels[a].strip().lower() == labels[b].strip().lower() or ta[0] == tb[0] or jaccard >= 0.5:
            pairs.append((a, b))

    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    grouped: Dict[int, List[int]] = {}
    for topic_id in labels:
        grouped.setdefault(find(topic_id), []).append(topic_id)
    return sorted((sorted(v) for v in grouped.values() if len(v) > 1), key=lambda g: g[0])


def discriminative_terms(
    group: Sequence[int],
    top_words_by_topic: Dict[int, List[Tuple[str, float]]],
    top_n: int = 8,
) -> Tuple[List[str], Dict[int, List[str]]]:
    """Terms that separate each topic in a group from its siblings.

    Weights are normalized per topic first: c-TF-IDF magnitudes are not
    comparable across topics, so raw differences would mostly measure topic size.
    """
    normalized = {}
    for topic_id in group:
        weights = dict(top_words_by_topic[topic_id])
        peak = max(weights.values(), default=0.0) or 1.0
        normalized[topic_id] = {w: v / peak for w, v in weights.items()}

    common = set.intersection(*(set(normalized[t]) for t in group)) if group else set()
    shared = sorted(common, key=lambda w: -sum(normalized[t][w] for t in group))[:top_n]

    distinctive: Dict[int, List[str]] = {}
    for topic_id in group:
        mine = normalized[topic_id]
        others = [normalized[t] for t in group if t != topic_id]
        ranked = sorted(mine, key=lambda w: -(mine[w] - max((o.get(w, 0.0) for o in others), default=0.0)))
        distinctive[topic_id] = ranked[:top_n]
    return shared, distinctive


def is_juxtaposition(label: str, conjunctions: set | None = None) -> bool:
    """True for labels of the form "<word> and <word>".

    The defect is the juxtaposition itself, not where the words came from: a
    label naming two things has dodged the job of naming the theme. A single
    term is fine; only the conjunction is disqualifying.

    Detection needs a conjunction between two space-separated words, so it does
    nothing for languages that do not separate words that way (Chinese,
    Japanese, Thai) or that attach the conjunction as a clitic (Latin -que).
    """
    match = _JUXTAPOSITION_RE.match(label.strip())
    if not match:
        return False
    return _fold(match.group(2)) in (conjunctions or _DEFAULT_CONJUNCTIONS)


def _format_payload(top_words: List[Tuple[str, float]]) -> str:
    """Render the top-words list as a compact JSON array the model can parse."""
    payload = [{"word": w, "weight": round(float(weight), 4)} for w, weight in top_words]
    return json.dumps(payload, ensure_ascii=False)


def _extract_top_words(entry: Dict[str, Any]) -> List[Tuple[str, float]]:
    """Prefer the rich `top_words` field. Fall back to parsing `description`."""
    raw = entry.get("top_words")
    if raw:
        # Stored as [[word, weight], ...].
        return [(item[0], float(item[1])) for item in raw]
    # Legacy fallback: 10-word comma list, no weights — synthesize rank-based weights
    # so the prompt still conveys ordering.
    words = [w.strip() for w in entry.get("description", "").split(",") if w.strip()]
    n = len(words) or 1
    return [(w, round((n - i) / n, 3)) for i, w in enumerate(words)]


def _clean_label(raw: str) -> str:
    label = raw.strip().strip('"\'`.').splitlines()[0].strip() if raw.strip() else ""
    for prefix in ("Label:", "label:"):
        if label.startswith(prefix):
            label = label[len(prefix):].strip()
    return label.strip('"\'`.')[:80]


def _generate(pipe, prompt: str, max_new_tokens: int = 24) -> str:
    """One turn against the chat model. Gemma's template rejects `system`."""
    out = pipe(
        [{"role": "user", "content": prompt}],
        max_new_tokens=max_new_tokens,
        max_length=None,
        do_sample=False,
    )
    generated = out[0]["generated_text"]
    return generated[-1]["content"] if isinstance(generated, list) else generated


def repair_concatenations(
    pipe,
    labels: Dict[int, str],
    top_words_by_topic: Dict[int, List[Tuple[str, float]]],
    language: str,
) -> int:
    """Re-ask for any label that names two things joined by a conjunction."""
    conjunctions = conjunctions_for(language)
    offenders = [t for t, label in labels.items() if is_juxtaposition(label, conjunctions)]
    if not offenders:
        return 0
    repaired = 0
    for topic_id in tqdm(offenders, desc="Repairing concatenated labels"):
        prompt = REPAIR_INSTRUCTION.format(
            label=labels[topic_id],
            language=language,
            payload=_format_payload(top_words_by_topic[topic_id]),
        )
        for attempt in range(2):
            nudge = "" if attempt == 0 else (
                "\n\nYour previous answer still used a conjunction. Give a single noun "
                "phrase naming the theme, with no conjunction at all."
            )
            try:
                candidate = _clean_label(_generate(pipe, prompt + nudge))
            except Exception as e:
                print(f"cluster labeler: repair of topic {topic_id} failed: {e}", file=sys.stderr)
                break
            # Only accept a replacement that is not itself a juxtaposition.
            if candidate and not is_juxtaposition(candidate, conjunctions):
                print(f"  t{topic_id}: {labels[topic_id]!r} -> {candidate!r}", flush=True)
                labels[topic_id] = candidate
                repaired += 1
                break
    return repaired


def _parse_group_reply(reply: str, letters: Sequence[str]) -> Dict[str, Tuple[str, List[str]]]:
    """Parse "A: <label> || <term>, <term>" lines into {letter: (label, terms)}."""
    parsed: Dict[str, Tuple[str, List[str]]] = {}
    for line in reply.splitlines():
        match = re.match(r"^\s*\(?([A-Z])\)?\s*[:.\)-]\s*(.+?)\s*$", line)
        if not match or match.group(1) not in letters:
            continue
        body = match.group(2)
        label_part, _, evidence_part = body.partition("||")
        label = _clean_label(label_part)
        cited = [t.strip().strip('"\'`.') for t in re.split(r"[,;]", evidence_part) if t.strip()]
        if label:
            parsed[match.group(1)] = (label, cited)
    return parsed


def _evidence_supports(cited: Sequence[str], distinctive: Sequence[str]) -> bool:
    """At least one cited term must really be one of that topic's terms.

    Stops the model naming a subject none of the terms supports.
    """
    if not cited:
        return False
    pool = {_fold(t)[:6] for t in distinctive}
    return any(_fold(t)[:6] in pool for t in cited)


def refine_groups(
    pipe,
    labels: Dict[int, str],
    top_words_by_topic: Dict[int, List[Tuple[str, float]]],
    language: str,
) -> int:
    """Relabel topics whose labels may be indistinguishable from a sibling's."""
    groups = candidate_groups(labels)
    if not groups:
        return 0
    print(f"Second pass: {len(groups)} candidate group(s) of similar labels.", flush=True)

    changed = 0
    kept = 0
    for group in tqdm(groups, desc="Distinguishing similar labels"):
        letters = [chr(ord("A") + i) for i in range(len(group))]
        shared, distinctive = discriminative_terms(group, top_words_by_topic)

        # Identical labels cannot discriminate, so there is nothing to test,
        # and with two members a blind guess passes half the time.
        folded = [_fold(labels[t]).strip() for t in group]
        duplicated = len(set(folded)) != len(folded)
        if not duplicated and labels_discriminate(pipe, group, labels, distinctive):
            kept += 1
            continue
        # Main terms lead: they name the subject, the distinctive terms only
        # separate it from its siblings. The existing label is withheld so the
        # rewrite is not anchored on the wording it replaces.
        blocks = "".join(
            f"{letter}. main terms: {', '.join(w for w, _ in top_words_by_topic[topic_id][:12])}\n"
            f"   sets it apart from the others: {', '.join(distinctive[topic_id])}\n\n"
            for letter, topic_id in zip(letters, group)
        )
        outside = sorted({labels[t] for t in labels if t not in group})
        prompt = SECOND_PASS_INSTRUCTION.format(
            n=len(group),
            language=language,
            shared=", ".join(shared) or "(none)",
            blocks=blocks,
            taken=", ".join(outside) or "(none)",
            example="\n".join(f"{letter}: <label> || <term>, <term>" for letter in letters),
        )

        accepted: Dict[str, str] = {}
        for attempt in range(2):
            nudge = "" if attempt == 0 else (
                "\n\nYour previous answer was rejected. Give exactly one line per letter, "
                "every label different, and cite distinctive terms copied from that "
                "topic's own list."
            )
            try:
                reply = _generate(pipe, prompt + nudge, max_new_tokens=40 * len(group) + 24)
            except Exception as e:
                print(f"cluster labeler: group {group} failed: {e}", file=sys.stderr)
                break

            parsed = _parse_group_reply(reply, letters)
            if len(parsed) != len(group):
                continue
            proposed = [label for label, _ in parsed.values()]
            if len({label.lower() for label in proposed}) != len(group):
                continue
            # Enforce the prompt's no-juxtaposition rule, so this pass cannot
            # undo a repair.
            if any(is_juxtaposition(label, conjunctions_for(language)) for label in proposed):
                continue
            # A rewrite that trades a collision inside the group for one
            # outside it has not helped.
            outside_tokens = {frozenset(_label_tokens(labels[t])) for t in labels if t not in group}
            if any(frozenset(_label_tokens(label)) in outside_tokens for label in proposed):
                continue
            grounded = all(
                _evidence_supports(
                    cited,
                    list(distinctive[topic_id]) + [w for w, _ in top_words_by_topic[topic_id][:12]],
                )
                for (letter, topic_id), (_, cited) in zip(zip(letters, group), parsed.values())
            )
            if not grounded:
                continue
            accepted = {letter: label for letter, (label, _) in parsed.items()}
            break

        if not accepted:
            print(f"  group {group}: no usable reply; keeping first-pass labels.", flush=True)
            continue
        for letter, topic_id in zip(letters, group):
            new_label = accepted[letter]
            if new_label != labels[topic_id]:
                print(f"  t{topic_id}: {labels[topic_id]!r} -> {new_label!r}", flush=True)
                labels[topic_id] = new_label
                changed += 1
    if kept:
        print(f"  {kept} group(s) already discriminated; left unchanged.", flush=True)
    return changed


def label_topics(
    top_words_by_topic: Dict[int, List[Tuple[str, float]]],
    model_id: str,
    language: str = "English",
    existing_labels: Dict[int, str] | None = None,
    second_pass: bool = True,
) -> Dict[int, str]:
    try:
        from transformers import pipeline
        from transformers import logging as transformers_logging
    except ImportError as e:
        print(f"cluster labeler: transformers not available ({e})", file=sys.stderr)
        return {}

    # Silence the per-generation "both max_new_tokens and max_length set"
    # warning — it's benign (max_new_tokens correctly wins) but floods stderr.
    transformers_logging.set_verbosity_error()

    try:
        pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype="auto",
            device_map="auto",
        )
    except Exception as e:
        print(f"cluster labeler: failed to load {model_id} ({e})", file=sys.stderr)
        return {}

    labels: Dict[int, str] = dict(existing_labels or {})
    if existing_labels:
        print(f"Reusing {len(labels)} existing labels; skipping the first pass.", flush=True)
    else:
        instruction = PROMPT_INSTRUCTION.format(language=language)
        for topic_id in tqdm(sorted(top_words_by_topic), desc="Labeling topics"):
            payload = _format_payload(top_words_by_topic[topic_id])
            try:
                raw = _generate(pipe, f"{instruction}\nterms: {payload}\n\nLabel:")
            except Exception as e:
                print(f"cluster labeler: topic {topic_id} failed: {e}", file=sys.stderr)
                continue
            label = _clean_label(raw)
            if label:
                labels[topic_id] = label

    if labels:
        repaired = repair_concatenations(pipe, labels, top_words_by_topic, language)
        refined = refine_groups(pipe, labels, top_words_by_topic, language) if second_pass else 0
        if repaired or refined:
            print(f"Repaired {repaired} concatenated label(s); "
                  f"rewrote {refined} label(s) to distinguish similar topics.", flush=True)

        # Report rather than leave it to be found in the web app.
        seen: Dict[str, List[int]] = {}
        for topic_id, label in labels.items():
            seen.setdefault(_fold(label).strip(), []).append(topic_id)
        duplicates = {k: v for k, v in seen.items() if len(v) > 1}
        if duplicates:
            for label_key, topic_ids in sorted(duplicates.items()):
                print(f"Warning: topics {topic_ids} still share the label "
                      f"{labels[topic_ids[0]]!r}.", flush=True)
        leftover = [t for t, label in labels.items()
                    if is_juxtaposition(label, conjunctions_for(language))]
        if leftover:
            print(f"Warning: {len(leftover)} label(s) still name two things instead of "
                  f"one: {[labels[t] for t in leftover]}", flush=True)

    return labels


def relabel_json(
    path: str,
    model_id: str,
    language: str,
    second_pass: bool = True,
    second_pass_only: bool = False,
) -> int:
    with open(path, "r", encoding="utf-8") as f:
        topics = json.load(f)

    top_words_by_topic = {int(entry["name"]): _extract_top_words(entry) for entry in topics}

    existing = None
    if second_pass_only:
        existing = {int(e["name"]): e["label"] for e in topics if e.get("label")}
        if not existing:
            print(
                "cluster labeler: --second-pass-only needs existing labels, and this "
                "file has none; file not modified.",
                file=sys.stderr,
            )
            return 0

    labels = label_topics(
        top_words_by_topic,
        model_id=model_id,
        language=language,
        existing_labels=existing,
        second_pass=second_pass,
    )
    if not labels:
        print("cluster labeler: no labels produced; file not modified.", file=sys.stderr)
        return 0

    for entry in topics:
        label = labels.get(int(entry["name"]))
        if label:
            entry["label"] = label
        else:
            entry.pop("label", None)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(topics, f, indent=None)
    return len(labels)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m textpair_graph.topic_labeler",
        description="Relabel topics in an existing topic_words.json using an LLM.",
    )
    parser.add_argument("topic_words_path", help="Path to topic_words.json to update in place")
    parser.add_argument("--model", required=True, help="HuggingFace instruction-tuned model ID")
    parser.add_argument("--language", default="English", help="Language for generated labels")
    parser.add_argument(
        "--no-second-pass",
        action="store_true",
        help="Skip the pass that distinguishes topics whose labels are too alike.",
    )
    parser.add_argument(
        "--second-pass-only",
        action="store_true",
        help="Keep the labels already in the file and only run the refinement passes. "
             "Lets an existing model be re-labelled without regenerating every label.",
    )
    args = parser.parse_args()

    n = relabel_json(
        args.topic_words_path,
        args.model,
        args.language,
        second_pass=not args.no_second_pass,
        second_pass_only=args.second_pass_only,
    )
    print(f"cluster labeler: wrote {n} labels to {args.topic_words_path}")


if __name__ == "__main__":
    main()
