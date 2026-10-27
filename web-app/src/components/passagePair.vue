<template>
    <div class="local-pair" :class="{ 'showing-diff': showingDiff }">
        <div class="pair-grid">
            <div class="pair-side side-source">
                <component :is="`h${headingLevel}`" class="side-label">
                    <span class="pair-number" aria-hidden="true" v-if="number">{{ number }}</span>
                    <span class="side-mark source" aria-hidden="true"></span>
                    <span class="side-label-text" v-html="globalConfig.sourceLabel"></span>
                </component>
                <citations class="side-citation" :citation="globalConfig.sourceCitation" :alignment="alignment"
                    :link-to-doc="globalConfig.sourceLinkToDocMetadata"></citations>
                <p class="passage-text" :lang="globalConfig.sourceLanguage || undefined">
                    {{ alignment.source_context_before }}
                    <span class="source-passage" v-if="sourceHtml === null">{{ alignment.source_passage }}</span>
                    <span class="source-passage" v-else v-html="sourceHtml"></span>
                    {{ alignment.source_context_after }}
                </p>
                <div class="side-actions">
                    <button type="button" class="context-btn" v-if="globalConfig.sourceTextAvailable"
                        @click="goToContext(alignment, 'source')">
                        View passage in context<span class="visually-hidden">: {{ sourceLabel }}</span>
                    </button>
                </div>
            </div>

            <div class="pair-bridge">
                <div class="bridge-relation">
                    <p class="bridge-gap" v-if="yearGap !== null">
                        <span aria-hidden="true">
                            <template v-if="yearGap === 0">same year</template>
                            <template v-else><span class="gap-number"><span class="gap-sign">{{ yearGap > 0 ? "+" : "−"
                            }}</span>{{ Math.abs(yearGap).toLocaleString() }}</span> {{ gapYears }}</template>
                        </span>
                        <span class="visually-hidden">{{ gapSpoken }}</span>
                    </p>
                    <div class="bridge-similarity" v-if="globalConfig.matchingAlgorithm == 'vsa'">
                        <span class="similarity-number">{{ Math.round(alignment.similarity * 100) }}%</span>
                        similar
                    </div>
                    <span v-if="globalConfig.matchingAlgorithm == 'vsa' && alignment.llm_stance && alignment.llm_stance !== 'Unknown'"
                        :class="stanceClass(alignment.llm_stance)" class="stance-badge">
                        {{ alignment.llm_stance }}
                    </span>
                    <p class="bridge-length" v-if="alignment.source_passage_length">
                        {{ alignment.source_passage_length.toLocaleString() }}
                        {{ alignment.source_passage_length == 1 ? "word" : "words" }}
                    </p>
                </div>
                <div class="bridge-spine" aria-hidden="true"></div>
            </div>

            <div class="pair-side side-target">
                <component :is="`h${headingLevel}`" class="side-label">
                    <span class="side-mark target" aria-hidden="true"></span>
                    <span class="side-label-text" v-html="globalConfig.targetLabel"></span>
                </component>
                <citations class="side-citation" :citation="globalConfig.targetCitation" :alignment="alignment"
                    :link-to-doc="globalConfig.targetLinkToDocMetadata"></citations>
                <div class="classification-tags"
                    v-if="alignment.target_first_class || alignment.target_second_class || alignment.target_third_class">
                    <span v-if="alignment.target_first_class" class="classification-tag tag-primary">
                        {{ alignment.target_first_class.replace(/_/g, ' ') }}
                    </span>
                    <span v-if="alignment.target_second_class" class="classification-tag tag-secondary">
                        {{ alignment.target_second_class.replace(/_/g, ' ') }}
                    </span>
                    <span v-if="alignment.target_third_class" class="classification-tag tag-tertiary">
                        {{ alignment.target_third_class.replace(/_/g, ' ') }}
                    </span>
                </div>
                <p class="passage-text" :lang="globalConfig.targetLanguage || undefined">
                    {{ alignment.target_context_before }}
                    <span class="target-passage" v-if="targetHtml === null">{{ alignment.target_passage }}</span>
                    <span class="target-passage" v-else v-html="targetHtml"></span>
                    {{ alignment.target_context_after }}
                </p>
                <div class="side-actions">
                    <button type="button" class="context-btn" v-if="globalConfig.targetTextAvailable"
                        @click="goToContext(alignment, 'target')">
                        View passage in context<span class="visually-hidden">: {{ targetLabel }}</span>
                    </button>
                </div>
            </div>
        </div>

        <p v-if="globalConfig.matchingAlgorithm == 'vsa' && alignment.llm_reasoning && !alignment.llm_reasoning.startsWith('Error:')"
            class="reasoning-text">
            {{ alignment.llm_reasoning }}
        </p>
        <!-- The pair's family on the left; comparing the two texts in the middle; their key on the right -->
        <div class="pair-footer">
            <router-link class="reuse-link" :to="`/group/${alignment.group_id}`"
                v-if="globalConfig.matchingAlgorithm == 'sa' && alignment.count > 1">
                <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
                    <path d="M3 3v5.5a2 2 0 0 0 2 2h7.5M10 8l2.5 2.5L10 13" fill="none" stroke="currentColor"
                        stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" />
                </svg>
                Passage reused in {{ alignment.count }} different titles
            </router-link>
            <div class="pair-actions">
                <template v-if="globalConfig.matchingAlgorithm == 'sa'">
                    <button type="button" class="diff-btn" :class="{ active: showingDiff }" @click="showDifferences">
                        <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
                            <path d="M8 2.5v5M5.5 5h5M5.5 12h5" fill="none" stroke="currentColor" stroke-width="1.5"
                                stroke-linecap="round" />
                        </svg>
                        {{ showingDiff ? "Hide differences" : "Show differences" }}
                    </button>
                </template>
                <button type="button" class="diff-btn" :class="{ active: showingDiff }"
                    v-if="globalConfig.matchingAlgorithm == 'vsa'" @click="showMatches">
                    <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
                        <path d="M8 2.5v5M5.5 5h5M5.5 12h5" fill="none" stroke="currentColor" stroke-width="1.5"
                            stroke-linecap="round" />
                    </svg>
                    {{ showingDiff ? "Hide matching words" : "Show matching words" }}
                </button>
            </div>
            <div class="pair-status">
                <span class="spinner-border spinner-border-sm diff-spinner" role="status" v-if="diffLoading">
                    <span class="visually-hidden">Loading...</span>
                </span>
                <span class="diff-key" v-if="showingDiff && globalConfig.matchingAlgorithm == 'sa'" aria-hidden="true">
                    <del class="removed">removed</del>
                    <ins class="added">added</ins>
                </span>
            </div>
        </div>
    </div>
</template>

<script setup>
import emitter from "tiny-emitter/instance";
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import { useRouter } from "vue-router";
import globalConfig from "../config.js";
import { plainText } from "../utils.js";
import citations from "./citations";
import Worker from "./diffStrings?worker";

const props = defineProps({
    alignment: Object,
    index: Number,
    number: Number,
    diffed: Boolean,
    headingLevel: { type: Number, default: 4 },
    startByteIndex: Number,
    endByteIndex: Number,
});

const router = useRouter();
const sourceLabel = plainText(globalConfig.sourceLabel);
const targetLabel = plainText(globalConfig.targetLabel);
// Passages as HTML while differences or matches are shown, null otherwise
const sourceHtml = ref(null);
const targetHtml = ref(null);
const showingDiff = ref(false);
const diffLoading = ref(false);
let worker = null;

// Years from the earlier text to the later one, when both are dated
const yearGap = computed(() => {
    const from = parseInt(props.alignment.source_year);
    const to = parseInt(props.alignment.target_year);
    return isNaN(from) || isNaN(to) ? null : to - from;
});
const gapYears = computed(() => (Math.abs(yearGap.value) == 1 ? "year" : "years"));
const gapSpoken = computed(() => (yearGap.value === 0 ? "Same year" : `${Math.abs(yearGap.value)} ${gapYears.value} apart`));

onMounted(() => {
    if (props.diffed) {
        showDifferences();
    }
});

function stopWorker() {
    if (worker) {
        worker.terminate();
        worker = null;
    }
}

// Lists and modals reuse this component for the next pair: drop the previous pair's differences
watch(
    () => props.alignment,
    () => {
        stopWorker();
        sourceHtml.value = null;
        targetHtml.value = null;
        showingDiff.value = false;
        diffLoading.value = false;
        if (props.diffed) {
            showDifferences();
        }
    }
);
onUnmounted(stopWorker);

function escapeHtml(text) {
    return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function showDifferences() {
    if (showingDiff.value) {
        sourceHtml.value = null;
        targetHtml.value = null;
        showingDiff.value = false;
        emitter.emit("announce", "Differences hidden");
        return;
    }
    const sourceText = props.alignment.source_passage.replace(/\s\s+/g, " ");
    const targetText = props.alignment.target_passage.replace(/\s\s+/g, " ");
    if (props.alignment.source_passage_length > 10000 || props.alignment.target_passage.length > 10000) {
        alert("Passage of 10000 words or more may take up a long time to compare");
    }
    diffLoading.value = true;
    stopWorker();
    const current = new Worker();
    worker = current;
    const startByteIndex = props.startByteIndex;
    const endByteIndex = props.endByteIndex;
    // works around issue with diff-match-patch where no matches are found if the match is far after the beginning of the string
    const slicing = startByteIndex != undefined && startByteIndex > 0;
    if (slicing) {
        current.postMessage([sourceText.slice(startByteIndex, endByteIndex), targetText]);
    } else {
        current.postMessage([sourceText, targetText]);
    }
    current.onmessage = (response) => {
        if (current !== worker) return;
        let newSourceString = "";
        let newTargetString = "";
        for (const [diffCode, text] of response.data) {
            if (diffCode === 0) {
                newSourceString += escapeHtml(text);
                newTargetString += escapeHtml(text);
            } else if (diffCode === -1) {
                newSourceString += `<del class="removed">${escapeHtml(text)}</del>`;
            } else if (diffCode === 1) {
                newTargetString += `<ins class="added">${escapeHtml(text)}</ins>`;
            }
        }
        if (slicing) {
            newSourceString = `<del class="removed">${escapeHtml(sourceText.slice(0, startByteIndex))}</del>${newSourceString}<del class="removed">${escapeHtml(sourceText.slice(endByteIndex))}</del>`;
        }
        sourceHtml.value = newSourceString;
        targetHtml.value = newTargetString;
        showingDiff.value = true;
        diffLoading.value = false;
        stopWorker();
        emitter.emit("announce", "Differences shown");
    };
}

function stanceClass(stance) {
    const map = {
        Agree: "stance-agree",
        Disagree: "stance-disagree",
        Neutral: "stance-neutral",
        Unrelated: "stance-unrelated",
    };
    return map[stance] || "";
}

function showMatches() {
    if (showingDiff.value) {
        sourceHtml.value = null;
        targetHtml.value = null;
        showingDiff.value = false;
        emitter.emit("announce", "Matching words hidden");
        return;
    }
    sourceHtml.value = props.alignment.source_passage_with_matches.replace(/&gt;/g, ">").replace(/&lt;/g, "<");
    targetHtml.value = props.alignment.target_passage_with_matches.replace(/&gt;/g, ">").replace(/&lt;/g, "<");
    showingDiff.value = true;
    emitter.emit("announce", "Matching words shown");
}

function goToContext(alignment, direction) {
    router.push({
        path: "/text-view/",
        query: {
            db_table: globalConfig.databaseName,
            doc_id: alignment[`${direction}_doc_id`],
            start_byte: alignment[`${direction}_start_byte`],
            end_byte: alignment[`${direction}_end_byte`],
            directionSelected: direction,
        },
    });
}
</script>

<style lang="scss" scoped>
@import "../assets/theme.module.scss";

// Facing passages: earlier | bridge | later, rows shared so both texts start level
.pair-grid {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 9.5rem minmax(0, 1fr);
    grid-template-rows: auto auto auto 1fr auto;
}

.pair-side,
.pair-bridge {
    display: grid;
    grid-row: 1 / span 5;
    grid-template-rows: subgrid;
    min-width: 0;
}

.side-source {
    grid-column: 1;
    color: $source-color;
}

.pair-bridge {
    grid-column: 2;
}

.side-target {
    grid-column: 3;
    color: $target-color;
}

// The label sits on the thread that runs from one text to the other
.side-label {
    @include kicker;
    grid-row: 1;
    display: flex;
    align-items: center;
    margin: 0;
    padding: 0.15rem 0;
    color: inherit;
}

.side-label::after {
    content: "";
    flex: 1 1 auto;
    min-width: 1rem;
    height: 1.5px;
    margin-left: 0.75rem;
}

.side-source .side-label::after {
    background: linear-gradient(90deg, rgba($source-color, 0.25), $source-color);
}

.side-target .side-label::after {
    background: linear-gradient(90deg, rgba($target-color, 0.6), rgba($target-color, 0));
}

// Running number leading the source label; negative margins keep it from growing the row
.pair-number {
    margin: -1em 0.6rem -1em 0;
    color: $header-color;
    @include display-type;
    font-size: 1.05rem;
    font-variant-numeric: lining-nums tabular-nums;
    font-weight: 600;
    line-height: 1;
}

.side-label-text {
    min-width: 0;
}

.side-citation {
    grid-row: 2;
    margin: 0.6rem 0 0 !important;
    padding: 0 !important;
}

.classification-tags {
    grid-row: 3;
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin-top: 0.6rem;
}

.passage-text {
    grid-row: 4;
    margin: 0.9rem 0 0;
    color: $muted;
    font-size: 1.02rem;
    line-height: 1.62;
    hyphens: auto;
    overflow-wrap: break-word;
}

.side-actions {
    grid-row: 5;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.5rem 1rem;
    margin-top: 1rem;
}

.context-btn {
    padding: 0.3rem 0.75rem;
    border: 1px solid currentColor;
    border-radius: 2rem;
    background-color: #fff;
    color: inherit;
    font-family: $font-ui;
    font-size: 0.75rem;
    font-weight: 600;
    line-height: 1.2;
    transition: background-color 0.15s ease;
}

.side-source .context-btn:hover {
    background-color: $source-tint;
}

.side-target .context-btn:hover {
    background-color: $target-tint;
}

.side-target .context-btn:focus-visible {
    outline-color: $target-color !important;
}

// The bridge: what separates the two texts in time, and the control comparing them
.bridge-relation {
    grid-column: 1;
    grid-row: 1 / span 3;
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
}

// Thread across the gutter, ending in an arrowhead: centred in the label row, like the labels' own threads
@media (min-width: 768px) {
    .pair-bridge::before,
    .pair-bridge::after {
        content: "";
        grid-column: 1;
        grid-row: 1;
        align-self: center;
        margin-right: 0.35rem;
    }

    .pair-bridge::before {
        height: 1.5px;
        background: linear-gradient(90deg, $source-color, $target-color);
    }

    .pair-bridge::after {
        justify-self: end;
        width: 9px;
        height: 9px;
        border-top: 1.5px solid $target-color;
        border-right: 1.5px solid $target-color;
        transform: translateX(1px) rotate(45deg);
    }
}

.bridge-gap {
    position: relative;
    z-index: 1;
    margin: 0;
    padding: 0 0.5rem;
    background-color: #fff;
    color: $ink;
    font-family: $font-ui;
    font-size: 0.7rem;
    font-weight: 600;
    line-height: 1.2;
    white-space: nowrap;
}

.gap-number {
    display: block;
    @include display-type;
    font-size: 1.6rem;
    font-weight: 600;
    line-height: 1;
    letter-spacing: -0.01em;
}

// The sign in the sans, sturdier than the display face's hairline
.gap-sign {
    margin-right: 0.05em;
    font-family: $font-ui;
    font-size: 0.7em;
    font-weight: 600;
    vertical-align: 0.12em;
}

.bridge-similarity {
    position: relative;
    z-index: 1;
    margin-top: 0.4rem;
    padding: 0 0.5rem;
    background-color: #fff;
    color: $ink;
    font-family: $font-ui;
    font-size: 0.7rem;
    font-weight: 600;
}

.similarity-number {
    display: block;
    @include display-type;
    font-size: 1.35rem;
    line-height: 1;
}

.bridge-length {
    margin: 0.35rem 0 0;
    color: $muted;
    font-family: $font-ui;
    font-size: 0.7rem;
}

.bridge-spine {
    grid-column: 1;
    grid-row: 4 / span 2;
    justify-self: center;
    width: 0;
    margin: 0.9rem 0 0.25rem;
    border-left: 1px dashed $rule-strong;
}

// The reused passage in its side's ink, its context in grey
.source-passage,
.target-passage {
    font-weight: 600;
}

.source-passage {
    color: $source-deep;
}

.target-passage {
    color: $target-deep;
}

// Differences: colour, a tint, and a strike or an underline
:deep(.added),
:deep(.removed) {
    padding: 0 0.12em;
    border-radius: 2px;
    font-weight: 600;
    text-decoration-thickness: 1.5px;
    -webkit-box-decoration-break: clone;
    box-decoration-break: clone;
}

:deep(.added) {
    color: $added-color;
    background-color: $added-tint;
    text-decoration-line: underline;
    text-underline-offset: 0.2em;
}

:deep(.removed) {
    color: $removed-color;
    background-color: $removed-tint;
    text-decoration-line: line-through;
}

:deep(.token-match) {
    color: $ink;
    font-weight: 700;
    background-image: linear-gradient(transparent 58%, rgba(#d9b45a, 0.45) 58%);
}

:deep(.filtered-token) {
    color: $muted;
    font-weight: 400;
}

// Full width under the pair: its family at the left, the comparison in the middle, its key at the right
.pair-footer {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
    align-items: center;
    gap: 0.5rem 1rem;
    margin: 1.1rem calc(-1 * var(--pair-pad-x, 0px)) calc(-1 * var(--pair-pad-bottom, 0px));
    padding: 0.55rem var(--pair-pad-x, 0.75rem);
    border-top: 1px solid $rule-color;
    border-radius: 0 0 3px 3px;
    background-color: $paper-2;
    font-family: $font-ui;
}

.reuse-link {
    display: inline-flex;
    align-items: center;
    gap: 0.375rem;
    grid-column: 1;
    grid-row: 1;
    justify-self: start;
    margin-left: -0.25rem;
    padding: 0.125rem 0.25rem;
    font-size: 0.82rem;
    font-weight: 600;
}

.reuse-link svg,
.diff-btn svg {
    flex: 0 0 auto;
    width: 0.9375rem;
    height: 0.9375rem;
}

.pair-actions {
    grid-column: 2;
    grid-row: 1;
    display: flex;
    justify-content: center;
}

.pair-status {
    grid-column: 3;
    grid-row: 1;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.5rem 0.75rem;
}

// Key to the marks in the passages, while they are shown
.diff-key {
    display: inline-flex;
    gap: 0.5rem;
    font-family: $font-text;
    font-size: 0.85rem;
}

.diff-spinner {
    width: 1rem;
    height: 1rem;
}

.stance-badge {
    position: relative;
    z-index: 1;
    display: inline-block;
    margin-top: 0.4rem;
    font-family: $font-ui;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 2rem;
    color: #fff;
    text-transform: capitalize;
}

.stance-agree { background-color: #2c6e3f; }
.stance-disagree { background-color: #a03b1f; }
.stance-neutral { background-color: #5f6b6d; }
.stance-unrelated { background-color: #707070; }

.reasoning-text {
    max-width: 46rem;
    margin: 1rem auto 0;
    padding-left: 0.9rem;
    border-left: 2px solid $rule-strong;
    color: $muted;
    font-size: 0.9rem;
    font-style: italic;
    line-height: 1.45;
}

.classification-tag {
    display: inline-block;
    padding: 0.2rem 0.55rem;
    border: 1px solid currentColor;
    border-radius: 2rem;
    font-family: $font-ui;
    font-size: 0.72rem;
    font-weight: 600;
    line-height: 1.2;
    text-transform: capitalize;
    background-color: #fff;
}

.tag-primary {
    color: $target-deep;
    background-color: $target-tint;
}

.tag-secondary {
    color: #3d5a3a;
}

.tag-tertiary {
    color: #5c3a6b;
}

@media (max-width: 1199.98px) {
    .pair-grid {
        grid-template-columns: minmax(0, 1fr) 7.5rem minmax(0, 1fr);
    }
}

// One column: the thread turns vertical and runs down the left edge
@media (max-width: 767.98px) {
    .pair-grid {
        grid-template-columns: minmax(0, 1fr);
        grid-template-rows: none;
        padding-left: 1.1rem;
    }

    .pair-side,
    .pair-bridge {
        display: block;
        grid-row: auto;
        grid-column: 1;
        position: relative;
    }

    .pair-side::before,
    .pair-bridge::before {
        content: "";
        position: absolute;
        left: calc(-1.1rem + 0.31em);
        width: 1.5px;
    }

    .side-source::before {
        top: 1rem;
        bottom: 0;
        background-color: $source-color;
    }

    .pair-bridge::before {
        top: 0;
        bottom: 0;
        background: linear-gradient($source-color, $target-color);
    }

    .side-label .side-mark {
        position: absolute;
        left: -1.1rem;
        font-size: 0.7rem;
    }

    .side-label {
        position: relative;
    }

    .pair-bridge {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.5rem 1rem;
        margin: 0.25rem 0;
        padding: 0.75rem 0;
    }

    .bridge-relation {
        flex-direction: row;
        align-items: baseline;
        gap: 0.75rem;
        text-align: left;
    }

    .bridge-spine {
        display: none;
    }

    .bridge-gap,
    .bridge-similarity {
        padding: 0;
        background: none;
    }

    .gap-number,
    .similarity-number {
        display: inline;
        font-size: 1.3rem;
        margin-right: 0.15rem;
    }

    .bridge-similarity,
    .bridge-length,
    .stance-badge {
        margin-top: 0;
    }

    .side-target .side-label {
        padding-top: 0.25rem;
    }

    .pair-footer {
        grid-template-columns: minmax(0, 1fr);
        justify-items: center;
    }

    .reuse-link,
    .pair-actions,
    .pair-status {
        grid-column: 1;
        grid-row: auto;
    }

    .reuse-link {
        justify-self: start;
    }

    .pair-status:empty {
        display: none;
    }
}
</style>
