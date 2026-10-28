<template>
    <div id="alignment-group" class="mt-4">
        <h2 class="group-heading">Reuses of a passage</h2>
        <figure class="epigraph side-source-color" v-if="sourcePassage.source_passage">
            <blockquote class="epigraph-text" :lang="globalConfig.sourceLanguage || undefined">
                <p>{{ sourcePassage.source_passage }}</p>
            </blockquote>
            <figcaption class="epigraph-source">
                <span class="epigraph-dash" aria-hidden="true">&mdash;</span>
                <citations :citation="globalConfig.sourceCitation" :alignment="sourcePassage" tag="span"></citations>
            </figcaption>
        </figure>
        <!-- Overview of the reuses along time, the list below gives the same dates -->
        <div class="lineage-strip" v-if="strip" aria-hidden="true">
            <span class="strip-end start">{{ strip.from }}</span>
            <div class="strip-axis">
                <span v-for="tick in strip.ticks" :key="tick.year" class="strip-tick" :class="tick.kind"
                    :style="{ left: `${tick.left}%` }"></span>
            </div>
            <span class="strip-end end">{{ strip.to }}</span>
        </div>
        <p class="lineage-dek" v-if="reuseCount">
            <strong>{{ reuseCount.toLocaleString() }}</strong> {{ reuseCount == 1 ? "reuse" : "reuses" }}<template
                v-if="documentCount"> in <strong>{{ documentCount.toLocaleString() }}</strong> other
                {{ documentCount == 1 ? "document" : "documents" }}</template><template v-if="timeline.length">, {{
                    periodText }}</template>
        </p>
        <div id="timeline-container" :class="{ 'origin-first': originIndex === 0 }">
            <div id="vertical-line"></div>
            <template v-for="(date, index) in timeline" :key="index">
                <div class="lineage-origin" v-if="index === originIndex" aria-hidden="true"
                    :style="{ paddingTop: `${originSpacing}rem` }">
                    <span class="origin-year">{{ anchorYear }}</span>
                    <span class="lineage-node source"></span>
                </div>
                <section class="timeline-dates" :style="{ paddingTop: `${spacing(index)}rem` }">
                    <div class="date-marker">
                        <h3 class="year" :class="{ undated: yearOf(date) === null }">{{ yearOf(date) === null ?
                            "Undated" : date.year }}</h3>
                        <span class="year-gap" v-if="gapLabel(date.year)">{{ gapLabel(date.year) }}</span>
                    </div>
                    <span class="lineage-node" :class="date.result[0].direction" aria-hidden="true"></span>
                    <div class="date-reuses">
                        <div class="timeline-events card shadow-1" v-for="(reuse, reuseIndex) in date.result"
                            :key="reuseIndex" :class="`side-${reuse.direction}`">
                            <h4 class="h5 reuse-title">
                                <button type="button" class="reuse-toggle"
                                    :aria-expanded="String(isOpen(index, reuseIndex))"
                                    :aria-controls="`reuse-${index}-${reuseIndex}`"
                                    @click="togglePassage(index, reuseIndex)">
                                    <citations :citation="globalConfig[`${reuse.direction}Citation`]" :alignment="reuse"
                                        tag="span"></citations>
                                    <span class="toggle-chevron" aria-hidden="true"></span>
                                </button>
                            </h4>
                            <div class="timeline-text-content m-0" :class="{ show: isOpen(index, reuseIndex) }"
                                :id="`reuse-${index}-${reuseIndex}`">
                                <p class="text-content m-0"
                                    :lang="globalConfig[`${reuse.direction}Language`] || undefined">
                                    {{ reuse[`${reuse.direction}_context_before`] }}
                                    <span class="highlight">{{ reuse[`${reuse.direction}_passage`] }}</span> {{
                                        reuse[`${reuse.direction}_context_after`] }}
                                </p>
                                <span class="diff-row">
                                    <button type="button" class="group-diff-btn" @click="showDifferences(reuse)">Show
                                        differences</button>
                                </span>
                            </div>
                        </div>
                    </div>
                </section>
            </template>
            <div class="lineage-origin" v-if="originIndex === timeline.length" aria-hidden="true"
                :style="{ paddingTop: `${originSpacing}rem` }">
                <span class="origin-year">{{ anchorYear }}</span>
                <span class="lineage-node source"></span>
            </div>
        </div>
        <div id="passage-diff" ref="modalElement" class="modal fade" tabindex="-1" aria-labelledby="passage-diff-title">
            <div class="modal-dialog modal-xl">
                <div class="modal-content">
                    <div class="modal-header">
                        <h2 class="modal-title h5" id="passage-diff-title">Passage Pair</h2>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <passage-pair v-if="showPair" :alignment="localAlignment" :diffed="true" :index="0"
                            :heading-level="3" :startByteIndex="startByteIndex" :endByteIndex="endByteIndex"
                            :key="render"></passage-pair>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup>
import axios from "axios";
import { Modal } from "bootstrap";
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useRoute } from "vue-router";
import globalConfig from "../config.js";
import { paramsToUrl } from "../utils.js";
import citations from "./citations.vue";
import passagePair from "./passagePair.vue";

const route = useRoute();
const timeline = ref({});
const sourcePassage = ref({});
const openPassages = ref({});
const showPair = ref(false);
const localAlignment = ref({});
const startByteIndex = ref(0);
const endByteIndex = ref(0);
const render = ref(0); // used to force re-rendering of passagePair
const modalElement = ref(null);
let modal = null;
let modalOpener = null;

const anchorYear = computed(() => {
    const year = parseInt(sourcePassage.value.source_year);
    return isNaN(year) ? null : year;
});

// Year of a date in the list, or null when undated
function yearOf(date) {
    const year = parseInt(date && date.year);
    return isNaN(year) ? null : year;
}

// The passage's own date, in its place among the reuses
const originIndex = computed(() => {
    if (anchorYear.value === null || !Array.isArray(timeline.value)) return -1;
    const index = timeline.value.findIndex((date) => yearOf(date) === null || yearOf(date) >= anchorYear.value);
    return index < 0 ? timeline.value.length : index;
});

// Space before each date grows with the years since the one above it
function gapSpace(previous, current) {
    const years = previous === null || current === null ? 0 : Math.max(0, current - previous);
    return Math.min(6.5, 1 + Math.sqrt(years) * 0.55);
}

function spacing(index) {
    const previous = index === originIndex.value ? anchorYear.value : index ? yearOf(timeline.value[index - 1]) : null;
    return index || index === originIndex.value ? gapSpace(previous, yearOf(timeline.value[index])) : 0.25;
}

const originSpacing = computed(() =>
    originIndex.value > 0 ? gapSpace(yearOf(timeline.value[originIndex.value - 1]), anchorYear.value) : 0.25
);

function gapLabel(value) {
    const year = yearOf({ year: value });
    if (year === null || anchorYear.value === null || year == anchorYear.value) return "";
    const gap = year - anchorYear.value;
    const years = `${Math.abs(gap)} ${Math.abs(gap) == 1 ? "year" : "years"}`;
    return gap > 0 ? `+${years}` : `${years} before`;
}

const reuseCount = computed(() =>
    Array.isArray(timeline.value) ? timeline.value.reduce((sum, date) => sum + date.result.length, 0) : 0
);
// Documents other than the original's: the sorted results' count for the passage, less the original
const documentCount = computed(() => {
    if (!Array.isArray(timeline.value)) return 0;
    const files = new Set(timeline.value.flatMap((date) => date.result.map((reuse) => reuse.target_filename)));
    files.delete(sourcePassage.value.source_filename);
    return files.size;
});
const periodText = computed(() => {
    const years = timeline.value.map(yearOf).filter((year) => year !== null);
    if (!years.length) return "undated";
    const [first, last] = [Math.min(...years), Math.max(...years)];
    return first == last ? `${first}` : `${first}–${last}`;
});

const strip = computed(() => {
    const dates = (Array.isArray(timeline.value) ? timeline.value : []).filter((date) => yearOf(date) !== null);
    const years = dates.map(yearOf);
    if (anchorYear.value !== null) years.push(anchorYear.value);
    if (years.length < 2) return null;
    const from = Math.min(...years);
    const to = Math.max(...years);
    const left = (year) => (to == from ? 50 : (100 * (year - from)) / (to - from));
    const ticks = dates.map((date) => ({ year: date.year, kind: date.result[0].direction, left: left(date.year) }));
    if (anchorYear.value !== null) ticks.push({ year: "anchor", kind: "anchor", left: left(anchorYear.value) });
    return { from, to, ticks };
});

function returnFocus() {
    if (modalOpener) modalOpener.focus();
}

onMounted(() => {
    modal = new Modal(modalElement.value, { backdrop: "static" });
    modalElement.value.addEventListener("hidden.bs.modal", returnFocus);
});

// Hidden rather than disposed, so the next page doesn't inherit the modal's scroll lock
onBeforeUnmount(() => {
    modalElement.value.removeEventListener("hidden.bs.modal", returnFocus);
    modal.hide();
});

function fetchData() {
    const params = { db_table: globalConfig.databaseName };
    openPassages.value = {};
    axios
        .get(`${globalConfig.apiServer}/group/${route.params.groupId}/?${paramsToUrl(params)}`)
        .then((response) => {
            timeline.value = response.data.passageList;
            sourcePassage.value = response.data.original_passage;
        })
        .catch((error) => {
            console.log(error);
        });
}
fetchData();
watch(() => route.fullPath, fetchData);

function isOpen(index, reuseIndex) {
    return Boolean(openPassages.value[`${index}-${reuseIndex}`]);
}

function togglePassage(index, reuseIndex) {
    const key = `${index}-${reuseIndex}`;
    openPassages.value[key] = !openPassages.value[key];
}

function showDifferences(reuse) {
    modalOpener = document.activeElement;
    showPair.value = false;
    localAlignment.value = { ...sourcePassage.value };
    // The reuse's own side only: it holds both sides of its alignment
    const side = `${reuse.direction}_`;
    for (const key in reuse) {
        if (key.startsWith(side)) {
            localAlignment.value[`target_${key.slice(side.length)}`] = reuse[key];
        }
    }
    localAlignment.value.count = 1;
    reuse.source_passage = reuse.source_passage.replace(/\s\s+/g, " ");
    sourcePassage.value.source_passage = sourcePassage.value.source_passage.replace(/\s\s+/g, " ");
    startByteIndex.value = sourcePassage.value.source_passage.indexOf(reuse.source_passage);
    let remove = reuse.source_passage.length;
    while (startByteIndex.value == -1) { // if the passage is not found, remove one character from the end and try again
        startByteIndex.value = sourcePassage.value.source_passage.indexOf(reuse.source_passage.slice(0, remove));
        remove -= 1;
    }
    endByteIndex.value = startByteIndex.value + reuse.source_passage.length;
    render.value += 1;
    showPair.value = true;
    modal.show();
}
</script>

<style lang="scss" scoped>
@import "../assets/theme.module.scss";

#alignment-group {
    max-width: 62rem;
    margin: 0 auto;
}

// Title in spaced small caps over a fleuron; the font has no small caps, so they are synthesised
.group-heading {
    margin: 1.75rem 0 1.25rem;
    color: $link-color;
    font-family: $font-text;
    font-size: 1.35rem;
    font-weight: 600;
    font-variant-caps: small-caps;
    letter-spacing: 0.08em;
    line-height: 1.3;
    text-align: center;
}

.group-heading::after {
    content: "\2766";
    content: "\2766" / "";
    display: block;
    margin-top: 0.2rem;
    color: $button-color;
    font-size: 1.1rem;
    letter-spacing: 0;
}

// The passage as an epigraph, its source below and to the right
.epigraph {
    max-width: 48rem;
    margin: 0 auto;
    padding: 0 1.5rem;
}

.epigraph-text {
    position: relative;
    margin: 0;
    padding-left: 1.75rem;
    color: $ink;
    font-size: 1.35rem;
    font-style: italic;
    line-height: 1.5;
}

.epigraph-text::before {
    content: "\201C";
    content: "\201C" / "";
    position: absolute;
    left: -0.4rem;
    top: -0.55rem;
    color: $button-color;
    font-size: 4rem;
    font-style: normal;
    line-height: 1;
}

.epigraph-text p {
    margin: 0;
}

.epigraph-text p::after {
    content: "\2060\201D";
    content: "\2060\201D" / "";
    position: relative;
    top: 0.625rem;
    right: -0.735rem;
    vertical-align: -1.25rem;
    margin-right: -0.37em;
    color: $button-color;
    font-size: 4rem;
    font-style: normal;
    line-height: 0;
}

.epigraph-source {
    margin-top: 0.75rem;
    padding-left: 1.75rem;
    font-size: 1.05rem;
    text-align: right;
}

.epigraph-dash {
    margin-right: 0.35rem;
    color: $muted;
}

.epigraph-source :deep(span.d-block) {
    display: inline !important;
    margin: 0 !important;
    padding: 0 !important;
}

.epigraph-source :deep([data-field="author"]) {
    color: $ink;
    font-variant-caps: small-caps;
    letter-spacing: 0.03em;
}

.epigraph-source :deep([data-field="year"]) {
    color: $ink;
    font-weight: 400;
}

// Every date of the family on one axis
.lineage-strip {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    max-width: 40rem;
    margin: 2rem auto 0;
    color: $muted;
    font-family: $font-ui;
    font-size: 0.75rem;
    font-weight: 600;
}

.strip-axis {
    position: relative;
    flex: 1 1 auto;
    height: 1.5px;
    background: linear-gradient(90deg, $source-color, $target-color);
}

.strip-tick {
    position: absolute;
    top: 50%;
    width: 9px;
    height: 9px;
    margin: -4.5px 0 0 -4.5px;
    border: 1.5px solid $target-color;
    border-radius: 50%;
    background-color: $paper;
}

.strip-tick.source,
.strip-tick.anchor {
    border-color: $source-color;
}

.strip-tick.anchor {
    width: 11px;
    height: 11px;
    margin: -5.5px 0 0 -5.5px;
    background-color: $source-color;
}

.lineage-dek {
    margin: 0.75rem 0 0;
    color: $muted;
    font-family: $font-ui;
    font-size: 0.85rem;
    text-align: center;
}

.lineage-dek strong {
    color: $ink;
}

.strip-end {
    @include display-type;
    color: $ink;
    font-size: 1rem;
}

#timeline-container {
    position: relative;
    margin-top: 1.5rem;
    padding-bottom: 2rem;
}

// Axis down the left: years on one side, reuses on the other
#vertical-line {
    position: absolute;
    left: 7.5rem;
    top: 0.9rem;
    bottom: 2rem;
    width: 1.5px;
    background-color: $target-color;
    z-index: 0;
}

.origin-first #vertical-line {
    background: linear-gradient($source-color, $target-color 12rem, $target-color);
}

.lineage-origin,
.timeline-dates {
    position: relative;
    display: grid;
    grid-template-columns: 6.5rem 2rem minmax(0, 1fr);
    align-items: start;
}


.origin-year {
    grid-column: 1;
    color: $source-color;
    @include display-type;
    font-size: 1.6rem;
    font-weight: 600;
    line-height: 1;
    text-align: right;
}

.date-marker {
    grid-column: 1;
    text-align: right;
}

.year {
    margin: 0.35rem 0 0;
    color: $ink;
    @include display-type;
    font-size: 1.6rem;
    font-weight: 600;
    line-height: 1;
}

.year.undated {
    color: $muted;
    font-size: 1rem;
}

.year-gap {
    display: block;
    margin-top: 0.3rem;
    color: $muted;
    font-family: $font-ui;
    font-size: 0.72rem;
    font-weight: 600;
}

.lineage-node {
    grid-column: 2;
    justify-self: center;
    position: relative;
    z-index: 1;
    width: 13px;
    height: 13px;
    margin-top: 0.7rem;
    margin-left: 1px;
    border: 2px solid $target-color;
    border-radius: 50%;
    background-color: $paper;
}

.lineage-node.source {
    border-color: $source-color;
    background-color: $source-color;
}

.lineage-origin .lineage-node {
    margin-top: 0.45rem;
}

.date-reuses {
    grid-column: 3;
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
}

.timeline-events {
    border-left: 3px solid $target-color;
    border-radius: 3px;
}

.timeline-events.side-source {
    border-left-color: $source-color;
}

.reuse-title {
    margin: 0;
    font-size: 1.05rem;
}

.reuse-toggle {
    display: flex;
    align-items: center;
    gap: 1rem;
    width: 100%;
    padding: 0.7rem 1rem;
    border-width: 0;
    background-color: transparent;
    color: inherit;
    font: inherit;
    text-align: left;
}

.reuse-toggle:hover {
    background-color: rgba($target-color, 0.04);
}

.reuse-toggle :deep(.citation) {
    line-height: 1.4;
}

.reuse-toggle :deep(span.d-block) {
    flex: 1 1 auto;
    margin: 0 !important;
    padding: 0 !important;
}

.toggle-chevron {
    flex: 0 0 auto;
    width: 0.55rem;
    height: 0.55rem;
    margin-right: 0.25rem;
    border-right: 1.5px solid currentColor;
    border-bottom: 1.5px solid currentColor;
    transform: translateY(-25%) rotate(45deg);
    transition: transform 0.2s ease-out;
}

.reuse-toggle[aria-expanded="true"] .toggle-chevron {
    transform: translateY(25%) rotate(225deg);
}

.side-target .reuse-toggle {
    color: $target-color;
}

.side-source .reuse-toggle {
    color: $source-color;
}

.side-target .reuse-toggle:focus-visible {
    outline-color: $target-color !important;
}

.highlight {
    font-weight: 600;
}

.side-target .highlight {
    color: $target-deep;
}

.side-source .highlight {
    color: $source-deep;
}

.timeline-text-content {
    visibility: hidden;
    max-height: 0;
    font-size: 1rem;
    overflow: hidden;
    position: relative;
    line-height: 0;
    padding: 0 1rem;
    transform: translateY(-25%);
    opacity: 0;
    color: $muted;
    transition: all 200ms ease-out;
}

.timeline-text-content.show {
    visibility: visible;
    max-height: none;
    opacity: 1;
    padding-top: 0.15rem;
    padding-bottom: 0.9rem;
    line-height: 1.6;
    transform: translateY(0);
    border-top: 1px solid $rule-color;
}

.timeline-text-content.show .text-content {
    padding-top: 0.75rem;
}

.diff-row {
    display: block;
    margin-top: 0.75rem;
}

@media (max-width: 767.98px) {
    .epigraph {
        padding: 0 0.25rem;
    }

    .epigraph-text {
        font-size: 1.15rem;
    }

    #vertical-line {
        left: 4.2rem;
    }

    .lineage-origin,
    .timeline-dates {
        grid-template-columns: 3.4rem 1.6rem minmax(0, 1fr);
    }

    .year,
    .origin-year {
        font-size: 1.15rem;
    }

    .year-gap {
        font-size: 0.65rem;
    }

    .reuse-toggle {
        padding: 0.6rem 0.75rem;
        gap: 0.5rem;
    }
}
</style>
