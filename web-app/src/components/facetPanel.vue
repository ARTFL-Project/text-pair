<template>
    <aside id="facets" class="col-12 col-lg-3 pl-0 position-relative" aria-labelledby="facets-title">
        <div class="card facets-card shadow-1">
            <h3 id="facets-title" class="card-header facets-title">
                <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
                    <path d="M2.5 3.5h11M2.5 8h7M2.5 12.5h4" fill="none" stroke="currentColor" stroke-width="1.75"
                        stroke-linecap="round" />
                </svg>
                {{ heading }}
            </h3>
            <button type="button" id="metadata-list" :class="{ show: facetsCollapsed }" aria-expanded="false"
                aria-controls="facet-fields" @click="toggleFacetList()">Show
                options</button>
            <div id="facet-fields" ref="facetFields" :aria-busy="facetLoading ? 'true' : undefined">
                <div v-for="side in sides" :key="side.name" class="facet-list" :class="[side.color, { hide: facetsCollapsed }]">
                    <h4 class="facet-side" :class="{ stacked: passages }">
                        <span class="side-mark" :class="side.name" aria-hidden="true"></span>
                        <span v-if="passages" class="facet-side-text">
                            <span>{{ side.lead }}</span><span class="visually-hidden">, </span>
                            <span class="facet-side-label" v-html="side.label"></span>
                        </span>
                        <span v-else>in <span v-html="side.label"></span></span>
                    </h4>
                    <div class="list-group list-group-flush">
                        <button type="button" class="list-group-item list-group-item-action facet-field"
                            v-for="(field, index) in fields(side.name)" :key="index"
                            @click="facetSearch(field.value)">
                            {{ field.label }}
                        </button>
                    </div>
                </div>
            </div>
        </div>
        <p class="facet-error" v-if="facetError">{{ facetError }}</p>
        <div class="loading position-absolute" style="left: 50%; transform: translateX(-50%)" v-if="facetLoading">
            <div class="d-flex justify-content-center position-relative">
                <div class="spinner-border" style="width: 4rem; height: 4rem; position: absolute; z-index: 50; top: 30px"
                    role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            </div>
        </div>
        <section class="card facet-results shadow-1 mt-3" v-if="facetResults"
            :class="facetResults.facet.startsWith('target_') ? 'side-target-color' : 'side-source-color'"
            aria-labelledby="facet-results-title">
            <button type="button" class="corner-btn destroy right" aria-label="Close counts"
                @click="closeFacetResults()"><span aria-hidden="true">&times;</span></button>
            <h4 id="facet-results-title" ref="facetResultsTitle" class="card-header facet-results-title" tabindex="-1">
                <template v-if="passages">Reused passages by {{ resultsField.toLowerCase() }}, {{ resultsSide.lead.toLowerCase() }}</template>
                <template v-else>
                    Frequency by
                    <span v-html="resultsSide.label"></span>&nbsp;
                    {{ resultsField }}
                </template>
            </h4>
            <div class="facet-results-body">
                <div class="facet-note">Showing the top 100 values</div>
                <div class="facet-note" v-if="passages && facetResults.counted_once === false">{{ severalValues }}</div>
                <div class="list-group list-group-flush">
                    <button type="button" class="list-group-item list-group-item-action facet-result"
                        v-for="(field, index) in facetResults.results.slice(0, 100)" :key="index"
                        @click="filteredSearch(facetResults.facet, field.field)">
                        <span class="facet-bar" aria-hidden="true"
                            :style="{ width: `calc((100% - 1rem) * ${field.count / facetMax})` }"></span>
                        <span class="facet-name">{{ field.field || "N/A" }}</span>
                        <span class="facet-count">{{ field.count.toLocaleString() }}</span>
                    </button>
                </div>
            </div>
        </section>
    </aside>
</template>

<script setup>
import axios from "axios";
import emitter from "tiny-emitter/instance";
import { computed, nextTick, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import globalConfig from "../config.js";
import { paramsToUrl } from "../utils.js";

// The counts of shared passages (pairs of texts), or of reused passages (families, as first found or found again)
const props = defineProps({ mode: { type: String, default: "pairs" } });

const route = useRoute();
const router = useRouter();
const passages = computed(() => props.mode == "passages");
const heading = computed(() => `Count ${passages.value ? "reused" : "shared"} passages by`);
const sides = computed(() => [
    { name: "source", color: "side-source-color", lead: "First found", label: globalConfig.sourceLabel },
    { name: "target", color: "side-target-color", lead: "Found again", label: globalConfig.targetLabel },
]);
// Every passage is its own value of its text, so the reused passages are not counted by it
const fields = (side) =>
    globalConfig.facetsFields[side].filter((field) => !(passages.value && field.value.endsWith("_passage")));
const facetResults = ref(null);
const facetLoading = ref(false);
const facetError = ref("");
const facetsCollapsed = ref(false);
const facetFields = ref(null);
const facetResultsTitle = ref(null);
// Largest count, for the bars under the frequencies
const facetMax = computed(() => Math.max(1, ...facetResults.value.results.slice(0, 100).map((r) => r.count)));
const resultsSide = computed(() => sides.value[facetResults.value.facet.startsWith("target_") ? 1 : 0]);
const resultsField = computed(() => {
    const { facet } = facetResults.value;
    const field = globalConfig.facetsFields[resultsSide.value.name].find((field) => field.value == facet);
    return field ? field.label : facet.split("_").slice(1).join(" ");
});

// Said where the API says a passage counts under more than one value
const severalValues = computed(() =>
    facetResults.value.facet.endsWith("passage_length")
        ? "A passage counts under every length it has."
        : "A passage counts under every value it was found again in."
);

// The counts of a broad search take seconds: only the latest request is read, and a new search drops it
let request = 0;

// A new search clears the counts
watch(
    () => route.fullPath,
    () => {
        request++;
        facetLoading.value = false;
        facetResults.value = null;
        facetError.value = "";
        facetsCollapsed.value = false;
    }
);

function facetSearch(field) {
    const queryParams = { ...route.query };
    queryParams.db_table = globalConfig.databaseName;
    queryParams.facet = field;
    facetLoading.value = true;
    facetError.value = "";
    const current = ++request;
    const endpoint = passages.value ? "passage_facets" : "facets";
    axios
        .post(`${globalConfig.apiServer}/${endpoint}/?${paramsToUrl(queryParams)}`, {
            metadata: globalConfig.metadataTypes,
        })
        .then((response) => {
            if (current != request) return;
            facetResults.value = response.data;
            toggleFacetList();
            facetLoading.value = false;
            nextTick(() => facetResultsTitle.value.focus());
        })
        .catch((err) => {
            if (current != request) return;
            facetLoading.value = false;
            // The API says why a search cannot be counted that way
            const detail = err.response && err.response.data && err.response.data.detail;
            facetError.value = typeof detail == "string" ? detail : "The counts could not be loaded";
            emitter.emit("announce", "The counts could not be loaded");
            console.log("ERROR", err);
        });
}

function toggleFacetList() {
    facetsCollapsed.value = !facetsCollapsed.value;
}

function closeFacetResults() {
    facetResults.value = null;
    toggleFacetList();
    nextTick(() => facetFields.value.querySelector("button").focus());
}

function filteredSearch(fieldName, value) {
    const queryParams = { ...route.query };
    delete queryParams.page;
    delete queryParams.id_anchor;
    queryParams.db_table = globalConfig.databaseName;
    queryParams[fieldName] = `"${value}"`;
    emitter.emit("urlUpdate", queryParams);
    facetResults.value = null;
    router.push(`${route.path}?${paramsToUrl(queryParams)}`).then(() => emitter.emit("focusMain"));
}
</script>

<style lang="scss" scoped>
@import "../assets/theme.module.scss";

#facets {
    font-family: $font-ui;
}

// A quiet card: a bar-chart glyph by the title, side marks on the two lists
.facets-card,
.facet-results {
    overflow: hidden;
    border-radius: 3px;
}

.facets-title,
.facet-results-title {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 0;
    padding: 0.75rem 1rem;
    color: $ink;
    font-family: $font-ui;
    font-size: 0.95rem;
    font-weight: 600;
    line-height: 1.25;
}

.facets-title svg {
    flex: 0 0 auto;
    width: 1rem;
    height: 1rem;
    color: $button-color;
}

.facet-results-title {
    display: block;
    padding-right: 2.75rem;
}

.facet-list {
    padding: 0.75rem 0.5rem 0.25rem;
    transition: all 0.2s ease-out, visibility 0s;
}

.facet-list+.facet-list {
    border-top: 1px solid $rule-color;
}

.facet-list.hide {
    max-height: 0px;
    opacity: 0;
    margin: 0 !important;
    padding: 0;
    border: 0;
    visibility: hidden;
    transition: all 0.2s ease-out, visibility 0s 0.2s;
}

// "in" and the label are one flex item; apart, the space after "in" would collapse
.facet-side {
    @include kicker;
    display: flex;
    align-items: center;
    margin: 0 0 0.3rem;
    padding: 0 0.5rem;
    color: inherit;
}

// Where the passage was, over the side's own name, which the form and the search's pills use
.facet-side-text {
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
}

// The mark sits by the first line
.facet-side.stacked {
    align-items: flex-start;
}

.facet-side.stacked .side-mark {
    margin-top: 0.22rem;
}

.facet-side-label {
    color: $muted;
    font-size: 0.75rem;
    font-weight: 400;
    letter-spacing: 0;
    text-transform: none;
}

// Each field in its side's ink, with a glyph for the counts it opens
.facet-field {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.45rem 0.5rem;
    border: 0;
    border-radius: 3px !important;
    background-color: transparent;
    font-size: 0.88rem;
    font-weight: 500;
    cursor: pointer;
}

.side-source-color .facet-field {
    color: $source-deep;
}

.side-target-color .facet-field {
    color: $target-deep;
}

.facet-field::after {
    content: "";
    flex: 0 0 auto;
    width: 0.875rem;
    height: 0.875rem;
    background-color: currentColor;
    opacity: 0.55;
    -webkit-mask: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><path d='M2 3h12v2H2zM2 7h8v2H2zM2 11h4v2H2z'/></svg>") center / contain no-repeat;
    mask: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><path d='M2 3h12v2H2zM2 7h8v2H2zM2 11h4v2H2z'/></svg>") center / contain no-repeat;
    transition: opacity 0.12s ease;
}

.facet-field:hover::after,
.facet-field:focus-visible::after {
    opacity: 1;
}

.side-source-color .list-group-item-action:hover,
.side-source-color .list-group-item-action:focus {
    background-color: $source-tint;
}

.side-target-color .list-group-item-action:hover,
.side-target-color .list-group-item-action:focus {
    background-color: $target-tint;
}

.side-target-color .list-group-item-action:focus-visible {
    outline-color: $target-color !important;
}

#metadata-list {
    display: none;
    width: 100%;
    padding: 0.75rem 1rem;
    border-width: 0;
    background-color: transparent;
    color: $link-color;
    font-weight: 600;
    text-align: left;
    cursor: pointer;
}

#metadata-list.show {
    display: block;
}

#metadata-list:hover {
    background-color: $source-tint;
    color: $source-deep;
}

.facet-results-body {
    padding: 0.5rem;
}

.facet-error {
    margin: 0.75rem 0 0;
    padding: 0 0.25rem;
    color: $ink-2;
    font-size: 0.85rem;
}

.facet-note {
    padding: 0.25rem 0.5rem 0.5rem;
    color: $muted;
    font-size: 0.8rem;
}

// Each frequency over a thin bar of its share of the largest, in the counted side's ink
.facet-result {
    position: relative;
    display: flex;
    gap: 0.75rem;
    align-items: baseline;
    justify-content: space-between;
    padding: 0.45rem 0.5rem 0.65rem;
    border: 0;
    border-radius: 3px !important;
    color: $ink;
    font-size: 0.85rem;
    line-height: 1.25;
    cursor: pointer;
}

.facet-bar {
    position: absolute;
    left: 0.5rem;
    bottom: 0.25rem;
    min-width: 2px;
    height: 3px;
    border-radius: 2px;
    background-color: currentColor;
    opacity: 0.75;
    transition: opacity 0.12s ease;
}

.facet-result:hover .facet-bar,
.facet-result:focus-visible .facet-bar {
    opacity: 1;
}

.side-source-color .facet-bar {
    color: $source-color;
}

.side-target-color .facet-bar {
    color: $target-color;
}

.facet-name {
    flex: 1 1 auto;
    min-width: 0;
    overflow-wrap: anywhere;
}

.facet-count {
    flex: 0 0 auto;
    color: $ink-2;
    font-variant-numeric: tabular-nums;
}

.corner-btn.destroy {
    z-index: 1;
    top: 0.45rem;
    right: 0.45rem;
    width: 2rem;
    height: 2rem;
    border: 0;
    border-radius: 3px;
    color: $muted;
    font-size: 1.2rem;
}
</style>
