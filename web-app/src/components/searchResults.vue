<template>
    <div class="mt-3">
        <h2 class="visually-hidden">Passage pairs</h2>
        <div class="row" style="padding: 0 0.75rem">
            <div class="m-4" style="font-size: 120%" v-if="error">No results for your query</div>
            <search-arguments></search-arguments>
        </div>
        <report-switcher />
        <div class="row">
            <div class="col-12 col-lg-9 position-relative">
                <div class="d-flex justify-content-center position-relative" v-if="loading">
                    <div class="spinner-border"
                        style="width: 8rem; height: 8rem; position: absolute; z-index: 50; top: 30px" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                </div>
                <transition-group name="staggered-fade" tag="ol" class="list-unstyled" v-bind:css="false"
                    v-on:before-enter="beforeEnter" v-on:enter="enter">
                    <li class="card pair-card mb-3 shadow-1" v-for="(alignment, index) in results.alignments"
                        :key="results.start_position + index + 1" v-bind:data-index="index">
                        <h3 class="visually-hidden">Passage pair {{ results.start_position + index + 1 }}</h3>
                        <passage-pair :alignment="alignment" :index="index" :diffed="false"
                            :number="results.start_position + index + 1"></passage-pair>
                    </li>
                </transition-group>
                <nav aria-label="Pages of results" v-if="done">
                    <ul class="pagination justify-content-center mb-4">
                        <li class="page-item" v-if="results.page > 1">
                            <button type="button" class="page-link" @click="previousPage()" aria-label="Previous page">
                                <span aria-hidden="true">&laquo;</span>
                            </button>
                        </li>
                        <li class="page-item active">
                            <span class="page-link" aria-current="page">Page {{ results.page }}</span>
                        </li>
                        <li class="page-item" v-if="resultsLeft > 0">
                            <button type="button" class="page-link" @click="nextPage()" aria-label="Next page">
                                <span aria-hidden="true">&raquo;</span>
                            </button>
                        </li>
                    </ul>
                </nav>
            </div>
            <aside id="facets" class="col-12 col-lg-3 pl-0 position-relative" aria-labelledby="facets-title">
                <div class="card facets-card shadow-1">
                    <h3 id="facets-title" class="card-header facets-title">
                        <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
                            <path d="M2.5 3.5h11M2.5 8h7M2.5 12.5h4" fill="none" stroke="currentColor"
                                stroke-width="1.75" stroke-linecap="round" />
                        </svg>
                        Top Occurrences by
                    </h3>
                    <button type="button" id="metadata-list" :class="{ show: facetsCollapsed }" aria-expanded="false"
                        aria-controls="facet-fields" @click="toggleFacetList()">Show
                        options</button>
                    <div id="facet-fields" ref="facetFields">
                        <div class="facet-list side-source-color" :class="{ hide: facetsCollapsed }">
                            <h4 class="facet-side"><span class="side-mark source" aria-hidden="true"></span>in <span
                                    v-html="globalConfig.sourceLabel"></span></h4>
                            <div class="list-group list-group-flush">
                                <button type="button" class="list-group-item list-group-item-action facet-field"
                                    v-for="(field, index) in globalConfig.facetsFields.source" :key="index"
                                    @click="facetSearch(field.value)">
                                    {{ field.label }}
                                </button>
                            </div>
                        </div>
                        <div class="facet-list side-target-color" :class="{ hide: facetsCollapsed }">
                            <h4 class="facet-side"><span class="side-mark target" aria-hidden="true"></span>in <span
                                    v-html="globalConfig.targetLabel"></span></h4>
                            <div class="list-group list-group-flush">
                                <button type="button" class="list-group-item list-group-item-action facet-field"
                                    v-for="(field, index) in globalConfig.facetsFields.target" :key="index"
                                    @click="facetSearch(field.value)">
                                    {{ field.label }}
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="loading position-absolute" style="left: 50%; transform: translateX(-50%)"
                    v-if="facetLoading">
                    <div class="d-flex justify-content-center position-relative">
                        <div class="spinner-border"
                            style="width: 4rem; height: 4rem; position: absolute; z-index: 50; top: 30px" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                    </div>
                </div>
                <section class="card facet-results shadow-1 mt-3" v-if="facetResults"
                    :class="facetResults.facet.startsWith('target_') ? 'side-target-color' : 'side-source-color'"
                    aria-labelledby="facet-results-title">
                    <button type="button" class="corner-btn destroy right" aria-label="Close frequency results"
                        @click="closeFacetResults()"><span aria-hidden="true">&times;</span></button>
                    <h4 id="facet-results-title" ref="facetResultsTitle" class="card-header facet-results-title"
                        tabindex="-1">
                        Frequency by
                        <span v-html="facetDirectionLabel"></span>&nbsp;
                        <span class="text-capitalize">{{ facetResults.facet.split("_")[1] }}</span>
                    </h4>
                    <div class="facet-results-body">
                        <div class="facet-note">Showing top 100 results</div>
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
        </div>
    </div>
</template>

<script setup>
import axios from "axios";
import emitter from "tiny-emitter/instance";
import Velocity from "velocity-animate";
import { computed, nextTick, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import globalConfig from "../config.js";
import { paramsToUrl, prefersReducedMotion } from "../utils.js";
import passagePair from "./passagePair";
import reportSwitcher from "./reportSwitcher";
import searchArguments from "./searchArguments";

const route = useRoute();
const router = useRouter();
const loading = ref(false);
const done = ref(false);
const results = ref({ alignments: [] });
const resultsLeft = ref(0);
const error = ref(null);
const facetResults = ref(null);
const facetLoading = ref(null);
const facetDirectionLabel = ref("");
const facetsCollapsed = ref(false);
const facetFields = ref(null);
const facetResultsTitle = ref(null);
// Largest count, for the bars under the frequencies
const facetMax = computed(() => Math.max(1, ...facetResults.value.results.slice(0, 100).map((r) => r.count)));

function fetchData() {
    results.value = { alignments: [] }; // clear alignments with new search
    facetResults.value = null; // clear facet results with new search
    error.value = null;
    loading.value = true;
    const params = { ...route.query };
    params.db_table = globalConfig.databaseName;
    emitter.emit("searchArgsUpdate", {
        counts: "",
        searchParams: params,
    });
    axios
        .get(`${globalConfig.apiServer}/search_alignments/?${paramsToUrl(params)}`)
        .then((response) => {
            loading.value = false;
            if (!response.data.alignments || !response.data.alignments.length) {
                error.value = "No results";
                emitter.emit("announce", "No results for your query");
                return;
            }
            results.value = response.data;
            done.value = true;
            axios
                .get(`${globalConfig.apiServer}/count_results/?${paramsToUrl(params)}`)
                .then((response) => {
                    const counts = response.data.counts;
                    emitter.emit("searchArgsUpdate", {
                        counts: counts,
                        searchParams: params,
                    });
                    resultsLeft.value = counts - (results.value.start_position + results.value.alignments.length);
                })
                .catch((error) => {
                    console.log(error);
                });
            facetsCollapsed.value = false;
        })
        .catch((err) => {
            loading.value = false;
            error.value = err.toString();
            emitter.emit("announce", "No results for your query");
            console.log(err);
        });
}
fetchData();
watch(() => route.fullPath, fetchData);

function goToPage(page, direction, anchor) {
    const queryParams = { ...route.query };
    queryParams.page = page;
    queryParams.direction = direction;
    queryParams.id_anchor = anchor;
    router.push(`/search?${paramsToUrl(queryParams)}`).then(() => emitter.emit("focusMain"));
}

function previousPage() {
    goToPage(parseInt(results.value.page) - 1, "previous", results.value.alignments[0].page_cursor);
}

function nextPage() {
    const alignments = results.value.alignments;
    goToPage(parseInt(results.value.page) + 1, "next", alignments[alignments.length - 1].page_cursor);
}

function facetSearch(field) {
    const queryParams = { ...route.query };
    queryParams.db_table = globalConfig.databaseName;
    queryParams.facet = field;
    facetLoading.value = true;
    axios
        .post(`${globalConfig.apiServer}/facets/?${paramsToUrl(queryParams)}`, {
            metadata: globalConfig.metadataTypes,
        })
        .then((response) => {
            facetDirectionLabel.value = globalConfig[`${response.data.facet.split("_")[0]}Label`];
            facetResults.value = response.data;
            toggleFacetList();
            facetLoading.value = false;
            nextTick(() => facetResultsTitle.value.focus());
        })
        .catch((err) => {
            facetLoading.value = false;
            error.value = err.toString();
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
    results.value = { alignments: [] };
    router.push(`/search?${paramsToUrl(queryParams)}`).then(() => emitter.emit("focusMain"));
}

function beforeEnter(el) {
    if (prefersReducedMotion()) return;
    el.style.opacity = 0;
    el.style.height = 0;
}

function enter(el, complete) {
    if (prefersReducedMotion()) {
        complete();
        return;
    }
    const delay = Math.min(el.dataset.index, 10) * 100;
    setTimeout(function () {
        Velocity(el, { opacity: 1, height: "100%" }, { complete });
    }, delay);
}
</script>

<style lang="scss" scoped>
@import "../assets/theme.module.scss";

// The pair's footer reaches the card's edges through these
.pair-card {
    --pair-pad-x: 1.5rem;
    --pair-pad-bottom: 0px;
    position: relative;
    padding: 1.35rem var(--pair-pad-x) 0;
    border-radius: 3px;
}

.page-link {
    cursor: pointer;
}

.pagination .page-item.active .page-link {
    @include display-type;
    font-weight: 600;
}

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

.facet-side {
    @include kicker;
    display: flex;
    align-items: center;
    margin: 0 0 0.3rem;
    padding: 0 0.5rem;
    color: inherit;
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

@media (max-width: 767.98px) {
    .pair-card {
        --pair-pad-x: 1rem;
        padding-top: 1rem;
    }
}
</style>
