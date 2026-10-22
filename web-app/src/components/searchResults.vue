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
                    <li class="card mb-3 rounded-0 shadow-1" style="position: relative;"
                        v-for="(alignment, index) in results.alignments" :key="results.start_position + index + 1"
                        v-bind:data-index="index">
                        <h3 class="visually-hidden">Passage pair {{ results.start_position + index + 1 }}</h3>
                        <div class="corner-btn left" aria-hidden="true">{{ results.start_position + index + 1 }}</div>
                        <passage-pair :alignment="alignment" :index="index" :diffed="false"></passage-pair>
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
                <div class="card shadow-1">
                    <h3 id="facets-title" class="h6 card-header text-center">Browse by Counts</h3>
                    <button type="button" id="metadata-list" class="mx-auto p-2" :class="{ show: facetsCollapsed }"
                        aria-expanded="false" aria-controls="facet-fields" @click="toggleFacetList()">Show
                        options</button>
                    <div id="facet-fields" ref="facetFields">
                        <div class="mt-2 mb-3 pr-3 pl-3 facet-list" :class="{ hide: facetsCollapsed }">
                            <h4 class="dropdown-header text-center" v-html="globalConfig.sourceLabel"></h4>
                            <div class="list-group">
                                <button type="button" class="list-group-item list-group-item-action"
                                    v-for="(field, index) in globalConfig.facetsFields.source" :key="index"
                                    @click="facetSearch(field.value)">
                                    {{ field.label }}
                                </button>
                            </div>
                        </div>
                        <div class="mb-3 pr-3 pl-3 facet-list" :class="{ hide: facetsCollapsed }">
                            <h4 class="dropdown-header text-center" v-html="globalConfig.targetLabel"></h4>
                            <div class="list-group">
                                <button type="button" class="list-group-item list-group-item-action"
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
                <section class="card rounded-0 shadow-1 mt-3" v-if="facetResults" aria-labelledby="facet-results-title">
                    <button type="button" class="corner-btn destroy right" aria-label="Close frequency results"
                        @click="closeFacetResults()">&times;</button>
                    <h4 id="facet-results-title" ref="facetResultsTitle" class="h6 card-header text-center"
                        tabindex="-1">
                        Frequency by
                        <span v-html="facetDirectionLabel"></span>&nbsp;
                        <span class="text-capitalize">{{ facetResults.facet.split("_")[1] }}</span>
                    </h4>
                    <div class="mt-1 p-2">
                        <div class="pb-2 text-center text-muted">Showing top 100 results</div>
                        <div class="list-group">
                            <button type="button" class="list-group-item list-group-item-action facet-result"
                                v-for="(field, index) in facetResults.results.slice(0, 100)" :key="index"
                                @click="filteredSearch(facetResults.facet, field.field)">
                                <span class="row">
                                    <span class="col pr-1 pl-1">{{ field.field || "N/A" }}</span>
                                    <span class="col-4 pr-1 pl-1 facet-count">{{ field.count.toLocaleString() }}</span>
                                </span>
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
import { nextTick, ref, watch } from "vue";
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
    const delay = el.dataset.index * 100;
    setTimeout(function () {
        Velocity(el, { opacity: 1, height: "100%" }, { complete });
    }, delay);
}
</script>

<style scoped>
#facets,
.corner-btn {
    font-family: "Open-Sans", sans-serif;
}

.passage-label {
    font-family: "Open-Sans", sans-serif;
}

.card-link {
    color: #007bff !important;
}

.card-link:hover,
.page-link {
    cursor: pointer;
}

.list-group-item:first-child,
.list-group-item:last-child {
    border-radius: 0 !important;
}

.facet-result {
    cursor: pointer;
}

.facet-count {
    float: right;
}

.corner-btn.destroy {
    z-index: 1;
}

.facet-list {
    transition: all 0.2s ease-out, visibility 0s;
}

.facet-list button:hover {
    cursor: pointer;
}

.facet-list.hide {
    max-height: 0px;
    opacity: 0;
    margin: 0 !important;
    visibility: hidden;
    transition: all 0.2s ease-out, visibility 0s 0.2s;
}

#metadata-list {
    display: none;
    opacity: 0;
    border-width: 0;
    background-color: transparent;
    color: inherit;
    cursor: pointer;
    transition: all 0.2s ease-out;
}

#metadata-list.show {
    display: block;
    opacity: 1;
}

#metadata-list:hover {
    color: #565656;
}



.dropdown-header {
    display: block;
    padding: 0.5rem 1rem;
    margin-bottom: 0;
    font-size: 0.875rem;
    color: #6c757d;
    white-space: nowrap;
}

.facet-list .list-group-item {
    border-left-width: 0 !important;
    border-right-width: 0 !important;
}
</style>