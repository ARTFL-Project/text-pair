<template>
    <div class="mt-3">
        <h2 class="visually-hidden">Reused passages</h2>
        <div class="row" style="padding: 0 0.75rem">
            <div class="m-4" style="font-size: 120%" v-if="error">No reused passages for your query</div>
            <search-arguments unit="reused"></search-arguments>
        </div>
        <report-switcher></report-switcher>
        <div class="row">
            <div class="col position-relative">
                <div class="d-flex justify-content-center position-relative" v-if="loading">
                    <div class="spinner-border"
                        style="width: 8rem; height: 8rem; position: absolute; z-index: 50; top: 30px" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                </div>
                <transition-group name="staggered-fade" tag="ol" class="list-unstyled" v-bind:css="false"
                    v-on:before-enter="beforeEnter" v-on:enter="enter">
                    <li class="card family-card mb-3 shadow-1" v-for="(sourcePassage, index) in results.groups"
                        :key="sourcePassage.group_id" v-bind:data-index="index">
                        <div class="family-rank" aria-hidden="true">{{ results.start_position + index + 1 }}</div>
                        <div class="family-body side-source-color">
                            <div class="family-figure" aria-hidden="true" v-if="sourcePassage.count">
                                <span class="family-figure-number">{{ sourcePassage.count.toLocaleString() }}</span>
                                <span class="family-figure-label">{{ sourcePassage.count == 1 ? "document" : "documents"
                                }}</span>
                                <span class="family-figure-track"><span class="family-figure-bar"
                                        :style="{ width: `${(100 * sourcePassage.count) / maxCount}%` }"></span></span>
                            </div>
                            <h3 class="family-citation">
                                <span class="visually-hidden">{{ results.start_position + index + 1 }}. </span>
                                <span class="family-origin">First found in:</span>
                                <citations :citation="globalConfig.sourceCitation" :alignment="sourcePassage"
                                    :link-to-doc="globalConfig.sourceLinkToDocMetadata" tag="span"></citations>
                            </h3>
                            <span class="family-count" v-if="sourcePassage.count">The following passage appears (in
                                whole or in part) in <strong>{{ sourcePassage.count.toLocaleString() }}</strong>
                                {{ sourcePassage.count == 1 ? "document" : "documents" }}:</span>
                            <q class="family-quote passages" :lang="globalConfig.sourceLanguage || undefined">{{
                                sourcePassage.source_passage }}</q>
                            <p class="family-found" v-if="sourcePassage.found_again">
                                <span class="family-found-lead">{{ foundLead }}:</span>{{ " " + foundDocuments(sourcePassage) }}<span
                                    class="family-found-count">{{ " \u2014 " + foundCount(sourcePassage) }}</span>
                            </p>
                            <div>
                                <router-link class="btn btn-outline-secondary btn-sm family-link"
                                    :to="`/group/${sourcePassage.group_id}`">
                                    View all reuses of this passage
                                    <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
                                        <path d="M3 8h9.5M9 4.5L12.5 8 9 11.5" fill="none" stroke="currentColor"
                                            stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" />
                                    </svg>
                                </router-link>
                            </div>
                        </div>
                    </li>
                </transition-group>
                <nav aria-label="Pages of reused passages" v-if="pages > 1">
                    <ul class="pagination justify-content-center mb-4">
                        <li class="page-item" v-if="results.page > 1">
                            <button type="button" class="page-link" @click="goToPage(results.page - 1)"
                                aria-label="Previous page">
                                <span aria-hidden="true">&laquo;</span>
                            </button>
                        </li>
                        <li class="page-item active">
                            <span class="page-link" aria-current="page">Page {{ results.page.toLocaleString() }} of {{
                                pages.toLocaleString() }}</span>
                        </li>
                        <li class="page-item" v-if="results.page < pages">
                            <button type="button" class="page-link" @click="goToPage(results.page + 1)"
                                aria-label="Next page">
                                <span aria-hidden="true">&raquo;</span>
                            </button>
                        </li>
                    </ul>
                </nav>
            </div>
        </div>
    </div>
</template>

<script setup>
import axios from "axios";
import emitter from "tiny-emitter/instance";
import Velocity from "velocity-animate";
import { computed, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import globalConfig from "../config.js";
import { fieldPhrase, paramsToUrl, prefersReducedMotion } from "../utils.js";
import citations from "./citations";
import reportSwitcher from "./reportSwitcher";
import searchArguments from "./searchArguments";

const route = useRoute();
const router = useRouter();
const loading = ref(false);
const results = ref({ groups: [], start_position: 0 });
const error = ref(null);
// Most reused passage of the whole result, for the bars of the others
const maxCount = computed(() => results.value.max_count || 1);
const pages = computed(() => Math.ceil((results.value.total_count || 0) / (results.value.page_size || 1)));
let request = 0;

// What the cards that show where a passage was found again say it was found in, from the later side's fields
const foundLead = computed(() => {
    const phrases = globalConfig.metadataFields.target
        .filter((field) => route.query[field.value] && !field.value.endsWith("passage_length"))
        .map((field) => fieldPhrase(field.label, route.query[field.value]));
    return `Found again in ${phrases.join(" and ")}`;
});

// A document of the later side as its citation fields give it: author, title (year)
function citationText(occurrence) {
    const parts = (globalConfig.targetCitation || [])
        .filter((cite) => occurrence[cite.field])
        .map((cite) => (/year$/.test(cite.field) ? `(${occurrence[cite.field]})` : occurrence[cite.field]));
    return parts.join(", ").replace(", (", " (");
}

// The documents the later side's fields found a passage again in, as many as the card has room for
function foundDocuments(group) {
    return group.found_again.documents.map(citationText).join("; ");
}

function foundCount(group) {
    return `${group.found_again.matching.toLocaleString()} of ${group.count.toLocaleString()} documents`;
}

function fetchData() {
    results.value = { groups: [], start_position: 0 }; // clear alignments with new search
    error.value = null;
    loading.value = true;
    const current = ++request;
    const params = { ...route.query };
    params.db_table = globalConfig.databaseName;
    emitter.emit("searchArgsUpdate", {
        counts: "",
        searchParams: params,
        scope: "origin",
    });
    axios
        .get(`${globalConfig.apiServer}/sorted_results/?${paramsToUrl(params)}`)
        .then((response) => {
            if (current != request) return;
            results.value = response.data;
            loading.value = false;
            emitter.emit("searchArgsUpdate", {
                counts: response.data.total_count,
                searchParams: params,
                scope: response.data.scope || "any",
            });
        })
        .catch((err) => {
            if (current != request) return;
            loading.value = false;
            error.value = err.toString();
            emitter.emit("announce", "No reused passages for your query");
            console.log(err);
        });
}
fetchData();
watch(() => route.fullPath, fetchData);

function goToPage(page) {
    const queryParams = { ...route.query };
    queryParams.page = page;
    router.push(`${route.path}?${paramsToUrl(queryParams)}`).then(() => emitter.emit("focusMain"));
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

// Rank in the left margin, the count large at the top right
.family-card {
    display: grid;
    grid-template-columns: minmax(3rem, auto) minmax(0, 1fr);
    border-radius: 3px;
}

.family-rank {
    padding: 1.2rem 0.4rem 0;
    color: $header-color;
    @include display-type;
    font-size: 1.05rem;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    line-height: 1.2;
    text-align: center;
    white-space: nowrap;
}

.family-body {
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 0.5rem;
    padding: 1.1rem calc(9rem + 1.5rem) 1.1rem 0;
}

.family-figure {
    position: absolute;
    top: 1.1rem;
    right: 1.5rem;
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    justify-content: flex-end;
    gap: 0 0.35rem;
    width: 9rem;
}

.family-figure-number {
    color: $ink;
    @include display-type;
    font-size: 2rem;
    font-weight: 600;
    line-height: 1;
}

.family-figure-label {
    color: $muted;
    font-family: $font-ui;
    font-size: 0.8rem;
}

// The count against the top family's, carried from the earlier text to the later ones
.family-figure-track {
    display: flex;
    justify-content: flex-end;
    flex: 0 0 100%;
    height: 3px;
    margin-top: 0.5rem;
    border-radius: 2px;
    background-color: $well;
}

.family-figure-bar {
    min-width: 3px;
    border-radius: inherit;
    background: linear-gradient(90deg, $source-color, $target-color);
}

.family-citation {
    margin: 0;
    font-size: 1rem;
    font-weight: 400;
    line-height: 1.4;
}

.family-citation :deep(span.d-block) {
    margin: 0 !important;
    padding: 0 !important;
}

// Says what the citation is: the passage's earliest text, which need not be the one a search matched
.family-origin {
    display: block;
    margin-bottom: 0.1rem;
    color: $muted;
    font-family: $font-ui;
    font-size: 0.75rem;
    font-weight: 600;
    line-height: 1.4;
}

// Where the later side's fields found the passage again, in the later side's ink
.family-found {
    max-width: 90ch;
    margin: 0 0 0.25rem;
    padding-left: 0.65rem;
    border-left: 3px solid rgba($target-color, 0.5);
    color: $ink;
    font-family: $font-ui;
    font-size: 0.82rem;
    line-height: 1.5;
}

.family-found-lead {
    color: $target-deep;
    font-weight: 600;
}

.family-found-count {
    color: $muted;
    white-space: nowrap;
}

.family-count {
    color: $muted;
    font-family: $font-ui;
    font-size: 0.82rem;
}

.family-count strong {
    color: $ink;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
}

.family-quote {
    display: block;
    max-width: 90ch;
    margin-bottom: 0.25rem;
    color: $source-deep;
    font-size: 1.05rem;
    font-style: italic;
    line-height: 1.6;
}

// Quiet, like the pair footer's toggle
.family-link {
    --bs-btn-border-color: #{$rule-strong};
    --bs-btn-hover-color: #{$source-deep};
    --bs-btn-hover-bg: #{$source-tint};
    --bs-btn-hover-border-color: #{rgba($button-color, 0.55)};
    display: inline-flex;
    align-items: center;
    gap: 0.375rem;
    border-radius: 3px;
    font-size: 0.8rem;
    font-weight: 600;
}

// Over the theme's white text for hovered outline links
a.family-link:hover,
a.family-link:focus {
    color: $source-deep !important;
}

.family-link svg {
    width: 0.875rem;
    height: 0.875rem;
}

.page-link {
    cursor: pointer;
}

.pagination .page-item.active .page-link {
    @include display-type;
    font-weight: 600;
}

@media (max-width: 767.98px) {
    .family-body {
        padding-right: 1.5rem;
    }

    .family-figure {
        display: none;
    }
}

@media (max-width: 575.98px) {
    .family-card {
        grid-template-columns: minmax(2.25rem, auto) minmax(0, 1fr);
    }

    .family-rank {
        padding-inline: 0.15rem;
    }

    .family-body {
        padding-right: 1rem;
    }
}
</style>
