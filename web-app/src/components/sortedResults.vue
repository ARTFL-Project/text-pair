<template>
    <div class="mt-3">
        <h2 class="visually-hidden">Passages by frequency of reuse</h2>
        <div class="row" style="padding: 0 0.75rem">
            <div class="m-4" style="font-size: 120%" v-if="error">No results for your query</div>
            <search-arguments></search-arguments>
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
                        :key="index + 1" v-bind:data-index="index">
                        <div class="family-rank" aria-hidden="true">{{ index + 1 }}</div>
                        <div class="family-body side-source-color">
                            <div class="family-figure" aria-hidden="true" v-if="sourcePassage.count">
                                <span class="family-figure-number">{{ sourcePassage.count.toLocaleString() }}</span>
                                <span class="family-figure-label">{{ sourcePassage.count == 1 ? "reuse" : "reuses"
                                }}</span>
                                <span class="family-figure-track"><span class="family-figure-bar"
                                        :style="{ width: `${(100 * sourcePassage.count) / maxCount}%` }"></span></span>
                            </div>
                            <h3 class="family-citation">
                                <span class="visually-hidden">{{ index + 1 }}. </span>
                                <citations :citation="globalConfig.sourceCitation" :alignment="sourcePassage"
                                    :link-to-doc="globalConfig.sourceLinkToDocMetadata" tag="span"></citations>
                            </h3>
                            <span class="family-count" v-if="sourcePassage.count">The following passage is reused (in
                                whole or in part) <strong>{{ sourcePassage.count.toLocaleString() }}</strong>
                                times:</span>
                            <q class="family-quote passages" :lang="globalConfig.sourceLanguage || undefined">{{
                                sourcePassage.source_passage }}</q>
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
            </div>
        </div>
        <p class="top-note">For performance reasons, only the top 100 are displayed.</p>
    </div>
</template>

<script setup>
import axios from "axios";
import emitter from "tiny-emitter/instance";
import Velocity from "velocity-animate";
import { computed, ref, watch } from "vue";
import { useRoute } from "vue-router";
import globalConfig from "../config.js";
import { paramsToUrl, prefersReducedMotion } from "../utils.js";
import citations from "./citations";
import reportSwitcher from "./reportSwitcher";
import searchArguments from "./searchArguments";

const route = useRoute();
const loading = ref(false);
const results = ref({ groups: [] });
const error = ref(null);
// Most reused passage, for the bars of the others
const maxCount = computed(() => Math.max(1, ...results.value.groups.map((group) => group.count || 0)));

function fetchData() {
    results.value = { groups: [] }; // clear alignments with new search
    error.value = null;
    loading.value = true;
    const params = { ...route.query };
    params.db_table = globalConfig.databaseName;
    emitter.emit("searchArgsUpdate", {
        counts: "",
        searchParams: params,
    });
    axios
        .get(`${globalConfig.apiServer}/sorted_results/?${paramsToUrl(params)}`)
        .then((response) => {
            results.value.groups = response.data.groups;
            loading.value = false;
            emitter.emit("searchArgsUpdate", {
                counts: response.data.total_count,
                searchParams: params,
            });
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

<style lang="scss" scoped>
@import "../assets/theme.module.scss";

// Rank in the left margin, the count large at the top right
.family-card {
    display: grid;
    grid-template-columns: 3rem minmax(0, 1fr);
    border-radius: 3px;
}

.family-rank {
    padding-top: 1.2rem;
    color: $header-color;
    @include display-type;
    font-size: 1.05rem;
    font-weight: 600;
    line-height: 1.2;
    text-align: center;
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

// Reuses against the most reused passage, carried from the earlier text to the later ones
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
.family-link:hover,
.family-link:focus {
    color: $source-deep !important;
}

.family-link svg {
    width: 0.875rem;
    height: 0.875rem;
}

.top-note {
    margin: 1rem 0 2.5rem;
    color: $muted;
    font-family: $font-ui;
    font-size: 0.85rem;
    text-align: center;
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
        grid-template-columns: 2.25rem minmax(0, 1fr);
    }

    .family-body {
        padding-right: 1rem;
    }
}
</style>
