<template>
    <div class="mt-3">
        <h2 class="visually-hidden">Shared passages</h2>
        <div class="row" style="padding: 0 0.75rem">
            <div class="m-4" style="font-size: 120%" v-if="error">No shared passages for your query</div>
            <search-arguments></search-arguments>
        </div>
        <div class="row">
            <div class="col-12 col-lg-9">
                <report-switcher />
            </div>
        </div>
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
                        <h3 class="visually-hidden">Shared passage {{ results.start_position + index + 1 }}</h3>
                        <passage-pair :alignment="alignment" :index="index" :diffed="false"
                            :number="results.start_position + index + 1"></passage-pair>
                    </li>
                </transition-group>
                <nav aria-label="Pages of shared passages" v-if="done">
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
            <facet-panel mode="pairs"></facet-panel>
        </div>
    </div>
</template>

<script setup>
import axios from "axios";
import emitter from "tiny-emitter/instance";
import Velocity from "velocity-animate";
import { ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import globalConfig from "../config.js";
import { paramsToUrl, prefersReducedMotion } from "../utils.js";
import facetPanel from "./facetPanel";
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

function fetchData() {
    results.value = { alignments: [] }; // clear alignments with new search
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
                error.value = "No shared passages";
                emitter.emit("announce", "No shared passages for your query");
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
        })
        .catch((err) => {
            loading.value = false;
            error.value = err.toString();
            emitter.emit("announce", "No shared passages for your query");
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

@media (max-width: 767.98px) {
    .pair-card {
        --pair-pad-x: 1rem;
        padding-top: 1rem;
    }
}
</style>
