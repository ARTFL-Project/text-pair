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
                    <li class="card mb-3 rounded-0 shadow-1" style="position: relative"
                        v-for="(sourcePassage, index) in results.groups" :key="index + 1" v-bind:data-index="index">
                        <div class="corner-btn left" aria-hidden="true">{{ index + 1 }}</div>
                        <h3 class="h6 mt-2 pt-3" style="text-align: center;">
                            <span class="visually-hidden">{{ index + 1 }}. </span>
                            <citations :citation="globalConfig.sourceCitation" :alignment="sourcePassage"
                                :link-to-doc="globalConfig.sourceLinkToDocMetadata" tag="span"></citations>
                        </h3>
                        <span class="px-3 pb-2" v-if="sourcePassage.count">The following passage is reused (in whole or in
                            part) {{
                                sourcePassage.count.toLocaleString() }}
                            times:</span>
                        <i class="passages px-3 pb-3" :lang="globalConfig.sourceLanguage || undefined">"{{
                            sourcePassage.source_passage }}"</i>
                        <div class="ps-3">
                            <router-link class="btn btn-secondary btn-sm mb-3" :to="`/group/${sourcePassage.group_id}`">
                                View all reuses of this passage
                            </router-link>
                        </div>
                    </li>
                </transition-group>
            </div>
        </div>
        <p class="h5 mt-2 mb-4" style="text-align: center">For performance reasons, only the top 100 are displayed.</p>
    </div>
</template>

<script setup>
import axios from "axios";
import emitter from "tiny-emitter/instance";
import Velocity from "velocity-animate";
import { ref, watch } from "vue";
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

<style scoped>
.corner-btn {
    font-family: "Open-Sans", sans-serif;
}

#metadata-list {
    display: none;
    opacity: 0;
    cursor: pointer;
    transition: all 0.2s ease-out;
}

#metadata-list:hover {
    color: #565656;
}
</style>