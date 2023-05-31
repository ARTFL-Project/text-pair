<template>
    <div class="mt-3">
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
                <transition-group name="staggered-fade" tag="div" v-bind:css="false" v-on:before-enter="beforeEnter"
                    v-on:enter="enter">
                    <div class="card mb-3 rounded-0 shadow-1" style="position: relative"
                        v-for="(sourcePassage, index) in results.groups" :key="index + 1" v-bind:data-index="index">
                        <div class="corner-btn left">{{ index + 1 }}</div>
                        <h6 class="mt-2 pt-3" style="text-align: center;">
                            <citations :citation="globalConfig.sourceCitation" :alignment="sourcePassage"
                                :link-to-doc="globalConfig.sourceLinkToDocMetadata"></citations>
                        </h6>
                        <span class="px-3 pb-2" v-if="sourcePassage.count">The following passage is reused (in whole or in
                            part) {{
                                sourcePassage.count.toLocaleString() }}
                            times:</span>
                        <i class="passages px-3 pb-3">"{{ sourcePassage.source_passage }}"</i>
                        <div class="ps-3">
                            <router-link class="" :to="`/group/${sourcePassage.group_id}`">
                                <button class="btn btn-secondary btn-sm mb-3">
                                    View all reuses of this passage
                                </button>
                            </router-link>
                        </div>
                    </div>
                </transition-group>
            </div>
        </div>
        <h5 class="mt-2 mb-4" style="text-align: center">For performance reasons, only the top 100 are displayed.</h5>
    </div>
</template>

<script>
import searchArguments from "./searchArguments";
import passagePair from "./passagePair";
import reportSwitcher from "./reportSwitcher";
import citations from "./citations";
import Velocity from "velocity-animate";

export default {
    name: "sortedResults",
    components: {
        searchArguments, passagePair, reportSwitcher, citations
    },
    inject: ["$http"],
    data() {
        return {
            loading: false,
            done: false,
            results: { groups: [] },
            counts: null,
            error: null,
            globalConfig: this.$globalConfig,
        };
    },
    created() {
        // fetch the data when the view is created and the data is
        // already being observed
        this.fetchData();
    },
    watch: {
        // call again the method if the route changes
        $route: "fetchData",
    },
    methods: {
        fetchData() {
            this.results = { groups: [] }; // clear alignments with new search
            this.facetResults = null; // clear facet results with new search
            this.error = null;
            this.loading = true;
            let params = { ...this.$route.query };
            params.db_table = this.$globalConfig.databaseName;
            this.emitter.emit("searchArgsUpdate", {
                counts: "",
                searchParams: params,
            });
            this.$http
                .get(`${this.$globalConfig.apiServer}/sorted_results/?${this.paramsToUrl(params)}`)
                .then((response) => {
                    this.results.groups = response.data.groups;
                    this.loading = false;
                    this.done = true;
                    this.emitter.emit("searchArgsUpdate", {
                        counts: response.data.total_count,
                        searchParams: params,
                    });
                })
                .catch((error) => {
                    this.loading = false;
                    this.error = error.toString();
                    console.log(error);
                });
        },
        previousPage() {
            let queryParams = { ...this.$route.query };
            queryParams.page = parseInt(this.results.page) - 1;
            queryParams.direction = "previous";
            queryParams.id_anchor = this.results.alignments[0].rowid_ordered;
            this.$router.push(`/search?${this.paramsToUrl(queryParams)}`);
        },
        nextPage() {
            let queryParams = { ...this.$route.query };
            queryParams.page = parseInt(this.results.page) + 1;
            queryParams.direction = "next";
            queryParams.id_anchor = this.results.alignments[this.results.alignments.length - 1].rowid_ordered;
            this.$router.push(`/search?${this.paramsToUrl(queryParams)}`);
        },
        facetSearch(field) {
            let queryParams = { ...this.$route.query };
            queryParams.db_table = this.$globalConfig.databaseName;
            queryParams.facet = field;
            this.facetLoading = true;
            this.$http
                .post(`${this.$globalConfig.apiServer}/facets/?${this.paramsToUrl(queryParams)}`, {
                    metadata: this.$globalConfig.metadataTypes,
                })
                .then((response) => {
                    this.facetDirectionLabel = this.$globalConfig[`${response.data.facet.split("_")[0]}Label`];
                    this.facetResults = response.data;
                    this.toggleFacetList();
                    this.facetLoading = false;
                })
                .catch((error) => {
                    this.facetLoading = false;
                    this.error = error.toString();
                    console.log("ERROR", error);
                });
        },
        toggleFacetList() {
            Array.from(document.getElementsByClassName("facet-list")).forEach(function (element) {
                element.classList.toggle("hide");
            });
            document.querySelector("#metadata-list").classList.toggle("show");
        },
        closeFacetResults() {
            this.facetResults = null;
            this.toggleFacetList();
        },
        filteredSearch(fieldName, value) {
            let queryParams = { ...this.$route.query };
            delete queryParams.page;
            delete queryParams.id_anchor;
            queryParams.db_table = this.$globalConfig.databaseName;
            queryParams[fieldName] = `"${value}"`;
            this.emitter.emit("urlUpdate", queryParams);
            this.facetResults = null;
            this.results = { alignments: [] };
            this.$router.push(`/search?${this.paramsToUrl(queryParams)}`);
        },
        beforeEnter: function (el) {
            el.style.opacity = 0;
            el.style.height = 0;
        },
        enter: function (el, done) {
            var delay = el.dataset.index * 100;
            setTimeout(function () {
                Velocity(el, { opacity: 1, height: "100%" }, { complete: done });
            }, delay);
        },
        toggleSearchForm() {
            this.emitter.emit("toggleSearchForm");
        },
    },
};
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