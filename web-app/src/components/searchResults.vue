<template>
    <div class="mt-3">
        <div class="row" style="padding: 0 0.75rem">
            <div class="m-4" style="font-size: 120%" v-if="error">No results for your query</div>
            <search-arguments></search-arguments>
        </div>
        <report-switcher />
        <div class="row">
            <div class="col-9 position-relative">
                <div class="d-flex justify-content-center position-relative" v-if="loading">
                    <div class="spinner-border"
                        style="width: 8rem; height: 8rem; position: absolute; z-index: 50; top: 30px" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                </div>
                <transition-group name="staggered-fade" tag="div" v-bind:css="false" v-on:before-enter="beforeEnter"
                    v-on:enter="enter">
                    <div class="card mb-3 rounded-0 shadow-1" style="position: relative;"
                        v-for="(alignment, index) in results.alignments" :key="results.start_position + index + 1"
                        v-bind:data-index="index">
                        <div class="corner-btn left">{{ results.start_position + index + 1 }}</div>
                        <passage-pair :alignment="alignment" :index="index" :diffed="false"></passage-pair>
                    </div>
                </transition-group>
                <nav aria-label="Page navigation" v-if="done">
                    <ul class="pagination justify-content-center mb-4">
                        <li class="page-item" v-if="results.page > 1">
                            <a class="page-link" v-on:click="previousPage()" aria-label="Previous">
                                <span aria-hidden="true">&laquo;</span>
                            </a>
                        </li>
                        <li class="page-item">
                            <a class="page-link">Page {{ results.page }}</a>
                        </li>
                        <li class="page-item" v-if="this.resultsLeft > 0">
                            <a class="page-link" v-on:click="nextPage()" aria-label="Next">
                                <span aria-hidden="true">&raquo;</span>
                            </a>
                        </li>
                    </ul>
                </nav>
            </div>
            <div id="facets" class="col-3 pl-0 position-relative">
                <div class="card shadow-1">
                    <h6 class="card-header text-center">Browse by Counts</h6>
                    <div id="metadata-list" class="mx-auto p-2" @click="toggleFacetList()">Show options</div>
                    <div class="mt-2 mb-3 pr-3 pl-3 facet-list">
                        <span class="dropdown-header text-center" v-html="globalConfig.sourceLabel"></span>
                        <div class="list-group">
                            <button type="button" class="list-group-item list-group-item-action"
                                v-for="(field, index) in globalConfig.facetsFields.source" :key="index"
                                v-on:click="facetSearch(field.value)">
                                {{ field.label }}
                            </button>
                        </div>
                    </div>
                    <div class="mb-3 pr-3 pl-3 facet-list">
                        <h6 class="dropdown-header text-center" v-html="globalConfig.targetLabel"></h6>
                        <div class="list-group">
                            <button type="button" class="list-group-item list-group-item-action"
                                v-for="(field, index) in globalConfig.facetsFields.target" :key="index"
                                v-on:click="facetSearch(field.value)">
                                {{ field.label }}
                            </button>
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
                <div class="card rounded-0 shadow-1 mt-3" v-if="facetResults">
                    <div class="corner-btn destroy right" @click="closeFacetResults()">X</div>
                    <h6 class="card-header text-center">
                        Frequency by
                        <span v-html="facetDirectionLabel"></span>&nbsp;
                        <span class="text-capitalize">{{ facetResults.facet.split("_")[1] }}</span>
                    </h6>
                    <div class="mt-1 p-2">
                        <div class="pb-2 text-center" style="opacity: 0.5">Showing top 100 results</div>
                        <div class="list-group">
                            <div class="list-group-item list-group-item-action facet-result"
                                v-for="(field, index) in facetResults.results.slice(0, 100)" :key="index"
                                v-on:click="filteredSearch(facetResults.facet, field.field)">
                                <div class="row">
                                    <div class="col pr-1 pl-1">{{ field.field || "N/A" }}</div>
                                    <div class="col-4 pr-1 pl-1 facet-count">{{ field.count.toLocaleString() }}</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
import Velocity from "velocity-animate";
import passagePair from "./passagePair";
import reportSwitcher from "./reportSwitcher";
import searchArguments from "./searchArguments";

export default {
    name: "searchResults",
    components: {
        searchArguments, passagePair, reportSwitcher
    },
    inject: ["$http"],
    data() {
        return {
            loading: false,
            done: false,
            results: { alignments: [] },
            counts: null,
            resultsLeft: 0,
            lastRowID: null,
            page: 0,
            error: null,
            globalConfig: this.$globalConfig,
            facetResults: null,
            facetLoading: null,
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
            this.results = { alignments: [] }; // clear alignments with new search
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
                .get(`${this.$globalConfig.apiServer}/search_alignments/?${this.paramsToUrl(params)}`)
                .then((response) => {
                    this.results = response.data;
                    this.lastRowID = this.results.alignments[this.results.alignments.length - 1].rowid_ordered;
                    this.page++;
                    this.loading = false;
                    this.done = true;
                    this.$http
                        .get(`${this.$globalConfig.apiServer}/count_results/?${this.paramsToUrl(params)}`)
                        .then((response) => {
                            let counts = response.data.counts;
                            this.emitter.emit("searchArgsUpdate", {
                                counts: counts,
                                searchParams: params,
                            });
                            this.resultsLeft = counts - (this.results.start_position + this.results.alignments.length);
                        })
                        .catch((error) => {
                            console.log(error);
                        });
                    Array.from(document.getElementsByClassName("facet-list")).forEach(function (element) {
                        element.classList.remove("hide");
                    });
                    document.querySelector("#metadata-list").classList.remove("show");
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

.list-group-item:focus,
.list-group-item:active {
    outline: none !important;
}

.facet-list {
    transition: all 0.2s ease-out;
}

.facet-list button:hover {
    cursor: pointer;
}

.facet-list.hide {
    max-height: 0px;
    opacity: 0;
    margin: 0 !important;
}

#metadata-list {
    display: none;
    opacity: 0;
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