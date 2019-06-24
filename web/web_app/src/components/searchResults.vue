<template>
    <div class="mt-3">
        <div class="row">
            <div class="m-4" style="font-size: 120%" v-if="error">No results for your query</div>
            <search-arguments></search-arguments>
        </div>
        <div class="row">
            <div class="col position-relative">
                <div
                    class="loading position-absolute"
                    style="left: 50%; transform: translateX(-50%);"
                    v-if="loading"
                >
                    <atom-spinner :animation-duration="800" :size="65" color="#000"/>
                </div>
                <transition-group
                    name="staggered-fade"
                    tag="div"
                    v-bind:css="false"
                    v-on:before-enter="beforeEnter"
                    v-on:enter="enter"
                >
                    <div
                        class="card mb-3 rounded-0 shadow-1"
                        style="position: relative"
                        v-for="(alignment, index) in results.alignments"
                        :key="results.start_position + index + 1"
                        v-bind:data-index="index"
                    >
                        <div class="corner-btn left">{{ results.start_position + index + 1 }}</div>
                        <div class="row">
                            <div class="col mt-4">
                                <h6 class="text-center pb-2" v-html="globalConfig.sourceLabel"></h6>
                                <p class="pt-3 px-3">
                                    <span
                                        v-for="(citation, citationIndex) in globalConfig.sourceCitation"
                                        :key="citation.field"
                                        v-if="alignment[citation.field]"
                                    >
                                        <span
                                            :style="citation.style"
                                        >{{ alignment[citation.field] }}</span>
                                        <span
                                            class="separator"
                                            v-if="citationIndex != globalConfig.sourceCitation.length - 1"
                                        >&#9679;&nbsp;</span>
                                    </span>
                                </p>
                            </div>
                            <div
                                class="col mt-4 border border-top-0 border-right-0 border-bottom-0"
                            >
                                <h6 class="text-center pb-2" v-html="globalConfig.targetLabel"></h6>
                                <p class="pt-3 px-3">
                                    <span
                                        v-for="(citation, citationIndex) in globalConfig.targetCitation"
                                        :key="citation.field"
                                        v-if="alignment[citation.field]"
                                    >
                                        <span
                                            :style="citation.style"
                                        >{{ alignment[citation.field] }}</span>
                                        <span
                                            class="separator"
                                            v-if="citationIndex != globalConfig.targetCitation.length - 1"
                                        >&#9679;&nbsp;</span>
                                    </span>
                                </p>
                            </div>
                        </div>
                        <div class="row passages">
                            <div class="col mb-2">
                                <p class="card-text text-justify px-3 pt-2 pb-4 mb-4">
                                    {{ alignment.source_context_before }}
                                    <span
                                        class="source-passage"
                                    >{{ alignment.source_passage }}</span>
                                    {{ alignment.source_context_after }}
                                </p>
                                <a
                                    class="card-link px-3 pt-2"
                                    style="position: absolute; bottom: 0"
                                    v-if="globalConfig.sourcePhiloDBLink"
                                    @click="goToContext(alignment, 'source')"
                                >View passage in context</a>
                            </div>
                            <div
                                class="col mb-2 border border-top-0 border-right-0 border-bottom-0"
                            >
                                <p class="card-text text-justify px-3 pt-2 pb-4 mb-4">
                                    {{ alignment.target_context_before }}
                                    <span
                                        class="target-passage"
                                    >{{ alignment.target_passage }}</span>
                                    {{ alignment.target_context_after }}
                                </p>
                                <a
                                    class="card-link px-3 pt-2"
                                    style="position: absolute; bottom: 0"
                                    v-if="globalConfig.targetPhiloDBLink"
                                    @click="goToContext(alignment, 'target')"
                                >View passage in context</a>
                            </div>
                        </div>
                        <div class="text-muted text-center mb-2">
                            <div v-if="globalConfig.matchingAlgorithm == 'vsa'">
                                <div>{{ alignment.similarity.toFixed(2) *100 }} % similar</div>
                                <a
                                    class="diff-btn"
                                    diffed="false"
                                    @click="showMatches(alignment)"
                                >Show matching words</a>
                                <div
                                    class="loading position-absolute"
                                    style="display:none; left: 50%; transform: translateX(-50%);"
                                >
                                    <atom-spinner
                                        :animation-duration="800"
                                        :size="25"
                                        color="#000"
                                    />
                                </div>
                            </div>
                            <div v-if="globalConfig.matchingAlgorithm == 'sa'">
                                <a
                                    class="diff-btn"
                                    diffed="false"
                                    @click="showDifferences(alignment.source_passage, alignment.target_passage, alignment.source_passage_length, alignment.target_passage.length)"
                                >Show differences</a>
                                <div
                                    class="loading position-absolute"
                                    style="display:none; left: 50%; transform: translateX(-50%);"
                                >
                                    <atom-spinner
                                        :animation-duration="800"
                                        :size="25"
                                        color="#000"
                                    />
                                </div>
                            </div>
                        </div>
                    </div>
                </transition-group>
                <nav aria-label="Page navigation" v-if="done">
                    <ul class="pagination justify-content-center mb-4">
                        <li class="page-item" v-if="results.page > 1">
                            <a class="page-link" v-on:click="previousPage()" aria-label="Previous">
                                <span aria-hidden="true">&laquo;</span>
                                <span class="sr-only">Previous</span>
                            </a>
                        </li>
                        <li class="page-item">
                            <a class="page-link">Page {{ results.page }}</a>
                        </li>
                        <li class="page-item" v-if="this.resultsLeft > 0">
                            <a class="page-link" v-on:click="nextPage()" aria-label="Next">
                                <span aria-hidden="true">&raquo;</span>
                                <span class="sr-only">Next</span>
                            </a>
                        </li>
                    </ul>
                </nav>
            </div>
            <div class="col-3 pl-0 position-relative">
                <div class="card rounded-0 shadow-1">
                    <h6 class="card-header text-center">Browse by Metadata Counts</h6>
                    <div
                        id="metadata-list"
                        class="mx-auto p-2"
                        @click="toggleFacetList()"
                    >Show options</div>
                    <div class="mt-3 mb-3 pr-3 pl-3 facet-list">
                        <h6 class="text-center" v-html="globalConfig.sourceLabel"></h6>
                        <div class="list-group">
                            <button
                                type="button"
                                class="list-group-item list-group-item-action"
                                v-for="(field, index) in globalConfig.facetsFields.source"
                                :key="index"
                                v-on:click="facetSearch(field.value)"
                            >{{ field.label }}</button>
                        </div>
                    </div>
                    <div class="mb-3 pr-3 pl-3 facet-list">
                        <h6 class="text-center" v-html="globalConfig.targetLabel"></h6>
                        <div class="list-group">
                            <button
                                type="button"
                                class="list-group-item list-group-item-action"
                                v-for="(field, index) in globalConfig.facetsFields.target"
                                :key="index"
                                v-on:click="facetSearch(field.value)"
                            >{{ field.label }}</button>
                        </div>
                    </div>
                </div>
                <div
                    class="loading position-absolute"
                    style="left: 50%; transform: translateX(-50%);"
                    v-if="facetLoading"
                >
                    <atom-spinner :animation-duration="800" :size="65" color="#000"/>
                </div>
                <div class="card rounded-0 shadow-1 mt-3" v-if="facetResults">
                    <div class="corner-btn destroy right" @click="closeFacetResults()">X</div>
                    <h6 class="card-header text-center">
                        Frequency by
                        <span v-html="facetDirectionLabel"></span>&nbsp;
                        <span class="text-capitalize">{{ facetResults.facet.split("_")[1] }}</span>
                    </h6>
                    <div class="mt-1 p-2">
                        <div class="pb-2 text-center" style="opacity: .5">Showing top 100 results</div>
                        <div class="list-group">
                            <div
                                class="list-group-item list-group-item-action facet-result"
                                v-for="(field, index) in facetResults.results.slice(0, 100)"
                                :key="index"
                                v-on:click="filteredSearch(facetResults.facet, field.field)"
                            >
                                <div class="row">
                                    <div class="col pr-1 pl-1">{{ field.field || "N/A"}}</div>
                                    <div
                                        class="col-4 pr-1 pl-1 facet-count"
                                    >{{ field.count.toLocaleString() }}</div>
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
import { EventBus } from "../main.js";
import searchArguments from "./searchArguments";
import { AtomSpinner } from "epic-spinners";
import Worker from "worker-loader!./diffStrings";
import Velocity from "velocity-animate";

export default {
    name: "searchResults",
    components: {
        searchArguments,
        AtomSpinner
    },
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
            facetLoading: null
        };
    },
    created() {
        // fetch the data when the view is created and the data is
        // already being observed
        this.fetchData();
    },
    watch: {
        // call again the method if the route changes
        $route: "fetchData"
    },
    methods: {
        fetchData() {
            this.results = { alignments: [] }; // clear alignments with new search
            this.facetResults = null; // clear facet results with new search
            this.error = null;
            this.loading = true;
            let params = { ...this.$route.query };
            params.db_table = this.$globalConfig.databaseName;
            EventBus.$emit("searchArgsUpdate", {
                counts: "",
                searchParams: params
            });
            this.$http
                .post(
                    `${
                        this.$globalConfig.apiServer
                    }/search_alignments/?${this.paramsToUrl(params)}`,
                    {
                        metadata: this.$globalConfig.metadataTypes
                    }
                )
                .then(response => {
                    this.results = response.data;
                    this.lastRowID = this.results.alignments[
                        this.results.alignments.length - 1
                    ].rowid_ordered;
                    this.page++;
                    this.loading = false;
                    this.done = true;
                    this.$http
                        .post(
                            `${
                                this.$globalConfig.apiServer
                            }/count_results/?${this.paramsToUrl(params)}`,
                            {
                                metadata: this.$globalConfig.metadataTypes
                            }
                        )
                        .then(response => {
                            let counts = response.data.counts;
                            EventBus.$emit("searchArgsUpdate", {
                                counts: counts,
                                searchParams: params
                            });
                            this.resultsLeft =
                                counts -
                                (this.results.start_position +
                                    this.results.alignments.length);
                        })
                        .catch(error => {
                            console.log(error);
                        });
                    Array.from(
                        document.getElementsByClassName("facet-list")
                    ).forEach(function(element) {
                        element.classList.remove("hide");
                    });
                    document
                        .querySelector("#metadata-list")
                        .classList.remove("show");
                })
                .catch(error => {
                    this.loading = false;
                    this.error = error.toString();
                    console.log(error);
                });
        },
        goToContext(alignment, direction) {
            let rootURL = "";
            let params = {};
            if (direction == "source") {
                rootURL = this.globalConfig.sourcePhiloDBLink;
                params = {
                    filename: alignment.source_filename.substr(
                        alignment.source_filename.lastIndexOf("/") + 1
                    ),
                    start_byte: alignment.source_start_byte,
                    end_byte: alignment.source_end_byte
                };
            } else {
                rootURL = this.globalConfig.targetPhiloDBLink;
                params = {
                    filename: alignment.target_filename.substr(
                        alignment.target_filename.lastIndexOf("/") + 1
                    ),
                    start_byte: alignment.target_start_byte,
                    end_byte: alignment.target_end_byte
                };
            }
            this.$http
                .get(`${rootURL}/scripts/alignment_to_text.py?`, {
                    params: params
                })
                .then(response => {
                    window.open(`${rootURL}/${response.data.link}`, "_blank");
                })
                .catch(error => {
                    alert(error);
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
            queryParams.id_anchor = this.results.alignments[
                this.results.alignments.length - 1
            ].rowid_ordered;
            this.$router.push(`/search?${this.paramsToUrl(queryParams)}`);
        },
        facetSearch(field) {
            let queryParams = { ...this.$route.query };
            queryParams.db_table = this.$globalConfig.databaseName;
            queryParams.facet = field;
            this.facetLoading = true;
            this.$http
                .post(
                    `${this.$globalConfig.apiServer}/facets/?${this.paramsToUrl(
                        queryParams
                    )}`,
                    {
                        metadata: this.$globalConfig.metadataTypes
                    }
                )
                .then(response => {
                    this.facetDirectionLabel = this.$globalConfig[
                        `${response.data.facet.split("_")[0]}Label`
                    ];
                    this.facetResults = response.data;
                    this.toggleFacetList();
                    this.facetLoading = false;
                })
                .catch(error => {
                    this.facetLoading = false;
                    this.error = error.toString();
                    console.log("ERROR", error);
                });
        },
        toggleFacetList() {
            Array.from(document.getElementsByClassName("facet-list")).forEach(
                function(element) {
                    element.classList.toggle("hide");
                }
            );
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
            EventBus.$emit("urlUpdate", queryParams);
            this.facetResults = null;
            this.results = { alignments: [] };
            this.$router.push(`/search?${this.paramsToUrl(queryParams)}`);
        },
        showDifferences(
            sourceText,
            targetText,
            sourcePassageLength,
            targetPassageLength
        ) {
            if (sourcePassageLength > 10000 || targetPassageLength > 10000) {
                alert(
                    "Passage of 10000 words or more may take up a long time to compare"
                );
            }
            let parent = event.srcElement.parentNode.parentNode.parentNode;
            let loading = parent.querySelector(".loading");
            let sourceElement = parent.querySelector(".source-passage");
            let targetElement = parent.querySelector(".target-passage");
            if (event.srcElement.getAttribute("diffed") == "false") {
                loading.style.display = "initial";
                let outerEvent = event;
                this.worker = new Worker();
                this.worker.postMessage([sourceText, targetText]);
                this.worker.onmessage = function(response) {
                    let differences = response.data;
                    let newSourceString = "";
                    let newTargetString = "";
                    for (let diffObj of differences) {
                        let [diffCode, text] = diffObj;
                        if (diffCode === 0) {
                            newSourceString += text;
                            newTargetString += text;
                        } else if (diffCode === -1) {
                            newSourceString += `<span class="removed">${text}</span>`;
                        } else if (diffCode === 1) {
                            newTargetString += `<span class="added">${text}</span>`;
                        }
                    }
                    sourceElement.innerHTML = newSourceString;
                    targetElement.innerHTML = newTargetString;
                    outerEvent.srcElement.setAttribute("diffed", "true");
                    loading.style.display = "none";
                    outerEvent.srcElement.textContent = "Hide differences";
                };
            } else {
                sourceElement.innerHTML = sourceText;
                targetElement.innerHTML = targetText;
                event.srcElement.setAttribute("diffed", "false");
                event.srcElement.textContent = "Show differences";
            }
        },
        showMatches: function(alignment) {
            let parent = event.srcElement.parentNode.parentNode.parentNode;
            let sourceElement = parent.querySelector(".source-passage");
            let targetElement = parent.querySelector(".target-passage");
            if (event.srcElement.getAttribute("diffed") == "false") {
                let source = alignment.source_passage_with_matches
                    .replace(/&gt;/g, ">")
                    .replace(/&lt;/g, "<");
                sourceElement.innerHTML = source;
                let target = alignment.target_passage_with_matches
                    .replace(/&gt;/g, ">")
                    .replace(/&lt;/g, "<");
                targetElement.innerHTML = target;
                event.srcElement.setAttribute("diffed", "true");
                event.srcElement.textContent = "Hide matching words";
            } else {
                sourceElement.innerHTML = alignment.source_passage;
                targetElement.innerHTML = alignment.target_passage;
                event.srcElement.setAttribute("diffed", "false");
                event.srcElement.textContent = "Show matching words";
            }
        },
        beforeEnter: function(el) {
            el.style.opacity = 0;
            el.style.height = 0;
        },
        enter: function(el, done) {
            var delay = el.dataset.index * 100;
            setTimeout(function() {
                Velocity(
                    el,
                    { opacity: 1, height: "100%" },
                    { complete: done }
                );
            }, delay);
        },
        toggleSearchForm() {
            EventBus.$emit("toggleSearchForm");
        }
    }
};
</script>

<style scoped>
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
    text-pair: right;
}

.list-group-item:focus,
.list-group-item:active {
    outline: none !important;
}

.source-passage,
.target-passage {
    color: dodgerblue;
}

/deep/ .added {
    color: darkblue;
    font-weight: 700;
}

/deep/ .removed {
    color: green;
    font-weight: 700;
    text-decoration: line-through;
}

/deep/ .token-match {
    color: darkblue;
    font-weight: 700;
    letter-spacing: -0.0075em;
}

/deep/ .filtered-token {
    letter-spacing: -0.0075em;
    opacity: 0.25;
}

.diff-btn {
    display: inline-block;
    padding: 0.2rem;
    margin-bottom: 2px;
    border: solid 1px #ddd;
    cursor: pointer;
}

.diff-btn:hover {
    color: #565656 !important;
    background-color: #f8f8f8;
}

.separator {
    padding: 5px;
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
</style>