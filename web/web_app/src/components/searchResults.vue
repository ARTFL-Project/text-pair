<template>
    <div class="mt-3">
        <div class="row">
            <div class="loading" v-if="loading">
                Loading...
            </div>
            <div v-if="error" class="error">
                {{ error }}
            </div>
            <div class="col">
                <transition-group name="staggered-fade" tag="div" v-bind:css="false" v-on:before-enter="beforeEnter" v-on:enter="enter">
                    <div class="card mb-3 rounded-0 shadow-1" style="position: relative" v-for="(alignment, index) in results.alignments" :key="results.start_position + index + 1" v-bind:data-index="index">
                        <div class="border border-top-0 border-left-0" style="position: absolute; padding: 5px; overflow: hidden;">
                            {{ results.start_position + index + 1 }}
                        </div>
                        <div class="row">
                            <div class="col mt-4">
                                <h6 class="text-center pb-2">
                                    Source
                                </h6>
                                <p class="pt-3 px-3">
                                    <span v-for="(citation, citationIndex) in globalConfig.sourceCitation" :key="citation.field" v-if="alignment[citation.field]">
                                        <span :style="citation.style">{{ alignment[citation.field] }}</span>
                                        <span class="separator" v-if="citationIndex != globalConfig.sourceCitation.length - 1">&#9679;&nbsp;</span>
                                    </span>
                                </p>
                            </div>
                            <div class="col mt-4 border border-top-0 border-right-0 border-bottom-0">
                                <h6 class="text-center pb-2">
                                    Target
                                </h6>
                                <p class="pt-3 px-3">
                                    <span v-for="(citation, citationIndex) in globalConfig.targetCitation" :key="citation.field" v-if="alignment[citation.field]">
                                        <span :style="citation.style">{{ alignment[citation.field] }}</span>
                                        <span class="separator" v-if="citationIndex != globalConfig.targetCitation.length - 1">&#9679;&nbsp;</span>
                                    </span>
                                </p>
                            </div>
                        </div>
                        <div class="row passages">
                            <div class="col mb-2">
                                <p class="card-text text-justify px-3 pt-2 pb-4 mb-4">
                                    {{ alignment.source_context_before }}
                                    <span class="source-passage">{{ alignment.source_passage }}</span>
                                    {{ alignment.source_context_after }}
                                </p>
                                <a class="card-link px-3 pt-2" style="position: absolute; bottom: 0" v-if="globalConfig.sourceDB.philoDB" v-on:click="goToContext(alignment, 'source')">View passage in context</a>
                            </div>
                            <div class="col mb-2 border border-top-0 border-right-0 border-bottom-0">
                                <p class="card-text text-justify px-3 pt-2 pb-4 mb-4">
                                    {{ alignment.target_context_before }}
                                    <span class="target-passage">{{ alignment.target_passage }}</span>
                                    {{ alignment.target_context_after }}
                                </p>
                                <a class="card-link px-3 pt-2" style="position: absolute; bottom: 0" v-if="globalConfig.targetDB.philoDB" v-on:click="goToContext(alignment, 'target')">View passage in context</a>
                            </div>
                        </div>
                        <div class="text-muted text-center">
                            <!-- <span style="padding: .25rem .5rem;">{{ alignment.passage_similarity }} similar:</span><br> -->
                            <a class="diff-btn" v-on:click="showDifferences(alignment.source_passage, alignment.target_passage)">Show differences</a>
                        </div>
                    </div>
                </transition-group>
                <!-- <infinite-loading v-if="results.alignments.length > 0" @infinite="infiniteHandler">
                        <span slot="no-more">
                            There is no more Hacker News :(
                        </span>
                    </infinite-loading> -->
            </div>
            <div class="col-3 pl-0">
                <div class="card rounded-0 shadow-1">
                    <h6 class="card-header text-center">Browse by Metadata Counts</h6>
                    <div class="mt-3 p-3">
                        <h6 class="text-center">Source</h6>
                        <div class="list-group">
                            <button type="button" class="list-group-item list-group-item-action" v-for="(field, index) in globalConfig.metadataFields.source" :key="index" v-on:click="facetSearch(field.value)">
                                {{ field.label }}
                            </button>
                        </div>
                    </div>
                    <div class="mt-3 p-3">
                        <h6 class="text-center">Target</h6>
                        <div class="list-group">
                            <button type="button" class="list-group-item list-group-item-action" v-for="(field, index) in globalConfig.metadataFields.target" :key="index" v-on:click="facetSearch(field.value)">
                                {{ field.label }}
                            </button>
                        </div>
                    </div>
                </div>
                <div class="card rounded-0 shadow-1 mt-3" v-if="facetResults">
                    <h6 class="card-header text-center">Frequency by {{ facetResults.facet }}</h6>
                    <div class="mt-3 p-3">
                        <div class="list-group">
                            <div class="list-group-item list-group-item-action facet-result" v-for="(field, index) in facetResults.results" :key="index" v-on:click="filteredSearch(facetResults.facet, field.field)">
                                <div class="row">
                                    <div class="col">{{ field.field }}</div>
                                    <div class="col-4 facet-count">{{ field.count }}</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
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
                <li class="page-item" v-if="results.next_url != ''">
                    <a class="page-link" v-on:click="nextPage()" aria-label="Next">
                        <span aria-hidden="true">&raquo;</span>
                        <span class="sr-only">Next</span>
                    </a>
                </li>
            </ul>
        </nav>
    </div>
</template>

<script>
import { EventBus } from '../main.js';
import * as differ from 'diff';
// import { InfiniteLoading } from '../main.js';

export default {
    name: "searchResults",
    data() {
        return {
            loading: false,
            done: false,
            results: { alignments: [] },
            lastRowID: null,
            page: 0,
            error: null,
            globalConfig: this.$globalConfig,
            facetResults: null,
            facetLoading: null
        }
    },
    created() {
        // fetch the data when the view is created and the data is
        // already being observed
        this.fetchData()
    },
    watch: {
        // call again the method if the route changes
        '$route': 'fetchData'
    },
    methods: {
        fetchData() {
            this.results = { alignments: [] } // clear alignments with new search
            this.facetResults = null // clear facet results with new search
            let params = { ...this.$route.query }
            params.db_table = this.$globalConfig.databaseName
            this.$http.post(`${this.$globalConfig.apiServer}/search_alignments/?${this.paramsToUrl(params)}`, {
                metadata: this.$globalConfig.metadataTypes
            }).then(response => {
                this.results = response.data
                this.lastRowID = this.results.alignments[this.results.alignments.length - 1].rowid_ordered
                this.page++
                this.loading = false
                this.done = true
            }).catch(error => {
                this.loading = false
                this.error = error.toString();
                console.log(error)
            });
        },
        goToContext(alignment, direction) {
            let rootURL = ""
            let params = {}
            if (direction == "source") {
                rootURL = this.globalConfig.sourceDB.link
                params = {
                    doc_id: alignment.source_doc_id,
                    start_byte: alignment.source_start_byte,
                    end_byte: alignment.source_end_byte
                }
            } else {
                rootURL = this.globalConfig.targetDB.link
                params = {
                    doc_id: alignment.target_doc_id,
                    start_byte: alignment.target_start_byte,
                    end_byte: alignment.target_end_byte
                }
            }
            this.$http.get(`${rootURL}/scripts/alignment_to_text.py?`, {
                params: params
            }).then(response => {
                window.open(`${rootURL}/${response.data.link}`, '_blank');
            }).catch(error => {
                alert(error)
            })
        },
        previousPage() {
            let queryParams = { ...this.$route.query }
            queryParams.page = parseInt(this.results.page) - 1
            queryParams.direction = "previous"
            queryParams.id_anchor = this.results.alignments[0].rowid_ordered
            this.$router.push(`/search?${this.paramsToUrl(queryParams)}`)
        },
        nextPage(urlEnd) {
            let queryParams = { ...this.$route.query }
            queryParams.page = parseInt(this.results.page) + 1
            queryParams.direction = "next"
            queryParams.id_anchor = this.results.alignments[this.results.alignments.length - 1].rowid_ordered
            this.$router.push(`/search?${this.paramsToUrl(queryParams)}`)
        },
        facetSearch(field) {
            let queryParams = { ...this.$route.query }
            queryParams.db_table = this.$globalConfig.databaseName
            queryParams.facet = field
            this.facetLoading = true
            this.$http.post(`${this.$globalConfig.apiServer}/facets/?${this.paramsToUrl(queryParams)}`, {
                metadata: this.$globalConfig.metadataTypes
            }).then(response => {
                this.facetResults = response.data
            }).catch(error => {
                this.facetLoading = false
                this.error = error.toString();
                console.log("ERROR", error)
            });
        },
        filteredSearch(fieldName, value) {
            let queryParams = { ...this.$route.query }
            delete queryParams.page
            delete queryParams.id_anchor
            queryParams.db_table = this.$globalConfig.databaseName
            if (this.$globalConfig.metadataTypes[fieldName] == "TEXT") {
                queryParams[fieldName] = `"${value}"`
            } else {
                queryParams[fieldName] = value
            }
            console.log(this.$globalConfig.metadataTypes[fieldName], queryParams)
            EventBus.$emit("urlUpdate", queryParams)
            this.facetResults = null
            this.results = { alignments: [] }
            this.$router.push(`/search?${this.paramsToUrl(queryParams)}`)
        },
        showDifferences(sourceText, targetText) {
            let sourceElement = event.srcElement.parentNode.parentNode.querySelector(".source-passage")
            let targetElement = event.srcElement.parentNode.parentNode.querySelector(".target-passage")
            let differences = differ.diffChars(sourceText, targetText, { ignoreCase: true })
            let newSourceString = ""
            let newTargetString = ""
            let deleted = ""
            for (let text of differences) {
                if (!text.hasOwnProperty("added") && !text.hasOwnProperty("removed")) {
                    newTargetString += text.value
                    newSourceString += text.value
                } else if (text.added) {
                    newTargetString += `<span class="added">${text.value}</span>`
                } else {
                    newSourceString += `<span class="removed">${text.value}</span>`
                }
            }
            sourceElement.innerHTML = newSourceString
            targetElement.innerHTML = newTargetString
        },
        beforeEnter: function(el) {
            el.style.opacity = 0
            el.style.height = 0
        },
        enter: function(el, done) {
            var delay = el.dataset.index * 100
            setTimeout(function() {
                Velocity(
                    el,
                    { opacity: 1, height: "100%" },
                    { complete: done }
                )
            }, delay)
        },
        infiniteHandler($state) {
            // setTimeout(function() {
            console.log("hi", this.results.alignments.length)
            let queryParams = { ...this.$route.query }
            queryParams.page = this.page + 1
            queryParams.direction = "next"
            queryParams.id_anchor = this.lastRowID
            queryParams.db_table = this.$globalConfig.databaseName
            this.$http.get(`${this.$globalConfig.apiServer}/search_alignments/?`, {
                params: queryParams
            }).then(response => {
                if (response.data.length) {
                    this.results.alignments = this.results.alignments.concat(response.data.alignments);
                    this.page++
                    this.lastRowID = this.results.alignments[this.results.alignments.length - 1].rowid_ordered
                    $state.loaded();
                } else {
                    $state.complete();
                }
            }).catch(error => {
                this.error = error.toString();
                console.log("ERROR", error)
            });
            // }, 10)
        },
    }
}
</script>

<style>
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
    text-align: right;
}

.list-group-item:focus,
.list-group-item:active {
    outline: none !important;
}

.source-passage,
.target-passage {
    color: dodgerblue;
}

.added {
    color: darkblue;
    font-weight: 700;
}

.removed {
    color: green;
    font-weight: 700;
    text-decoration: line-through;
}

.diff-btn {
    display: inline-block;
    padding: .2rem;
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
</style>