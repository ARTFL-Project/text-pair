<template>
    <div id="time-series-chart" class="mt-5">
        <div class="row">
            <div class="loading" v-if="loading">
                Loading...
            </div>
            <div v-if="error" class="error">
                {{ error }}
            </div>
            <div class="col">
                <canvas id="myChart" height="700"></canvas>
            </div>
        </div>
    </div>
</template>

<script>
import { EventBus } from "../main.js"
import Chart from "chart.js"

export default {
    name: "timeSeries",
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
            facetLoading: null,
            chart: null,
            interval: null
        }
    },
    created() {
        // fetch the data when the view is created and the data is
        // already being observed
        this.fetchData()
    },
    watch: {
        // call again the method if the route changes
        $route: "fetchData"
    },
    methods: {
        fetchData() {
            this.results = { alignments: [] } // clear alignments with new search
            this.facetResults = null // clear facet results with new search
            let params = { ...this.$route.query }
            this.interval = parseInt(params.timeSeriesInterval)
            params.db_table = this.$globalConfig.databaseName
            this.$http
                .post(
                    `${this.$globalConfig
                        .apiServer}/generate_time_series/?${this.paramsToUrl(
                        params
                    )}`,
                    {
                        metadata: this.$globalConfig.metadataTypes
                    }
                )
                .then(response => {
                    this.loading = false
                    this.done = true
                    this.drawChart(response.data.results)
                })
                .catch(error => {
                    this.loading = false
                    this.error = error.toString()
                    console.log(error)
                })
        },
        drawChart(results) {
            if (this.chart != null) {
                this.chart.destroy()
            }
            var ctx = document.getElementById("myChart")
            ctx.width = document.getElementById("search-form").offsetWidth
            var labels = []
            var data = []
            for (var key of results) {
                labels.push(key.year)
                data.push(key.count)
            }
            var vm = this
            console.log("INTERVAL", this.interval)
            vm.chart = new Chart(ctx, {
                type: "bar",
                data: {
                    labels: labels,
                    datasets: [
                        {
                            data: data,
                            borderWidth: 1,
                            backgroundColor: "#e9ecef"
                        }
                    ]
                },
                options: {
                    legend: {
                        display: false
                    },
                    layout: {
                        padding: {
                            right: 10,
                            left: 10
                        }
                    },
                    responsive: false,
                    scales: {
                        yAxes: [
                            {
                                ticks: {
                                    beginAtZero: true
                                },
                                gridLines: {
                                    color: "#eee",
                                    offsetGridLines: true
                                }
                            }
                        ],
                        xAxes: [
                            {
                                gridLines: {
                                    drawOnChartArea: false
                                },
                                scaleLabel: {
                                    labelString: "Years"
                                }
                            }
                        ]
                    },
                    tooltips: {
                        cornerRadius: 0,
                        callbacks: {
                            title: function(tooltipItem) {
                                if (vm.interval != 1) {
                                    return `${tooltipItem[0].xLabel}-${parseInt(
                                        tooltipItem[0].xLabel
                                    ) +
                                        vm.interval -
                                        1}`
                                } else {
                                    return tooltipItem[0].xLabel
                                }
                            },
                            label: function(tooltipItem) {
                                return `${tooltipItem.yLabel} shared passages`
                            },
                            displayColors: false
                        }
                    }
                }
            })
            ctx.onclick = function(evt) {
                var activePoints = vm.chart.getElementsAtEvent(evt)
                if (activePoints.length > 0) {
                    var clickedElementindex = activePoints[0]["_index"]
                    var label = vm.chart.data.labels[clickedElementindex]
                    var value =
                        vm.chart.data.datasets[0].data[clickedElementindex]
                    let params = { ...vm.$route.query }
                    if (vm.interval != 1) {
                        params[
                            `${params.directionSelected}_year`
                        ] = `${label}-${label + vm.interval - 1}`
                    } else {
                        params[`${params.directionSelected}_year`] = label
                    }
                    EventBus.$emit("urlUpdate", params)
                    vm.$router.push(`/search?${vm.paramsToUrl(params)}`)
                }
            }
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
            this.$http
                .get(`${rootURL}/scripts/alignment_to_text.py?`, {
                    params: params
                })
                .then(response => {
                    window.open(`${rootURL}/${response.data.link}`, "_blank")
                })
                .catch(error => {
                    alert(error)
                })
        },
        facetSearch(field) {
            let queryParams = { ...this.$route.query }
            queryParams.db_table = this.$globalConfig.databaseName
            queryParams.facet = field
            this.facetLoading = true
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
                    this.facetResults = response.data
                })
                .catch(error => {
                    this.facetLoading = false
                    this.error = error.toString()
                    console.log("ERROR", error)
                })
        },
        filteredSearch(fieldName, value) {
            let queryParams = { ...this.$route.query }
            delete queryParams.page
            delete queryParams.id_anchor
            queryParams.db_table = this.$globalConfig.databaseName
            if (this.$globalConfig.metadataTypes[fieldName] == "text") {
                queryParams[fieldName] = `"${value}"`
            } else {
                queryParams[fieldName] = value
            }
            EventBus.$emit("urlUpdate", queryParams)
            this.facetResults = null
            this.results = { alignments: [] }
            this.$router.push(`/search?${this.paramsToUrl(queryParams)}`)
        },
        showDifferences(sourceText, targetText) {
            let sourceElement = event.srcElement.parentNode.parentNode.querySelector(
                ".source-passage"
            )
            let targetElement = event.srcElement.parentNode.parentNode.querySelector(
                ".target-passage"
            )
            let differences = differ.diffChars(sourceText, targetText, {
                ignoreCase: true
            })
            let newSourceString = ""
            let newTargetString = ""
            let deleted = ""
            for (let text of differences) {
                if (
                    !text.hasOwnProperty("added") &&
                    !text.hasOwnProperty("removed")
                ) {
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
                Velocity(el, { opacity: 1, height: "100%" }, { complete: done })
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
            this.$http
                .get(`${this.$globalConfig.apiServer}/search_alignments/?`, {
                    params: queryParams
                })
                .then(response => {
                    if (response.data.length) {
                        this.results.alignments = this.results.alignments.concat(
                            response.data.alignments
                        )
                        this.page++
                        this.lastRowID = this.results.alignments[
                            this.results.alignments.length - 1
                        ].rowid_ordered
                        $state.loaded()
                    } else {
                        $state.complete()
                    }
                })
                .catch(error => {
                    this.error = error.toString()
                    console.log("ERROR", error)
                })
            // }, 10)
        }
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
</style>