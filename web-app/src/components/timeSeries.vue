<template>
    <div id="time-series-chart" class="mt-3">
         <div class="row">
            <div class="m-4" style="font-size: 120%" v-if="error">
                No results for your query
            </div>
            <search-arguments></search-arguments>
        </div>
        <div class="row mt-3">
            <div class="loading position-absolute" style="left: 50%; transform: translateX(-50%);" v-if="loading">
                <atom-spinner :animation-duration="800" :size="65" color="#000"/>
            </div>
            <div class="col">
                <canvas id="myChart" height="800"></canvas>
            </div>
        </div>
    </div>
</template>

<script>
import { EventBus } from "../main.js"
import Chart from "chart.js/dist/Chart.js"
import searchArguments from "./searchArguments";
import { AtomSpinner } from 'epic-spinners';


export default {
    name: "timeSeries",
    components: {
        searchArguments,
        AtomSpinner
    },
    data() {
        return {
            loading: true,
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
            this.loading = true
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
                    EventBus.$emit("searchArgsUpdate", {counts: response.data.counts, searchParams : params})
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
                                return `${tooltipItem.yLabel.toLocaleString()} shared passages`
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