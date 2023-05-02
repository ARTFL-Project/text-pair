<template>
    <div id="time-series-chart" class="mt-3">
        <div class="row" style="padding: 0 0.75rem">
            <div class="m-4" style="font-size: 120%;" v-if="error">No results for your query</div>
            <search-arguments></search-arguments>
        </div>
        <report-switcher></report-switcher>
        <div>
            <div class="loading position-absolute" style="left: 50%; transform: translateX(-50%)" v-if="loading">
                <div class="d-flex justify-content-center position-relative">
                    <div class="spinner-border"
                        style="width: 8rem; height: 8rem; position: absolute; z-index: 50; top: 30px" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                </div>
            </div>
            <div class="card p-4">
                <div id="time-series-options">
                    Group
                    <div class="my-dropdown" style="display: inline-block">
                        <button type="button" class="btn btn-sm btn-light rounded-0" @click="toggleDropdown()">
                            {{ directionSelected.label }} &#9662;
                        </button>
                        <ul class="my-dropdown-menu shadow-1">
                            <li class="my-dropdown-item" v-for="direction in directions" :key="direction.label"
                                @click="selectItem('directionSelected', direction)">
                                {{ direction.label }}
                            </li>
                        </ul>
                    </div>
                    &nbsp;results by
                    <div class="my-dropdown" style="display: inline-block">
                        <button type="button" class="btn btn-sm btn-light rounded-0" @click="toggleDropdown()">
                            {{ timeSeriesInterval.label }} &#9662;
                        </button>
                        <ul class="my-dropdown-menu shadow-1">
                            <li class="my-dropdown-item text-nowrap" v-for="interval in globalConfig.timeSeriesIntervals"
                                :key="interval.label" @click="selectItem('timeSeriesInterval', interval)">
                                {{ interval.label }}
                            </li>
                        </ul>
                    </div>
                    <div class="d-inline-block ms-2">
                        <button class="btn btn-sm btn-secondary rounded-0" type="button" @click="displayTimeSeries()">
                            Reload Time Series
                        </button>
                    </div>
                </div>
                <canvas id="myChart" style="margin-left: -2rem" height="800"></canvas>
            </div>
        </div>
    </div>
</template>

<script>
import Chart from "chart.js/dist/Chart.js";
import searchArguments from "./searchArguments";
import reportSwitcher from "./reportSwitcher";
import cssVariables from "../assets/theme.module.scss";

export default {
    name: "timeSeries",
    components: {
        searchArguments, reportSwitcher
    },
    inject: ["$http"],
    data() {
        return {
            formValues: this.getFormValues(),
            timeSeriesInterval: this.getTimeSeriesInterval(),
            directions: [
                {
                    label: this.$globalConfig.sourceLabel,
                    value: "source",
                },
                {
                    label: this.$globalConfig.targetLabel,
                    value: "target",
                },
            ],
            directionSelected: {
                label: this.$globalConfig.sourceLabel,
                value: "source",
            },
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
            interval: null,
        };
    },
    created() {
        this.fetchData();
    },
    watch: {
        // call again the method if the route changes
        $route: "fetchData",
    },
    methods: {
        getTimeSeriesInterval() {
            for (const interval of this.$globalConfig.timeSeriesIntervals) {
                if (interval.value == this.$route.query.timeSeriesInterval) {
                    return interval;
                }
            }
        },
        getFormValues() {
            let formValues = {};
            for (const key of this.$globalConfig.metadataFields.source) {
                if (key.value in this.$route.query) {
                    formValues[key.value] = this.$route.query[key.value];
                } else {
                    formValues[key.value] = "";
                }
            }
            for (const key of this.$globalConfig.metadataFields.target) {
                if (key.value in this.$route.query) {
                    formValues[key.value] = this.$route.query[key.value];
                } else {
                    formValues[key.value] = "";
                }
            }
            formValues.banality = "";
            formValues.timeSeriesInterval = this.$globalConfig.timeSeriesIntervals[0].value;
            formValues.directionSelected = "source";
            return formValues;
        },
        fetchData() {
            this.results = { alignments: [] }; // clear alignments with new search
            this.facetResults = null; // clear facet results with new search
            let params = { ...this.$route.query };
            this.interval = parseInt(params.timeSeriesInterval);
            params.db_table = this.$globalConfig.databaseName;
            this.loading = true;
            this.$http
                .post(`${this.$globalConfig.apiServer}/generate_time_series/?${this.paramsToUrl(params)}`, {
                    metadata: this.$globalConfig.metadataTypes,
                })
                .then((response) => {
                    this.loading = false;
                    this.done = true;
                    this.emitter.emit("searchArgsUpdate", { counts: response.data.counts, searchParams: params });
                    this.drawChart(response.data.results);
                })
                .catch((error) => {
                    this.loading = false;
                    this.error = error.toString();
                    console.log(error);
                });
        },
        drawChart(results) {
            if (this.chart != null) {
                this.chart.destroy();
            }
            var ctx = document.getElementById("myChart");
            ctx.width = document.getElementById("search-form").offsetWidth;
            var labels = [];
            var data = [];
            for (var key of results) {
                labels.push(key.year);
                data.push(key.count);
            }
            var vm = this;
            vm.chart = new Chart(ctx, {
                type: "bar",
                data: {
                    labels: labels,
                    datasets: [
                        {
                            data: data,
                            borderWidth: 1,
                            backgroundColor: cssVariables.color,
                            hoverBackgroundColor: this.hexToRGBA(cssVariables.color),
                        },
                    ],
                },
                options: {
                    legend: {
                        display: false,
                    },
                    layout: {
                        padding: {
                            right: 10,
                            left: 10,
                        },
                    },
                    responsive: false,
                    scales: {
                        yAxes: [
                            {
                                ticks: {
                                    beginAtZero: true,
                                },
                                gridLines: {
                                    color: "#eee",
                                    offsetGridLines: true,
                                },
                            },
                        ],
                        xAxes: [
                            {
                                gridLines: {
                                    drawOnChartArea: false,
                                },
                                scaleLabel: {
                                    labelString: "Years",
                                },
                            },
                        ],
                    },
                    tooltips: {
                        cornerRadius: 0,
                        callbacks: {
                            title: function (tooltipItem) {
                                if (vm.interval != 1) {
                                    return `${tooltipItem[0].xLabel}-${parseInt(tooltipItem[0].xLabel) + vm.interval - 1
                                        }`;
                                } else {
                                    return tooltipItem[0].xLabel;
                                }
                            },
                            label: function (tooltipItem) {
                                return `${tooltipItem.yLabel.toLocaleString()} shared passages`;
                            },
                            displayColors: false,
                        },
                    },
                },
            });
            ctx.onclick = function (evt) {
                var activePoints = vm.chart.getElementsAtEvent(evt);
                if (activePoints.length > 0) {
                    var clickedElementindex = activePoints[0]["_index"];
                    var label = vm.chart.data.labels[clickedElementindex];
                    let params = { ...vm.$route.query };
                    if (vm.interval != 1) {
                        params[`${params.directionSelected}_year`] = `${label}-${label + vm.interval - 1}`;
                    } else {
                        params[`${params.directionSelected}_year`] = label;
                    }
                    vm.emitter.emit("urlUpdate", params);
                    vm.$router.push(`/search?${vm.paramsToUrl(params)}`);
                }
            };
        },
        hexToRGBA(h) {
            let r = 0,
                g = 0,
                b = 0;

            // 3 digits
            if (h.length == 4) {
                r = "0x" + h[1] + h[1];
                g = "0x" + h[2] + h[2];
                b = "0x" + h[3] + h[3];

                // 6 digits
            } else if (h.length == 7) {
                r = "0x" + h[1] + h[2];
                g = "0x" + h[3] + h[4];
                b = "0x" + h[5] + h[6];
            }
            return "rgba(" + +r + "," + +g + "," + +b + ", .7)";
        },
        selectItem(key, item) {
            this[key] = item;
            this.formValues[key] = item.value;
            this.toggleDropdown();
        },
        toggleDropdown() {
            let element = event.srcElement.closest(".my-dropdown").querySelector("ul");
            if (element.style.display != "inline-block") {
                element.style.display = "inline-block";
            } else {
                element.style.display = "none";
            }
        },
        displayTimeSeries() {
            this.$router.push(`/time?${this.paramsToUrl(this.formValues)}`);
        },
    },
};
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
    float: right;
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

#time-series-options {
    margin-left: -1rem;
    margin-bottom: 1rem;
}
</style>