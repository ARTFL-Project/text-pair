<template>
    <div id="time-series-chart" class="mt-3">
        <div class="row" style="padding: 0 0.75rem">
            <div class="m-4" style="font-size: 120%;" v-if="error">No results for your query</div>
            <search-arguments></search-arguments>
        </div>
        <report-switcher></report-switcher>
        <div class="card p-4">
            <div id="time-series-options">
                Group results by
                <div class="my-dropdown" style="display: inline-block">
                    <button type="button" class="btn btn-sm btn-light rounded-0" @click="toggleDropdown()">
                        {{ timeSeriesInterval.label }} &#9662;
                    </button>
                    <ul class="my-dropdown-menu shadow-1">
                        <li class="my-dropdown-item text-nowrap" v-for="interval in globalConfig.timeSeriesIntervals"
                            :key="interval.label" @click="selectInterval(interval)">
                            {{ interval.label }}
                        </li>
                    </ul>
                </div>
                <span class="ms-3">from</span>
                <input type="number" class="form-control form-control-sm rounded-0 year-input" :value="range[0]"
                    @change="setRange(0, $event)" aria-label="First year shown">
                to
                <input type="number" class="form-control form-control-sm rounded-0 year-input" :value="range[1]"
                    @change="setRange(1, $event)" aria-label="Last year shown">
                <button type="button" class="btn btn-sm btn-light rounded-0" v-if="customRange"
                    @click="customRange = null">Fit to data</button>
            </div>
            <div class="range-note" v-if="hiddenCounts.length">
                Not shown:
                <span v-for="(hidden, i) in hiddenCounts" :key="hidden.direction">{{ i ? " and " : "" }}{{
                    hidden.count.toLocaleString() }} <span v-html="hidden.label"></span></span>
                passages dated outside {{ shownRange }}.
            </div>
            <div class="d-flex justify-content-center" v-if="loading && !bins.length">
                <div class="spinner-border" style="width: 8rem; height: 8rem; margin-top: 30px" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            </div>
            <div class="chart-area" ref="chartArea" :class="{ refreshing: loading }">
                <template v-if="bins.length">
                    <div class="time-panel" v-for="panel in panels" :key="panel.direction">
                        <div class="panel-title">
                            <span v-html="panel.label"></span>
                            <span class="panel-total">{{ panel.total.toLocaleString() }} dated passages</span>
                        </div>
                        <svg :width="width" :height="svgHeight" class="time-svg" tabindex="0" role="img"
                            :aria-label="`${plainText(panel.label)} passages by year. Use arrow keys to move between periods and Enter to see their passages.`"
                            @keydown="onKeydown(panel.direction, $event)"
                            @focus="focusBin(hover ? hover.index : 0, panel.direction, $event.currentTarget)"
                            @blur="hover = null" @mouseleave="hover = null">
                            <g v-for="tick in yTicks[panel.direction]" :key="tick">
                                <line class="grid-line" :x1="marginLeft" :x2="width - margin.right"
                                    :y1="yScale(tick, panel.direction)" :y2="yScale(tick, panel.direction)" />
                                <text class="tick-label" :x="marginLeft - 8" :y="yScale(tick, panel.direction)"
                                    dy="0.32em" text-anchor="end">{{ tick.toLocaleString() }}</text>
                            </g>
                            <rect class="hover-wash" v-if="isHovered(panel.direction)" :x="bandX(hover.index)" :y="margin.top" :width="band"
                                :height="plotHeight" />
                            <path v-for="(bin, i) in visibleBins" :key="bin.year" :d="barPath(i, bin, panel.direction)"
                                :fill="barColor" :class="{ hovered: isHovered(panel.direction) && hover.index === i }" />
                            <line class="baseline" :x1="marginLeft" :x2="width - margin.right" :y1="baseline"
                                :y2="baseline" />
                            <text class="tick-label" v-for="tick in xTicks" :key="tick.year"
                                :x="bandX(tick.index) + band / 2" :y="baseline + 18" text-anchor="middle">{{ tick.year
                                }}</text>
                            <rect class="hit-area" v-for="(bin, i) in visibleBins" :key="`hit-${bin.year}`" :x="bandX(i)"
                                :y="margin.top" :width="band" :height="plotHeight"
                                @mousemove="hoverBin(i, panel.direction, $event)"
                                @click="showPassages(panel.direction, bin.year)" />
                        </svg>
                    </div>
                </template>
                <div class="time-tooltip shadow-1" v-if="hoveredBin" :style="tooltipStyle">
                    <div class="tooltip-period">{{ periodLabel(hoveredBin.year) }}</div>
                    <div v-for="panel in panels" :key="panel.direction">
                        <strong>{{ hoveredBin[panel.direction].toLocaleString() }}</strong>
                        <span v-html="panel.label" class="ms-1"></span>
                    </div>
                    <div class="tooltip-hint" v-if="hoveredBin[hover.direction]">
                        Click to see these <span v-html="hoveredLabel"></span> passages
                    </div>
                </div>
            </div>
            <details class="mt-3" v-if="visibleBins.length">
                <summary>Data table</summary>
                <table class="table table-sm time-table mt-2">
                    <thead>
                        <tr>
                            <th>Period</th>
                            <th v-for="panel in panels" :key="panel.direction" v-html="panel.label"></th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr v-for="bin in visibleBins.filter((b) => b.source || b.target)" :key="bin.year">
                            <td>{{ periodLabel(bin.year) }}</td>
                            <td v-for="panel in panels" :key="panel.direction">
                                <a href="#" v-if="bin[panel.direction]"
                                    @click.prevent="showPassages(panel.direction, bin.year)">{{
                                        bin[panel.direction].toLocaleString() }}</a>
                                <span v-else>0</span>
                            </td>
                        </tr>
                    </tbody>
                </table>
            </details>
        </div>
    </div>
</template>

<script>
import searchArguments from "./searchArguments";
import reportSwitcher from "./reportSwitcher";
import cssVariables from "../assets/theme.module.scss";

const MARGIN = { top: 10, right: 20, bottom: 28 };
const PLOT_HEIGHT = 220;
const MAX_BAR_WIDTH = 24;
// Share of passages the auto-fitted range must hold before widening to adjacent non-empty bins
const AUTO_FIT_SHARE = 0.99;
const LABEL_STEPS = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000];
const MIN_LABEL_SPACING = 64;

export default {
    name: "timeSeries",
    components: {
        searchArguments, reportSwitcher
    },
    inject: ["$http"],
    data() {
        return {
            globalConfig: this.$globalConfig,
            panels: [
                { direction: "source", label: this.$globalConfig.sourceLabel, total: 0 },
                { direction: "target", label: this.$globalConfig.targetLabel, total: 0 },
            ],
            bins: [],
            customRange: null,
            searchKey: null,
            requestId: 0,
            loading: true,
            error: null,
            width: 0,
            hover: null,
            barColor: cssVariables.color,
            margin: MARGIN,
            plotHeight: PLOT_HEIGHT,
        };
    },
    computed: {
        timeSeriesInterval() {
            const intervals = this.$globalConfig.timeSeriesIntervals;
            return intervals.find((i) => i.value == this.$route.query.timeSeriesInterval) || intervals[0];
        },
        interval() {
            return parseInt(this.timeSeriesInterval.value);
        },
        autoRange() {
            const bins = this.bins;
            if (!bins.length) {
                return null;
            }
            const mass = bins.map((bin) => bin.source + bin.target);
            const total = mass.reduce((sum, count) => sum + count, 0);
            let [first, last] = [0, bins.length - 1];
            if (total > 0) {
                // Shortest window of bins holding AUTO_FIT_SHARE of the passages
                let lo = 0;
                let sum = 0;
                for (let hi = 0; hi < bins.length; hi++) {
                    sum += mass[hi];
                    while (lo < hi && sum - mass[lo] >= AUTO_FIT_SHARE * total) {
                        sum -= mass[lo];
                        lo++;
                    }
                    if (sum >= AUTO_FIT_SHARE * total && hi - lo < last - first) {
                        [first, last] = [lo, hi];
                    }
                }
                while (first > 0 && mass[first - 1] > 0) first--;
                while (last < bins.length - 1 && mass[last + 1] > 0) last++;
            }
            return [bins[first].year, bins[last].year + this.interval - 1];
        },
        range() {
            return this.customRange || this.autoRange || ["", ""];
        },
        visibleBins() {
            const [from, to] = this.range;
            return this.bins.filter((bin) => bin.year + this.interval - 1 >= from && bin.year <= to);
        },
        shownRange() {
            const bins = this.visibleBins;
            return bins.length ? `${bins[0].year}–${bins[bins.length - 1].year + this.interval - 1}` : "";
        },
        hiddenCounts() {
            const hidden = [];
            for (const panel of this.panels) {
                const shown = this.visibleBins.reduce((sum, bin) => sum + bin[panel.direction], 0);
                if (panel.total > shown) {
                    hidden.push({ direction: panel.direction, label: panel.label, count: panel.total - shown });
                }
            }
            return hidden;
        },
        // Each panel scales to its own peak so a scattered series isn't flattened by a concentrated one
        yTicks() {
            const ticks = {};
            for (const panel of this.panels) {
                const max = this.visibleBins.reduce((top, bin) => Math.max(top, bin[panel.direction]), 1);
                const raw = max / 4;
                const magnitude = 10 ** Math.floor(Math.log10(raw));
                const step = Math.max(1, [1, 2, 5, 10].map((m) => m * magnitude).find((s) => s >= raw));
                ticks[panel.direction] = [0];
                while (ticks[panel.direction].at(-1) < max) {
                    ticks[panel.direction].push(ticks[panel.direction].at(-1) + step);
                }
            }
            return ticks;
        },
        // Shared by both panels so their year axes line up
        marginLeft() {
            const longest = Math.max(...Object.values(this.yTicks).map((ticks) => ticks.at(-1).toLocaleString().length));
            return longest * 7 + 16;
        },
        svgHeight() {
            return MARGIN.top + PLOT_HEIGHT + MARGIN.bottom;
        },
        baseline() {
            return MARGIN.top + PLOT_HEIGHT;
        },
        band() {
            return Math.max(0, this.width - this.marginLeft - MARGIN.right) / Math.max(1, this.visibleBins.length);
        },
        xTicks() {
            const step = LABEL_STEPS.find((s) => s % this.interval === 0 && (s / this.interval) * this.band >= MIN_LABEL_SPACING)
                || LABEL_STEPS[LABEL_STEPS.length - 1];
            const ticks = [];
            this.visibleBins.forEach((bin, index) => {
                if (bin.year % step === 0) {
                    ticks.push({ year: bin.year, index });
                }
            });
            return ticks;
        },
        hoveredBin() {
            return this.hover ? this.visibleBins[this.hover.index] : null;
        },
        hoveredLabel() {
            return this.panels.find((panel) => panel.direction === this.hover.direction).label;
        },
        tooltipStyle() {
            const flip = this.hover.x > this.width - 240;
            return {
                left: `${this.hover.x + (flip ? -14 : 14)}px`,
                top: `${this.hover.y + 14}px`,
                transform: flip ? "translateX(-100%)" : "none",
            };
        },
    },
    created() {
        this.fetchData();
    },
    mounted() {
        this.resizeObserver = new ResizeObserver(([entry]) => {
            this.width = entry.contentRect.width;
        });
        this.resizeObserver.observe(this.$refs.chartArea);
    },
    beforeUnmount() {
        this.resizeObserver.disconnect();
    },
    watch: {
        // call again the method if the route changes
        $route: "fetchData",
    },
    methods: {
        fetchData() {
            const params = {
                ...this.$route.query,
                db_table: this.$globalConfig.databaseName,
                timeSeriesInterval: this.interval,
            };
            delete params.directionSelected;
            // A new search drops the typed year range; a new interval keeps it
            const searchKey = this.paramsToUrl({ ...params, timeSeriesInterval: "" });
            if (searchKey !== this.searchKey) {
                this.customRange = null;
                this.searchKey = searchKey;
            }
            const requestId = ++this.requestId;
            this.loading = true;
            this.error = null;
            this.hover = null;
            const api = this.$globalConfig.apiServer;
            this.emitter.emit("searchArgsUpdate", { counts: "", searchParams: params });
            this.$http
                .get(`${api}/count_results/?${this.paramsToUrl(params)}`)
                .then((response) => {
                    if (requestId === this.requestId) {
                        this.emitter.emit("searchArgsUpdate", { counts: response.data.counts, searchParams: params });
                    }
                })
                .catch((error) => console.log(error));
            Promise.all(
                this.panels.map((panel) =>
                    this.$http.post(
                        `${api}/generate_time_series/?${this.paramsToUrl({ ...params, directionSelected: panel.direction })}`,
                        { metadata: this.$globalConfig.metadataTypes }
                    )
                )
            )
                .then((responses) => {
                    if (requestId === this.requestId) {
                        this.setBins(responses.map((response) => response.data));
                        this.loading = false;
                    }
                })
                .catch((error) => {
                    if (requestId === this.requestId) {
                        this.loading = false;
                        this.error = error.toString();
                    }
                    console.log(error);
                });
        },
        setBins(seriesData) {
            const counts = new Map();
            this.panels.forEach((panel, index) => {
                panel.total = seriesData[index].counts;
                for (const { year, count } of seriesData[index].results) {
                    counts.set(`${panel.direction}:${Number(year)}`, count);
                }
            });
            const years = [...counts.keys()].map((key) => Number(key.split(":")[1]));
            const bins = [];
            if (years.length) {
                const last = years.reduce((a, b) => Math.max(a, b));
                for (let year = years.reduce((a, b) => Math.min(a, b)); year <= last; year += this.interval) {
                    bins.push({
                        year,
                        source: counts.get(`source:${year}`) || 0,
                        target: counts.get(`target:${year}`) || 0,
                    });
                }
            }
            this.bins = bins;
        },
        setRange(index, event) {
            const year = parseInt(event.target.value);
            if (isNaN(year)) {
                event.target.value = this.range[index];
                return;
            }
            const range = [...this.range];
            range[index] = year;
            this.customRange = range[0] <= range[1] ? range : [range[1], range[0]];
        },
        yScale(value, direction) {
            return this.baseline - (value / this.yTicks[direction].at(-1)) * PLOT_HEIGHT;
        },
        bandX(index) {
            return this.marginLeft + index * this.band;
        },
        barPath(index, bin, direction) {
            const value = bin[direction];
            if (!value) {
                return "";
            }
            const gap = this.band >= 6 ? 2 : this.band >= 3 ? 1 : 0;
            const width = Math.min(MAX_BAR_WIDTH, this.band - gap);
            const x = this.bandX(index) + (this.band - width) / 2;
            const height = Math.max(1, this.baseline - this.yScale(value, direction));
            const top = this.baseline - height;
            const r = Math.min(4, width / 2, height);
            return `M${x},${this.baseline}V${top + r}A${r},${r} 0 0 1 ${x + r},${top}`
                + `H${x + width - r}A${r},${r} 0 0 1 ${x + width},${top + r}V${this.baseline}Z`;
        },
        isHovered(direction) {
            return this.hover !== null && this.hover.direction === direction;
        },
        periodLabel(year) {
            return this.interval === 1 ? `${year}` : `${year}–${year + this.interval - 1}`;
        },
        plainText(html) {
            return html.replace(/<[^>]*>/g, "");
        },
        hoverBin(index, direction, event) {
            const box = this.$refs.chartArea.getBoundingClientRect();
            this.hover = { index, direction, x: event.clientX - box.left, y: event.clientY - box.top };
        },
        focusBin(index, direction, svg) {
            if (!this.visibleBins.length) {
                return;
            }
            const box = this.$refs.chartArea.getBoundingClientRect();
            const svgBox = svg.getBoundingClientRect();
            this.hover = {
                index,
                direction,
                x: svgBox.left - box.left + this.bandX(index) + this.band / 2,
                y: svgBox.top - box.top + MARGIN.top + PLOT_HEIGHT / 2,
            };
        },
        onKeydown(direction, event) {
            const last = this.visibleBins.length - 1;
            let index = this.hover ? this.hover.index : 0;
            if (event.key === "ArrowRight") {
                index = Math.min(last, index + 1);
            } else if (event.key === "ArrowLeft") {
                index = Math.max(0, index - 1);
            } else if (event.key === "Home") {
                index = 0;
            } else if (event.key === "End") {
                index = last;
            } else if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                if (this.visibleBins[index] && this.visibleBins[index][direction]) {
                    this.showPassages(direction, this.visibleBins[index].year);
                }
                return;
            } else {
                return;
            }
            event.preventDefault();
            this.focusBin(index, direction, event.currentTarget);
        },
        showPassages(direction, year) {
            const params = { ...this.$route.query };
            params[`${direction}_year`] = this.interval === 1 ? `${year}` : `${year}-${year + this.interval - 1}`;
            this.emitter.emit("urlUpdate", params);
            this.$router.push(`/search?${this.paramsToUrl(params)}`);
        },
        selectInterval(interval) {
            this.toggleDropdown();
            const params = { ...this.$route.query, timeSeriesInterval: interval.value };
            this.emitter.emit("urlUpdate", params);
            this.$router.push(`/time?${this.paramsToUrl(params)}`);
        },
        toggleDropdown() {
            let element = event.srcElement.closest(".my-dropdown").querySelector("ul");
            if (element.style.display != "inline-block") {
                element.style.display = "inline-block";
            } else {
                element.style.display = "none";
            }
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

</style>

<style scoped>
#time-series-options {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.4rem;
}

.year-input {
    width: 5.5rem;
    font-variant-numeric: tabular-nums;
}

.range-note {
    margin-top: 0.5rem;
    color: #6c757d;
    font-size: 0.85rem;
}

.chart-area {
    position: relative;
    margin-top: 1rem;
    transition: opacity 0.2s ease-out;
}

.chart-area.refreshing {
    opacity: 0.5;
}

.time-panel + .time-panel {
    margin-top: 1rem;
}

.panel-title {
    font-weight: 600;
    margin-left: 0.5rem;
}

.panel-total {
    font-weight: 400;
    color: #6c757d;
    margin-left: 0.5rem;
    font-size: 0.85rem;
}

.time-svg {
    display: block;
    outline: none;
}

.time-svg:focus-visible {
    outline: 2px solid #6c757d;
    outline-offset: 2px;
}

.grid-line {
    stroke: #eee;
    stroke-width: 1;
}

.baseline {
    stroke: #ccc;
    stroke-width: 1;
}

.tick-label {
    font-family: "Open-Sans", sans-serif;
    font-size: 12px;
    font-variant-numeric: tabular-nums;
    fill: #6c757d;
}

.hover-wash {
    fill: #f2f2f2;
}

path.hovered {
    opacity: 0.7;
}

.hit-area {
    fill: transparent;
    cursor: pointer;
}

.time-tooltip {
    position: absolute;
    z-index: 10;
    pointer-events: none;
    background-color: #fff;
    border: 1px solid #ddd;
    padding: 0.4rem 0.6rem;
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
}

.tooltip-period,
.tooltip-hint {
    color: #6c757d;
    font-size: 0.85rem;
}

.tooltip-hint {
    margin-top: 0.2rem;
}

.time-table {
    width: auto;
    font-variant-numeric: tabular-nums;
}

.time-table td:not(:first-child),
.time-table th:not(:first-child) {
    text-align: right;
    padding-left: 2rem;
}
</style>
