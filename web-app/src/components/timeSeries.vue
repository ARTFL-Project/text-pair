<template>
    <div id="time-series-chart" class="mt-3">
        <h2 class="visually-hidden">Passages in timeline</h2>
        <div class="row" style="padding: 0 0.75rem">
            <div class="m-4" style="font-size: 120%;" v-if="error">No shared passages for your query</div>
            <search-arguments></search-arguments>
        </div>
        <report-switcher></report-switcher>
        <div class="card time-card shadow-1">
            <div id="time-series-options">
                <span id="time-interval-label">Group shared passages by</span>
                <div class="interval-dropdown" @focusout="onDropdownFocusout" @keydown.esc="closeDropdown">
                    <button type="button" id="time-interval-toggle" ref="intervalToggle"
                        class="btn btn-sm btn-light rounded-0" aria-labelledby="time-interval-label time-interval-toggle"
                        :aria-expanded="dropdownOpen ? 'true' : 'false'" aria-controls="time-interval-menu"
                        @click="dropdownOpen = !dropdownOpen">
                        {{ timeSeriesInterval.label }} <span aria-hidden="true">&#9662;</span>
                    </button>
                    <ul id="time-interval-menu" class="my-dropdown-menu interval-menu shadow-1" v-show="dropdownOpen">
                        <li v-for="option in globalConfig.timeSeriesIntervals" :key="option.label">
                            <button type="button" class="my-dropdown-item interval-option text-nowrap"
                                :aria-pressed="option.value == timeSeriesInterval.value ? 'true' : 'false'"
                                @click="selectInterval(option)">{{ option.label }}</button>
                        </li>
                    </ul>
                </div>
                <div class="year-range ms-3" role="group" aria-label="Years shown">
                    <label for="time-range-from">from</label>
                    <input type="number" id="time-range-from" class="form-control form-control-sm rounded-0 year-input"
                        :value="range[0]" :aria-describedby="rangeError ? 'time-range-error' : undefined"
                        @change="setRange(0, $event)">
                    <label for="time-range-to">to</label>
                    <input type="number" id="time-range-to" ref="rangeTo"
                        class="form-control form-control-sm rounded-0 year-input" :value="range[1]"
                        :aria-describedby="rangeError ? 'time-range-error' : undefined" @change="setRange(1, $event)">
                    <button type="button" class="btn btn-sm btn-light rounded-0" v-if="customRange"
                        @click="fitToData">Fit to data</button>
                </div>
            </div>
            <div id="time-range-error" class="range-error" v-if="rangeError">{{ rangeError }}</div>
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
            <p class="mt-3 mb-0" v-if="!loading && !error && !bins.length">No dated passages for your query.</p>
            <div class="chart-area" ref="chartArea" :class="{ refreshing: loading }"
                :aria-busy="loading ? 'true' : 'false'">
                <template v-if="bins.length">
                    <p id="time-chart-help" class="visually-hidden">
                        Each bar links to the passages of its period. Use the left and right arrow keys to move
                        between periods, Home and End for the first and last.
                    </p>
                    <div class="time-panel" :class="`side-${panel.direction}-color`" v-for="panel in panels"
                        :key="panel.direction">
                        <h3 class="panel-title">
                            <span class="side-mark" :class="panel.direction" aria-hidden="true"></span>
                            <span v-html="panel.label"></span> <span class="panel-total">{{
                                panel.total.toLocaleString() }} dated passages</span>
                        </h3>
                        <svg :width="width" :height="panelHeight(panel.direction)" class="time-svg" role="group"
                            :aria-label="chartLabel(panel)" aria-describedby="time-chart-help"
                            @keydown="onKeydown(panel.direction, $event)" @mouseleave="onChartLeave">
                            <g aria-hidden="true">
                                <g v-for="tick in yTicks[panel.direction]" :key="tick">
                                    <line class="grid-line" :x1="marginLeft" :x2="width - MARGIN.right"
                                        :y1="yScale(tick, panel.direction)" :y2="yScale(tick, panel.direction)" />
                                    <text class="tick-label" :x="marginLeft - 8" :y="yScale(tick, panel.direction)"
                                        dy="0.32em" text-anchor="end" v-if="tick || !flipped(panel.direction)">{{
                                            tick.toLocaleString() }}</text>
                                </g>
                                <rect class="hover-wash" v-if="isActive(panel.direction)" :x="bandX(active.index)"
                                    :y="plotTop(panel.direction)" :width="band" :height="PLOT_HEIGHT" />
                                <path v-for="(bin, i) in visibleBins" :key="bin.year"
                                    :d="barPath(i, bin, panel.direction)" :fill="barColors[panel.direction]"
                                    :class="{ hovered: isActive(panel.direction) && active.index === i }" />
                                <line class="baseline" :x1="marginLeft" :x2="width - MARGIN.right"
                                    :y1="baselineOf(panel.direction)" :y2="baselineOf(panel.direction)" />
                                <template v-if="!flipped(panel.direction)">
                                    <text class="tick-label year-label" v-for="tick in xTicks" :key="tick.year"
                                        :x="bandX(tick.index) + band / 2" :y="baseline + 19" text-anchor="middle">{{
                                            tick.year }}</text>
                                </template>
                            </g>
                            <template v-for="(bin, i) in visibleBins" :key="`hit-${bin.year}`">
                                <a v-if="bin[panel.direction]" class="bar-link" :data-index="i"
                                    :href="hrefs[`${panel.direction}:${bin.year}`]"
                                    :tabindex="i === cursorIndex[panel.direction] ? 0 : -1"
                                    :aria-label="barLabel(bin, panel.direction)"
                                    @click="openPassages($event, panel.direction, bin.year)"
                                    @focus="onBarFocus(i, panel.direction, $event)" @blur="focused = null">
                                    <rect class="hit-area" :x="bandX(i)" :y="plotTop(panel.direction)" :width="band"
                                        :height="PLOT_HEIGHT" @mousemove="hoverBin(i, panel.direction, $event)" />
                                </a>
                                <rect v-else class="hit-area empty" aria-hidden="true" :x="bandX(i)"
                                    :y="plotTop(panel.direction)" :width="band" :height="PLOT_HEIGHT"
                                    @mousemove="hoverBin(i, panel.direction, $event)" />
                            </template>
                        </svg>
                    </div>
                </template>
                <!-- Mouse shortcut for the bar it describes; keyboard users have the bar link -->
                <div class="time-tooltip shadow-1" ref="tooltip" v-if="activeBin && !tooltipDismissed"
                    :class="{ clickable: activeBin[active.direction] }" :style="tooltipStyle" aria-hidden="true"
                    @mouseleave="onTooltipLeave" @click="onTooltipClick">
                    <div class="tooltip-period">{{ periodLabel(activeBin.year) }}</div>
                    <div v-for="panel in panels" :key="panel.direction" class="tooltip-row">
                        <span class="side-mark" :class="[panel.direction, `side-${panel.direction}-color`]"></span>
                        <strong>{{ activeBin[panel.direction].toLocaleString() }}</strong>
                        <span v-html="panel.label" class="ms-1"></span>
                    </div>
                    <div class="tooltip-hint" v-if="activeBin[active.direction]">
                        <template v-if="active.keyboard">
                            Press Enter to see these <span v-html="activeLabel"></span> passages; arrow keys move
                            between periods
                        </template>
                        <template v-else>Click to see these <span v-html="activeLabel"></span> passages</template>
                    </div>
                </div>
            </div>
            <details class="mt-3" v-if="visibleBins.length">
                <summary>Data table</summary>
                <table class="table table-sm time-table mt-2">
                    <caption class="visually-hidden">Dated passages by {{ intervalName }}, {{ shownRange }}</caption>
                    <thead>
                        <tr>
                            <th scope="col">Period</th>
                            <th scope="col" v-for="panel in panels" :key="panel.direction" v-html="panel.label"></th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr v-for="bin in visibleBins.filter((b) => b.source || b.target)" :key="bin.year">
                            <th scope="row" class="fw-normal">{{ periodLabel(bin.year) }}</th>
                            <td v-for="panel in panels" :key="panel.direction">
                                <a v-if="bin[panel.direction]" :href="hrefs[`${panel.direction}:${bin.year}`]"
                                    @click="openPassages($event, panel.direction, bin.year)">{{
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

<script setup>
import axios from "axios";
import emitter from "tiny-emitter/instance";
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import cssVariables from "../assets/theme.module.scss";
import globalConfig from "../config.js";
import { paramsToUrl, plainText } from "../utils.js";
import reportSwitcher from "./reportSwitcher";
import searchArguments from "./searchArguments";

const MARGIN = { top: 10, right: 20, bottom: 28 };
const PLOT_HEIGHT = 220;
const MAX_BAR_WIDTH = 24;
// Share of passages the auto-fitted range must hold before widening to adjacent non-empty bins
const AUTO_FIT_SHARE = 0.99;
const LABEL_STEPS = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000];
const MIN_LABEL_SPACING = 64;

const route = useRoute();
const router = useRouter();
const panels = reactive([
    { direction: "source", label: globalConfig.sourceLabel, total: 0 },
    { direction: "target", label: globalConfig.targetLabel, total: 0 },
]);
const labelText = { source: plainText(globalConfig.sourceLabel), target: plainText(globalConfig.targetLabel) };
const bins = ref([]);
const customRange = ref(null);
const rangeError = ref("");
let searchKey = null;
let requestId = 0;
const loading = ref(true);
const error = ref(null);
const width = ref(0);
// Bar under the mouse, and bar with keyboard focus; the latest one drives the tooltip
const pointer = ref(null);
const focused = ref(null);
const tooltipDismissed = ref(false);
const tooltipWidth = ref(0);
// Year of the bar each chart's Tab stop lands on
const cursorYear = reactive({ source: null, target: null });
const dropdownOpen = ref(false);
const barColors = { source: cssVariables.sourceColor, target: cssVariables.targetColor };
const chartArea = ref(null);
const tooltip = ref(null);
const intervalToggle = ref(null);
const rangeTo = ref(null);

const timeSeriesInterval = computed(() => {
    const intervals = globalConfig.timeSeriesIntervals;
    return intervals.find((i) => i.value == route.query.timeSeriesInterval) || intervals[0];
});
const interval = computed(() => parseInt(timeSeriesInterval.value.value));
const intervalName = computed(() => timeSeriesInterval.value.label.toLowerCase());
const autoRange = computed(() => {
    if (!bins.value.length) {
        return null;
    }
    const mass = bins.value.map((bin) => bin.source + bin.target);
    const total = mass.reduce((sum, count) => sum + count, 0);
    let [first, last] = [0, bins.value.length - 1];
    if (total > 0) {
        // Shortest window of bins holding AUTO_FIT_SHARE of the passages
        let lo = 0;
        let sum = 0;
        for (let hi = 0; hi < mass.length; hi++) {
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
        while (last < mass.length - 1 && mass[last + 1] > 0) last++;
    }
    return [bins.value[first].year, bins.value[last].year + interval.value - 1];
});
const range = computed(() => customRange.value || autoRange.value || ["", ""]);
const visibleBins = computed(() => {
    const [from, to] = range.value;
    return bins.value.filter((bin) => bin.year + interval.value - 1 >= from && bin.year <= to);
});
const shownRange = computed(() => {
    const shown = visibleBins.value;
    return shown.length ? `${shown[0].year}–${shown[shown.length - 1].year + interval.value - 1}` : "";
});
const hiddenCounts = computed(() => {
    const hidden = [];
    for (const panel of panels) {
        const shown = visibleBins.value.reduce((sum, bin) => sum + bin[panel.direction], 0);
        if (panel.total > shown) {
            hidden.push({ direction: panel.direction, label: panel.label, count: panel.total - shown });
        }
    }
    return hidden;
});
// Each panel scales to its own peak so a scattered series isn't flattened by a concentrated one
const yTicks = computed(() => {
    const ticks = {};
    for (const panel of panels) {
        const max = visibleBins.value.reduce((top, bin) => Math.max(top, bin[panel.direction]), 1);
        const raw = max / 4;
        const magnitude = 10 ** Math.floor(Math.log10(raw));
        const step = Math.max(1, [1, 2, 5, 10].map((m) => m * magnitude).find((s) => s >= raw));
        ticks[panel.direction] = [0];
        while (ticks[panel.direction].at(-1) < max) {
            ticks[panel.direction].push(ticks[panel.direction].at(-1) + step);
        }
    }
    return ticks;
});
// Shared by both panels so their year axes line up
const marginLeft = computed(() => {
    const longest = Math.max(...Object.values(yTicks.value).map((ticks) => ticks.at(-1).toLocaleString().length));
    return longest * 7 + 16;
});
const svgHeight = MARGIN.top + PLOT_HEIGHT + MARGIN.bottom;
const baseline = MARGIN.top + PLOT_HEIGHT;
// The later series hangs from its top edge, mirroring the earlier one across the year axis
const FLIPPED_TOP = 1;
const band = computed(
    () => Math.max(0, width.value - marginLeft.value - MARGIN.right) / Math.max(1, visibleBins.value.length)
);
const xTicks = computed(() => {
    const step =
        LABEL_STEPS.find((s) => s % interval.value === 0 && (s / interval.value) * band.value >= MIN_LABEL_SPACING) ||
        LABEL_STEPS[LABEL_STEPS.length - 1];
    const ticks = [];
    visibleBins.value.forEach((bin, index) => {
        if (bin.year % step === 0) {
            ticks.push({ year: bin.year, index });
        }
    });
    return ticks;
});
const hrefs = computed(() => {
    const links = {};
    for (const bin of visibleBins.value) {
        for (const panel of panels) {
            if (bin[panel.direction]) {
                const path = `/search?${paramsToUrl(passageParams(panel.direction, bin.year))}`;
                links[`${panel.direction}:${bin.year}`] = router.resolve(path).href;
            }
        }
    }
    return links;
});
const cursorIndex = computed(() => {
    const indexes = {};
    for (const panel of panels) {
        const direction = panel.direction;
        const index = visibleBins.value.findIndex((bin) => bin.year === cursorYear[direction] && bin[direction]);
        indexes[direction] = index >= 0 ? index : visibleBins.value.findIndex((bin) => bin[direction]);
    }
    return indexes;
});
const active = computed(() => pointer.value || focused.value);
const activeBin = computed(() => (active.value ? visibleBins.value[active.value.index] : null));
const activeLabel = computed(() => panels.find((panel) => panel.direction === active.value.direction).label);
// Under the pointer's column so the pointer can move onto it; beside a focused bar; inside the chart
const tooltipStyle = computed(() => {
    const { x, y, keyboard, bandLeft, bandRight } = active.value;
    let left = x - 8;
    let top = y + 20;
    if (keyboard) {
        const fitsAfter = bandRight + 8 + tooltipWidth.value <= width.value;
        left = fitsAfter ? bandRight + 8 : bandLeft - 8 - tooltipWidth.value;
        top = y;
    }
    return { left: `${Math.max(0, Math.min(left, width.value - tooltipWidth.value))}px`, top: `${top}px` };
});

watch(active, async () => {
    await nextTick();
    tooltipWidth.value = tooltip.value ? tooltip.value.offsetWidth : 0;
});

function fetchData() {
    const params = {
        ...route.query,
        db_table: globalConfig.databaseName,
        timeSeriesInterval: interval.value,
    };
    delete params.directionSelected;
    // A new search drops the typed year range; a new interval keeps it
    const key = paramsToUrl({ ...params, timeSeriesInterval: "" });
    if (key !== searchKey) {
        customRange.value = null;
        rangeError.value = "";
        searchKey = key;
    }
    const id = ++requestId;
    loading.value = true;
    error.value = null;
    pointer.value = null;
    focused.value = null;
    const api = globalConfig.apiServer;
    emitter.emit("searchArgsUpdate", { counts: "", searchParams: params });
    axios
        .get(`${api}/count_results/?${paramsToUrl(params)}`)
        .then((response) => {
            if (id === requestId) {
                emitter.emit("searchArgsUpdate", { counts: response.data.counts, searchParams: params });
            }
        })
        .catch((err) => console.log(err));
    Promise.all(
        panels.map((panel) =>
            axios.post(
                `${api}/generate_time_series/?${paramsToUrl({ ...params, directionSelected: panel.direction })}`,
                { metadata: globalConfig.metadataTypes }
            )
        )
    )
        .then((responses) => {
            if (id === requestId) {
                setBins(responses.map((response) => response.data));
                loading.value = false;
                announceChart();
            }
        })
        .catch((err) => {
            if (id === requestId) {
                loading.value = false;
                error.value = err.toString();
                emitter.emit("announce", "No shared passages for your query");
            }
            console.log(err);
        });
}

function setBins(seriesData) {
    const counts = new Map();
    panels.forEach((panel, index) => {
        panel.total = seriesData[index].counts;
        for (const { year, count } of seriesData[index].results) {
            counts.set(`${panel.direction}:${Number(year)}`, count);
        }
    });
    const years = [...counts.keys()].map((key) => Number(key.split(":")[1]));
    const newBins = [];
    if (years.length) {
        const last = years.reduce((a, b) => Math.max(a, b));
        for (let year = years.reduce((a, b) => Math.min(a, b)); year <= last; year += interval.value) {
            newBins.push({
                year,
                source: counts.get(`source:${year}`) || 0,
                target: counts.get(`target:${year}`) || 0,
            });
        }
    }
    bins.value = newBins;
}

function announceChart() {
    if (!bins.value.length) {
        emitter.emit("announce", "No dated passages for your query");
        return;
    }
    const totals = panels.map((panel) => `${panel.total.toLocaleString()} ${labelText[panel.direction]}`);
    emitter.emit("announce", `Timeline by ${intervalName.value}, ${shownRange.value}: ${totals.join(" and ")} dated passages`);
}

function announceRange() {
    emitter.emit("announce", shownRange.value ? `Showing ${shownRange.value}` : "No periods in this range");
}

function setRange(index, event) {
    const year = parseInt(event.target.value);
    if (isNaN(year)) {
        event.target.value = range.value[index];
        rangeError.value = `Enter a year as a whole number; kept ${range.value[index]}.`;
        emitter.emit("announce", rangeError.value);
        return;
    }
    rangeError.value = "";
    const newRange = [...range.value];
    newRange[index] = year;
    customRange.value = newRange[0] <= newRange[1] ? newRange : [newRange[1], newRange[0]];
    announceRange();
}

function fitToData() {
    customRange.value = null;
    rangeError.value = "";
    // The button goes away, so focus goes back to the field before it
    rangeTo.value.focus();
    announceRange();
}

function flipped(direction) {
    return direction === "target";
}

function plotTop(direction) {
    return flipped(direction) ? FLIPPED_TOP : MARGIN.top;
}

function baselineOf(direction) {
    return flipped(direction) ? FLIPPED_TOP : baseline;
}

function panelHeight(direction) {
    return flipped(direction) ? FLIPPED_TOP + PLOT_HEIGHT + MARGIN.top : svgHeight;
}

function yScale(value, direction) {
    const offset = (value / yTicks.value[direction].at(-1)) * PLOT_HEIGHT;
    return flipped(direction) ? FLIPPED_TOP + offset : baseline - offset;
}

function bandX(index) {
    return marginLeft.value + index * band.value;
}

function barPath(index, bin, direction) {
    const value = bin[direction];
    if (!value) {
        return "";
    }
    const gap = band.value >= 6 ? 2 : band.value >= 3 ? 1 : 0;
    const barWidth = Math.min(MAX_BAR_WIDTH, band.value - gap);
    const x = bandX(index) + (band.value - barWidth) / 2;
    const height = Math.max(1, Math.abs(baselineOf(direction) - yScale(value, direction)));
    const r = Math.min(4, barWidth / 2, height);
    if (flipped(direction)) {
        const bottom = FLIPPED_TOP + height;
        return (
            `M${x},${FLIPPED_TOP}V${bottom - r}A${r},${r} 0 0 0 ${x + r},${bottom}` +
            `H${x + barWidth - r}A${r},${r} 0 0 0 ${x + barWidth},${bottom - r}V${FLIPPED_TOP}Z`
        );
    }
    const top = baseline - height;
    return (
        `M${x},${baseline}V${top + r}A${r},${r} 0 0 1 ${x + r},${top}` +
        `H${x + barWidth - r}A${r},${r} 0 0 1 ${x + barWidth},${top + r}V${baseline}Z`
    );
}

function isActive(direction) {
    return active.value !== null && active.value.direction === direction;
}

function periodLabel(year) {
    return interval.value === 1 ? `${year}` : `${year}–${year + interval.value - 1}`;
}

function chartLabel(panel) {
    const name = `${labelText[panel.direction]} passages by ${intervalName.value}, ${shownRange.value}`;
    return cursorIndex.value[panel.direction] < 0 ? `${name}: none in this range` : name;
}

function barLabel(bin, direction) {
    const count = bin[direction];
    return `${periodLabel(bin.year)}: ${count.toLocaleString()} ${labelText[direction]} passage${count === 1 ? "" : "s"}`;
}

function hoverBin(index, direction, event) {
    const current = pointer.value;
    if (current && current.index === index && current.direction === direction) {
        return;
    }
    const box = chartArea.value.getBoundingClientRect();
    pointer.value = { index, direction, x: event.clientX - box.left, y: event.clientY - box.top };
    tooltipDismissed.value = false;
}

function onChartLeave(event) {
    if (!event.relatedTarget || !event.relatedTarget.closest(".time-tooltip")) {
        pointer.value = null;
    }
}

function onTooltipLeave(event) {
    if (!event.relatedTarget || !event.relatedTarget.closest(".time-svg")) {
        pointer.value = null;
    }
}

function onTooltipClick() {
    if (activeBin.value[active.value.direction]) {
        showPassages(active.value.direction, activeBin.value.year);
    }
}

function onBarFocus(index, direction, event) {
    cursorYear[direction] = visibleBins.value[index].year;
    if (!event.currentTarget.matches(":focus-visible")) {
        return;
    }
    const box = chartArea.value.getBoundingClientRect();
    const svgBox = event.currentTarget.ownerSVGElement.getBoundingClientRect();
    pointer.value = null;
    focused.value = {
        index,
        direction,
        keyboard: true,
        bandLeft: svgBox.left - box.left + bandX(index),
        bandRight: svgBox.left - box.left + bandX(index) + band.value,
        y: svgBox.top - box.top + plotTop(direction),
    };
    tooltipDismissed.value = false;
}

function onKeydown(direction, event) {
    const current = event.target.getAttribute("data-index");
    if (current === null) {
        return;
    }
    const index = Number(current);
    const filled = [];
    visibleBins.value.forEach((bin, i) => bin[direction] && filled.push(i));
    let next;
    if (event.key === "ArrowRight") {
        next = filled.find((i) => i > index);
    } else if (event.key === "ArrowLeft") {
        next = filled.findLast((i) => i < index);
    } else if (event.key === "Home") {
        next = filled[0];
    } else if (event.key === "End") {
        next = filled.at(-1);
    } else if (event.key === " ") {
        event.preventDefault();
        showPassages(direction, visibleBins.value[index].year);
        return;
    } else {
        return;
    }
    event.preventDefault();
    if (next !== undefined) {
        event.currentTarget.querySelector(`[data-index="${next}"]`).focus();
    }
}

// Escape hides the tooltip without moving focus
function onWindowKeydown(event) {
    if (event.key === "Escape" && activeBin.value) {
        tooltipDismissed.value = true;
    }
}

function passageParams(direction, year) {
    const params = { ...route.query };
    params[`${direction}_year`] = interval.value === 1 ? `${year}` : `${year}-${year + interval.value - 1}`;
    return params;
}

// Plain clicks route in the app; modified clicks keep the browser's link behavior
function openPassages(event, direction, year) {
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || event.button) {
        return;
    }
    event.preventDefault();
    showPassages(direction, year);
}

function showPassages(direction, year) {
    const params = passageParams(direction, year);
    emitter.emit("urlUpdate", params);
    router.push(`/search?${paramsToUrl(params)}`);
}

function selectInterval(option) {
    closeDropdown();
    const params = { ...route.query, timeSeriesInterval: option.value };
    emitter.emit("urlUpdate", params);
    // Same path, so focus stays on the toggle
    router.push(`${route.path}?${paramsToUrl(params)}`);
}

function closeDropdown() {
    if (dropdownOpen.value) {
        dropdownOpen.value = false;
        intervalToggle.value.focus();
    }
}

function onDropdownFocusout(event) {
    if (!event.currentTarget.contains(event.relatedTarget)) {
        dropdownOpen.value = false;
    }
}

fetchData();
watch(() => route.fullPath, fetchData);

let resizeObserver = null;
onMounted(() => {
    resizeObserver = new ResizeObserver(([entry]) => {
        width.value = entry.contentRect.width;
    });
    resizeObserver.observe(chartArea.value);
    window.addEventListener("keydown", onWindowKeydown);
});
onBeforeUnmount(() => {
    resizeObserver.disconnect();
    window.removeEventListener("keydown", onWindowKeydown);
});
</script>

<style scoped lang="scss">
@use "../assets/theme.module.scss" as theme;

#time-series-options,
.year-range {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.4rem;
}

#time-series-options {
    padding-bottom: 1rem;
    border-bottom: 1px solid theme.$rule-color;
    font-family: theme.$font-ui;
    font-size: 0.85rem;
    color: theme.$muted;
}

#time-series-options .btn-light {
    font-weight: 600;
}

.time-card {
    padding: 1.5rem 1.75rem;
    border-radius: 3px;
}

.time-table {
    font-family: theme.$font-ui;
    font-size: 0.85rem;
}

.interval-dropdown {
    position: relative;
    display: inline-block;
}

.interval-menu {
    display: block;
    top: calc(100% + 0.4rem);
}

.interval-option {
    display: block;
    width: 100%;
    border: 0;
    background: transparent;
    color: inherit;
    text-align: left;
}

.interval-option:hover {
    background: #ddd;
}

.year-input {
    width: 5.5rem;
    font-variant-numeric: tabular-nums;
}

.range-note {
    margin-top: 0.5rem;
    color: theme.$muted;
    font-family: theme.$font-ui;
    font-size: 0.8rem;
}

.range-error {
    margin-top: 0.5rem;
    color: #b02a37;
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

@media (prefers-reduced-motion: reduce) {
    .chart-area {
        transition: none;
    }
}

.time-panel {
    display: flex;
    flex-direction: column;
}

// The later series' title goes under its bars, which hang below the year axis
.time-panel.side-target-color .panel-title {
    order: 2;
    margin-top: 0.6rem;
}

.panel-title {
    @include theme.kicker;
    display: flex;
    align-items: center;
    margin: 0 0 0.6rem 0.25rem;
    color: inherit;
    font-size: 0.74rem;
}

.panel-total {
    margin-left: 0.6rem;
    color: theme.$muted;
    font-weight: 600;
    letter-spacing: 0;
    text-transform: none;
    font-size: 0.8rem;
}

.time-svg {
    display: block;
}

.grid-line {
    stroke: theme.$rule-color;
    stroke-width: 1;
    stroke-dasharray: 2 3;
}

.baseline {
    stroke: theme.$rule-strong;
    stroke-width: 1;
}

.tick-label {
    font-family: theme.$font-ui;
    font-size: 11px;
    font-variant-numeric: tabular-nums;
    fill: theme.$muted;
}

// Years between the two series, shared by both
.year-label {
    font-size: 12.5px;
    font-weight: 600;
    fill: theme.$ink;
}

.hover-wash {
    fill: theme.$paper;
}

path.hovered {
    opacity: 0.7;
}

.hit-area {
    fill: transparent;
    cursor: pointer;
}

.hit-area.empty {
    cursor: default;
}

// Outline doesn't draw on SVG in every browser, so the focused bar's band gets a stroke instead
.bar-link:focus-visible {
    outline: none !important;
}

.bar-link:focus-visible .hit-area {
    stroke: theme.$focus-color;
    stroke-width: 3px;
}

.time-tooltip {
    position: absolute;
    z-index: 10;
    background-color: #fff;
    border: 1px solid theme.$rule-strong;
    border-radius: 3px;
    padding: 0.45rem 0.7rem;
    font-family: theme.$font-ui;
    font-size: 0.85rem;
    width: max-content;
    max-width: 100%;
    font-variant-numeric: tabular-nums;
}

.time-tooltip.clickable {
    cursor: pointer;
}

.tooltip-period,
.tooltip-hint {
    color: theme.$muted;
    font-size: 0.8rem;
}

.tooltip-row .side-mark {
    font-size: 0.8rem;
}

.tooltip-period {
    color: theme.$ink;
    @include theme.display-type;
    font-size: 1.05rem;
    font-weight: 600;
}

.tooltip-hint {
    margin-top: 0.2rem;
    max-width: 16rem;
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
