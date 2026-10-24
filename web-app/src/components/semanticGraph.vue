<template>
    <div class="mt-3" ref="root">
        <div class="container-fluid">
            <h2 class="visually-hidden">Semantic graph</h2>
            <div class="row" style="padding: 0 0.75rem">
                <div class="m-4" style="font-size: 120%" v-if="error">{{ error }}</div>
                <search-arguments></search-arguments>
            </div>
            <report-switcher />

            <!-- Graph Container with Controls -->
            <div class="card shadow-1" style="position: relative;">
                <!-- Loading Spinner -->
                <div class="d-flex justify-content-center position-relative flex-column align-items-center"
                    v-if="loading">
                    <div class="spinner-border"
                        style="width: 10rem; height: 10rem; position: absolute; z-index: 50; top: 300px"
                        role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <div class="position-absolute fw-bold fs-6 text-center loading-message"
                        style="z-index: 51; top: 370px; width: 200px;">
                        {{ loadingMessage }}
                    </div>
                </div>

                <!-- What is selected, and the two global toggles. Choosing
                     values happens in the sidebar. -->
                <div class="card-body p-2 border-bottom">
                    <div class="d-flex flex-wrap align-items-center gap-2" ref="filterBar">
                        <span ref="selectionSummary" tabindex="-1">
                            <strong>{{ selectionTotals.selected.toLocaleString() }}</strong>
                            of {{ selectionTotals.universe.toLocaleString() }} passages
                            <span class="text-muted" v-if="activeFilters.length">
                                ({{ Math.round(selectionTotals.share * 100) }}%)
                            </span>
                            in {{ visibleClusters }} themes
                        </span>
                        <button v-for="(f, i) in activeFilters" :key="f.key" type="button"
                            class="badge bg-secondary border-0 filter-chip" @click="removeChip(f.key, i)"
                            :title="'Remove ' + f.label + ' filter'">
                            {{ f.label }}: <span :lang="optionLang(f.key, f.value)">{{ f.display }}</span>
                            <span aria-hidden="true">&times;</span>
                            <span class="visually-hidden">(remove filter)</span>
                        </button>
                        <button v-if="activeFilters.length" type="button" class="btn btn-sm btn-link text-decoration-none p-0"
                            @click="clearAll">clear all<span class="visually-hidden"> filters</span></button>

                        <div class="btn-group btn-group-sm ms-auto" role="radiogroup"
                            aria-label="Which side of the pair to match" title="Which side of the pair to match">
                            <input type="radio" class="btn-check" name="semanticAuthorRole" id="roleSource"
                                value="source" v-model="authorRole">
                            <label class="btn btn-outline-secondary" for="roleSource"><span class="side-mark source"
                                    aria-hidden="true"></span>source</label>
                            <input type="radio" class="btn-check" name="semanticAuthorRole" id="roleTarget"
                                value="target" v-model="authorRole">
                            <label class="btn btn-outline-secondary" for="roleTarget"><span class="side-mark target"
                                    aria-hidden="true"></span>target</label>
                            <input type="radio" class="btn-check" name="semanticAuthorRole" id="roleEither"
                                value="either" v-model="authorRole">
                            <label class="btn btn-outline-secondary" for="roleEither">either</label>
                        </div>
                        <div class="form-check form-check-inline mb-0 ms-2" v-if="mergedCount">
                            <input class="form-check-input" type="checkbox" id="showNoiseToggle"
                                v-model="showNoise">
                            <label class="form-check-label" for="showNoiseToggle">
                                Show {{ mergedCount }} unclustered
                            </label>
                        </div>
                    </div>
                </div>

                <!-- Scatter canvas host -->
                <div class="graph-layout">
                    <div class="graph-main">
                        <div id="sigma-container" ref="sigmaContainer" class="vector-space-bg"
                            :class="{ 'stars-paused': starsPaused }" role="img" :aria-label="mapLabel"
                            style="height: calc(100vh - 300px); min-height: 600px;"></div>

                        <!-- Keyboard alternatives to wheel zoom and drag -->
                        <div class="map-controls btn-group-vertical btn-group-sm" role="group" aria-label="Map view"
                            v-if="graphReady">
                            <button type="button" class="btn btn-light" aria-label="Zoom in" title="Zoom in"
                                @click="zoomBy(ZOOM_STEP)"><span aria-hidden="true">+</span></button>
                            <button type="button" class="btn btn-light" aria-label="Zoom out" title="Zoom out"
                                @click="zoomBy(1 / ZOOM_STEP)"><span aria-hidden="true">&minus;</span></button>
                            <button type="button" class="btn btn-light" @click="resetView">
                                Reset<span class="visually-hidden"> view</span>
                            </button>
                            <button v-if="!reducedMotion" type="button" class="btn btn-light"
                                aria-label="Pause background animation" title="Pause background animation"
                                :aria-pressed="starsPaused ? 'true' : 'false'" @click="starsPaused = !starsPaused">
                                <span aria-hidden="true">{{ starsPaused ? "&#9654;" : "&#10074;&#10074;" }}</span>
                            </button>
                        </div>

                        <!-- Theme legend. Clicking highlights that theme's
                             points and nothing else. -->
                        <div class="theme-legend" ref="themeLegend" v-if="legendThemes.length">
                            <h3 class="theme-legend-title">
                                <button type="button" class="theme-legend-head"
                                    :aria-expanded="legendOpen ? 'true' : 'false'" aria-controls="semantic-theme-legend"
                                    @click="legendOpen = !legendOpen">
                                    <span><strong>Themes</strong> ({{ legendThemes.length }})</span>
                                    <span aria-hidden="true">{{ legendOpen ? '&minus;' : '+' }}</span>
                                </button>
                            </h3>
                            <ul class="theme-legend-body list-unstyled mb-0" id="semantic-theme-legend"
                                v-show="legendOpen">
                                <li v-for="t in legendThemes" :key="t.id">
                                    <button type="button" class="theme-legend-item"
                                        :class="{ 'theme-legend-item-on': highlightedTheme === t.id }"
                                        :aria-pressed="highlightedTheme === t.id ? 'true' : 'false'"
                                        @click="toggleLegendTheme(t.id)"
                                        :title="t.label + ' — ' + t.count + ' passages'">
                                        <span class="legend-color-box" aria-hidden="true"
                                            :style="{ backgroundColor: getClusterColor(t.id) }"></span>
                                        <span class="theme-legend-label" :lang="themeLabelLang(t.id)">{{ t.label
                                            }}</span>
                                        <span class="visually-hidden">, {{ t.count.toLocaleString() }} passages</span>
                                    </button>
                                </li>
                            </ul>
                        </div>
                    </div>

                    <!-- One sidebar for everything: themes, authors, and the
                         readout for whatever is hovered or selected. -->
                    <div class="graph-sidebar card shadow-lg">
                        <h3 class="visually-hidden">Filters</h3>
                        <!-- Readout: hover wins, then selection. Stays until
                             another point is hovered or Escape is pressed. -->
                        <div v-if="hoveredPoint" class="sidebar-readout p-2 border-bottom">
                            <div class="d-flex align-items-center">
                                <div class="legend-color-box" aria-hidden="true"
                                    :style="{ backgroundColor: getClusterColor(hoveredPoint.theme) }"></div>
                                <span class="ms-2" :lang="themeLabelLang(hoveredPoint.theme)"><strong>{{
                                    hoveredPoint.themeLabel }}</strong></span>
                            </div>
                            <div class="text-truncate">{{ hoveredPoint.sourceAuthor || '—' }}</div>
                            <div class="text-truncate readout-secondary">&harr; {{ hoveredPoint.targetAuthor
                                || '—' }}</div>
                            <div v-if="hoveredPoint.merged" class="readout-warning">unclustered</div>
                        </div>

                        <!-- One dimension at a time. Each row is a filter and
                             its own proportion bar; the bar is the share of
                             what is currently in view. -->
                        <div class="sidebar-dim-picker p-2 border-bottom">
                            <div class="btn-group btn-group-sm w-100 mb-2" role="group" aria-label="Browse by">
                                <button v-for="d in filterDimensions" :key="d.key" type="button"
                                    class="btn btn-outline-secondary"
                                    :class="{ active: activeDimension === d.key }"
                                    :aria-pressed="activeDimension === d.key ? 'true' : 'false'"
                                    aria-controls="semantic-dim-list" @click="setDimension(d.key)">
                                    {{ d.label }}
                                    <template v-if="filters[d.key] !== null && filters[d.key] !== undefined">
                                        <span aria-hidden="true">&nbsp;&#9679;</span>
                                        <span class="visually-hidden">(filtered)</span>
                                    </template>
                                </button>
                            </div>
                            <label for="semantic-dim-search" class="visually-hidden">{{ searchPlaceholder }}</label>
                            <input type="search" id="semantic-dim-search" class="form-control form-control-sm"
                                :placeholder="searchPlaceholder" v-model="dimensionQuery">
                        </div>

                        <div class="sidebar-body p-2" id="semantic-dim-list" ref="sidebarBody">
                            <div v-if="activeDimensionFiltered" class="text-muted" ref="dimSummary" tabindex="-1">
                                Filtered to <strong :lang="optionLang(activeDimension, filters[activeDimension])">{{
                                    activeDimensionFilterLabel }}</strong>.
                                <button type="button" class="btn btn-sm btn-outline-secondary py-0 px-1 align-baseline"
                                    @click="clearActiveDimension">Clear</button>
                                to see the spread again.
                            </div>
                            <template v-else>
                                <div class="d-flex justify-content-between align-items-baseline mb-1" ref="dimSummary"
                                    tabindex="-1">
                                    <span class="text-muted">{{ activeDimensionOptions.length.toLocaleString() }}
                                        {{ activeDimensionNoun(activeDimensionOptions.length) }}
                                        in view</span>
                                    <span class="text-muted">share of view</span>
                                </div>
                                <ul class="list-unstyled mb-0">
                                    <li v-for="o in activeDimensionOptions.slice(0, dimensionLimit)" :key="o.value">
                                        <button type="button" class="dim-row" @click="pickOption(o.value)"
                                            @mouseenter="hoveredOption = o.value" @mouseleave="hoveredOption = null"
                                            @focus="hoveredOption = o.value" @blur="hoveredOption = null"
                                            :title="o.label">
                                            <span class="dim-fill" aria-hidden="true"
                                                :style="{ width: (o.shareOfContext * 100) + '%', backgroundColor: activeDimension === 'theme' ? getClusterColor(o.value) : '#d9c9bd' }">
                                            </span>
                                            <span class="dim-row-body">
                                                <span v-if="activeDimension === 'theme'" class="legend-color-box me-1"
                                                    aria-hidden="true"
                                                    :style="{ backgroundColor: getClusterColor(o.value) }"></span>
                                                <span class="dim-label" :lang="optionLang(activeDimension, o.value)">{{
                                                    o.label }}</span>
                                                <span class="dim-metric" aria-hidden="true"
                                                    v-if="hoveredOption === o.value">
                                                    {{ Math.round(o.shareOfValue * 100) }}% of its own total
                                                </span>
                                                <span class="dim-metric" aria-hidden="true" v-else>
                                                    {{ o.count.toLocaleString() }} &middot;
                                                    {{ Math.round(o.shareOfContext * 100) }}%
                                                </span>
                                                <span class="visually-hidden">: {{ o.count.toLocaleString() }} passages,
                                                    {{ Math.round(o.shareOfContext * 100) }}% of view,
                                                    {{ Math.round(o.shareOfValue * 100) }}% of its own total</span>
                                            </span>
                                        </button>
                                    </li>
                                </ul>
                                <button v-if="activeDimensionOptions.length > dimensionLimit" type="button"
                                    class="btn btn-sm btn-link text-decoration-none p-0" @click="showMoreOptions">
                                    show more<span class="visually-hidden"> {{ activeDimensionNoun(2) }}</span>
                                </button>
                                <div v-if="!activeDimensionOptions.length" class="text-muted">
                                    {{ dimensionQuery ? 'No match in view.' : 'Nothing in view.' }}
                                </div>
                            </template>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Text equivalent of the map, and the keyboard route to a pair -->
            <section class="mt-3 mb-4" aria-labelledby="semantic-passages-heading" v-if="rawData.points">
                <div class="d-flex flex-wrap align-items-center gap-2 mb-2">
                    <h3 class="h6 mb-0" id="semantic-passages-heading">Passages</h3>
                    <button type="button" class="btn btn-sm btn-outline-secondary"
                        :aria-expanded="passageListOpen ? 'true' : 'false'" aria-controls="semantic-passage-list"
                        @click="passageListOpen = !passageListOpen">
                        {{ passageListOpen ? "Hide" : "Show" }} passage table
                    </button>
                </div>
                <div id="semantic-passage-list" v-show="passageListOpen" ref="passageList">
                    <div class="table-responsive position-relative">
                        <table class="table table-sm align-middle caption-top mb-2">
                            <caption>
                                Showing {{ passageRows.length.toLocaleString() }} of
                                {{ passageScope.indices.length.toLocaleString() }} passages
                                <template v-if="passageScope.kind === 'lasso'">selected on the map</template>
                                <template v-else-if="passageScope.kind === 'theme'">
                                    in the theme <span :lang="themeLabelLang(passageScope.theme)">{{
                                        facetValueName("theme", passageScope.theme) }}</span>
                                </template>
                                <template v-else-if="passageScope.kind === 'filters'">matching the filters</template>
                            </caption>
                            <thead>
                                <tr>
                                    <th scope="col">Theme</th>
                                    <th scope="col">{{ sourceLabel }}</th>
                                    <th scope="col">{{ targetLabel }}</th>
                                    <th scope="col"><span class="visually-hidden">Passage pair</span></th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr v-for="row in passageRows" :key="row.index">
                                    <td>
                                        <span :lang="themeLabelLang(row.theme)">{{ row.themeLabel }}</span>
                                        <span v-if="row.merged" class="text-muted"> (unclustered)</span>
                                    </td>
                                    <td>
                                        {{ row.sourceAuthor || "—" }}<template v-if="row.sourceTitle">,
                                            <i :lang="sourceLang">{{ row.sourceTitle }}</i></template>
                                    </td>
                                    <td>
                                        {{ row.targetAuthor || "—" }}<template v-if="row.targetTitle">,
                                            <i :lang="targetLang">{{ row.targetTitle }}</i></template>
                                    </td>
                                    <td>
                                        <button type="button"
                                            class="btn btn-sm btn-outline-secondary text-nowrap passage-pair-button"
                                            @click="showAlignment(row.index)">
                                            View pair<span class="visually-hidden">: {{ row.sourceAuthor }} and
                                                {{ row.targetAuthor }}</span>
                                        </button>
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                    <button v-if="passageScope.indices.length > passageLimit" type="button"
                        class="btn btn-sm btn-link text-decoration-none p-0" @click="showMorePassages">show more passages</button>
                </div>
            </section>

            <!-- Full alignment for a clicked passage -->
            <div class="modal modal-xl modal-dialog-scrollable fade" id="semantic-passage-pair" ref="modalElement"
                tabindex="-1" aria-labelledby="semantic-passage-pair-title" aria-hidden="true">
                <div class="modal-dialog">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h2 class="modal-title fs-5" id="semantic-passage-pair-title">Alignment</h2>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"
                                aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                            <div v-if="alignmentLoading" class="text-center p-3">
                                <div class="spinner-border spinner-border-sm" role="status">
                                    <span class="visually-hidden">Loading...</span>
                                </div>
                            </div>
                            <div v-else-if="alignmentError" class="alert alert-warning mb-0">
                                {{ alignmentError }}
                            </div>
                            <passage-pair v-else-if="clickedAlignment" :alignment="clickedAlignment" :index="0"
                                :diffed="false" :heading-level="3"></passage-pair>
                        </div>
                    </div>
                </div>
            </div>

        </div>
    </div>
</template>

<script setup>
import axios from "axios";
import { Modal } from "bootstrap";
import createScatterplot from "regl-scatterplot";
import emitter from "tiny-emitter/instance";
import { computed, nextTick, onBeforeUnmount, onMounted, ref, shallowRef, watch } from "vue";
import { useRoute } from "vue-router";
import globalConfig from "../config.js";
import { paramsToUrl, plainText, prefersReducedMotion } from "../utils.js";
import passagePair from "./passagePair";
import reportSwitcher from "./reportSwitcher";
import searchArguments from "./searchArguments";

// Opacity of points outside the highlight, and the smallest area the camera
// will frame, in data units (the layout spans [-1, 1]).
const DIM_OPACITY = 0.2;
const MIN_ZOOM_SPAN = 0.05;
const ZOOM_PADDING = 0.06;
const ZOOM_STEP = 1.5;
const PAGE_SIZE = 50;

const themeLang = globalConfig.themeLanguage || undefined;
const sourceLang = globalConfig.sourceLanguage || undefined;
const targetLang = globalConfig.targetLanguage || undefined;
const sourceLabel = plainText(globalConfig.sourceLabel) || "Source";
const targetLabel = plainText(globalConfig.targetLabel) || "Target";

const route = useRoute();
const viewName = route.name;

const loading = ref(false);
const loadingMessage = ref("Fetching data...");
const error = ref(null);
// Frozen payload of large arrays: replaced whole, never proxied.
const rawData = shallowRef({});

// UI state
const hoveredPoint = ref(null);
const selectedPoints = shallowRef([]);
const clickedAlignment = shallowRef(null);
const alignmentLoading = ref(false);
const alignmentError = ref(null);
// One filter per dimension, combined with AND. null means unset.
// theme is always available; the rest come from rawData.facets.
const filters = ref({ theme: null });
// Which dimension the sidebar is browsing. Author by default: the
// question this view exists for is who is being reused, and in what.
const activeDimension = ref("author");
const dimensionQuery = ref("");
// Theme highlighted from the legend. Independent of the filters --
// the map shows one highlight, and the last thing touched wins.
const highlightedTheme = ref(null);
const hoveredOption = ref(null);
const legendOpen = ref(true);
const dimensionLimit = ref(PAGE_SIZE);
// Which side of the alignment a facet filter applies to.
const authorRole = ref("source");
// Off by default. HDBSCAN rejected these points as noise; the theme
// on them is a nearest-centroid assignment, not a density finding.
const showNoise = ref(false);
const graphReady = ref(false);
const starsPaused = ref(false);
const reducedMotion = prefersReducedMotion();
const passageListOpen = ref(false);
const passageLimit = ref(PAGE_SIZE);

// Template refs
const root = ref(null);
const sigmaContainer = ref(null);
const themeLegend = ref(null);
const modalElement = ref(null);
const filterBar = ref(null);
const selectionSummary = ref(null);
const sidebarBody = ref(null);
const dimSummary = ref(null);
const passageList = ref(null);

// Renderer state, kept out of Vue's reactivity
let scatterplot = null;
let pointThemes = null;
let noiseCategory = null;
let dimOffset = 0;
let painting = false;
let pendingPaint = null;
let alignmentModal = null;
let modalOpener = null;
let modalOpen = false;
let unmounted = false;
let queryTimer = null;
let loadedTimer = null;
const clusterColorMap = new Map();

const mergedCount = computed(() => rawData.value?.metadata?.merged_in || 0);

const clusterLabels = computed(() => {
    const labels = {};
    for (const t of rawData.value?.themes || []) labels[t.id] = t.label || "";
    return labels;
});

// Themes the current selection touches, not the payload total.
const visibleClusters = computed(() => {
    const pts = rawData.value?.points;
    if (!pts || !pts.theme) return 0;
    if (!activeFilters.value.length) return (rawData.value.themes || []).length;
    const seen = new Set();
    for (const i of filteredIndices.value) seen.add(pts.theme[i]);
    return seen.size;
});

const availableFacets = computed(() => rawData.value?.facets || [{ key: "author", label: "Author", names: null }]);

// Dimensions the user can filter on: theme plus whatever facets the
// payload carries. Role applies to the facet dimensions only -- a theme
// belongs to the passage, not to one side of it.
const filterDimensions = computed(() => {
    const dims = [{ key: "theme", label: "Theme" }];
    for (const f of availableFacets.value) {
        dims.push({ key: f.key, label: f.label });
    }
    return dims;
});

// Dimensions with no value chosen. Once a dimension is filtered its
// spread describes passages outside the selection, so it gets no
// breakdown.
const openDimensions = computed(() =>
    filterDimensions.value.filter((d) => filters.value[d.key] === null || filters.value[d.key] === undefined)
);

const activeDimensionLabel = computed(() => {
    const d = filterDimensions.value.find((x) => x.key === activeDimension.value);
    return d ? d.label : "";
});

const searchPlaceholder = computed(() => "Search " + activeDimensionLabel.value.toLowerCase() + "s");

const activeDimensionFiltered = computed(() => {
    const v = filters.value[activeDimension.value];
    return v !== null && v !== undefined;
});

const activeDimensionFilterLabel = computed(() =>
    facetValueName(activeDimension.value, filters.value[activeDimension.value])
);

const activeDimensionOptions = computed(() => {
    if (activeDimensionFiltered.value) return [];
    const options = optionsFor(activeDimension.value);
    const q = dimensionQuery.value.trim().toLowerCase();
    return q ? options.filter((o) => o.label.toLowerCase().includes(q)) : options;
});

// Legend covers every theme in the payload, largest first, so the
// colours stay stable as filters change.
const legendThemes = computed(() =>
    [...(rawData.value?.themes || [])]
        .map((t) => ({ id: t.id, label: t.label || `Theme ${t.id}`, count: t.count }))
        .sort((a, b) => b.count - a.count)
);

const activeFilters = computed(() =>
    filterDimensions.value
        .filter((d) => filters.value[d.key] !== null && filters.value[d.key] !== undefined)
        .map((d) => ({
            key: d.key,
            label: d.label,
            value: filters.value[d.key],
            display: d.key === "theme"
                ? (clusterLabels.value[filters.value[d.key]] || `Theme ${filters.value[d.key]}`)
                : facetValueName(d.key, filters.value[d.key]),
        }))
);

// Point indices satisfying every active filter. The basis for both the
// map highlight and every count shown in the sidebar.
const filteredIndices = computed(() => {
    const pts = rawData.value?.points;
    if (!pts || !pts.theme) return [];
    const active = activeFilters.value;
    const out = [];
    for (let i = 0; i < pts.theme.length; i++) {
        if (!showNoise.value && pts.merged[i] === 1) continue;
        let ok = true;
        for (const f of active) {
            if (f.key === "theme") {
                if (pts.theme[i] !== f.value) { ok = false; break; }
            } else if (!matchesFacet(pts, f, i)) {
                ok = false;
                break;
            }
        }
        if (ok) out.push(i);
    }
    return out;
});

// Totals, which stay visible at every level of drill-down.
const selectionTotals = computed(() => {
    const pts = rawData.value?.points;
    const universe = pts && pts.theme
        ? (showNoise.value ? pts.theme.length : pts.merged.filter((m) => m === 0).length)
        : 0;
    const n = activeFilters.value.length ? filteredIndices.value.length : universe;
    return { selected: n, universe, share: universe ? n / universe : 0 };
});

// What the passage table lists: the lasso, else the lit theme, else the
// filters, else everything in view.
const passageScope = computed(() => {
    const pts = rawData.value?.points;
    if (!pts || !pts.theme) return { kind: "all", indices: [] };
    if (selectedPoints.value.length > 1) return { kind: "lasso", indices: selectedPoints.value };
    if (highlightedTheme.value !== null) {
        return { kind: "theme", theme: highlightedTheme.value, indices: themeIndices(highlightedTheme.value) };
    }
    if (activeFilters.value.length) return { kind: "filters", indices: filteredIndices.value };
    const all = [];
    for (let i = 0; i < pts.theme.length; i++) {
        if (showNoise.value || pts.merged[i] !== 1) all.push(i);
    }
    return { kind: "all", indices: all };
});

const passageRows = computed(() => {
    const pts = rawData.value?.points;
    if (!pts || !pts.theme) return [];
    const names = rawData.value.author_names || {};
    return passageScope.value.indices.slice(0, passageLimit.value).map((i) => ({
        index: i,
        theme: pts.theme[i],
        themeLabel: facetValueName("theme", pts.theme[i]),
        merged: pts.merged[i] === 1,
        sourceAuthor: names[pts.source_author[i]] || "",
        targetAuthor: names[pts.target_author[i]] || "",
        sourceTitle: pts.source_title ? facetValueName("title", pts.source_title[i]) : "",
        targetTitle: pts.target_title ? facetValueName("title", pts.target_title[i]) : "",
    }));
});

const mapLabel = computed(() => {
    if (error.value) return "Semantic map unavailable.";
    if (!rawData.value?.points) return "Semantic map, loading.";
    const totals = selectionTotals.value;
    let label = `Semantic map of ${totals.universe.toLocaleString()} passages in ${legendThemes.value.length} themes, placed by similarity of content.`;
    if (selectedPoints.value.length > 1) {
        label += ` ${selectedPoints.value.length.toLocaleString()} passages selected.`;
    } else if (highlightedTheme.value !== null) {
        label += ` Highlighted theme: ${facetValueName("theme", highlightedTheme.value)}.`;
    } else if (activeFilters.value.length) {
        label += ` Highlighted: ${totals.selected.toLocaleString()} passages matching the filters.`;
    }
    return label + " The themes list and the passage table give the same information.";
});

function matchesFacet(pts, f, i) {
    const s = pts[`source_${f.key}`];
    const t = pts[`target_${f.key}`];
    return (authorRole.value === "source" && s && s[i] === f.value)
        || (authorRole.value === "target" && t && t[i] === f.value)
        || (authorRole.value === "either" && ((s && s[i] === f.value) || (t && t[i] === f.value)));
}

function themeIndices(id) {
    const pts = rawData.value?.points;
    const out = [];
    if (!pts || !pts.theme) return out;
    for (let i = 0; i < pts.theme.length; i++) {
        if (!showNoise.value && pts.merged[i] === 1) continue;
        if (pts.theme[i] === id) out.push(i);
    }
    return out;
}

// Generated labels are in the theme language; the fallback is English.
function themeLabelLang(id) {
    return clusterLabels.value[id] ? themeLang : undefined;
}

// Language of a facet value: theme labels, or corpus titles on the side
// being matched. Author names are proper names and get none.
function optionLang(key, value) {
    if (key === "theme") return themeLabelLang(value);
    if (key === "author") return undefined;
    if (authorRole.value === "source") return sourceLang;
    if (authorRole.value === "target") return targetLang;
    return sourceLang === targetLang ? sourceLang : undefined;
}

function activeDimensionNoun(n) {
    return activeDimensionLabel.value.toLowerCase() + (n === 1 ? "" : "s");
}

function hslToHex(h, s, l) {
    const sN = s / 100;
    const lN = l / 100;
    const k = (n) => (n + h / 30) % 12;
    const a = sN * Math.min(lN, 1 - lN);
    const f = (n) => {
        const v = lN - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
        return Math.round(255 * v).toString(16).padStart(2, "0");
    };
    return `#${f(0)}${f(8)}${f(4)}`;
}

function cameraTransition(duration) {
    return prefersReducedMotion() ? {} : { transition: true, transitionDuration: duration };
}

function fetchSemanticData() {
    loading.value = true;
    loadingMessage.value = "Fetching semantic graph data...";
    error.value = null;
    hoveredPoint.value = null;
    graphReady.value = false;
    clearTimeout(loadedTimer);

    // Destroy existing renderer and graph
    if (scatterplot) {
        scatterplot.destroy();
        scatterplot = null;
    }

    let params = { ...route.query };
    params.db_table = globalConfig.databaseName;

    emitter.emit("searchArgsUpdate", {
        counts: "",
        searchParams: params,
    });

    axios
        .get(`${globalConfig.apiServer}/semantic_scatter_data/?${paramsToUrl(params)}`)
        .then((response) => {
            if (unmounted) return;
            if (response.data.error) {
                error.value = response.data.error;
                loading.value = false;
                emitter.emit("announce", `Semantic graph error: ${response.data.error}`);
                return;
            }

            rawData.value = Object.freeze(response.data);

            loadingMessage.value = "Initializing graph...";

            // A corpus may carry no author facet at all.
            if (!filterDimensions.value.some((d) => d.key === activeDimension.value)) {
                activeDimension.value = filterDimensions.value[0].key;
            }
            highlightedTheme.value = null;

            // Use setTimeout to allow the UI to update
            setTimeout(() => {
                if (unmounted) return;
                initializeGraph();
                loading.value = false;
                graphReady.value = true;
            }, 10);

            emitter.emit("searchArgsUpdate", {
                counts: response.data.metadata?.total_points || 0,
                searchParams: params,
            });
            // Later than the result count searchArguments announces
            loadedTimer = setTimeout(() => {
                emitter.emit("announce", `Semantic graph ready: ${legendThemes.value.length} themes.`);
            }, 1500);
        })
        .catch((err) => {
            if (unmounted) return;
            loading.value = false;
            error.value = err.toString();
            emitter.emit("announce", `Semantic graph error: ${error.value}`);
            console.log(err);
        });
}

function getClusterColor(clusterId) {
    // Check if this is a mini-cluster (singleton/noise cluster)
    // Mini-clusters have IDs >= n_clusters (the real clusters)
    if (rawData.value?.points && clusterId >= (rawData.value.metadata?.n_themes || 0)) {
        return '#a39a90';  // Warm grey for mini-clusters
    }

    // Check if we already assigned a color to this cluster
    if (clusterColorMap.has(clusterId)) {
        return clusterColorMap.get(clusterId);
    }

    // Golden-angle hue spacing, so colours stay separated for any
    // number of clusters. Hex, not hsl(), which the renderer rejects.
    const index = clusterColorMap.size;
    // Deep tones for the paper map; yellows and greens, which read lighter, go darker
    const hue = (index * 137.508) % 360;
    const light = 36 + (index % 3) * 7 - (hue > 40 && hue < 200 ? 12 : 0);
    const color = hslToHex(hue, 58 + (index % 2) * 14, light);
    clusterColorMap.set(clusterId, color);
    return color;
}

function initializeGraph() {
    // regl-scatterplot, not a graph renderer: this view needs points,
    // colour, hover, zoom and lasso -- no node/edge model.
    const pts = rawData.value.points;
    const themes = rawData.value.themes || [];
    const themeIds = themes.map((t) => t.id);

    // Colour is a per-point categorical encoding: `z` carries the theme
    // id and pointColor maps it. Noise gets an extra slot at the end.
    const maxTheme = themeIds.length ? Math.max(...themeIds) : 0;
    const vivid = [];
    for (let i = 0; i <= maxTheme; i++) vivid.push(getClusterColor(i));
    noiseCategory = maxTheme + 1;
    vivid.push("#c9c0b4");
    // Second half of the palette is the same colours desaturated, for
    // points outside the current highlight.
    dimOffset = vivid.length;
    const palette = [...vivid, ...vivid.map((c) => desaturate(c))];

    // Noise is given its own category rather than its theme's, so it is
    // never coloured as if HDBSCAN had placed it there.
    const z = new Float32Array(pts.theme.length);
    for (let i = 0; i < z.length; i++) {
        z[i] = pts.merged[i] === 1 ? noiseCategory : pts.theme[i];
    }

    if (scatterplot) scatterplot.destroy();
    const host = sigmaContainer.value;
    const canvas = host.querySelector("canvas") || document.createElement("canvas");
    if (!canvas.parentNode) {
        canvas.style.width = "100%";
        canvas.style.height = "100%";
        host.appendChild(canvas);
    }

    scatterplot = createScatterplot({
        canvas,
        width: "auto",
        height: "auto",
        pointSize: 2.4,
        pointSizeSelected: 4,
        pointColor: palette,
        opacityBy: "density",
        lassoOnLongPress: true,
        // Escape is handled below, so closing the modal keeps the lasso
        deselectOnEscape: false,
        backgroundColor: [0.976, 0.965, 0.945, 1],
        xScale: null,
        yScale: null,
    });
    scatterplot.set({ colorBy: "z" });

    scatterplot.subscribe("pointOver", (i) => describePoint(i));
    scatterplot.subscribe("select", ({ points }) => {
        selectedPoints.value = points || [];
        // A single point is a click; a lasso hands back many, and should
        // not open a modal.
        if (selectedPoints.value.length === 1) {
            showAlignment(selectedPoints.value[0]);
        } else if (selectedPoints.value.length > 1) {
            emitter.emit("announce", `${selectedPoints.value.length.toLocaleString()} passages selected on the map.`);
        }
    });
    scatterplot.subscribe("deselect", () => { selectedPoints.value = []; });

    pointThemes = z;
    paintHighlight(null);
}

// Mix a colour most of the way to its own luminance, so a dimmed point
// still hints at its theme without reading as one.
function desaturate(hex) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    const grey = 0.3 * r + 0.59 * g + 0.11 * b;
    const f = (c) => Math.round(0.9 * (0.25 * c + 0.75 * grey))
        .toString(16).padStart(2, "0");
    return `#${f(r)}${f(g)}${f(b)}`;
}

// Values of a facet on the side(s) being matched, for one point.
function facetIds(pts, key, i) {
    const source = pts[`source_${key}`] && pts[`source_${key}`][i];
    const target = pts[`target_${key}`] && pts[`target_${key}`][i];
    if (authorRole.value === "target") return [target];
    if (authorRole.value === "either") return [source, target];
    return [source];
}

// Values for one dimension, counted under every *other* active filter.
function optionsFor(key) {
    const pts = rawData.value?.points;
    if (!pts || !pts.theme) return [];
    const others = activeFilters.value.filter((f) => f.key !== key);
    const counts = new Map();
    let inContext = 0;
    for (let i = 0; i < pts.theme.length; i++) {
        if (!showNoise.value && pts.merged[i] === 1) continue;
        let ok = true;
        for (const f of others) {
            if (f.key === "theme") {
                if (pts.theme[i] !== f.value) { ok = false; break; }
            } else if (!matchesFacet(pts, f, i)) {
                ok = false;
                break;
            }
        }
        if (!ok) continue;
        inContext += 1;
        if (key === "theme") {
            counts.set(pts.theme[i], (counts.get(pts.theme[i]) || 0) + 1);
        } else {
            for (const id of new Set(facetIds(pts, key, i))) {
                if (id === undefined || id === null || id < 0) continue;
                if (!facetValueName(key, id)) continue;
                counts.set(id, (counts.get(id) || 0) + 1);
            }
        }
    }
    // total for each value across the whole corpus, for the second
    // proportion -- "how much of this author is in view"
    const overall = overallCounts(key);
    return [...counts.entries()]
        .map(([value, count]) => ({
            value,
            label: key === "theme"
                ? (clusterLabels.value[value] || `Theme ${value}`)
                : facetValueName(key, value),
            count,
            shareOfContext: inContext ? count / inContext : 0,
            shareOfValue: overall.get(value) ? count / overall.get(value) : 0,
        }))
        .sort((a, b) => b.count - a.count);
}

function overallCounts(key) {
    const pts = rawData.value?.points;
    const out = new Map();
    if (!pts || !pts.theme) return out;
    for (let i = 0; i < pts.theme.length; i++) {
        if (!showNoise.value && pts.merged[i] === 1) continue;
        if (key === "theme") {
            out.set(pts.theme[i], (out.get(pts.theme[i]) || 0) + 1);
        } else {
            for (const id of new Set(facetIds(pts, key, i))) {
                if (id === undefined || id === null || id < 0) continue;
                out.set(id, (out.get(id) || 0) + 1);
            }
        }
    }
    return out;
}

function facetValueName(key, id) {
    if (key === "theme") return clusterLabels.value[id] || `Theme ${id}`;
    if (key === "author") {
        const names = rawData.value?.author_names || {};
        return names[String(id)] || "";
    }
    const facet = (rawData.value?.facets || []).find((f) => f.key === key);
    if (!facet || !facet.names) return "";
    return facet.names[id] || "";
}

function setDimension(key) {
    activeDimension.value = key;
    dimensionQuery.value = "";
    dimensionLimit.value = PAGE_SIZE;
}

function setFilter(key, value) {
    highlightedTheme.value = null;
    filters.value = { ...filters.value, [key]: value === "" || value === null ? null : value };
    // Drill down: move to the next unfiltered dimension so the sidebar
    // shows what is inside the selection just made.
    if (key === activeDimension.value && filters.value[key] !== null) {
        const next = openDimensions.value[0];
        if (next) setDimension(next.key);
    }
    applySelection();
}

function clearFilter(key) {
    setFilter(key, null);
}

function clearAllFilters() {
    highlightedTheme.value = null;
    const cleared = {};
    for (const d of filterDimensions.value) cleared[d.key] = null;
    filters.value = cleared;
    applySelection();
}

function announceSelection() {
    const totals = selectionTotals.value;
    emitter.emit("announce",
        `${totals.selected.toLocaleString()} of ${totals.universe.toLocaleString()} passages in ${visibleClusters.value} themes.`);
}

function focusLater(target) {
    nextTick(() => target.value?.focus());
}

// Sidebar and chip handlers: the control used is re-rendered away, so
// focus moves to what replaced it.
function pickOption(value) {
    setFilter(activeDimension.value, value);
    announceSelection();
    focusLater(dimSummary);
}

function clearActiveDimension() {
    clearFilter(activeDimension.value);
    announceSelection();
    focusLater(dimSummary);
}

function removeChip(key, position) {
    clearFilter(key);
    announceSelection();
    nextTick(() => {
        const chips = filterBar.value ? filterBar.value.querySelectorAll(".filter-chip") : [];
        const next = chips[Math.min(position, chips.length - 1)];
        (next || selectionSummary.value)?.focus();
    });
}

function clearAll() {
    clearAllFilters();
    announceSelection();
    focusLater(selectionSummary);
}

function showMoreOptions() {
    const first = dimensionLimit.value;
    dimensionLimit.value += PAGE_SIZE;
    nextTick(() => sidebarBody.value?.querySelectorAll(".dim-row")[first]?.focus());
}

function showMorePassages() {
    const first = passageLimit.value;
    passageLimit.value += PAGE_SIZE;
    nextTick(() => passageList.value?.querySelectorAll(".passage-pair-button")[first]?.focus());
}

// One highlight at a time: a legend theme if one is lit, otherwise the
// filter intersection, otherwise nothing.
function applySelection() {
    if (!scatterplot) return;
    let idxs = null;
    const pts = rawData.value?.points;
    if (highlightedTheme.value !== null && pts && pts.theme) {
        idxs = themeIndices(highlightedTheme.value);
    } else if (activeFilters.value.length) {
        idxs = filteredIndices.value;
    }
    scatterplot.deselect();
    paintHighlight(idxs);
    if (idxs === null) scatterplot.zoomToOrigin(cameraTransition(400));
    else if (idxs.length) frameSelection(idxs);
}

// Highlighted points keep their colour at full opacity and a larger
// size; everything else is desaturated and faded but still there, so
// the selection reads against the rest of the map.
function paintHighlight(idxs) {
    const pts = rawData.value?.points;
    if (!scatterplot || !pts || !pointThemes) return;
    if (painting) {
        pendingPaint = { idxs };
        return;
    }
    const n = pointThemes.length;
    const z = new Float32Array(n);
    const w = new Float32Array(n);
    if (idxs) {
        const on = new Uint8Array(n);
        for (const i of idxs) on[i] = 1;
        for (let i = 0; i < n; i++) {
            z[i] = on[i] ? pointThemes[i] : pointThemes[i] + dimOffset;
            w[i] = on[i];
        }
        scatterplot.set({
            opacityBy: "valueW",
            sizeBy: "valueW",
            opacity: [DIM_OPACITY, 1],
            pointSize: [1.8, 4],
        });
    } else {
        z.set(pointThemes);
        scatterplot.set({ opacityBy: "density", sizeBy: null, pointSize: 2.4 });
    }
    painting = true;
    scatterplot
        .draw({ x: pts.x, y: pts.y, z: Array.from(z), w: Array.from(w) },
            { preventFilterReset: true, zDataType: "categorical", wDataType: "categorical" })
        .then(() => applyNoiseFilter())
        .catch(() => { })
        .then(() => {
            painting = false;
            const pending = pendingPaint;
            pendingPaint = null;
            if (pending) paintHighlight(pending.idxs);
        });
}

// Pan and zoom onto the highlight, as tight as the highlight allows.
function frameSelection(idxs) {
    const pts = rawData.value.points;
    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (const i of idxs) {
        if (pts.x[i] < xMin) xMin = pts.x[i];
        if (pts.x[i] > xMax) xMax = pts.x[i];
        if (pts.y[i] < yMin) yMin = pts.y[i];
        if (pts.y[i] > yMax) yMax = pts.y[i];
    }
    // The real bounding box, not a square: the viewport is wide, so a
    // square leaves the short axis mostly empty. Floors keep a single
    // point or a straight line framable.
    let width = Math.max(xMax - xMin, MIN_ZOOM_SPAN) * (1 + ZOOM_PADDING);
    const height = Math.max(yMax - yMin, MIN_ZOOM_SPAN) * (1 + ZOOM_PADDING);
    const x = (xMin + xMax) / 2 - width / 2;
    const y = (yMin + yMax) / 2 - height / 2;
    // Grow rightwards by the strip the legend covers, so framed points
    // land clear of it rather than underneath.
    const reserved = legendWidthFraction();
    if (reserved > 0) width += (width * reserved) / (1 - reserved);
    scatterplot.zoomToArea({ x, y, width, height }, cameraTransition(500));
}

function legendWidthFraction() {
    const host = sigmaContainer.value;
    const legend = themeLegend.value;
    if (!host || !legend) return 0;
    // On narrow screens the legend sits below the map, not over it.
    if (getComputedStyle(legend).position !== "absolute") return 0;
    const canvasBox = host.getBoundingClientRect();
    const legendBox = legend.getBoundingClientRect();
    if (!canvasBox.width || !legendBox.width) return 0;
    return Math.min(0.3, (legendBox.width + 16) / canvasBox.width);
}

function zoomBy(factor) {
    if (!scatterplot) return;
    const target = [...scatterplot.get("cameraTarget")];
    scatterplot.zoomToLocation(target, scatterplot.get("cameraDistance") / factor, cameraTransition(250));
}

function resetView() {
    if (scatterplot) scatterplot.zoomToOrigin(cameraTransition(400));
}

function toggleLegendTheme(id) {
    highlightedTheme.value = highlightedTheme.value === id ? null : id;
    applySelection();
    if (highlightedTheme.value === null) {
        emitter.emit("announce", "Theme highlight cleared.");
    } else {
        const n = themeIndices(id).length;
        emitter.emit("announce", `${facetValueName("theme", id)} highlighted: ${n.toLocaleString()} passages.`);
    }
}

function applyNoiseFilter() {
    // Only the noise toggle. Metadata filters are shown as a selection
    // highlight instead of hiding points, so the filtered set stays
    // legible against the rest of the map.
    if (!scatterplot || !pointThemes) return;
    if (showNoise.value) return scatterplot.unfilter();
    const keep = [];
    for (let i = 0; i < pointThemes.length; i++) {
        if (pointThemes[i] !== noiseCategory) keep.push(i);
    }
    return scatterplot.filter(keep);
}

function getModal() {
    if (!alignmentModal) {
        alignmentModal = new Modal(modalElement.value);
        modalElement.value.addEventListener("show.bs.modal", () => { modalOpen = true; });
        modalElement.value.addEventListener("hidden.bs.modal", () => {
            modalOpen = false;
            if (modalOpener && modalOpener.isConnected) modalOpener.focus();
            modalOpener = null;
        });
    }
    return alignmentModal;
}

function showAlignment(index) {
    const pts = rawData.value.points;
    const rowid = pts && pts.rowid ? pts.rowid[index] : null;
    if (rowid === null || rowid === undefined) return;
    clickedAlignment.value = null;
    alignmentError.value = null;
    alignmentLoading.value = true;
    const active = document.activeElement;
    modalOpener = active && active !== document.body ? active : null;
    getModal().show();
    axios
        .get(`${globalConfig.apiServer}/alignment_by_rowid/`, {
            params: { db_table: globalConfig.databaseName, rowid },
        })
        .then((response) => {
            alignmentLoading.value = false;
            if (response.data.error) {
                alignmentError.value = response.data.error;
                emitter.emit("announce", `Passage pair error: ${response.data.error}`);
                return;
            }
            clickedAlignment.value = response.data.alignment;
            emitter.emit("announce", "Passage pair loaded.");
        })
        .catch((err) => {
            alignmentLoading.value = false;
            alignmentError.value = err.toString();
            emitter.emit("announce", `Passage pair error: ${alignmentError.value}`);
        });
}

function describePoint(index) {
    const pts = rawData.value.points;
    const names = rawData.value.author_names || {};
    const themeId = pts.theme[index];
    hoveredPoint.value = {
        index,
        theme: themeId,
        themeLabel: clusterLabels.value[themeId] || `Theme ${themeId}`,
        sourceAuthor: names[pts.source_author[index]] || "",
        targetAuthor: names[pts.target_author[index]] || "",
        merged: pts.merged[index] === 1,
        rowid: pts.rowid[index],
    };
}

// Escape dismisses hover content first, then the lasso selection.
function onKeydown(event) {
    if (event.key !== "Escape" || event.defaultPrevented || modalOpen) return;
    if (hoveredOption.value !== null || hoveredPoint.value) {
        hoveredOption.value = null;
        hoveredPoint.value = null;
        return;
    }
    const active = document.activeElement;
    if (scatterplot && selectedPoints.value.length
        && (active === document.body || root.value?.contains(active))) {
        scatterplot.deselect();
    }
}

watch(() => route.fullPath, () => {
    if (route.name === viewName) fetchSemanticData();
});

watch(showNoise, (shown) => {
    if (!shown && hoveredPoint.value?.merged) hoveredPoint.value = null;
    applySelection();
    announceSelection();
});

watch(authorRole, () => {
    // A facet filter means something different per side, so drop the
    // facet filters and keep the theme, which has no side.
    filters.value = { theme: filters.value.theme ?? null };
    applySelection();
    announceSelection();
});

watch(passageScope, () => {
    passageLimit.value = PAGE_SIZE;
});

watch(dimensionQuery, (query) => {
    clearTimeout(queryTimer);
    if (!query.trim()) return;
    queryTimer = setTimeout(() => {
        const n = activeDimensionOptions.value.length;
        emitter.emit("announce",
            n ? `${n.toLocaleString()} ${activeDimensionNoun(n)} match.` : "No match in view.");
    }, 600);
});

fetchSemanticData();

onMounted(() => {
    document.addEventListener("keydown", onKeydown);
});

onBeforeUnmount(() => {
    unmounted = true;
    document.removeEventListener("keydown", onKeydown);
    clearTimeout(queryTimer);
    clearTimeout(loadedTimer);
    // Leaving from inside the modal (a link in the pair) must not strand
    // its backdrop and focus trap.
    if (alignmentModal && modalOpen) alignmentModal.hide();
    if (scatterplot) {
        scatterplot.destroy();
        scatterplot = null;
    }
});
</script>

<style scoped lang="scss">
@use "../assets/theme.module.scss" as theme;

.node-info-panel {
    position: absolute;
    top: 0px;
    left: 0px;
    width: fit-content;
    max-width: 200px;
    z-index: 10;
    background-color: rgba(256, 256, 256, 0.9);
    border-color: theme.$graph-btn-panel-color;
}

.graph-layout {
    display: flex;
    align-items: stretch;
}

.graph-main {
    flex: 1 1 auto;
    min-width: 0;
    position: relative;
}

.graph-sidebar {
    /* order, rather than moving the markup: the sidebar reads as the plot's
       annotation, so it stays after it in the DOM. */
    order: -1;
    flex: 0 0 380px;
    align-self: stretch;
    border-right: 1px solid theme.$rule-strong;
    display: flex;
    flex-direction: column;
    background-color: #fff;
    border-color: theme.$rule-strong;
    border-radius: 0;
    box-shadow: none !important;
}

.graph-sidebar .nav-tabs {
    flex: 0 0 auto;
}

.sidebar-readout {
    flex: 0 0 auto;
    background-color: rgba(0, 0, 0, 0.03);
}

.readout-secondary {
    color: #5c636a;
}

.readout-warning {
    color: #8a5300;
}

/* The only scrolling region, so the tabs and readout stay put. */
.sidebar-body {
    flex: 1 1 auto;
    overflow-y: auto;
    min-height: 0;
}

/* One row per value: the fill behind the text is its share of the view, so
   the proportion reads without a separate column. */
.dim-row {
    position: relative;
    display: block;
    width: 100%;
    padding: 0;
    border: 0;
    background: none;
    color: inherit;
    font: inherit;
    text-align: left;
    cursor: pointer;
    border-radius: 3px;
    margin-bottom: 1px;
    overflow: hidden;
}

.dim-row:hover {
    outline: 1px solid rgba(0, 0, 0, 0.15);
}

.dim-row:focus-visible {
    z-index: 1;
}

.dim-fill {
    position: absolute;
    top: 0;
    bottom: 0;
    left: 0;
    min-width: 2px;
    opacity: 0.5;
}

.dim-row-body {
    position: relative;
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 6px;
    line-height: 1.25;
}

.dim-label {
    flex: 1 1 auto;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* Full label on keyboard focus, where the title tooltip never shows */
.dim-row:focus-visible .dim-label,
.theme-legend-item:focus-visible .theme-legend-label {
    white-space: normal;
}

.dim-metric {
    flex: 0 0 auto;
    color: #343a40;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
}

.sidebar-dim-picker {
    flex: 0 0 auto;
}

.btn-check:focus-visible + .btn {
    outline: 2px solid theme.$focus-color;
    outline-offset: 2px;
}

/* Legend floats over the plot, top right. Highlight only -- it never filters. */
.theme-legend {
    position: absolute;
    top: 8px;
    right: 8px;
    z-index: 20;
    width: 265px;
    max-height: calc(100% - 24px);
    display: flex;
    flex-direction: column;
    background-color: rgba(255, 255, 255, 0.95);
    border: 1px solid theme.$rule-strong;
    border-top: 3px solid theme.$header-color;
    border-radius: 3px;
    box-shadow: 0 6px 24px -10px rgba(64, 40, 24, 0.4);
}

.theme-legend-title {
    margin: 0;
    font-size: inherit;
    font-weight: normal;
    line-height: inherit;
}

.theme-legend-head {
    display: flex;
    width: 100%;
    justify-content: space-between;
    align-items: center;
    padding: 4px 8px;
    cursor: pointer;
    border: 0;
    border-bottom: 1px solid rgba(0, 0, 0, 0.1);
    background: none;
    color: inherit;
    font: inherit;
    text-align: left;
}

.theme-legend-body {
    overflow-y: auto;
    min-height: 0;
    padding: 6px;
}

.theme-legend-item {
    display: flex;
    width: 100%;
    align-items: center;
    gap: 5px;
    padding: 2px 4px;
    border: 0;
    border-radius: 3px;
    background: none;
    color: inherit;
    font: inherit;
    text-align: left;
    cursor: pointer;
}

.theme-legend-item:hover {
    background-color: rgba(0, 0, 0, 0.07);
}

.theme-legend-item-on {
    background-color: theme.$source-tint-strong;
    font-weight: 600;
}

.theme-legend-label {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.theme-legend .legend-color-box {
    width: 13px;
    height: 13px;
}

.map-controls {
    position: absolute;
    top: 8px;
    left: 8px;
    z-index: 20;
}

.map-controls .btn {
    min-width: 2.25rem;
}



.filter-chip {
    font-size: 0.9rem;
    font-weight: 500;
    padding: 0.35em 0.6em;
    white-space: normal;
    text-align: left;
}

.graph-sidebar .nav-link {
    background-color: rgba(0, 0, 0, 0.05);
}

.legend-item {
    padding: 4px 6px;
    border-radius: 4px;
    transition: background-color 0.2s;
}

.legend-item:hover {
    background-color: rgba(0, 0, 0, 0.05);
}

.legend-item-selected {
    background-color: rgba(0, 123, 255, 0.1);
    border-left: 3px solid rgba(0, 123, 255, 0.5);
    padding-left: 3px;
}

.legend-color-box {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    flex-shrink: 0;
    border: 1px solid rgba(0, 0, 0, 0.35);
}

.node-info-panel .node-label-text {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    display: block;
    min-width: 0;
    flex: 1;
}

.node-info-panel .btn-outline-secondary {
    border-color: theme.$graph-btn-panel-color;
    color: theme.$graph-btn-panel-color;
}

.node-info-panel .btn-outline-secondary:hover {
    color: #fff;
}

.cluster-label-tag {
    font-size: 10px;
    font-weight: normal;
    color: #666;
    margin-top: 4px;
    font-style: italic;
}

.form-label {
    font-weight: 600;
    font-size: 0.9rem;
}

.loading-message {
    color: theme.$ink;
}

.theme-legend-head strong {
    @include theme.display-type;
    color: theme.$header-color;
    font-size: 1rem;
}

// Paper canvas, with a slow drift of ink specks behind the map
#sigma-container.vector-space-bg {
    background-color: theme.$paper;
    position: relative;
    overflow: hidden;
}

#sigma-container.vector-space-bg::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 1;
    background-image:
        radial-gradient(2px 2px at 20% 30%, rgba(142, 50, 50, 0.5), transparent),
        radial-gradient(2px 2px at 60% 70%, rgba(30, 84, 135, 0.5), transparent),
        radial-gradient(1.5px 1.5px at 50% 50%, rgba(42, 37, 34, 0.45), transparent),
        radial-gradient(1.5px 1.5px at 80% 10%, rgba(142, 50, 50, 0.45), transparent),
        radial-gradient(2px 2px at 90% 60%, rgba(30, 84, 135, 0.45), transparent),
        radial-gradient(1.5px 1.5px at 33% 80%, rgba(42, 37, 34, 0.4), transparent),
        radial-gradient(1.5px 1.5px at 15% 95%, rgba(30, 84, 135, 0.45), transparent);
    background-size: 200% 200%, 180% 180%, 220% 220%, 190% 190%, 210% 210%, 240% 240%, 230% 230%;
    background-position: 0% 0%, 10% 10%, 20% 20%, 30% 30%, 40% 40%, 50% 50%, 60% 60%;
    background-repeat: repeat;
    animation: stars 200s linear infinite;
    opacity: 0.5;
    pointer-events: none;
}

#sigma-container.stars-paused::before {
    animation-play-state: paused;
}

@media (prefers-reduced-motion: reduce) {
    #sigma-container.vector-space-bg::before {
        animation: none;
    }
}

@keyframes stars {
    from {
        background-position: 0% 0%, 10% 10%, 20% 20%, 30% 30%, 40% 40%, 50% 50%, 60% 60%;
    }

    to {
        background-position: 100% 100%, 110% 110%, 120% 120%, 130% 130%, 140% 140%, 150% 150%, 160% 160%;
    }
}

/* Narrow screens and large zoom: sidebar and legend stack under the map. */
@media (max-width: 767.98px) {
    .graph-layout {
        flex-direction: column;
    }

    .graph-sidebar {
        order: 0;
        flex: 0 0 auto;
        border-right: 0;
        border-top: 1px solid theme.$rule-strong;
    }

    .theme-legend {
        position: static;
        width: auto;
        max-height: none;
        margin: 8px;
    }
}

/* Fade animation for legend */
.fade-enter-active,
.fade-leave-active {
    transition: opacity 0.2s ease;
}

.fade-enter-from,
.fade-leave-to {
    opacity: 0;
}
</style>
