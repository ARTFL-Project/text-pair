<template>
    <div ref="rootEl" class="mt-3">
        <h2 class="visually-hidden">Network graph</h2>
        <div class="container-fluid">
            <div class="row" style="padding: 0 0.75rem">
                <div class="m-4" style="font-size: 120%" v-if="error">{{ error }}</div>
                <search-arguments unit="reused"></search-arguments>
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

                <!-- Network Controls -->
                <div class="card-body graph-controls border-bottom">
                    <div class="row align-items-center">
                        <div class="col-md-2">
                            <label for="network-aggregation" class="form-label mb-1">Aggregate by:</label>
                            <select id="network-aggregation" class="form-select form-select-sm"
                                v-model="aggregationField" @change="reloadNetwork">
                                <option v-for="option in availableAggregations" :key="option.value"
                                    :value="option.value">
                                    {{ option.label }}
                                </option>
                            </select>
                        </div>
                        <div class="col-md-2">
                            <label for="network-graph-type" class="form-label mb-1">Connect by:</label>
                            <select id="network-graph-type" class="form-select form-select-sm" v-model="graphType"
                                @change="changeGraph">
                                <option value="provenance">Reuse (source → reuser)</option>
                                <option value="canon">Shared passages</option>
                            </select>
                        </div>
                        <div class="col-md-2">
                            <label for="network-centrality" class="form-label mb-1">Measure Importance By:</label>
                            <select id="network-centrality" class="form-select form-select-sm" v-model="centralityMode"
                                @change="reloadNetwork">
                                <option v-for="option in centralityOptions" :key="option.value" :value="option.value">
                                    {{ option.label }}
                                </option>
                            </select>
                        </div>
                        <div class="col-md-2">
                            <label for="network-threshold" class="form-label mb-1">
                                Min. shared passages: <strong>{{ minThreshold }}</strong>
                            </label>
                            <input id="network-threshold" type="range" class="form-range" v-model.number="minThreshold"
                                min="1" max="100" @change="applyThreshold">
                        </div>
                        <div class="col-md-2">
                            <label for="network-layout" class="form-label mb-1">Arrange by:</label>
                            <select id="network-layout" class="form-select form-select-sm" v-model="layoutType"
                                @change="applyLayout">
                                <option value="communities">Community</option>
                                <option value="chronological">Chronological</option>
                            </select>
                        </div>
                        <div class="col-md-2">
                            <button type="button" class="btn btn-sm btn-secondary w-100 mt-3" @click="resetView">
                                Reset View
                            </button>
                        </div>
                    </div>
                    <div class="row mt-2">
                        <div class="col-md-12">
                            <span class="text-muted">
                                Showing {{ visibleNodes }} nodes
                                <span v-if="focus"> |
                                    <template v-if="focusRole === 'anchor'">
                                        Passages of <strong :lang="labelLang">{{ focus }}</strong> reused by others
                                    </template>
                                    <template v-else-if="focusRole === 'reuser'">
                                        Passages <strong :lang="labelLang">{{ focus }}</strong> reuses from others
                                    </template>
                                    <template v-else>
                                        Passages <strong :lang="labelLang">{{ focus }}</strong> shares with others
                                    </template>
                                    <button type="button"
                                        class="btn btn-outline-secondary btn-sm py-0 px-2 ms-1 align-baseline"
                                        @click="showAll">Show all</button>
                                </span>
                            </span>
                        </div>
                    </div>
                </div>

                <!-- Sigma Graph Container with overlays -->
                <div style="position: relative;">
                    <div id="sigma-container" ref="sigmaContainer" class="vector-space-bg"
                        :class="{ 'stars-paused': starsPaused }" role="img" :aria-label="graphSummary"
                        style="width: 100%; height: calc(100vh - 360px); min-height: 600px;"></div>

                    <!-- Year axis for the chronological layout -->
                    <div v-if="yearTicks.length" class="year-axis" aria-hidden="true">
                        <span v-for="tick in yearTicks" :key="tick.label" class="year-tick"
                            :style="{ left: `${tick.left}px` }">{{ tick.label }}</span>
                    </div>

                    <!-- Node Info Panel -->
                    <section v-if="selectedNode" ref="nodePanel" class="node-info-panel card shadow-lg"
                        aria-labelledby="network-node-title">
                        <div class="card-header d-flex justify-content-between align-items-center">
                            <h3 id="network-node-title" class="panel-title node-label-text" :lang="labelLang">
                                {{ selectedLabel }}
                            </h3>
                        </div>
                        <div class="card-body p-2">
                            <div class="d-grid gap-2">
                                <template v-if="graphType === 'provenance'">
                                    <button type="button" class="btn btn-sm btn-outline-secondary"
                                        @click="focusOn(selectedLabel, 'anchor')">
                                        <i class="bi bi-arrow-right" aria-hidden="true"></i> Reused by others
                                    </button>
                                    <button type="button" class="btn btn-sm btn-outline-secondary"
                                        @click="focusOn(selectedLabel, 'reuser')">
                                        <i class="bi bi-arrow-left" aria-hidden="true"></i> Reuses others
                                    </button>
                                </template>
                                <button v-else type="button" class="btn btn-sm btn-outline-secondary"
                                    @click="focusOn(selectedLabel)">
                                    <i class="bi bi-share" aria-hidden="true"></i> Shared passages
                                </button>
                            </div>
                        </div>
                    </section>

                    <!-- Edge Info Panel: the passages behind the edge -->
                    <section v-if="selectedEdge" ref="edgePanel" class="edge-info-panel card shadow-lg"
                        aria-labelledby="network-edge-title">
                        <div class="card-header d-flex justify-content-between align-items-start">
                            <h3 id="network-edge-title" ref="edgeTitle" class="panel-title" tabindex="-1">
                                <span :lang="labelLang">{{ selectedEdge.sourceLabel }}</span>
                                <span aria-hidden="true">{{ graphType === 'provenance' ? ' → ' : ' — ' }}</span>
                                <span class="visually-hidden">{{ graphType === 'provenance' ? ' reused by ' : ' and ' }}</span>
                                <span :lang="labelLang">{{ selectedEdge.targetLabel }}</span>
                            </h3>
                            <button type="button" class="btn-close ms-2" aria-label="Close passages"
                                @click="closeEdgePanel"></button>
                        </div>
                        <div class="card-body p-2">
                            <div class="mb-2 text-muted">
                                {{ selectedEdge.passages.toLocaleString() }}
                                {{ selectedEdge.passages === 1 ? 'passage' : 'passages' }}
                                {{ graphType === 'provenance' ? 'reused' : 'shared' }}
                            </div>
                            <div v-if="edgePassages.error" class="text-danger">{{ edgePassages.error }}</div>
                            <a v-for="passage in edgePassages.list" :key="passage.group_id" class="edge-passage"
                                :href="groupUrl(passage.group_id)" target="_blank">
                                <div class="edge-passage-text" :lang="sourceLang">{{ passage.source_passage }}</div>
                                <div class="edge-passage-meta">
                                    {{ passageDocuments(passage) }} ·
                                    <span :lang="sourceLang">{{ passage.source_title }}</span>
                                    <template v-if="passage.source_year">{{ ` (${passage.source_year})` }}</template>
                                </div>
                                <span class="visually-hidden">(opens in a new tab)</span>
                            </a>
                            <div v-if="edgePassages.loading" class="text-center p-2">
                                <div class="spinner-border spinner-border-sm" role="status">
                                    <span class="visually-hidden">Loading...</span>
                                </div>
                            </div>
                            <!-- Stays rendered while loading so that it keeps focus -->
                            <button v-if="edgePassages.list.length < edgePassages.total" type="button"
                                class="btn btn-sm btn-outline-secondary w-100 mt-1"
                                :aria-disabled="edgePassages.loading" @click="moreEdgePassages">
                                More passages
                            </button>
                        </div>
                    </section>

                    <!-- Zoom without the mouse wheel -->
                    <div class="graph-toolbar btn-group-vertical" role="group" aria-label="Graph view">
                        <button type="button" class="btn btn-sm btn-light" aria-label="Zoom in" title="Zoom in"
                            @click="zoomIn">
                            <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
                                <path d="M8 3v10M3 8h10" stroke="currentColor" stroke-width="2" fill="none" />
                            </svg>
                        </button>
                        <button type="button" class="btn btn-sm btn-light" aria-label="Zoom out" title="Zoom out"
                            @click="zoomOut">
                            <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
                                <path d="M3 8h10" stroke="currentColor" stroke-width="2" fill="none" />
                            </svg>
                        </button>
                        <button type="button" class="btn btn-sm btn-light" aria-label="Reset zoom" title="Reset zoom"
                            @click="resetZoom">
                            <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
                                <path d="M2 6V2h4M10 2h4v4M14 10v4h-4M6 14H2v-4" stroke="currentColor"
                                    stroke-width="1.75" fill="none" />
                            </svg>
                        </button>
                        <button v-if="!reducedMotion" type="button" class="btn btn-sm btn-light"
                            aria-label="Pause background animation" title="Pause background animation"
                            :aria-pressed="starsPaused" @click="starsPaused = !starsPaused">
                            <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
                                <path v-if="starsPaused" d="M5 3l8 5-8 5z" fill="currentColor" />
                                <path v-else d="M5 3v10M11 3v10" stroke="currentColor" stroke-width="2.5" fill="none" />
                            </svg>
                        </button>
                    </div>
                </div>
            </div>

            <!-- The graph as tables, for keyboard and screen reader use -->
            <div class="card shadow-1 mt-3 network-data">
                <div class="card-body">
                    <div class="row g-4">
                        <section ref="nodeTable" class="col-lg-7" aria-labelledby="network-nodes-title">
                            <h3 id="network-nodes-title" class="h6">Nodes</h3>
                            <div class="row g-2 align-items-end mb-2">
                                <div class="col-sm-6">
                                    <label for="network-node-filter" class="form-label mb-1">Find a node:</label>
                                    <input id="network-node-filter" v-model="nodeFilter" type="search"
                                        class="form-control form-control-sm" autocomplete="off">
                                </div>
                                <p id="network-nodes-summary" class="col-sm-6 mb-0 text-muted">
                                    {{ shownRows.length }} of {{ plural(matchingRows.length, "node") }}, ranked by
                                    {{ centralityLabel }}
                                </p>
                            </div>
                            <p v-if="!matchingRows.length" class="mb-0">
                                {{ loading ? "Loading..." : "No nodes to show." }}
                            </p>
                            <div v-else class="table-responsive">
                                <table class="table table-sm table-hover align-middle mb-2"
                                    aria-labelledby="network-nodes-title" aria-describedby="network-nodes-summary">
                                    <thead>
                                        <tr>
                                            <th scope="col">Rank</th>
                                            <th scope="col">{{ aggregationLabel }}</th>
                                            <th scope="col">{{ centralityLabel }}</th>
                                            <th scope="col">Links</th>
                                            <th scope="col">Passages</th>
                                            <th scope="col">Year</th>
                                            <th scope="col">Community</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr v-for="row in shownRows" :key="row.id"
                                            :class="{ 'table-active': row.id === selectedNode }">
                                            <td>{{ row.rank }}</td>
                                            <th scope="row">
                                                <button type="button" class="btn btn-link p-0 node-button"
                                                    :aria-pressed="row.id === selectedNode" :lang="labelLang"
                                                    @click="selectFromList(row.id)" @focus="highlightNode(row.id, true)"
                                                    @blur="highlightNode(row.id, false)">
                                                    {{ row.label }}
                                                </button>
                                            </th>
                                            <td>{{ formatMeasure(row.centrality) }}</td>
                                            <td>{{ row.links.toLocaleString() }}</td>
                                            <td>{{ row.passages.toLocaleString() }}</td>
                                            <td>
                                                <template v-if="row.year != null">{{ row.year }}</template>
                                                <template v-else>
                                                    <span aria-hidden="true">—</span>
                                                    <span class="visually-hidden">undated</span>
                                                </template>
                                            </td>
                                            <td class="text-nowrap">
                                                <span class="swatch" :style="{ backgroundColor: row.color }"
                                                    aria-hidden="true"></span>{{ row.community + 1 }}
                                            </td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                            <button v-if="shownRows.length < matchingRows.length" type="button"
                                class="btn btn-sm btn-outline-secondary" @click="showMoreNodes">
                                Show more nodes
                            </button>
                        </section>

                        <section ref="linksSection" class="col-lg-5" aria-labelledby="network-links-title">
                            <h3 id="network-links-title" class="h6">
                                Links<template v-if="selectedNode"> of <span :lang="labelLang">{{ selectedLabel }}</span></template>
                            </h3>
                            <template v-if="selectedNode">
                                <div class="d-flex flex-wrap gap-2 mb-2">
                                    <template v-if="graphType === 'provenance'">
                                        <button type="button" class="btn btn-sm btn-outline-secondary"
                                            @click="focusOn(selectedLabel, 'anchor')">
                                            Reused by others
                                        </button>
                                        <button type="button" class="btn btn-sm btn-outline-secondary"
                                            @click="focusOn(selectedLabel, 'reuser')">
                                            Reuses others
                                        </button>
                                    </template>
                                    <button v-else type="button" class="btn btn-sm btn-outline-secondary"
                                        @click="focusOn(selectedLabel)">
                                        Shared passages
                                    </button>
                                </div>
                                <p id="network-links-summary" class="text-muted mb-2">
                                    {{ plural(nodeLinks.length, "link") }}, by number of passages.
                                    <template v-if="graphType === 'provenance'">
                                        A reuser reuses passages of <span :lang="labelLang">{{ selectedLabel }}</span>;
                                        a source has passages it reuses.
                                    </template>
                                </p>
                                <div class="table-responsive">
                                    <table class="table table-sm table-hover align-middle mb-2"
                                        aria-labelledby="network-links-title" aria-describedby="network-links-summary">
                                        <thead>
                                            <tr>
                                                <th scope="col">Linked {{ aggregationLabel.toLowerCase() }}</th>
                                                <th v-if="graphType === 'provenance'" scope="col">Role</th>
                                                <th scope="col">Passages</th>
                                                <th scope="col"><span class="visually-hidden">Actions</span></th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr v-for="link in shownLinks" :key="link.edge"
                                                :class="{ 'table-active': selectedEdge && selectedEdge.edge === link.edge }">
                                                <th scope="row" class="fw-normal" :lang="labelLang">{{ link.label }}</th>
                                                <td v-if="graphType === 'provenance'">{{ link.role }}</td>
                                                <td>{{ link.passages.toLocaleString() }}</td>
                                                <td>
                                                    <button type="button"
                                                        class="btn btn-sm btn-outline-secondary text-nowrap link-button"
                                                        :aria-label="`View passages: ${linkName(link)}`"
                                                        @click="openEdge(link.edge, $event.currentTarget)">
                                                        View passages
                                                    </button>
                                                </td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>
                                <button v-if="shownLinks.length < nodeLinks.length" type="button"
                                    class="btn btn-sm btn-outline-secondary" @click="showMoreLinks">
                                    Show more links
                                </button>
                            </template>
                            <p v-else class="text-muted mb-0">
                                Select a node, in the graph or in the table, to list its links.
                            </p>
                        </section>
                    </div>
                </div>
            </div>

        </div> <!-- Close container-fluid -->
    </div> <!-- Close mt-3 wrapper -->
</template>

<script setup>
import { createNodeBorderProgram } from "@sigma/node-border";
import axios from "axios";
import Graph from "graphology";
import louvain from "graphology-communities-louvain";
import forceAtlas2 from "graphology-layout-forceatlas2";
import noverlap from "graphology-layout-noverlap";
import Sigma from "sigma";
import { EdgeArrowProgram, EdgeLineProgram } from "sigma/rendering";
import emitter from "tiny-emitter/instance";
import { computed, nextTick, onBeforeUnmount, onMounted, onUnmounted, ref, shallowRef, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import globalConfig from "../config.js";
import { paramsToUrl, prefersReducedMotion } from "../utils.js";
import reportSwitcher from "./reportSwitcher";
import searchArguments from "./searchArguments";

const FORCE_SETTINGS = {
    gravity: 1,
    scalingRatio: 10,
    strongGravityMode: true,
    barnesHutOptimize: true,
    barnesHutTheta: 0.5,
    edgeWeightInfluence: 1.0,  // Use edge weights for community structure
    slowDown: 5,
};

const CENTRALITY_OPTIONS = {
    provenance: [
        { value: "reused", label: "Reused by others" },
        { value: "reuses", label: "Reuses others" },
        { value: "authority", label: "Authority (PageRank)" },
        { value: "bridging", label: "Bridging role (betweenness)" },
    ],
    canon: [
        { value: "degree", label: "Shared passages (degree)" },
        { value: "bridging", label: "Bridging role (betweenness)" },
    ],
};

// Length of the chronological layout's time axis, in graph coordinates
const TIMELINE_WIDTH = 1000;

// Deep inks that hold 3:1 on the paper canvas, ordered so neighbours stay apart under colour blindness
const COLOR_PALETTE = [
    '#a33b3b', '#1f6aa5', '#b7791f', '#008c84', '#c75a2a',
    '#7b4b94', '#5f8a1c', '#b0466f', '#4a5fb0', '#2f7d4f'
];
const EDGE_COLOR = '#1e548780';

// Rows added to the node and link tables at a time
const PAGE_SIZE = 20;

const route = useRoute();
const router = useRouter();
const reducedMotion = prefersReducedMotion();
const sourceLang = globalConfig.sourceLanguage || undefined;

// Template refs
const rootEl = ref(null);
const sigmaContainer = ref(null);
const nodePanel = ref(null);
const edgePanel = ref(null);
const edgeTitle = ref(null);
const nodeTable = ref(null);
const linksSection = ref(null);

const loading = ref(false);
const loadingMessage = ref("Fetching data...");
const error = ref(null);

// Graph and renderer stay out of Vue's reactivity: deep proxies slow them down
let graph = null;
let renderer = null;
let rawData = { nodes: [], edges: [] };
let connectedNodes = new Set();
let timeline = null;
// The threshold the current data was fetched at. Edges below it were
// filtered out in SQL, so they cannot be revealed client-side.
let fetchedThreshold = 10;
// The control that opened the edge panel, to return focus to
let edgeOpener = null;
let filterTimer = null;
const nodeColorMap = new Map(); // Track assigned colors
const communityColors = new Map(); // Colors for communities
let colorIndex = 0;

// UI state
const aggregationField = ref("author");
const availableAggregations = ref([]);
const graphType = ref("provenance");  // provenance or canon
const centralityMode = ref("reused");
const minThreshold = ref(10);
const layoutType = ref("communities");  // communities or chronological
const selectedNode = ref(null);
const selectedEdge = shallowRef(null);
const edgePassages = ref({ list: [], total: 0, loading: false, error: null });
const yearTicks = ref([]);
const visibleNodes = ref(0);
const starsPaused = ref(reducedMotion);

// Table equivalent of the graph
const nodeRows = shallowRef([]);
const nodeLinks = shallowRef([]);
const nodeFilter = ref("");
const nodeLimit = ref(PAGE_SIZE);
const linkLimit = ref(PAGE_SIZE);

const centralityOptions = computed(() => CENTRALITY_OPTIONS[graphType.value]);
const centralityLabel = computed(() => {
    const option = centralityOptions.value.find((option) => option.value === centralityMode.value);
    return (option || centralityOptions.value[0]).label;
});
const aggregationLabel = computed(() => {
    const option = availableAggregations.value.find((option) => option.value === aggregationField.value);
    return option ? option.label : "Node";
});
const focus = computed(() => route.query.focus || "");
const focusRole = computed(() => route.query.focus_role || "");
const selectedLabel = computed(() => (selectedNode.value && graph ? getNodeLabel(selectedNode.value) : ""));
// Titles can come from either side, so they get a language only when both sides share it
const labelLang = computed(() => {
    if (aggregationField.value === "author") return undefined;
    const { sourceLanguage, targetLanguage } = globalConfig;
    return sourceLanguage && sourceLanguage === targetLanguage ? sourceLanguage : undefined;
});
const graphSummary = computed(() => {
    const kind = graphType.value === "provenance" ? "Reuse network" : "Shared passages network";
    if (selectedNode.value) {
        return `${kind}: ${selectedLabel.value} and its ${plural(visibleNodes.value - 1, "linked node")}. ` +
            "Its links are listed in a table below the graph.";
    }
    const arrangement = layoutType.value === "chronological" ? "year" : "community";
    return `${kind}: ${plural(visibleNodes.value, "node")} sized by ${centralityLabel.value}, ` +
        `colored by community, arranged by ${arrangement}. The nodes are listed in a table below the graph.`;
});
const matchingRows = computed(() => {
    const query = searchKey(nodeFilter.value.trim());
    return query ? nodeRows.value.filter((row) => row.key.includes(query)) : nodeRows.value;
});
const shownRows = computed(() => matchingRows.value.slice(0, nodeLimit.value));
const shownLinks = computed(() => nodeLinks.value.slice(0, linkLimit.value));

watch(() => route.fullPath, () => {
    // Also fires while leaving the view
    if (route.name !== "networkGraph") return;
    // Check if aggregation_field changed in URL
    if (route.query.aggregation_field &&
        route.query.aggregation_field !== aggregationField.value &&
        availableAggregations.value.some(opt => opt.value === route.query.aggregation_field)) {
        aggregationField.value = route.query.aggregation_field;
    }
    if (CENTRALITY_OPTIONS[route.query.graph] && route.query.graph !== graphType.value) {
        graphType.value = route.query.graph;
        centralityMode.value = centralityOptions.value[0].value;
    }
    fetchNetworkData();
});

watch(nodeFilter, () => {
    nodeLimit.value = PAGE_SIZE;
    clearTimeout(filterTimer);
    filterTimer = setTimeout(() => {
        const count = plural(matchingRows.value.length, "node");
        emitter.emit("announce", nodeFilter.value.trim() ? `${count} found` : `${count} listed`);
    }, 600);
});

onMounted(() => {
    document.addEventListener("keydown", onKeydown);
});

onBeforeUnmount(() => {
    if (renderer) {
        renderer.kill();
        renderer = null;
    }
});

onUnmounted(() => {
    document.removeEventListener("keydown", onKeydown);
    clearTimeout(filterTimer);
});

function plural(count, word) {
    return `${count.toLocaleString()} ${word}${count === 1 ? "" : "s"}`;
}

// The reuse network lists the passages a source has that others reuse, so it counts the documents that reuse them
function passageDocuments(passage) {
    return graphType.value === "provenance"
        ? `Reused in ${plural(passage.count - 1, "other document")}`
        : `In ${plural(passage.count, "document")}`;
}

function searchKey(text) {
    return text.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
}

function formatMeasure(value) {
    if (value == null) return "";
    return Number.isInteger(value) ? value.toLocaleString() : String(Number(value.toPrecision(3)));
}

function motionDuration(duration) {
    // Sigma's camera needs a non-zero duration
    return reducedMotion ? 1 : duration;
}

function initializeAggregationOptions() {
    // Build aggregation options from metadata fields
    const fieldMap = new Map();

    // Check source and target fields
    const sourceFields = globalConfig.metadataFields.source || [];
    const targetFields = globalConfig.metadataFields.target || [];

    sourceFields.forEach(field => {
        // Extract field type (author, title, etc.) from value like "source_author"
        const fieldType = field.value.replace('source_', '');
        if (!fieldMap.has(fieldType)) {
            fieldMap.set(fieldType, field.label);
        }
    });

    targetFields.forEach(field => {
        const fieldType = field.value.replace('target_', '');
        if (!fieldMap.has(fieldType)) {
            fieldMap.set(fieldType, field.label);
        }
    });

    // Build available aggregations array
    availableAggregations.value = Array.from(fieldMap.entries())
        .map(([value, label]) => ({ value, label }))
        .filter(option => {
            // Exclude passage fields and year: the passage is what links nodes
            if (['passage', 'passage_length', 'year'].includes(option.value)) {
                return false;
            }
            // Only include if both source and target have this field
            const hasSource = sourceFields.some(f => f.value === `source_${option.value}`);
            const hasTarget = targetFields.some(f => f.value === `target_${option.value}`);
            return hasSource && hasTarget;
        });

    // Check if aggregation_field is specified in URL
    const urlAggField = route.query.aggregation_field;

    if (urlAggField && availableAggregations.value.some(opt => opt.value === urlAggField)) {
        aggregationField.value = urlAggField;
    }
    // Otherwise set default to author if available
    else if (availableAggregations.value.some(opt => opt.value === 'author')) {
        aggregationField.value = 'author';
    } else if (availableAggregations.value.length > 0) {
        aggregationField.value = availableAggregations.value[0].value;
    }
}

function fetchNetworkData() {
    loading.value = true;
    loadingMessage.value = "Fetching data...";
    error.value = null;
    selectedNode.value = null;
    selectedEdge.value = null;
    nodeLinks.value = [];
    nodeRows.value = [];
    nodeLimit.value = PAGE_SIZE;

    // Destroy existing renderer and graph
    if (renderer) {
        renderer.kill();
        renderer = null;
    }
    if (graph) {
        graph.clear();
        graph = null;
    }

    let params = { ...route.query };
    params.db_table = globalConfig.databaseName;
    params.aggregation_field = aggregationField.value;
    params.graph = graphType.value;
    params.centrality = centralityMode.value;
    params.min_threshold = minThreshold.value;
    params.max_nodes = 10000;
    fetchedThreshold = minThreshold.value;

    emitter.emit("searchArgsUpdate", {
        counts: "",
        searchParams: params,
    });

    axios
        .get(`${globalConfig.apiServer}/network_data/?${paramsToUrl(params)}`)
        .then((response) => {
            if (response.data.error) {
                error.value = response.data.error;
                loading.value = false;
                emitter.emit("announce", error.value);
                return;
            }

            rawData = Object.freeze(response.data);
            loadingMessage.value = "Calculating layout...";

            // Use setTimeout to allow the UI to update with the new message
            setTimeout(() => {
                initializeGraph();
                loading.value = false;
            }, 10);

            emitter.emit("searchArgsUpdate", {
                counts: response.data.total_passages,
                searchParams: params,
            });
        })
        .catch((err) => {
            loading.value = false;
            error.value = err.toString();
            emitter.emit("announce", error.value);
            console.log(err);
        });
}

function initializeGraph() {
    console.time('Total graph initialization');

    // Reuse runs from a passage's source to its reusers; shared passages have no direction
    graph = new Graph({ type: rawData.graph === "provenance" ? "directed" : "undirected" });

    // Use centrality for node sizing
    const centralities = rawData.nodes.map(n => n.centrality || 0);
    const maxCentrality = Math.max(...centralities);
    const minCentrality = Math.min(...centralities);

    // Define size range (min and max pixel sizes)
    // For small graphs (< 1000 nodes), use larger min size so all nodes are visible
    const nodeCount = rawData.nodes.length;
    const minNodeSize = nodeCount < 1000 ? 3 : 2;
    const maxNodeSize = 30;

    // Add nodes with normalized sizes based on centrality
    console.time('Add nodes');
    rawData.nodes.forEach(node => {
        // Normalize centrality to range [minNodeSize, maxNodeSize]
        let normalizedSize;
        if (maxCentrality === minCentrality) {
            // All nodes have same centrality, use uniform size
            normalizedSize = (minNodeSize + maxNodeSize) / 2;
        } else {
            normalizedSize = minNodeSize +
                (node.centrality - minCentrality) /
                (maxCentrality - minCentrality) *
                (maxNodeSize - minNodeSize);
        }

        const nodeColor = getUniqueNodeColor(node.id);
        const label = node.label;

        // Truncate label: at second comma or 20 chars, whichever comes first
        let displayLabel = label;
        const firstComma = label.indexOf(',');
        if (firstComma !== -1) {
            const secondComma = label.indexOf(',', firstComma + 1);
            if (secondComma !== -1) {
                // Truncate at second comma (no ellipsis), but respect 20 char limit
                const commaLimit = secondComma;
                if (commaLimit <= 20) {
                    displayLabel = label.substring(0, commaLimit);
                } else {
                    displayLabel = label.substring(0, 20) + '...';
                }
            } else if (label.length > 20) {
                // No second comma, but still too long
                displayLabel = label.substring(0, 20) + '...';
            }
        } else if (label.length > 20) {
            // No commas, just truncate at 20
            displayLabel = label.substring(0, 20) + '...';
        }

        graph.addNode(node.id, {
            label: displayLabel,
            fullLabel: label,  // Store full label for reference
            size: normalizedSize,
            color: nodeColor,
            originalColor: nodeColor, // Store original color
            passages: node.passages,
            centrality: node.centrality,
            year: node.year,
            hidden: false,  // Initialize as visible
            zIndex: 0,  // Initialize with default zIndex
            x: Math.random(),
            y: Math.random()
        });
    });
    console.timeEnd('Add nodes');

    // Add edges
    console.time('Add edges');
    const edgeType = rawData.graph === "provenance" ? "arrow" : "line";
    rawData.edges.forEach((edge) => {
        if (graph.hasNode(edge.source) && graph.hasNode(edge.target)) {
            graph.addEdge(edge.source, edge.target, {
                weight: edge.weight,
                passages: edge.passages,
                size: Math.log(edge.passages + 1) * 2, // Log scale for edge thickness
                type: edgeType,
                color: EDGE_COLOR,
                hidden: true  // Initialize as hidden
            });
        }
    });
    console.timeEnd('Add edges');
    console.timeEnd('Total graph initialization');

    // Apply layout and initialize renderer after layout is complete
    applyLayoutAndInitRenderer();
}

function applyLayoutAndInitRenderer() {
    if (!graph) return;

    // Apply layout first
    applyLayoutOnly();

    // Wait for layout to complete, then initialize renderer
    setTimeout(() => {
        initRenderer();
        if (renderer) {
            emitter.emit("announce", visibleNodes.value
                ? `Network graph: ${plural(visibleNodes.value, "node")}`
                : "No nodes to show in the network graph");
        }
    }, 100);
}

function initRenderer() {
    // The view was left while the layout ran
    if (!graph || !sigmaContainer.value) return;
    console.time('Renderer initialization');

    // Initialize or update renderer
    if (renderer) {
        renderer.kill();
    }

    renderer = new Sigma(graph, sigmaContainer.value, {
        edgeProgramClasses: {
            line: EdgeLineProgram,
            arrow: EdgeArrowProgram,
        },
        nodeProgramClasses: {
            // Use border program to create hollow circles with colored border
            circle: createNodeBorderProgram({
                borders: [
                    {
                        size: { value: 0.3 },  // Ring of 30% of the node size, to hold on the paper canvas
                        color: { attribute: "color" }  // Use node's color for border
                    }
                ]
            })
        },
        renderEdgeLabels: false,
        defaultNodeColor: "#857d75",
        defaultEdgeColor: EDGE_COLOR,
        labelFont: '"Open Sans", sans-serif',
        labelSize: 12,
        labelWeight: "600",
        labelColor: { attribute: "labelColor", color: "#2a2522" },  // Use node's labelColor attribute
        enableEdgeEvents: true,  // Make edges clickable
        labelDensity: 0.1,  // Show 10% of labels initially
        labelGridCellSize: 100,  // Grid cell size for label positioning
        labelRenderedSizeThreshold: 15,  // Only show labels for nodes with rendered size > 10px
        zIndex: true,  // Enable zIndex for node layering
        hideLabelsOnMove: true,
        hideEdgesOnMove: true,
        ...(reducedMotion ? { zoomDuration: 1, doubleClickZoomingDuration: 1, inertiaRatio: 0 } : {}),
    });

    // Set up reducers using setSetting (best practice from Sigma examples)
    renderer.setSetting('nodeReducer', (node, data) => {
        const res = { ...data };

        // Highlight on hover - use black label on white background
        if (data.highlighted) {
            res.color = "#5f0202";
            res.zIndex = 10;
            res.labelColor = "#000000";  // Black label for hover
            return res;
        }

        // Default: ink label for the paper background
        res.labelColor = "#2a2522";

        // When a node is selected
        if (selectedNode.value) {
            if (connectedNodes.has(node)) {
                // Connected nodes - show in color and explicitly unhide
                res.hidden = false;
                res.color = data.originalColor || data.color;
                res.zIndex = 5;
            } else {
                // Unconnected nodes - hide them completely
                res.hidden = true;
            }
        } else {
            // Default state - make sure nodes are visible
            res.hidden = false;
            res.zIndex = 1;
        }

        return res;
    });

    renderer.setSetting('edgeReducer', (edge, data) => {
        const res = { ...data };

        if (selectedEdge.value && selectedEdge.value.edge === edge) {
            // The edge whose passages are listed
            res.hidden = false;
            res.color = "#ad4242";
            res.zIndex = 10;
        } else if (selectedNode.value) {
            // Show edge only if it connects to the selected node
            if (!graph.hasExtremity(edge, selectedNode.value)) {
                res.hidden = true;
            } else {
                res.hidden = false;
            }
        } else {
            // Hide all edges by default
            res.hidden = true;
        }

        return res;
    });

    // Set camera to default position
    const camera = renderer.getCamera();
    camera.setState({ ratio: 1.0 });

    // Keep the year axis under the chronological layout as the camera moves
    renderer.on("afterRender", () => updateYearTicks());

    // Update counts
    updateCounts();

    // Setup interactions
    setupInteractions();

    console.timeEnd('Renderer initialization');
}

function applyLayoutOnly() {
    if (!graph) return;

    console.time('Total layout');
    assignCommunityColors();
    if (layoutType.value === "chronological") {
        chronologicalLayout();
    } else {
        forceLayout();
    }
    console.timeEnd('Total layout');
    buildNodeRows();
}

function assignCommunityColors() {
    // Detect communities using Louvain algorithm and assign to nodes
    console.time('Louvain community detection');
    louvain.assign(graph, {});
    console.timeEnd('Louvain community detection');

    communityColors.clear();
    graph.forEachNode((node) => {
        const community = graph.getNodeAttribute(node, 'community');

        // Assign color for this community if not yet assigned
        if (!communityColors.has(community)) {
            const index = communityColors.size % COLOR_PALETTE.length;
            communityColors.set(community, COLOR_PALETTE[index]);
        }

        graph.setNodeAttribute(node, 'color', communityColors.get(community));
    });
}

function forceLayout() {
    timeline = null;

    // Apply force-directed layout with linLogMode for better community separation
    console.time('ForceAtlas2 layout');
    forceAtlas2.assign(graph, { iterations: 100, settings: FORCE_SETTINGS });
    console.timeEnd('ForceAtlas2 layout');

    // Apply noverlap to prevent node overlapping
    console.time('Noverlap layout');
    noverlap.assign(graph, {
        maxIterations: 50,
        settings: {
            margin: 5,
            ratio: 1.2,
            expansion: 1.1
        }
    });
    console.timeEnd('Noverlap layout');
}

function chronologicalLayout() {
    // x is the node's median year; the force layout only places nodes vertically
    const years = [];
    graph.forEachNode((node, { year }) => {
        if (year != null) years.push(year);
    });
    if (!years.length) {
        forceLayout();
        return;
    }
    years.sort((a, b) => a - b);
    // The axis runs out from the central 90% of years until a gap wider than a quarter of
    // their span; years past such a gap sit at its ends rather than stretch it
    const quantile = (q) => Math.min(years.length - 1, Math.floor(q * years.length));
    const maxGap = Math.max(50, 0.25 * (years[quantile(0.95)] - years[quantile(0.05)]));
    let low = quantile(0.05);
    while (low > 0 && years[low] - years[low - 1] <= maxGap) low--;
    let high = quantile(0.95);
    while (high < years.length - 1 && years[high + 1] - years[high] <= maxGap) high++;
    const minYear = years[low];
    const maxYear = years[high];
    const span = Math.max(maxYear - minYear, 1);
    // Undated nodes get a column of their own, left of the axis
    const xOf = (year) => (year == null
        ? -0.08 * TIMELINE_WIDTH
        : ((Math.min(Math.max(year, minYear), maxYear) - minYear) / span) * TIMELINE_WIDTH);
    const pinToYears = () => {
        graph.forEachNode((node, { year }) => graph.setNodeAttribute(node, 'x', xOf(year)));
    };

    console.time('Chronological layout');
    graph.forEachNode((node) => graph.setNodeAttribute(node, 'y', Math.random() * TIMELINE_WIDTH));
    pinToYears();
    for (let i = 0; i < 100; i++) {
        forceAtlas2.assign(graph, { iterations: 1, settings: FORCE_SETTINGS });
        pinToYears();
    }

    // Keep the vertical spread in proportion to the time axis
    let minY = Infinity;
    let maxY = -Infinity;
    graph.forEachNode((node, { y }) => {
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
    });
    const scale = (0.6 * TIMELINE_WIDTH) / Math.max(maxY - minY, 1e-9);
    graph.forEachNode((node, { y }) => graph.setNodeAttribute(node, 'y', (y - minY) * scale));
    console.timeEnd('Chronological layout');

    timeline = {
        minYear,
        maxYear,
        xOf,
        undated: years.length < graph.order,
        earlier: years[0] < minYear,
        later: years[years.length - 1] > maxYear,
    };
}

function updateYearTicks() {
    if (!renderer || !timeline) {
        if (yearTicks.value.length) yearTicks.value = [];
        return;
    }
    const { minYear, maxYear, xOf, undated, earlier, later } = timeline;
    const width = sigmaContainer.value.clientWidth;
    const left = (year) => renderer.graphToViewport({ x: xOf(year), y: 0 }).x;

    // A labelled year about every 100px at the current zoom
    const pixelsPerYear = (left(maxYear) - left(minYear)) / Math.max(maxYear - minYear, 1);
    const step = [1, 2, 5, 10, 20, 25, 50, 100, 200, 500, 1000].find((s) => s * pixelsPerYear >= 100) || 1000;
    const ticks = [];
    for (let year = Math.ceil(minYear / step) * step; year <= maxYear; year += step) {
        if ((earlier && year - minYear < step / 2) || (later && maxYear - year < step / 2)) continue;
        ticks.push({ label: String(year), left: left(year) });
    }
    if (earlier) ticks.push({ label: `≤ ${minYear}`, left: left(minYear) });
    if (later) ticks.push({ label: `≥ ${maxYear}`, left: left(maxYear) });
    if (undated) ticks.push({ label: "undated", left: left(null) });
    yearTicks.value = ticks.filter((tick) => tick.left >= 0 && tick.left <= width);
}

function applyLayout() {
    if (!graph) return;

    loading.value = true;
    loadingMessage.value = "Applying layout...";

    // Kill the renderer to hide the graph during layout calculation
    if (renderer) {
        renderer.kill();
        renderer = null;
    }

    // Use setTimeout to allow UI to update before heavy computation
    setTimeout(() => {
        applyLayoutOnly();

        // Reinitialize the renderer
        initRenderer();

        // Give renderer time to complete rendering before hiding spinner
        setTimeout(() => {
            loading.value = false;
            emitter.emit("announce", `Arranged by ${layoutType.value === "chronological" ? "year" : "community"}`);
        }, 50);
    }, 10);
}

function setupInteractions() {
    // Click on node - show edges and desaturate unconnected nodes
    renderer.on("clickNode", ({ node }) => toggleNode(node));

    // Click on edge - list the passages behind it, keeping the node's edges in view
    renderer.on("clickEdge", ({ edge }) => openEdge(edge));

    // Click on stage (background)
    renderer.on("clickStage", () => {
        clearSelection();
        setSelectedNode(null);
    });

    // Hover effects - use graph attributes, reducer handles rendering
    renderer.on("enterNode", ({ node }) => {
        graph.setNodeAttribute(node, 'highlighted', true);
    });

    renderer.on("leaveNode", ({ node }) => {
        graph.setNodeAttribute(node, 'highlighted', false);
    });
}

// Select or deselect a node, from the graph or the node table
function toggleNode(node) {
    selectedEdge.value = null;
    setSelectedNode(selectedNode.value === node ? null : node);
    emitter.emit("announce", selectedNode.value
        ? `${selectedLabel.value}: ${plural(nodeLinks.value.length, "link")}`
        : `Showing ${plural(visibleNodes.value, "node")}`);
}

function selectFromList(node) {
    toggleNode(node);
    if (selectedNode.value) revealNode(node);
}

// Pan to a node chosen in the table when it is out of view
function revealNode(node) {
    if (!renderer) return;
    const { x, y } = renderer.graphToViewport(graph.getNodeAttributes(node));
    const { width, height } = renderer.getDimensions();
    if (x >= 0 && y >= 0 && x <= width && y <= height) return;
    const target = renderer.getNodeDisplayData(node);
    renderer.getCamera().animate({ x: target.x, y: target.y }, { duration: motionDuration(300) });
}

function highlightNode(node, highlighted) {
    if (graph && graph.hasNode(node)) {
        graph.setNodeAttribute(node, 'highlighted', highlighted);
    }
}

function zoomIn() {
    if (renderer) renderer.getCamera().animatedZoom({ duration: motionDuration(300) });
}

function zoomOut() {
    if (renderer) renderer.getCamera().animatedUnzoom({ duration: motionDuration(300) });
}

function resetZoom() {
    if (renderer) renderer.getCamera().animatedReset({ duration: motionDuration(300) });
}

function openEdge(edge, opener = null) {
    const source = graph.source(edge);
    const target = graph.target(edge);
    selectedEdge.value = {
        edge: edge,
        source: source,
        target: target,
        sourceLabel: getNodeLabel(source),
        targetLabel: getNodeLabel(target),
        passages: graph.getEdgeAttribute(edge, 'passages')
    };
    edgeOpener = opener;
    fetchEdgePassages(0);
    renderer.refresh({ skipIndexation: true });
    if (opener) {
        nextTick(() => edgeTitle.value && edgeTitle.value.focus());
    }
}

function fetchEdgePassages(offset) {
    const edge = selectedEdge.value;
    if (!edge) return;
    if (offset === 0) {
        edgePassages.value = { list: [], total: 0, loading: true, error: null };
    } else {
        edgePassages.value.loading = true;
    }

    const params = {
        ...route.query,
        db_table: globalConfig.databaseName,
        aggregation_field: aggregationField.value,
        graph: graphType.value,
        source: edge.source,
        target: edge.target,
        limit: 20,
        offset: offset,
    };
    axios
        .get(`${globalConfig.apiServer}/network_passages/?${paramsToUrl(params)}`)
        .then((response) => {
            if (selectedEdge.value !== edge) return;  // another edge was clicked meanwhile
            edgePassages.value.loading = false;
            if (response.data.error) {
                edgePassages.value.error = response.data.error;
                emitter.emit("announce", response.data.error);
                return;
            }
            edgePassages.value.list = edgePassages.value.list.concat(response.data.passages);
            edgePassages.value.total = response.data.total_count;
            const { list, total } = edgePassages.value;
            emitter.emit("announce",
                total ? `Showing ${list.length} of ${plural(total, "passage")}` : "No passages found");
            if (offset > 0) {
                // Continue from the first passage added
                nextTick(() => {
                    const links = edgePanel.value ? edgePanel.value.querySelectorAll(".edge-passage") : [];
                    if (links[offset]) links[offset].focus();
                });
            }
        })
        .catch((err) => {
            if (selectedEdge.value !== edge) return;
            edgePassages.value.loading = false;
            edgePassages.value.error = err.toString();
            emitter.emit("announce", edgePassages.value.error);
        });
}

function moreEdgePassages() {
    if (!edgePassages.value.loading) {
        fetchEdgePassages(edgePassages.value.list.length);
    }
}

function closeEdgePanel() {
    const hadFocus = edgePanel.value && edgePanel.value.contains(document.activeElement);
    selectedEdge.value = null;
    if (renderer) {
        renderer.refresh({ skipIndexation: true });
    }
    if (hadFocus && edgeOpener && edgeOpener.isConnected) {
        edgeOpener.focus();
    }
    edgeOpener = null;
}

// Escape closes the edge panel, then the node selection
function onKeydown(event) {
    if (event.key !== "Escape" || event.defaultPrevented) return;
    const target = event.target;
    // Focus in the view, or on a container of it such as <main> after a click on the graph
    if (!rootEl.value || !(rootEl.value.contains(target) || target.contains(rootEl.value))) return;
    // Escape clears the node search field
    if (target.type === "search") return;
    if (selectedEdge.value) {
        closeEdgePanel();
    } else if (selectedNode.value) {
        clearNodeSelection();
    } else {
        return;
    }
    event.preventDefault();
}

function clearNodeSelection() {
    const node = selectedNode.value;
    const active = document.activeElement;
    const hadFocus = (nodePanel.value && nodePanel.value.contains(active)) ||
        (linksSection.value && linksSection.value.contains(active));
    toggleNode(node);
    if (hadFocus) {
        nextTick(() => {
            if (document.activeElement === document.body && !focusNodeButton(node)) {
                emitter.emit("focusMain");
            }
        });
    }
}

function focusNodeButton(node) {
    const index = shownRows.value.findIndex((row) => row.id === node);
    const button = index >= 0 && nodeTable.value ? nodeTable.value.querySelectorAll(".node-button")[index] : null;
    if (button) button.focus();
    return Boolean(button);
}

function showMoreNodes() {
    const first = shownRows.value.length;
    nodeLimit.value += PAGE_SIZE;
    nextTick(() => {
        const button = nodeTable.value && nodeTable.value.querySelectorAll(".node-button")[first];
        if (button) button.focus();
    });
}

function showMoreLinks() {
    const first = shownLinks.value.length;
    linkLimit.value += PAGE_SIZE;
    nextTick(() => {
        const button = linksSection.value && linksSection.value.querySelectorAll(".link-button")[first];
        if (button) button.focus();
    });
}

function groupUrl(groupId) {
    return router.resolve({ path: `/group/${groupId}` }).href;
}

// Set selected node
function setSelectedNode(node) {
    if (node) {
        selectedNode.value = node;
        const neighbors = graph.neighbors(node);
        connectedNodes = new Set([node, ...neighbors]);
        nodeLinks.value = buildNodeLinks(node);
    } else {
        selectedNode.value = null;
        connectedNodes.clear();
        nodeLinks.value = [];
    }
    linkLimit.value = PAGE_SIZE;

    // Refresh with skipIndexation for better performance
    updateCounts();
    renderer.refresh({
        skipIndexation: true
    });
}

// Visible nodes, ranked by the importance measure
function buildNodeRows() {
    const rows = [];
    graph.forEachNode((node, attributes) => {
        if (attributes.hidden) return;
        const label = attributes.fullLabel || attributes.label;
        rows.push({
            id: node,
            label: label,
            key: searchKey(label),
            centrality: attributes.centrality,
            passages: attributes.passages || 0,
            year: attributes.year,
            community: attributes.community,
            color: attributes.color,
            links: graph.degree(node),
        });
    });
    rows.sort((a, b) => (b.centrality || 0) - (a.centrality || 0) || a.label.localeCompare(b.label));
    rows.forEach((row, index) => {
        row.rank = index + 1;
    });
    nodeRows.value = rows;
}

function buildNodeLinks(node) {
    const links = [];
    graph.forEachEdge(node, (edge, attributes, source, target) => {
        links.push({
            edge: edge,
            label: getNodeLabel(source === node ? target : source),
            // In the reuse graph, edges run from a passage's source to its reuser
            role: source === node ? "Reuser" : "Source",
            passages: attributes.passages,
            source: getNodeLabel(source),
            target: getNodeLabel(target),
        });
    });
    links.sort((a, b) => b.passages - a.passages || a.label.localeCompare(b.label));
    return links;
}

function linkName(link) {
    return graphType.value === "provenance"
        ? `${link.source} reused by ${link.target}`
        : `${link.source} and ${link.target}`;
}

function getUniqueNodeColor(nodeId) {
    // Check if we already assigned a color to this node
    if (nodeColorMap.has(nodeId)) {
        return nodeColorMap.get(nodeId);
    }

    // Assign a new color from the palette
    const color = COLOR_PALETTE[colorIndex % COLOR_PALETTE.length];
    colorIndex++;
    nodeColorMap.set(nodeId, color);
    return color;
}

function applyThreshold() {
    // Lowering the threshold needs a refetch: the API filters edges by
    // passages in SQL, so what is below the fetched threshold was never sent.
    if (minThreshold.value < fetchedThreshold) {
        fetchNetworkData();
        return;
    }
    if (!graph) return;

    // Hide edges below threshold
    graph.forEachEdge((edge, attributes) => {
        const hidden = attributes.passages < minThreshold.value;
        graph.setEdgeAttribute(edge, 'hidden', hidden);
    });

    // Hide nodes with no visible edges
    graph.forEachNode((node) => {
        const hasVisibleEdge = graph.edges(node).some(edge =>
            !graph.getEdgeAttribute(edge, 'hidden')
        );
        graph.setNodeAttribute(node, 'hidden', !hasVisibleEdge);
    });

    updateCounts();
    buildNodeRows();
    if (renderer) renderer.refresh();
    emitter.emit("announce", `Showing ${plural(visibleNodes.value, "node")}`);
}

function updateCounts() {
    // Count nodes that are actually visible based on selection state
    if (selectedNode.value) {
        // When a node is selected, only count connected nodes
        visibleNodes.value = connectedNodes.size;
    } else {
        // When no selection, count all non-hidden nodes
        visibleNodes.value = graph.filterNodes((node, attr) => !attr.hidden).length;
    }
}

function reloadNetwork() {
    fetchNetworkData();
}

function changeGraph() {
    if (!centralityOptions.value.some((option) => option.value === centralityMode.value)) {
        centralityMode.value = centralityOptions.value[0].value;
    }
    // A focus is on one side of a reuse, which shared passages do not have
    clearFocus();
}

function resetView() {
    selectedNode.value = null;
    selectedEdge.value = null;
    minThreshold.value = 10;
    fetchedThreshold = 10;
    if (focus.value) {
        clearFocus();
    } else {
        fetchNetworkData();
    }
}

function clearSelection() {
    selectedNode.value = null;
    selectedEdge.value = null;
    nodeLinks.value = [];
}

function getNodeLabel(nodeId) {
    // Return fullLabel for complete text, fallback to label if fullLabel doesn't exist
    return graph.getNodeAttribute(nodeId, 'fullLabel') || graph.getNodeAttribute(nodeId, 'label');
}

function focusOn(nodeLabel, role) {
    let queryParams = { ...route.query, focus: nodeLabel };
    if (role) {
        queryParams.focus_role = role;
    } else {
        delete queryParams.focus_role;
    }
    queryParams.graph = graphType.value;
    queryParams.db_table = globalConfig.databaseName;
    clearSelection();
    router.push(`/network?${paramsToUrl(queryParams)}`);
    // The panel holding the button is gone
    emitter.emit("focusMain");
}

function clearFocus() {
    let queryParams = { ...route.query };
    delete queryParams.focus;
    delete queryParams.focus_role;
    queryParams.graph = graphType.value;
    queryParams.db_table = globalConfig.databaseName;
    router.push(`/network?${paramsToUrl(queryParams)}`);
}

function showAll() {
    clearFocus();
    emitter.emit("focusMain");
}

if (CENTRALITY_OPTIONS[route.query.graph]) {
    graphType.value = route.query.graph;
    centralityMode.value = centralityOptions.value[0].value;
}
initializeAggregationOptions();
fetchNetworkData();
</script>

<style scoped lang="scss">
@use "../assets/theme.module.scss" as theme;

.node-info-panel {
    position: absolute;
    top: 0.75rem;
    left: 0.75rem;
    width: fit-content;
    max-width: 220px;
    z-index: 10;
    background-color: #fff;
    border-color: theme.$rule-strong;
    border-top: 3px solid theme.$header-color;
    box-shadow: 0 6px 24px -10px rgba(64, 40, 24, 0.4) !important;
}

.node-info-panel .card-body {
    background-color: #fff;
    border-radius: 0 0 var(--bs-card-inner-border-radius) var(--bs-card-inner-border-radius);
}

.node-info-panel .node-label-text {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    display: block;
    min-width: 0;
    flex: 1;
}


// Headings that keep the size of the panel headers
.panel-title {
    font-size: inherit;
    font-weight: inherit;
    line-height: inherit;
    margin: 0;
}


.edge-info-panel {
    position: absolute;
    top: 0px;
    right: 0px;
    width: 420px;
    max-width: 100%;
    max-height: 100%;
    overflow-y: auto;
    z-index: 10;
    background-color: #fff;
    border-color: theme.$rule-strong;
    border-top: 3px solid theme.$target-color;
    box-shadow: 0 6px 24px -10px rgba(64, 40, 24, 0.4) !important;
    font-size: 0.9rem;
}

// Names in the text face: the display face is for short headings
.node-info-panel .card-header,
.edge-info-panel .card-header {
    font-family: theme.$font-text;
    font-variation-settings: normal;
    font-size: 0.95rem;
}

.edge-info-panel .card-header {
    color: theme.$ink;
}

.edge-passage {
    display: block;
    padding: 0.5rem 0.4rem;
    border-top: 1px solid theme.$rule-color;
    color: theme.$ink !important;
    font-style: italic;
    text-decoration: none;
}

.edge-passage:hover {
    background-color: theme.$source-tint;
}

.edge-passage-text {
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

.edge-passage-meta {
    margin-top: 0.2rem;
    font-family: theme.$font-ui;
    font-size: 0.75rem;
    font-style: normal;
    font-weight: 600;
    color: theme.$link-color;
}

.graph-toolbar {
    position: absolute;
    left: 0.75rem;
    bottom: 2.25rem;
    z-index: 10;
}

.graph-toolbar svg {
    display: block;
    width: 1rem;
    height: 1rem;
}

// Years under the chronological layout, on a thread from earlier to later
.year-axis {
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    height: 1.75rem;
    border-top: 1.5px solid transparent;
    border-image: linear-gradient(90deg, theme.$source-color, theme.$target-color) 1;
    background-color: rgba(255, 255, 255, 0.85);
    pointer-events: none;
}

.year-tick {
    position: absolute;
    top: 0.25rem;
    transform: translateX(-50%);
    color: theme.$ink;
    @include theme.display-type;
    font-size: 0.85rem;
    font-weight: 600;
}

.graph-controls {
    padding: 1rem 1.25rem 0.85rem;
    font-family: theme.$font-ui;
}

.form-label {
    @include theme.kicker;
    color: theme.$muted;
}

.loading-message {
    color: theme.$ink;
}

.network-data h3 {
    @include theme.kicker;
    color: theme.$header-color;
    font-size: 0.78rem;
}

.network-data .table {
    font-size: 0.88rem;
}

.network-data thead th {
    font-family: theme.$font-ui;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    color: theme.$muted;
    border-bottom-color: theme.$rule-strong;
}

.network-data .node-button {
    color: theme.$link-color;
    text-align: left;
    text-decoration: none;
    font-size: inherit;
}

.network-data .node-button:hover {
    background-color: rgba(theme.$link-color, 0.08);
}

.network-data .node-button[aria-pressed="true"] {
    font-weight: bold;
}

.swatch {
    display: inline-block;
    width: 0.8em;
    height: 0.8em;
    margin-right: 0.35em;
    border: 1px solid rgba(0, 0, 0, 0.6);
    border-radius: 50%;
    vertical-align: -0.05em;
}

// Paper canvas, with a slow drift of ink specks behind the graph
#sigma-container.vector-space-bg {
    background: radial-gradient(ellipse at 50% 45%, #fffdf9 0%, theme.$paper 60%, #f1ebe2 100%);
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

@keyframes stars {
    from {
        background-position: 0% 0%, 10% 10%, 20% 20%, 30% 30%, 40% 40%, 50% 50%, 60% 60%;
    }

    to {
        background-position: 100% 100%, 110% 110%, 120% 120%, 130% 130%, 140% 140%, 150% 150%, 160% 160%;
    }
}
</style>
