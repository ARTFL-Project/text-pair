<template>
    <div class="mt-3">
        <div class="container-fluid">
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
                        style="width: 10rem; height: 10rem; position: absolute; z-index: 50; top: 300px; color: #fff"
                        role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <div class="position-absolute fw-bold fs-6 text-center"
                        style="z-index: 51; top: 370px; width: 200px; color: #fff;">
                        {{ loadingMessage }}
                    </div>
                </div>

                <!-- What is selected, and the two global toggles. Choosing
                     values happens in the sidebar. -->
                <div class="card-body p-2 border-bottom">
                    <div class="d-flex flex-wrap align-items-center gap-2">
                        <span>
                            <strong>{{ selectionTotals.selected.toLocaleString() }}</strong>
                            of {{ selectionTotals.universe.toLocaleString() }} passages
                            <span class="text-muted" v-if="activeFilters.length">
                                ({{ Math.round(selectionTotals.share * 100) }}%)
                            </span>
                            in {{ visibleClusters }} themes
                        </span>
                        <span v-for="f in activeFilters" :key="f.key" class="badge bg-secondary filter-chip"
                            style="cursor: pointer;" @click="clearFilter(f.key)"
                            :title="'Remove ' + f.label + ' filter'">
                            {{ f.label }}: {{ f.display }} &times;
                        </span>
                        <button v-if="activeFilters.length" class="btn btn-sm btn-link p-0"
                            @click="clearAllFilters">clear all</button>

                        <div class="btn-group btn-group-sm ms-auto" role="group" title="Which side of the pair to match">
                            <input type="radio" class="btn-check" id="roleSource" value="source"
                                v-model="authorRole">
                            <label class="btn btn-outline-secondary" for="roleSource">source</label>
                            <input type="radio" class="btn-check" id="roleTarget" value="target"
                                v-model="authorRole">
                            <label class="btn btn-outline-secondary" for="roleTarget">target</label>
                            <input type="radio" class="btn-check" id="roleEither" value="either"
                                v-model="authorRole">
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
                <div class="graph-layout" style="display: flex; align-items: stretch;">
                    <div style="flex: 1 1 auto; min-width: 0; position: relative;">
                        <div id="sigma-container" ref="sigmaContainer" class="vector-space-bg"
                            style="height: calc(100vh - 300px); min-height: 600px;"></div>

                        <!-- Theme legend. Clicking highlights that theme's
                             points and nothing else. -->
                        <div class="theme-legend" ref="themeLegend" v-if="legendThemes.length">
                            <div class="theme-legend-head" @click="legendOpen = !legendOpen">
                                <span><strong>Themes</strong> ({{ legendThemes.length }})</span>
                                <span>{{ legendOpen ? '&minus;' : '+' }}</span>
                            </div>
                            <div class="theme-legend-body" v-show="legendOpen">
                                <div v-for="t in legendThemes" :key="t.id" class="theme-legend-item"
                                    :class="{ 'theme-legend-item-on': highlightedTheme === t.id }"
                                    @click="toggleLegendTheme(t.id)" :title="t.label + ' — ' + t.count + ' passages'">
                                    <span class="legend-color-box"
                                        :style="{ backgroundColor: getClusterColor(t.id) }"></span>
                                    <span class="theme-legend-label">{{ t.label }}</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- One sidebar for everything: themes, authors, and the
                         readout for whatever is hovered or selected. -->
                    <div class="graph-sidebar card shadow-lg">
                        <!-- Readout: hover wins, then selection. -->
                        <div v-if="hoveredPoint" class="sidebar-readout p-2 border-bottom">
                            <div class="d-flex align-items-center">
                                <div class="legend-color-box"
                                    :style="{ backgroundColor: getClusterColor(hoveredPoint.theme) }"></div>
                                <span class="ms-2"><strong>{{ hoveredPoint.themeLabel }}</strong></span>
                            </div>
                            <div class="text-truncate">{{ hoveredPoint.sourceAuthor || '—' }}</div>
                            <div class="text-truncate text-muted">&harr; {{ hoveredPoint.targetAuthor
                                || '—' }}</div>
                            <div v-if="hoveredPoint.merged" class="text-warning">unclustered</div>
                        </div>

                        <!-- One dimension at a time. Each row is a filter and
                             its own proportion bar; the bar is the share of
                             what is currently in view. -->
                        <div class="sidebar-dim-picker p-2 border-bottom">
                            <div class="btn-group btn-group-sm w-100 mb-2" role="group">
                                <button v-for="d in filterDimensions" :key="d.key" type="button"
                                    class="btn btn-outline-secondary"
                                    :class="{ active: activeDimension === d.key }"
                                    @click="setDimension(d.key)">
                                    {{ d.label }}
                                    <span v-if="filters[d.key] !== null && filters[d.key] !== undefined">&nbsp;&#9679;</span>
                                </button>
                            </div>
                            <input type="search" class="form-control form-control-sm"
                                :placeholder="'Search ' + activeDimensionLabel.toLowerCase() + 's'"
                                v-model="dimensionQuery">
                        </div>

                        <div class="sidebar-body p-2">
                            <div v-if="activeDimensionFiltered" class="text-muted">
                                Filtered to <strong>{{ activeDimensionFilterLabel }}</strong>.
                                <a href="#" @click.prevent="clearFilter(activeDimension)">Clear</a>
                                to see the spread again.
                            </div>
                            <template v-else>
                                <div class="d-flex justify-content-between align-items-baseline mb-1">
                                    <span class="text-muted">{{ activeDimensionOptions.length.toLocaleString() }}
                                        {{ activeDimensionLabel.toLowerCase() }}{{ activeDimensionOptions.length === 1 ? '' : 's' }}
                                        in view</span>
                                    <span class="text-muted">share of view</span>
                                </div>
                                <div v-for="o in activeDimensionOptions.slice(0, dimensionLimit)" :key="o.value"
                                    class="dim-row" @click="setFilter(activeDimension, o.value)"
                                    @mouseenter="hoveredOption = o.value" @mouseleave="hoveredOption = null"
                                    :title="o.label">
                                    <div class="dim-fill"
                                        :style="{ width: (o.shareOfContext * 100) + '%', backgroundColor: activeDimension === 'theme' ? getClusterColor(o.value) : '#9ab0c8' }">
                                    </div>
                                    <div class="dim-row-body">
                                        <span v-if="activeDimension === 'theme'" class="legend-color-box me-1"
                                            :style="{ backgroundColor: getClusterColor(o.value) }"></span>
                                        <span class="dim-label">{{ o.label }}</span>
                                        <span class="dim-metric" v-if="hoveredOption === o.value">
                                            {{ Math.round(o.shareOfValue * 100) }}% of its own total
                                        </span>
                                        <span class="dim-metric" v-else>
                                            {{ o.count.toLocaleString() }} &middot;
                                            {{ Math.round(o.shareOfContext * 100) }}%
                                        </span>
                                    </div>
                                </div>
                                <button v-if="activeDimensionOptions.length > dimensionLimit"
                                    class="btn btn-sm btn-link p-0" @click="dimensionLimit += 50">
                                    show more
                                </button>
                                <div v-if="!activeDimensionOptions.length" class="text-muted">
                                    {{ dimensionQuery ? 'No match in view.' : 'Nothing in view.' }}
                                </div>
                            </template>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Full alignment for a clicked passage -->
            <div class="modal modal-xl modal-dialog-scrollable fade" id="semantic-passage-pair" tabindex="-1"
                aria-labelledby="semantic-passage-pair" aria-hidden="true">
                <div class="modal-dialog">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h1 class="modal-title fs-5">Alignment</h1>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"
                                aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                            <div v-if="alignmentLoading" class="text-center p-3">
                                <div class="spinner-border spinner-border-sm"></div>
                            </div>
                            <div v-else-if="alignmentError" class="alert alert-warning mb-0">
                                {{ alignmentError }}
                            </div>
                            <passage-pair v-else-if="clickedAlignment" :alignment="clickedAlignment" :index="0"
                                :diffed="false"></passage-pair>
                        </div>
                    </div>
                </div>
            </div>

        </div>
    </div>
</template>

<script>
import { Modal } from "bootstrap";
import createScatterplot from "regl-scatterplot";
import passagePair from "./passagePair";
import reportSwitcher from "./reportSwitcher";

import searchArguments from "./searchArguments";

// Opacity of points outside the highlight, and the smallest area the camera
// will frame, in data units (the layout spans [-1, 1]).
const DIM_OPACITY = 0.2;
const MIN_ZOOM_SPAN = 0.05;
const ZOOM_PADDING = 0.06;

export default {
    name: "semanticGraph",
    components: {
        searchArguments,
        reportSwitcher,
        passagePair
    },
    inject: ["$http"],
    data() {
        return {
            loading: false,
            loadingMessage: "Fetching data...",
            error: null,
            globalConfig: this.$globalConfig,

            // Network data
            graph: null,
            renderer: null,
            rawData: { nodes: [], edges: [] },
            clusterInfo: {}, // Track cluster statistics
            clusterLabels: {}, // Track cluster labels (cluster_id -> label)
            clusterCentroids: {}, // Track cluster centroid positions (cluster_id -> {x, y})
            clusterColorMap: new Map(), // Track assigned colors per cluster
            clusterMetadata: null, // Metadata about clusters (n_clusters, n_noise, etc)
            showClusterLabels: true, // Toggle for floating labels

            // Color palette for clusters
            colorPalette: [
                '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
                '#6C5CE7', '#A29BFE', '#FD79A8', '#FDCB6E', '#00B894',
                '#0984E3', '#E17055', '#D63031', '#FF7675', '#74B9FF',
                '#55EFC4', '#81ECEC', '#FDA7DF', '#FAB1A0', '#00CEC9',
                '#FF6348', '#1E90FF', '#FF1493', '#32CD32', '#FF6347',
                '#4169E1', '#FF69B4', '#00FA9A', '#FFD700', '#E84393'
            ],

            // UI state
            hoveredPoint: null,
            selectedPoints: [],
            clickedAlignment: null,
            alignmentLoading: false,
            alignmentError: null,
            scatterplot: null,
            pointThemes: null,
            themeIds: [],
            noiseCategory: null,
            visibleNodes: 0,
            legendExpanded: false,
            // One filter per dimension, combined with AND. null means unset.
            // theme is always available; the rest come from rawData.facets.
            filters: { theme: null },
            // Which dimension the sidebar is browsing. Author by default: the
            // question this view exists for is who is being reused, and in what.
            activeDimension: "author",
            dimensionQuery: "",
            // Theme highlighted from the legend. Independent of the filters --
            // the map shows one highlight, and the last thing touched wins.
            highlightedTheme: null,
            hoveredOption: null,
            painting: false,
            pendingPaint: null,
            dimOffset: 0,
            legendOpen: true,
            dimensionLimit: 50,
            // Set while a filter is applied programmatically, so the
            // single-point click handler does not mistake it for a click.
            suppressPointModal: false,
            // Which side of the alignment a facet filter applies to.
            authorRole: "source",
            // Off by default. HDBSCAN rejected these points as noise; the theme
            // on them is a nearest-centroid assignment, not a density finding.
            showNoise: false,

        };
    },
    created() {
        // Non-reactive property
        this.connectedNodes = new Set();
        this.fetchSemanticData();
    },
    beforeUnmount() {
        if (this.scatterplot) {
            this.scatterplot.destroy();
        }
    },
    watch: {
        $route(to, from) {
            this.fetchSemanticData();
        },
        showNoise() {
            this.applySelection();
        },
        authorRole() {
            // A facet filter means something different per side, so drop the
            // facet filters and keep the theme, which has no side.
            const kept = { theme: this.filters.theme ?? null };
            this.filters = kept;
            this.applySelection();
        }
    },
    computed: {
        coreCount() {
            return this.rawData?.metadata?.directly_themed || 0;
        },
        mergedCount() {
            return this.rawData?.metadata?.merged_in || 0;
        },
        // Themes the current selection touches, not the payload total.
        visibleClusters() {
            const pts = this.rawData?.points;
            if (!pts || !pts.theme) return 0;
            if (!this.activeFilters.length) return (this.rawData.themes || []).length;
            const seen = new Set();
            for (const i of this.filteredIndices) seen.add(pts.theme[i]);
            return seen.size;
        },

        availableFacets() {
            return this.rawData?.facets || [{ key: "author", label: "Author", names: null }];
        },


        // Dimensions the user can filter on: theme plus whatever facets the
        // payload carries. Role applies to the facet dimensions only -- a theme
        // belongs to the passage, not to one side of it.
        filterDimensions() {
            const dims = [{ key: "theme", label: "Theme" }];
            for (const f of this.availableFacets) {
                dims.push({ key: f.key, label: f.label });
            }
            return dims;
        },

        // Dimensions with no value chosen. Once a dimension is filtered its
        // spread describes passages outside the selection, so it gets no
        // breakdown.
        openDimensions() {
            return this.filterDimensions.filter(
                (d) => this.filters[d.key] === null || this.filters[d.key] === undefined
            );
        },

        activeDimensionLabel() {
            const d = this.filterDimensions.find((x) => x.key === this.activeDimension);
            return d ? d.label : "";
        },

        activeDimensionFiltered() {
            const v = this.filters[this.activeDimension];
            return v !== null && v !== undefined;
        },

        activeDimensionFilterLabel() {
            return this.facetValueName(this.activeDimension, this.filters[this.activeDimension]);
        },

        activeDimensionOptions() {
            if (this.activeDimensionFiltered) return [];
            const options = this.optionsFor(this.activeDimension);
            const q = this.dimensionQuery.trim().toLowerCase();
            return q ? options.filter((o) => o.label.toLowerCase().includes(q)) : options;
        },

        // Legend covers every theme in the payload, largest first, so the
        // colours stay stable as filters change.
        legendThemes() {
            return [...(this.rawData?.themes || [])]
                .map((t) => ({ id: t.id, label: t.label || `Theme ${t.id}`, count: t.count }))
                .sort((a, b) => b.count - a.count);
        },

        activeFilters() {
            return this.filterDimensions
                .filter((d) => this.filters[d.key] !== null && this.filters[d.key] !== undefined)
                .map((d) => ({
                    key: d.key,
                    label: d.label,
                    value: this.filters[d.key],
                    display: d.key === "theme"
                        ? (this.clusterLabels[this.filters[d.key]] || `Theme ${this.filters[d.key]}`)
                        : this.facetValueName(d.key, this.filters[d.key]),
                }));
        },

        // Point indices satisfying every active filter. The basis for both the
        // map highlight and every count shown in the sidebar.
        filteredIndices() {
            const pts = this.rawData?.points;
            if (!pts || !pts.theme) return [];
            const active = this.activeFilters;
            const out = [];
            for (let i = 0; i < pts.theme.length; i++) {
                if (!this.showNoise && pts.merged[i] === 1) continue;
                let ok = true;
                for (const f of active) {
                    if (f.key === "theme") {
                        if (pts.theme[i] !== f.value) { ok = false; break; }
                    } else {
                        const s = pts[`source_${f.key === "author" ? "author" : f.key}`];
                        const t = pts[`target_${f.key === "author" ? "author" : f.key}`];
                        const hit = (this.authorRole === "source" && s && s[i] === f.value)
                            || (this.authorRole === "target" && t && t[i] === f.value)
                            || (this.authorRole === "either" && ((s && s[i] === f.value) || (t && t[i] === f.value)));
                        if (!hit) { ok = false; break; }
                    }
                }
                if (ok) out.push(i);
            }
            return out;
        },

        // Totals, which stay visible at every level of drill-down.
        selectionTotals() {
            const pts = this.rawData?.points;
            const universe = pts && pts.theme
                ? (this.showNoise ? pts.theme.length : pts.merged.filter((m) => m === 0).length)
                : 0;
            const n = this.activeFilters.length ? this.filteredIndices.length : universe;
            return { selected: n, universe, share: universe ? n / universe : 0 };
        },


    },
    methods: {
        hslToHex(h, s, l) {
            const sN = s / 100;
            const lN = l / 100;
            const k = (n) => (n + h / 30) % 12;
            const a = sN * Math.min(lN, 1 - lN);
            const f = (n) => {
                const v = lN - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
                return Math.round(255 * v).toString(16).padStart(2, "0");
            };
            return `#${f(0)}${f(8)}${f(4)}`;
        },

        fetchSemanticData() {
            this.loading = true;
            this.loadingMessage = "Fetching semantic graph data...";
            this.error = null;

            // Destroy existing renderer and graph
            if (this.scatterplot) {
                this.scatterplot.destroy();
                this.scatterplot = null;
            }

            let params = { ...this.$route.query };
            params.db_table = this.globalConfig.databaseName;

            this.emitter.emit("searchArgsUpdate", {
                counts: "",
                searchParams: params,
            });

            this.$http
                .get(`${this.globalConfig.apiServer}/semantic_scatter_data/?${this.paramsToUrl(params)}`)
                .then((response) => {
                    if (response.data.error) {
                        this.error = response.data.error;
                        this.loading = false;
                        return;
                    }

                    this.rawData = Object.freeze(response.data);

                    // Extract n_clusters from metadata
                    this.clusterMetadata = {
                        n_clusters: response.data.metadata?.n_themes || 0
                    };

                    this.loadingMessage = "Initializing graph...";

                    // Calculate cluster statistics
                    this.calculateClusterInfo();
                    // A corpus may carry no author facet at all.
                    if (!this.filterDimensions.some((d) => d.key === this.activeDimension)) {
                        this.activeDimension = this.filterDimensions[0].key;
                    }
                    this.highlightedTheme = null;

                    // Use setTimeout to allow the UI to update
                    setTimeout(() => {
                        this.initializeGraph();
                        this.loading = false;
                    }, 10);

                    this.emitter.emit("searchArgsUpdate", {
                        counts: response.data.metadata?.total_points || 0,
                        searchParams: params,
                    });
                })
                .catch((error) => {
                    this.loading = false;
                    this.error = error.toString();
                    console.log(error);
                });
        },

        calculateClusterInfo() {
            this.clusterInfo = {};
            this.clusterLabels = {};
            this.clusterCentroids = {};

            const themes = this.rawData.themes || [];
            const authorMarkers = this.rawData.authors || [];

            // Distinct authors per theme, from the author markers rather than by
            // walking passage points: a passage carries two authors and would be
            // double counted.
            const authorsPerTheme = {};
            authorMarkers.forEach((m) => {
                if (!authorsPerTheme[m.theme]) authorsPerTheme[m.theme] = new Set();
                authorsPerTheme[m.theme].add(m.author);
            });

            themes.forEach((t) => {
                this.clusterLabels[t.id] = t.label || '';
                this.clusterCentroids[t.id] = { x: t.x, y: t.y };
                this.clusterInfo[t.id] = {
                    authors: authorsPerTheme[t.id] ? authorsPerTheme[t.id].size : 0,
                    passages: t.count,
                    coherence: t.coherence,
                    terms: t.terms || [],
                };
            });
        },
        getClusterColor(clusterId) {
            // Check if this is a mini-cluster (singleton/noise cluster)
            // Mini-clusters have IDs >= n_clusters (the real clusters)
            if (this.clusterMetadata && clusterId >= this.clusterMetadata.n_clusters) {
                return '#FFFFFF';  // White for mini-clusters
            }

            // Check if we already assigned a color to this cluster
            if (this.clusterColorMap.has(clusterId)) {
                return this.clusterColorMap.get(clusterId);
            }

            // Golden-angle hue spacing, so colours stay separated for any
            // number of clusters. Hex, not hsl(), which the renderer rejects.
            const index = this.clusterColorMap.size;
            const color = this.hslToHex((index * 137.508) % 360, 70 + (index % 2) * 15, 55 + (index % 3) * 8);
            this.clusterColorMap.set(clusterId, color);
            return color;
        },

        initializeGraph() {
            // regl-scatterplot, not a graph renderer: this view needs points,
            // colour, hover, zoom and lasso -- no node/edge model.
            const pts = this.rawData.points;
            const themes = this.rawData.themes || [];
            this.themeIds = themes.map((t) => t.id);

            // Colour is a per-point categorical encoding: `z` carries the theme
            // id and pointColor maps it. Noise gets an extra slot at the end.
            const maxTheme = this.themeIds.length ? Math.max(...this.themeIds) : 0;
            const vivid = [];
            for (let i = 0; i <= maxTheme; i++) vivid.push(this.getClusterColor(i));
            this.noiseCategory = maxTheme + 1;
            vivid.push("#39465a");
            // Second half of the palette is the same colours desaturated, for
            // points outside the current highlight.
            this.dimOffset = vivid.length;
            const palette = [...vivid, ...vivid.map((c) => this.desaturate(c))];

            // Noise is given its own category rather than its theme's, so it is
            // never coloured as if HDBSCAN had placed it there.
            const z = new Float32Array(pts.theme.length);
            for (let i = 0; i < z.length; i++) {
                z[i] = pts.merged[i] === 1 ? this.noiseCategory : pts.theme[i];
            }

            if (this.scatterplot) this.scatterplot.destroy();
            const canvas = this.$refs.sigmaContainer.querySelector("canvas")
                || document.createElement("canvas");
            if (!canvas.parentNode) {
                canvas.style.width = "100%";
                canvas.style.height = "100%";
                this.$refs.sigmaContainer.appendChild(canvas);
            }

            this.scatterplot = createScatterplot({
                canvas,
                width: "auto",
                height: "auto",
                pointSize: 2.4,
                pointSizeSelected: 4,
                pointColor: palette,
                opacityBy: "density",
                lassoOnLongPress: true,
                backgroundColor: [0.043, 0.047, 0.06, 1],
                xScale: null,
                yScale: null,
            });
            this.scatterplot.set({ colorBy: "z" });

            this.scatterplot.subscribe("pointOver", (i) => this.describePoint(i));
            this.scatterplot.subscribe("pointOut", () => { this.hoveredPoint = null; });
            this.scatterplot.subscribe("select", ({ points }) => {
                this.selectedPoints = points || [];
                // A single point is a click; a lasso or a filter hands back
                // many, and neither should open a modal.
                if (this.selectedPoints.length === 1 && !this.suppressPointModal) {
                    this.showAlignment(this.selectedPoints[0]);
                }
            });
            this.scatterplot.subscribe("deselect", () => { this.selectedPoints = []; });

            this.pointThemes = z;
            this.paintHighlight(null);
        },

        // Mix a colour most of the way to its own luminance, so a dimmed point
        // still hints at its theme without reading as one.
        desaturate(hex) {
            const r = parseInt(hex.slice(1, 3), 16);
            const g = parseInt(hex.slice(3, 5), 16);
            const b = parseInt(hex.slice(5, 7), 16);
            const grey = 0.3 * r + 0.59 * g + 0.11 * b;
            const f = (c) => Math.round(0.9 * (0.25 * c + 0.75 * grey))
                .toString(16).padStart(2, "0");
            return `#${f(r)}${f(g)}${f(b)}`;
        },


        // Values for one dimension, counted under every *other* active filter.
        optionsFor(key) {
            const pts = this.rawData?.points;
            if (!pts || !pts.theme) return [];
            const others = this.activeFilters.filter((f) => f.key !== key);
            const counts = new Map();
            let inContext = 0;
            for (let i = 0; i < pts.theme.length; i++) {
                if (!this.showNoise && pts.merged[i] === 1) continue;
                let ok = true;
                for (const f of others) {
                    if (f.key === "theme") {
                        if (pts.theme[i] !== f.value) { ok = false; break; }
                    } else {
                        const s = pts[`source_${f.key}`];
                        const t = pts[`target_${f.key}`];
                        const hit = (this.authorRole === "source" && s && s[i] === f.value)
                            || (this.authorRole === "target" && t && t[i] === f.value)
                            || (this.authorRole === "either" && ((s && s[i] === f.value) || (t && t[i] === f.value)));
                        if (!hit) { ok = false; break; }
                    }
                }
                if (!ok) continue;
                inContext += 1;
                if (key === "theme") {
                    counts.set(pts.theme[i], (counts.get(pts.theme[i]) || 0) + 1);
                } else {
                    const ids = this.authorRole === "target"
                        ? [pts[`target_${key}`] && pts[`target_${key}`][i]]
                        : this.authorRole === "either"
                            ? [pts[`source_${key}`] && pts[`source_${key}`][i], pts[`target_${key}`] && pts[`target_${key}`][i]]
                            : [pts[`source_${key}`] && pts[`source_${key}`][i]];
                    for (const id of new Set(ids)) {
                        if (id === undefined || id === null || id < 0) continue;
                        if (!this.facetValueName(key, id)) continue;
                        counts.set(id, (counts.get(id) || 0) + 1);
                    }
                }
            }
            // total for each value across the whole corpus, for the second
            // proportion -- "how much of this author is in view"
            const overall = this.overallCounts(key);
            return [...counts.entries()]
                .map(([value, count]) => ({
                    value,
                    label: key === "theme"
                        ? (this.clusterLabels[value] || `Theme ${value}`)
                        : this.facetValueName(key, value),
                    count,
                    shareOfContext: inContext ? count / inContext : 0,
                    shareOfValue: overall.get(value) ? count / overall.get(value) : 0,
                }))
                .sort((a, b) => b.count - a.count);
        },

        overallCounts(key) {
            const pts = this.rawData?.points;
            const out = new Map();
            if (!pts || !pts.theme) return out;
            for (let i = 0; i < pts.theme.length; i++) {
                if (!this.showNoise && pts.merged[i] === 1) continue;
                if (key === "theme") {
                    out.set(pts.theme[i], (out.get(pts.theme[i]) || 0) + 1);
                } else {
                    const ids = this.authorRole === "target"
                        ? [pts[`target_${key}`] && pts[`target_${key}`][i]]
                        : this.authorRole === "either"
                            ? [pts[`source_${key}`] && pts[`source_${key}`][i], pts[`target_${key}`] && pts[`target_${key}`][i]]
                            : [pts[`source_${key}`] && pts[`source_${key}`][i]];
                    for (const id of new Set(ids)) {
                        if (id === undefined || id === null || id < 0) continue;
                        out.set(id, (out.get(id) || 0) + 1);
                    }
                }
            }
            return out;
        },

        facetValueName(key, id) {
            if (key === "theme") return this.clusterLabels[id] || `Theme ${id}`;
            if (key === "author") {
                const names = this.rawData?.author_names || {};
                return names[String(id)] || "";
            }
            const facet = (this.rawData?.facets || []).find((f) => f.key === key);
            if (!facet || !facet.names) return "";
            return facet.names[id] || "";
        },

        setDimension(key) {
            this.activeDimension = key;
            this.dimensionQuery = "";
            this.dimensionLimit = 50;
        },

        setFilter(key, value) {
            this.highlightedTheme = null;
            this.filters = { ...this.filters, [key]: value === "" || value === null ? null : value };
            // Drill down: move to the next unfiltered dimension so the sidebar
            // shows what is inside the selection just made.
            if (key === this.activeDimension && this.filters[key] !== null) {
                const next = this.openDimensions[0];
                if (next) this.setDimension(next.key);
            }
            this.applySelection();
        },

        clearFilter(key) {
            this.setFilter(key, null);
        },

        clearAllFilters() {
            this.highlightedTheme = null;
            const cleared = {};
            for (const d of this.filterDimensions) cleared[d.key] = null;
            this.filters = cleared;
            this.applySelection();
        },

        // One highlight at a time: a legend theme if one is lit, otherwise the
        // filter intersection, otherwise nothing.
        applySelection() {
            if (!this.scatterplot) return;
            let idxs = null;
            const pts = this.rawData?.points;
            if (this.highlightedTheme !== null && pts && pts.theme) {
                idxs = [];
                for (let i = 0; i < pts.theme.length; i++) {
                    if (!this.showNoise && pts.merged[i] === 1) continue;
                    if (pts.theme[i] === this.highlightedTheme) idxs.push(i);
                }
            } else if (this.activeFilters.length) {
                idxs = this.filteredIndices;
            }
            this.scatterplot.deselect();
            this.paintHighlight(idxs);
            if (idxs === null) this.scatterplot.zoomToOrigin({ transition: true, transitionDuration: 400 });
            else if (idxs.length) this.frameSelection(idxs);
        },

        // Highlighted points keep their colour at full opacity and a larger
        // size; everything else is desaturated and faded but still there, so
        // the selection reads against the rest of the map.
        paintHighlight(idxs) {
            const pts = this.rawData?.points;
            if (!this.scatterplot || !pts || !this.pointThemes) return;
            if (this.painting) {
                this.pendingPaint = { idxs };
                return;
            }
            const n = this.pointThemes.length;
            const z = new Float32Array(n);
            const w = new Float32Array(n);
            if (idxs) {
                const on = new Uint8Array(n);
                for (const i of idxs) on[i] = 1;
                for (let i = 0; i < n; i++) {
                    z[i] = on[i] ? this.pointThemes[i] : this.pointThemes[i] + this.dimOffset;
                    w[i] = on[i];
                }
                this.scatterplot.set({
                    opacityBy: "valueW",
                    sizeBy: "valueW",
                    opacity: [DIM_OPACITY, 1],
                    pointSize: [1.8, 4],
                });
            } else {
                z.set(this.pointThemes);
                this.scatterplot.set({ opacityBy: "density", sizeBy: null, pointSize: 2.4 });
            }
            this.painting = true;
            this.scatterplot
                .draw({ x: pts.x, y: pts.y, z: Array.from(z), w: Array.from(w) },
                    { preventFilterReset: true, zDataType: "categorical", wDataType: "categorical" })
                .then(() => this.applyNoiseFilter())
                .catch(() => { })
                .then(() => {
                    this.painting = false;
                    const pending = this.pendingPaint;
                    this.pendingPaint = null;
                    if (pending) this.paintHighlight(pending.idxs);
                });
        },

        // Pan and zoom onto the highlight, as tight as the highlight allows.
        frameSelection(idxs) {
            const pts = this.rawData.points;
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
            const reserved = this.legendWidthFraction();
            if (reserved > 0) width += (width * reserved) / (1 - reserved);
            this.scatterplot.zoomToArea(
                { x, y, width, height },
                { transition: true, transitionDuration: 500 }
            );
        },

        legendWidthFraction() {
            const host = this.$refs.sigmaContainer;
            const legend = this.$refs.themeLegend;
            if (!host || !legend) return 0;
            const canvasBox = host.getBoundingClientRect();
            const legendBox = legend.getBoundingClientRect();
            if (!canvasBox.width || !legendBox.width) return 0;
            return Math.min(0.3, (legendBox.width + 16) / canvasBox.width);
        },

        toggleLegendTheme(id) {
            this.highlightedTheme = this.highlightedTheme === id ? null : id;
            this.applySelection();
        },





        applyNoiseFilter() {
            // Only the noise toggle. Metadata filters are shown as a selection
            // highlight instead of hiding points, so the filtered set stays
            // legible against the rest of the map.
            if (!this.scatterplot || !this.pointThemes) return;
            if (this.showNoise) return this.scatterplot.unfilter();
            const keep = [];
            for (let i = 0; i < this.pointThemes.length; i++) {
                if (this.pointThemes[i] !== this.noiseCategory) keep.push(i);
            }
            return this.scatterplot.filter(keep);
        },

        showAlignment(index) {
            const pts = this.rawData.points;
            const rowid = pts && pts.rowid ? pts.rowid[index] : null;
            if (rowid === null || rowid === undefined) return;
            this.clickedAlignment = null;
            this.alignmentError = null;
            this.alignmentLoading = true;
            if (!this.alignmentModal) {
                this.alignmentModal = new Modal(document.getElementById("semantic-passage-pair"));
            }
            this.alignmentModal.show();
            this.$http
                .get(`${this.globalConfig.apiServer}/alignment_by_rowid/`, {
                    params: { db_table: this.globalConfig.databaseName, rowid },
                })
                .then((response) => {
                    this.alignmentLoading = false;
                    if (response.data.error) {
                        this.alignmentError = response.data.error;
                        return;
                    }
                    this.clickedAlignment = response.data.alignment;
                })
                .catch((error) => {
                    this.alignmentLoading = false;
                    this.alignmentError = error.toString();
                });
        },

        describePoint(index) {
            const pts = this.rawData.points;
            const names = this.rawData.author_names || {};
            const themeId = pts.theme[index];
            this.hoveredPoint = {
                index,
                theme: themeId,
                themeLabel: this.clusterLabels[themeId] || `Theme ${themeId}`,
                sourceAuthor: names[pts.source_author[index]] || "",
                targetAuthor: names[pts.target_author[index]] || "",
                merged: pts.merged[index] === 1,
                rowid: pts.rowid[index],
            };
        },

        viewAuthorInCluster(author, cluster) {
            // Navigate to results view with filters
            let queryParams = { ...this.$route.query };
            queryParams.source_author = `"${author}"`;
            queryParams.db_table = this.globalConfig.databaseName;
            this.$router.push(`/?${this.paramsToUrl(queryParams)}`);
        },


        desaturateColor(hexColor, amount = 0.6) {
            // Convert hex to RGB
            const r = parseInt(hexColor.slice(1, 3), 16);
            const g = parseInt(hexColor.slice(3, 5), 16);
            const b = parseInt(hexColor.slice(5, 7), 16);

            // Convert to grayscale
            const gray = Math.round(0.299 * r + 0.587 * g + 0.114 * b);

            // Mix with original color
            const newR = Math.round(gray * amount + r * (1 - amount));
            const newG = Math.round(gray * amount + g * (1 - amount));
            const newB = Math.round(gray * amount + b * (1 - amount));

            return `#${newR.toString(16).padStart(2, '0')}${newG.toString(16).padStart(2, '0')}${newB.toString(16).padStart(2, '0')}`;
        }
    }
};
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

.graph-sidebar {
    /* order, rather than moving the markup: the sidebar reads as the plot's
       annotation, so it stays after it in the DOM. */
    order: -1;
    flex: 0 0 380px;
    align-self: stretch;
    border-right: 1px solid theme.$graph-btn-panel-color;
    display: flex;
    flex-direction: column;
    background-color: rgba(256, 256, 256, 0.94);
    border-color: theme.$graph-btn-panel-color;
    border-radius: 0;
}

.graph-sidebar .nav-tabs {
    flex: 0 0 auto;
}

.sidebar-readout {
    flex: 0 0 auto;
    background-color: rgba(0, 0, 0, 0.03);
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
    cursor: pointer;
    border-radius: 3px;
    margin-bottom: 1px;
    overflow: hidden;
}

.dim-row:hover {
    outline: 1px solid rgba(0, 0, 0, 0.15);
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

.dim-metric {
    flex: 0 0 auto;
    color: #555;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
}

.sidebar-dim-picker {
    flex: 0 0 auto;
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
    background-color: rgba(255, 255, 255, 0.92);
    border: 1px solid theme.$graph-btn-panel-color;
    border-radius: 4px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
}

.theme-legend-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 4px 8px;
    cursor: pointer;
    border-bottom: 1px solid rgba(0, 0, 0, 0.1);
}

.theme-legend-body {
    overflow-y: auto;
    min-height: 0;
    padding: 4px;
}

.theme-legend-item {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 2px 4px;
    border-radius: 3px;
    cursor: pointer;
}

.theme-legend-item:hover {
    background-color: rgba(0, 0, 0, 0.07);
}

.theme-legend-item-on {
    background-color: rgba(0, 123, 255, 0.18);
    font-weight: 600;
}

.theme-legend-label {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.theme-legend .legend-color-box {
    width: 13px;
    height: 13px;
}

.filter-chip {
    font-size: 0.9rem;
    font-weight: 500;
    padding: 0.35em 0.6em;
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
    width: 16px;
    height: 16px;
    border-radius: 3px;
    flex-shrink: 0;
    border: 1px solid rgba(0, 0, 0, 0.2);
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

#sigma-container.vector-space-bg {
    background: radial-gradient(ellipse at bottom, #1B2735 0%, #090A0F 100%);
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
        radial-gradient(2px 2px at 20% 30%, white, transparent),
        radial-gradient(2px 2px at 60% 70%, white, transparent),
        radial-gradient(1px 1px at 50% 50%, white, transparent),
        radial-gradient(1px 1px at 80% 10%, white, transparent),
        radial-gradient(2px 2px at 90% 60%, white, transparent),
        radial-gradient(1px 1px at 33% 80%, white, transparent),
        radial-gradient(1px 1px at 15% 95%, white, transparent);
    background-size: 200% 200%, 180% 180%, 220% 220%, 190% 190%, 210% 210%, 240% 240%, 230% 230%;
    background-position: 0% 0%, 10% 10%, 20% 20%, 30% 30%, 40% 40%, 50% 50%, 60% 60%;
    background-repeat: repeat;
    animation: stars 200s linear infinite;
    opacity: 0.5;
    pointer-events: none;
}

@keyframes stars {
    from {
        background-position: 0% 0%, 10% 10%, 20% 20%, 30% 30%, 40% 40%, 50% 50%, 60% 60%;
    }

    to {
        background-position: 100% 100%, 110% 110%, 120% 120%, 130% 130%, 140% 140%, 150% 150%, 160% 160%;
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
