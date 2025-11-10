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

                <!-- Semantic Graph Controls -->
                <div class="card-body p-2 border-bottom">
                    <div class="row align-items-center">
                        <div class="col-md-12">
                            <span>
                                Showing {{ visibleNodes }} author-topic pairs in {{ visibleClusters }} clusters
                            </span>
                        </div>
                    </div>
                </div>

                <!-- Sigma Graph Container -->
                <div style="position: relative;">
                    <div id="sigma-container" ref="sigmaContainer" class="vector-space-bg"
                        style="width: 100%; height: calc(100vh - 300px); min-height: 600px;"></div>

                    <!-- Cluster Legend Panel -->
                    <div class="cluster-legend-panel card shadow-lg">
                        <div class="card-header p-2 d-flex justify-content-between align-items-center"
                            @click="legendExpanded = !legendExpanded" style="cursor: pointer;">
                            <small><strong>Cluster Legend</strong></small>
                            <i :class="legendExpanded ? 'bi bi-chevron-up' : 'bi bi-chevron-down'"></i>
                        </div>
                        <transition name="fade">
                            <div v-if="legendExpanded" class="card-body p-2"
                                style="max-height: 400px; overflow-y: auto;">
                                <div v-for="clusterId in sortedClusterIds" :key="clusterId" class="legend-item mb-1"
                                    :class="{ 'legend-item-selected': selectedLegendCluster === clusterId }"
                                    @click.stop="selectLegendCluster(clusterId)" style="cursor: pointer;">
                                    <div class="d-flex align-items-center">
                                        <div class="legend-color-box"
                                            :style="{ backgroundColor: getClusterColor(clusterId) }"></div>
                                        <small class="ms-2">
                                            <strong>{{ clusterId }}:</strong> {{ clusterLabels[clusterId] || 'Unlabeled'
                                            }}
                                        </small>
                                    </div>
                                </div>
                            </div>
                        </transition>
                    </div>

                    <!-- Cluster Info Panel (shows cluster or selected author) -->
                    <div v-if="hoveredCluster !== null || selectedNode" class="cluster-info-panel card shadow-lg">
                        <div class="card-header">
                            <strong v-if="selectedNode">{{ selectedNodeInfo.author }}</strong>
                            <strong v-else>{{ clusterLabels[hoveredCluster] || `Cluster ${hoveredCluster}` }}</strong>
                        </div>
                        <div class="card-body p-2">
                            <template v-if="selectedNode">
                                <small class="text-muted">{{ clusterLabels[selectedNodeInfo.cluster] || `Cluster
                                    ${selectedNodeInfo.cluster}` }}</small><br>
                                <small>{{ selectedNodeInfo.passages }} passages</small>
                                <div class="d-grid gap-2 mt-2">
                                    <button class="btn btn-sm btn-outline-secondary"
                                        @click="viewAuthorInCluster(selectedNodeInfo.author, selectedNodeInfo.cluster)">
                                        <i class="bi bi-search"></i> View Passages
                                    </button>
                                </div>
                            </template>
                            <template v-else>
                                <small>{{ clusterInfo[hoveredCluster]?.authors || 0 }} authors</small><br>
                                <small>{{ clusterInfo[hoveredCluster]?.passages || 0 }} passages</small>
                            </template>
                        </div>
                    </div>
                </div>
            </div>

        </div>
    </div>
</template>

<script>
import { createNodeBorderProgram } from "@sigma/node-border";
import Graph from "graphology";
import forceAtlas2 from "graphology-layout-forceatlas2";
import Sigma from "sigma";
import { EdgeLineProgram } from "sigma/rendering";
import reportSwitcher from "./reportSwitcher";

import noverlap from "graphology-layout-noverlap";
import searchArguments from "./searchArguments";

export default {
    name: "semanticGraph",
    components: {
        searchArguments,
        reportSwitcher
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
            selectedNode: null,
            selectedNodeInfo: null,
            hoveredCluster: null,
            visibleNodes: 0,
            visibleClusters: 0,
            legendExpanded: false,
            selectedLegendCluster: null
        };
    },
    computed: {
        sortedClusterIds() {
            // Get all cluster IDs that have labels, sorted numerically
            return Object.keys(this.clusterLabels)
                .map(id => parseInt(id))
                .filter(id => {
                    // Exclude mini-clusters (noise)
                    if (this.clusterMetadata && id >= this.clusterMetadata.n_clusters) {
                        return false;
                    }
                    return true;
                })
                .sort((a, b) => a - b);
        }
    },
    created() {
        // Non-reactive property
        this.connectedNodes = new Set();
        this.fetchSemanticData();
    },
    beforeUnmount() {
        if (this.renderer) {
            this.renderer.kill();
        }
    },
    watch: {
        $route(to, from) {
            this.fetchSemanticData();
        }
    },
    methods: {
        fetchSemanticData() {
            this.loading = true;
            this.loadingMessage = "Fetching semantic graph data...";
            this.error = null;

            // Destroy existing renderer and graph
            if (this.renderer) {
                this.renderer.kill();
                this.renderer = null;
            }
            if (this.graph) {
                this.graph.clear();
                this.graph = null;
            }

            let params = { ...this.$route.query };
            params.db_table = this.globalConfig.databaseName;

            this.emitter.emit("searchArgsUpdate", {
                counts: "",
                searchParams: params,
            });

            this.$http
                .get(`${this.globalConfig.apiServer}/semantic_graph_data/?${this.paramsToUrl(params)}`)
                .then((response) => {
                    if (response.data.error) {
                        this.error = response.data.error;
                        this.loading = false;
                        return;
                    }

                    this.rawData = Object.freeze(response.data);

                    // Extract n_clusters from metadata
                    this.clusterMetadata = {
                        n_clusters: response.data.metadata?.n_clusters || 0
                    };

                    this.loadingMessage = "Initializing graph...";

                    // Calculate cluster statistics
                    this.calculateClusterInfo();

                    // Use setTimeout to allow the UI to update
                    setTimeout(() => {
                        this.initializeGraph();
                        this.loading = false;
                    }, 10);

                    // Update search args with total count
                    const nodes = response.data.nodes || [];
                    const totalPassages = nodes.reduce((sum, node) => {
                        const size = node.passages || node.size || 0;
                        return sum + size;
                    }, 0);

                    this.emitter.emit("searchArgsUpdate", {
                        counts: totalPassages,
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
            const clusterAuthors = {};
            const clusterPassages = {};

            // Simple nodes/edges format now
            const nodes = this.rawData.nodes || [];

            nodes.forEach(node => {
                // Store cluster labels and centroids from anchor nodes
                if (node && node.node_type === 'cluster_anchor') {
                    this.clusterLabels[node.cluster_id] = node.cluster_label || '';
                    this.clusterCentroids[node.cluster_id] = { x: node.x, y: node.y };
                    return;
                }

                if (!node || node.node_type === 'cluster_anchor') return;

                const clusterId = node.cluster_id;
                const authorId = node.author_id;
                const size = node.passages || node.size || 0;

                // Also collect labels from regular nodes (as fallback)
                if (node.cluster_label && !this.clusterLabels[clusterId]) {
                    this.clusterLabels[clusterId] = node.cluster_label;
                }

                if (!clusterAuthors[clusterId]) {
                    clusterAuthors[clusterId] = new Set();
                    clusterPassages[clusterId] = 0;
                }
                clusterAuthors[clusterId].add(authorId);
                clusterPassages[clusterId] += size;
            });

            Object.keys(clusterAuthors).forEach(cluster => {
                this.clusterInfo[cluster] = {
                    authors: clusterAuthors[cluster].size,
                    passages: clusterPassages[cluster]
                };
            });
        }, getClusterColor(clusterId) {
            // Check if this is a mini-cluster (singleton/noise cluster)
            // Mini-clusters have IDs >= n_clusters (the real clusters)
            if (this.clusterMetadata && clusterId >= this.clusterMetadata.n_clusters) {
                return '#FFFFFF';  // White for mini-clusters
            }

            // Check if we already assigned a color to this cluster
            if (this.clusterColorMap.has(clusterId)) {
                return this.clusterColorMap.get(clusterId);
            }

            // Assign a new color from the palette for real clusters
            const color = this.colorPalette[this.clusterColorMap.size % this.colorPalette.length];
            this.clusterColorMap.set(clusterId, color);
            return color;
        },

        initializeGraph() {
            console.time('Total graph initialization');

            // Create new graph and build from simple nodes/edges arrays (like network graph)
            console.time('Build graph');
            this.graph = new Graph();

            const nodes = this.rawData.nodes || [];
            const edges = this.rawData.edges || [];
            const n_clusters = this.clusterMetadata?.n_clusters || 0;

            // Normalize node sizes based on passage count using logarithmic scale
            const passageCounts = nodes
                .filter(n => n.node_type !== 'cluster_anchor')
                .map(n => n.passages || n.size || 0);

            // Use log scale for better distribution with skewed data
            const nonZeroPassages = passageCounts.filter(p => p > 0);
            const maxPassages = Math.max(...nonZeroPassages, 1);

            // Use threshold from API metadata, or actual minimum if no threshold
            const minPassagesThreshold = this.rawData.metadata?.min_passages_threshold || Math.min(...nonZeroPassages, 1);
            console.log(`Passage count range: threshold=${minPassagesThreshold}, max=${maxPassages}`);

            // Scale from threshold to ensure smallest nodes are 1px
            const minLog = Math.log(minPassagesThreshold + 1);
            const maxLog = Math.log(maxPassages + 1);
            console.log(`Log range: minLog=${minLog.toFixed(4)}, maxLog=${maxLog.toFixed(4)}`);

            // Define size range (min and max pixel sizes)
            const minNodeSize = 1;
            const maxNodeSize = 25;

            // Track min/max sizes for debugging
            let actualMinSize = Infinity;
            let actualMaxSize = -Infinity;

            // Add nodes
            nodes.forEach(node => {
                const clusterId = node.cluster_id;
                const nodeColor = this.getClusterColor(clusterId);

                // Normalize size for non-anchor nodes using log scale
                let normalizedSize;
                if (node.node_type === 'cluster_anchor') {
                    normalizedSize = 0.01;  // Tiny for anchors
                } else {
                    const passages = node.passages || node.size || 0;
                    const logValue = Math.log(passages + 1);

                    if (maxLog === minLog) {
                        normalizedSize = (minNodeSize + maxNodeSize) / 2;
                    } else {
                        normalizedSize = minNodeSize +
                            (logValue - minLog) /
                            (maxLog - minLog) *
                            (maxNodeSize - minNodeSize);
                    }
                }

                // Track actual min/max (excluding anchors)
                if (node.node_type !== 'cluster_anchor') {
                    actualMinSize = Math.min(actualMinSize, normalizedSize);
                    actualMaxSize = Math.max(actualMaxSize, normalizedSize);
                }

                // Truncate label: at second comma or 20 chars, whichever comes first
                const fullLabel = node.label || node.author_name || '';
                let displayLabel = fullLabel;

                // Find second comma
                const firstComma = fullLabel.indexOf(',');
                if (firstComma !== -1) {
                    const secondComma = fullLabel.indexOf(',', firstComma + 1);
                    if (secondComma !== -1) {
                        // Truncate at second comma (no ellipsis), but respect 20 char limit
                        const commaLimit = secondComma;
                        if (commaLimit <= 20) {
                            displayLabel = fullLabel.substring(0, commaLimit);
                        } else {
                            displayLabel = fullLabel.substring(0, 20) + '...';
                        }
                    } else if (fullLabel.length > 20) {
                        // No second comma, but still too long
                        displayLabel = fullLabel.substring(0, 20) + '...';
                    }
                } else if (fullLabel.length > 20) {
                    // No commas, just truncate at 20
                    displayLabel = fullLabel.substring(0, 20) + '...';
                }

                this.graph.addNode(node.id, {
                    label: displayLabel,
                    fullLabel: fullLabel,  // Store full label for reference
                    x: node.x,
                    y: node.y,
                    size: normalizedSize,
                    color: nodeColor,
                    originalColor: nodeColor,
                    cluster_id: clusterId,
                    node_type: node.node_type || 'author_cluster_pair',
                    author_id: node.author_id,
                    author_name: node.author_name || node.label,
                    passages: node.passages || node.size,
                    hidden: node.hidden || false,
                    mass: 1.0
                });
            });

            console.log(`Node size range: min=${actualMinSize.toFixed(2)}, max=${actualMaxSize.toFixed(2)}`);

            // Add edges
            edges.forEach(edge => {
                if (this.graph.hasNode(edge.source) && this.graph.hasNode(edge.target)) {
                    this.graph.addEdge(edge.source, edge.target, {
                        weight: edge.weight || 1.0,
                        edge_type: edge.edge_type || 'default',
                        size: edge.size || 1.0,
                        color: edge.color || '#999999'
                    });
                }
            });

            console.timeEnd('Build graph');

            // Apply ForceAtlas2 layout starting from UMAP positions
            console.time('ForceAtlas2 layout');
            forceAtlas2.assign(this.graph, {
                iterations: 100,
                settings: {
                    gravity: 0.5,
                    scalingRatio: 20,
                    barnesHutOptimize: true,
                    slowDown: 5,
                }
            });
            console.timeEnd('ForceAtlas2 layout');

            // Apply noverlap to prevent node overlap
            console.time('Noverlap layout');
            noverlap.assign(this.graph, {
                maxIterations: 50,
                settings: {
                    margin: 10,
                    ratio: 1.2,
                    expansion: 1.1
                }
            });
            console.timeEnd('Noverlap layout');

            console.timeEnd('Total graph initialization');
            this.initRenderer();
        },

        initRenderer() {
            console.time('Renderer initialization');

            if (this.renderer) {
                this.renderer.kill();
            }

            this.renderer = new Sigma(this.graph, this.$refs.sigmaContainer, {
                edgeProgramClasses: {
                    line: EdgeLineProgram,
                },
                nodeProgramClasses: {
                    circle: createNodeBorderProgram({
                        borders: [
                            {
                                size: { value: 0.2 },
                                color: { attribute: "color" }
                            }
                        ]
                    })
                },
                renderEdgeLabels: false,
                defaultNodeColor: "#999",
                defaultEdgeColor: "#666",
                labelSize: 12,
                labelWeight: "bold",
                labelColor: { attribute: "labelColor", color: "#ffffff" },
                enableEdgeEvents: true,
                labelDensity: 0.3,  // Increased from 0.1 to show more labels
                labelGridCellSize: 150,  // Increased from 100 to reduce crowding
                labelRenderedSizeThreshold: 8,  // Reduced from 15 to show labels for smaller nodes
                zIndex: true,
                hideLabelsOnMove: true,
                hideEdgesOnMove: true,
            });

            // Set up reducers
            this.renderer.setSetting('nodeReducer', (node, data) => {
                const res = { ...data };

                if (data.highlighted) {
                    res.color = "#f00";
                    res.zIndex = 10;
                    res.labelColor = "#000000";
                    return res;
                }

                res.labelColor = "#ffffff";

                if (this.selectedLegendCluster !== null) {
                    // Legend cluster is selected - desaturate others
                    res.hidden = data.node_type === 'cluster_anchor' ? true : false;
                    if (data.cluster_id === this.selectedLegendCluster) {
                        res.color = data.originalColor || data.color;
                        res.zIndex = 5;
                    } else {
                        res.color = this.desaturateColor(data.originalColor || data.color, 0.9);
                        res.zIndex = 1;
                    }
                } else {
                    res.hidden = data.node_type === 'cluster_anchor' ? true : false;
                    res.zIndex = 1;
                }

                return res;
            });

            this.renderer.setSetting('edgeReducer', (edge, data) => {
                const res = { ...data };

                // Only show centroid similarity edges by default
                res.hidden = data.edge_type !== 'centroid_similarity';

                return res;
            });

            const camera = this.renderer.getCamera();
            camera.setState({ ratio: 1.0 });

            this.updateCounts();
            this.setupInteractions();

            console.timeEnd('Renderer initialization');
        },

        setupInteractions() {
            this.renderer.on("clickNode", ({ node }) => {
                const nodeData = this.graph.getNodeAttributes(node);

                // Skip anchor nodes
                if (nodeData.node_type === 'cluster_anchor') {
                    return;
                }

                // Toggle selection
                if (this.selectedNode === node) {
                    this.setSelectedNode(null);
                } else {
                    this.setSelectedNode(node);
                }
            });

            this.renderer.on("clickStage", () => {
                this.setSelectedNode(null);
                // Also clear legend cluster selection
                if (this.selectedLegendCluster !== null) {
                    this.selectedLegendCluster = null;
                    this.renderer.refresh({ skipIndexation: true });
                }
                // Hide legend if expanded
                if (this.legendExpanded) {
                    this.legendExpanded = false;
                }
            });

            this.renderer.on("enterNode", ({ node }) => {
                const nodeData = this.graph.getNodeAttributes(node);

                if (nodeData.node_type !== 'cluster_anchor') {
                    this.graph.setNodeAttribute(node, 'highlighted', true);
                    this.hoveredCluster = nodeData.cluster_id;
                }
            });

            this.renderer.on("leaveNode", ({ node }) => {
                this.graph.setNodeAttribute(node, 'highlighted', false);
                this.hoveredCluster = null;
            });
        },

        setSelectedNode(node) {
            if (node) {
                this.selectedNode = node;
                const nodeData = this.graph.getNodeAttributes(node);

                this.selectedNodeInfo = {
                    author: nodeData.author_name,
                    cluster: nodeData.cluster_id,
                    passages: nodeData.passages
                };
            } else {
                this.selectedNode = null;
                this.selectedNodeInfo = null;
            }

            this.renderer.refresh({ skipIndexation: true });
        },

        updateCounts() {
            this.visibleNodes = this.graph.filterNodes((node, attr) =>
                !attr.hidden && attr.node_type !== 'cluster_anchor'
            ).length;

            // Count visible clusters
            const clusters = new Set();
            this.graph.forEachNode((node, attr) => {
                if (!attr.hidden && attr.node_type !== 'cluster_anchor') {
                    clusters.add(attr.cluster_id);
                }
            });
            this.visibleClusters = clusters.size;
        },

        viewAuthorInCluster(author, cluster) {
            // Navigate to results view with filters
            let queryParams = { ...this.$route.query };
            queryParams.source_author = `"${author}"`;
            queryParams.db_table = this.globalConfig.databaseName;
            this.$router.push(`/?${this.paramsToUrl(queryParams)}`);
        },

        selectLegendCluster(clusterId) {
            // Toggle selection
            if (this.selectedLegendCluster === clusterId) {
                this.selectedLegendCluster = null;
            } else {
                this.selectedLegendCluster = clusterId;
            }
            this.renderer.refresh({ skipIndexation: true });
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

.cluster-info-panel {
    position: absolute;
    top: 0px;
    right: 0px;
    width: fit-content;
    max-width: 300px;
    z-index: 10;
    background-color: rgba(256, 256, 256, 0.9);
    border-color: theme.$graph-btn-panel-color;
}

.cluster-legend-panel {
    position: absolute;
    top: 0;
    left: 0;
    width: fit-content;
    min-width: 250px;
    height: fit-content;
    z-index: 10;
}

.cluster-legend-panel .card-header:hover {
    background-color: rgba(0, 0, 0, 0.05);
}

.cluster-legend-panel .card-body {
    max-height: 600px;
    overflow-y: auto;
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

.node-info-panel .btn-outline-secondary,
.cluster-info-panel .btn-outline-secondary {
    border-color: theme.$graph-btn-panel-color;
    color: theme.$graph-btn-panel-color;
}

.node-info-panel .btn-outline-secondary:hover,
.cluster-info-panel .btn-outline-secondary:hover {
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
