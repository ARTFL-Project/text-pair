<template>
    <div class="mt-3">
        <div class="container-fluid">
            <div class="row" style="padding: 0 0.75rem">
                <div class="m-4" style="font-size: 120%" v-if="error">{{ error }}</div>
                <search-arguments></search-arguments>
            </div>
            <report-switcher />

            <!-- Loading Spinner -->
            <div class="d-flex justify-content-center position-relative" v-if="loading">
                <div class="spinner-border"
                    style="width: 8rem; height: 8rem; position: absolute; z-index: 50; top: 100px" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            </div>

            <!-- Graph Container with Controls -->
            <div class="card shadow-1" style="position: relative;">
                <!-- Network Controls -->
                <div class="card-body p-3 border-bottom">
                    <div class="row align-items-center">
                        <div class="col-md-3">
                            <label class="form-label mb-1">Aggregate by:</label>
                            <select class="form-select form-select-sm" v-model="aggregationField"
                                @change="reloadNetwork">
                                <option v-for="option in availableAggregations" :key="option.value"
                                    :value="option.value">
                                    {{ option.label }}
                                </option>
                            </select>
                        </div>
                        <div class="col-md-4">
                            <label class="form-label mb-1">
                                Min. alignments: <strong>{{ minThreshold }}</strong>
                            </label>
                            <input type="range" class="form-range" v-model.number="minThreshold" min="1" max="100"
                                @change="applyThreshold">
                        </div>
                        <div class="col-md-3">
                            <label class="form-label mb-1">Layout:</label>
                            <select class="form-select form-select-sm" v-model="layoutType" @change="applyLayout">
                                <option value="force">Force-directed</option>
                                <option value="communities">Communities (Louvain)</option>
                                <option value="circular">Circular</option>
                                <option value="random">Random</option>
                            </select>
                        </div>
                        <div class="col-md-2">
                            <button class="btn btn-sm btn-secondary w-100 mt-3" @click="resetView">
                                Reset View
                            </button>
                        </div>
                    </div>
                    <div class="row mt-2">
                        <div class="col-md-12">
                            <small class="text-muted">
                                Showing {{ visibleNodes }} nodes and {{ visibleEdges }} edges
                                <span v-if="expandedNode"> | Expanded: <strong>{{ expandedNode }}</strong></span>
                            </small>
                        </div>
                    </div>
                </div>

                <!-- Sigma Graph -->
                <div id="sigma-container" ref="sigmaContainer" style="width: 100%; height: 800px;"></div>

                <!-- Edge Info Panel - Direction Selector -->
                <div v-if="selectedEdge" class="edge-info-panel card shadow-lg">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">Select View</h5>
                        <button type="button" class="btn-close" @click="clearSelection" aria-label="Close"></button>
                    </div>
                    <div class="card-body">
                        <p class="mb-3">
                            <span>{{ selectedEdge.weight }} total alignments</span>
                        </p>
                        <div class="d-grid gap-2">
                            <button class="btn btn-secondary"
                                @click="viewEdgeDirection(selectedEdge.source, selectedEdge.target)">
                                <i class="bi bi-arrow-right"></i> {{ selectedEdge.source }} → {{ selectedEdge.target }}
                            </button>
                            <button class="btn btn-secondary"
                                @click="viewEdgeDirection(selectedEdge.target, selectedEdge.source)">
                                <i class="bi bi-arrow-left"></i> {{ selectedEdge.target }} → {{ selectedEdge.source }}
                            </button>
                            <button class="btn btn-secondary"
                                @click="viewEdgeTitles(selectedEdge.source, selectedEdge.target)" v-if="canDrillDown">
                                <i class="bi bi-diagram-3"></i> View {{ drillDownLabel }} Network
                            </button>
                        </div>
                    </div>
                </div>
            </div>

        </div> <!-- Close container-fluid -->
    </div> <!-- Close mt-3 wrapper -->
</template>

<script>
import Graph from "graphology";
import louvain from "graphology-communities-louvain";
import forceAtlas2 from "graphology-layout-forceatlas2";
import noverlap from "graphology-layout-noverlap";
import circular from "graphology-layout/circular";
import Sigma from "sigma";
import passagePair from "./passagePair";
import reportSwitcher from "./reportSwitcher";
import searchArguments from "./searchArguments";

export default {
    name: "networkGraph",
    components: {
        searchArguments,
        reportSwitcher,
        passagePair
    },
    inject: ["$http"],
    data() {
        return {
            loading: false,
            error: null,
            globalConfig: this.$globalConfig,

            // Network data
            graph: null,
            renderer: null,
            rawData: { nodes: [], edges: [] },
            nodeTypes: new Map(), // Track node types separately
            nodeColorMap: new Map(), // Track assigned colors
            nodeCommunities: new Map(), // Track community assignments
            communityColors: new Map(), // Colors for communities
            colorPalette: [
                '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
                '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
                '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
                '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
                '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7',
                '#dfe6e9', '#74b9ff', '#a29bfe', '#fd79a8', '#fdcb6e',
                '#6c5ce7', '#00b894', '#00cec9', '#0984e3', '#b2bec3',
                '#e17055', '#d63031', '#ff7675', '#fab1a0', '#636e72'
            ],
            colorIndex: 0,

            // UI state
            aggregationField: "author",
            availableAggregations: [],
            minThreshold: 5,
            layoutType: "force",  // Start with force, user can switch to communities
            expandedNode: null,
            selectedNode: null,
            selectedEdge: null,
            visibleNodes: 0,
            visibleEdges: 0
        };
    },
    computed: {
        // Can we drill down to a more detailed view?
        canDrillDown() {
            return this.aggregationField !== 'title';
        },
        // What's the next level to drill down to?
        drillDownLabel() {
            // Find title field in available aggregations, or default to "Title"
            const titleAgg = this.availableAggregations.find(agg => agg.value === 'title');
            return titleAgg ? titleAgg.label : 'Title';
        }
    },
    created() {
        this.initializeAggregationOptions();
        this.fetchNetworkData();
    },
    beforeUnmount() {
        if (this.renderer) {
            this.renderer.kill();
        }
    },
    watch: {
        $route(to, from) {
            // Check if aggregation_field changed in URL
            if (to.query.aggregation_field &&
                to.query.aggregation_field !== this.aggregationField &&
                this.availableAggregations.some(opt => opt.value === to.query.aggregation_field)) {
                this.aggregationField = to.query.aggregation_field;
            }
            this.fetchNetworkData();
        }
    },
    methods: {
        initializeAggregationOptions() {
            // Build aggregation options from metadata fields
            const fieldMap = new Map();

            // Check source and target fields
            const sourceFields = this.globalConfig.metadataFields.source || [];
            const targetFields = this.globalConfig.metadataFields.target || [];

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
            this.availableAggregations = Array.from(fieldMap.entries())
                .map(([value, label]) => ({ value, label }))
                .filter(option => {
                    // Exclude passage_length and year
                    if (option.value === 'passage_length' || option.value === 'year') {
                        return false;
                    }
                    // Only include if both source and target have this field
                    const hasSource = sourceFields.some(f => f.value === `source_${option.value}`);
                    const hasTarget = targetFields.some(f => f.value === `target_${option.value}`);
                    return hasSource && hasTarget;
                });

            // Check if aggregation_field is specified in URL
            const urlAggField = this.$route.query.aggregation_field;

            if (urlAggField && this.availableAggregations.some(opt => opt.value === urlAggField)) {
                this.aggregationField = urlAggField;
            }
            // Otherwise set default to author if available
            else if (this.availableAggregations.some(opt => opt.value === 'author')) {
                this.aggregationField = 'author';
            } else if (this.availableAggregations.length > 0) {
                this.aggregationField = this.availableAggregations[0].value;
            }
        },

        fetchNetworkData() {
            this.loading = true;
            this.error = null;

            let params = { ...this.$route.query };
            params.db_table = this.globalConfig.databaseName;
            params.aggregation_field = this.aggregationField;
            params.min_threshold = this.minThreshold;
            params.max_nodes = 10000;

            this.emitter.emit("searchArgsUpdate", {
                counts: "",
                searchParams: params,
            });

            this.$http
                .get(`${this.globalConfig.apiServer}/network_data/?${this.paramsToUrl(params)}`)
                .then((response) => {
                    if (response.data.error) {
                        this.error = response.data.error;
                        this.loading = false;
                        return;
                    }

                    this.rawData = response.data;
                    this.expandedNode = null;
                    this.initializeGraph();
                    this.loading = false;

                    // Update search args with total count
                    const totalAlignments = response.data.edges.reduce((sum, edge) => sum + edge.weight, 0);
                    this.emitter.emit("searchArgsUpdate", {
                        counts: totalAlignments,
                        searchParams: params,
                    });
                })
                .catch((error) => {
                    this.loading = false;
                    this.error = error.toString();
                    console.log(error);
                });
        },

        initializeGraph() {
            // Create new graph
            this.graph = new Graph();

            // Find min and max sizes for normalization
            const sizes = this.rawData.nodes.map(n => n.size);
            const maxSize = Math.max(...sizes);
            const minSize = Math.min(...sizes);

            // Define size range (min and max pixel sizes)
            const minNodeSize = 5;
            const maxNodeSize = 30;

            // Add nodes with normalized sizes
            this.rawData.nodes.forEach(node => {
                // Normalize size to range [minNodeSize, maxNodeSize]
                const normalizedSize = minNodeSize +
                    (Math.sqrt(node.size) - Math.sqrt(minSize)) /
                    (Math.sqrt(maxSize) - Math.sqrt(minSize)) *
                    (maxNodeSize - minNodeSize);

                this.graph.addNode(node.id, {
                    label: node.label,
                    size: normalizedSize,
                    color: this.getUniqueNodeColor(node.id),
                    connections: node.size,
                    hidden: false,  // Initialize as visible
                    x: Math.random(),
                    y: Math.random()
                });
                this.nodeTypes.set(node.id, node.type); // Store type separately
            });

            // Add edges
            let skippedEdges = 0;
            this.rawData.edges.forEach((edge, index) => {
                if (this.graph.hasNode(edge.source) && this.graph.hasNode(edge.target)) {
                    this.graph.addEdge(edge.source, edge.target, {
                        weight: edge.weight,
                        size: Math.log(edge.weight + 1) * 2, // Log scale for edge thickness
                        color: '#cccccc', // Light gray
                        hidden: false  // Initialize as visible
                    });
                }
            });

            // Apply layout
            this.applyLayout();

            // Initialize or update renderer
            if (this.renderer) {
                this.renderer.kill();
            }

            this.renderer = new Sigma(this.graph, this.$refs.sigmaContainer, {
                renderEdgeLabels: false,
                defaultNodeColor: "#999",
                defaultEdgeColor: "#ccc",
                labelSize: 12,
                labelWeight: "bold",
                enableEdgeEvents: true,  // Make edges clickable
                labelDensity: 0.5,  // Show 50% of labels initially
                labelGridCellSize: 100,  // Grid cell size for label positioning
                labelRenderedSizeThreshold: 6,  // Only show labels for nodes with rendered size > 6px
            });

            // Set camera to show full graph with some padding
            // Adjust camera to show full graph
            const camera = this.renderer.getCamera();
            camera.setState({ ratio: 1.2 }); // Zoom out more to show spread-out nodes

            // Update counts
            this.updateCounts();

            // Setup interactions
            this.setupInteractions();
        },

        applyLayout() {
            if (!this.graph) return;

            if (this.layoutType === "circular") {
                circular.assign(this.graph);
            } else if (this.layoutType === "communities") {
                // Detect communities using Louvain algorithm
                const communities = louvain(this.graph, {
                    resolution: 1.0,
                    randomWalk: true
                });

                // Assign colors based on community
                this.communityColors.clear();
                this.nodeCommunities.clear();

                this.graph.forEachNode((node) => {
                    const community = communities[node];
                    this.nodeCommunities.set(node, community);

                    // Assign color for this community if not yet assigned
                    if (!this.communityColors.has(community)) {
                        const colorIndex = this.communityColors.size % this.colorPalette.length;
                        this.communityColors.set(community, this.colorPalette[colorIndex]);
                    }

                    this.graph.setNodeAttribute(node, 'color', this.communityColors.get(community));
                });

                // Apply force-directed layout with linLogMode for better community separation
                forceAtlas2.assign(this.graph, {
                    iterations: 200,
                    settings: {
                        gravity: 0.5,
                        scalingRatio: 50,
                        strongGravityMode: false,
                        barnesHutOptimize: true,
                        barnesHutTheta: 0.5,
                        edgeWeightInfluence: 1.0,  // Use edge weights for community structure
                        slowDown: 5,
                        linLogMode: true  // Better for community detection
                    }
                });

                // Apply noverlap to prevent node overlapping
                noverlap.assign(this.graph, {
                    maxIterations: 50,
                    settings: {
                        margin: 5,
                        ratio: 1.2,
                        expansion: 1.1
                    }
                });

            } else if (this.layoutType === "force") {
                forceAtlas2.assign(this.graph, {
                    iterations: 150,
                    settings: {
                        gravity: 0.1,
                        scalingRatio: 100,
                        strongGravityMode: false,
                        barnesHutOptimize: true,
                        barnesHutTheta: 0.5,
                        edgeWeightInfluence: 0,
                        slowDown: 5,
                        linLogMode: false
                    }
                });
            } else {
                // Random layout (default positions)
                this.graph.forEachNode((node) => {
                    this.graph.setNodeAttribute(node, 'x', Math.random() * 2 - 1);
                    this.graph.setNodeAttribute(node, 'y', Math.random() * 2 - 1);
                });
            }

            if (this.renderer) {
                this.renderer.refresh();
            }
        }, setupInteractions() {
            // Click on node - filter network to show this node's connections
            this.renderer.on("clickNode", ({ node }) => {
                this.selectedEdge = null;
                const nodeLabel = this.graph.getNodeAttribute(node, 'label');

                // Reload network data filtered to this node's connections
                this.filterToNode(nodeLabel);
            });

            // Click on edge - show direction selector
            this.renderer.on("clickEdge", ({ edge }) => {
                this.selectedNode = null;
                const source = this.graph.source(edge);
                const target = this.graph.target(edge);
                this.selectedEdge = {
                    edge: edge,
                    source: this.graph.getNodeAttribute(source, 'label'),
                    target: this.graph.getNodeAttribute(target, 'label'),
                    sourceId: source,
                    targetId: target,
                    weight: this.graph.getEdgeAttribute(edge, 'weight')
                };
            });

            // Click on stage (background)
            this.renderer.on("clickStage", () => {
                this.clearSelection();
            });

            // Hover effects
            this.renderer.on("enterNode", ({ node }) => {
                this.graph.setNodeAttribute(node, 'highlighted', true);
                this.renderer.refresh();
            });

            this.renderer.on("leaveNode", ({ node }) => {
                this.graph.setNodeAttribute(node, 'highlighted', false);
                this.renderer.refresh();
            });
        },

        getUniqueNodeColor(nodeId) {
            // Check if we already assigned a color to this node
            if (this.nodeColorMap.has(nodeId)) {
                return this.nodeColorMap.get(nodeId);
            }

            // Assign a new color from the palette
            const color = this.colorPalette[this.colorIndex % this.colorPalette.length];
            this.colorIndex++;
            this.nodeColorMap.set(nodeId, color);
            return color;
        },

        applyThreshold() {
            if (!this.graph) return;

            // Hide edges below threshold
            this.graph.forEachEdge((edge, attributes) => {
                const hidden = attributes.weight < this.minThreshold;
                this.graph.setEdgeAttribute(edge, 'hidden', hidden);
            });

            // Hide nodes with no visible edges
            this.graph.forEachNode((node) => {
                const hasVisibleEdge = this.graph.edges(node).some(edge =>
                    !this.graph.getEdgeAttribute(edge, 'hidden')
                );
                this.graph.setNodeAttribute(node, 'hidden', !hasVisibleEdge);
            });

            this.updateCounts();
            this.renderer.refresh();
        },

        updateCounts() {
            this.visibleNodes = this.graph.filterNodes((node, attr) => !attr.hidden).length;
            this.visibleEdges = this.graph.filterEdges((edge, attr) => !attr.hidden).length;
        },

        reloadNetwork() {
            this.fetchNetworkData();
        },

        resetView() {
            this.expandedNode = null;
            this.selectedNode = null;
            this.selectedEdge = null;
            this.minThreshold = 5;
            this.fetchNetworkData();
        },

        clearSelection() {
            this.selectedNode = null;
            this.selectedEdge = null;
        },

        viewEdgeDirection(source, target) {
            // Navigate to search results with specific direction
            let queryParams = { ...this.$route.query };
            queryParams[`source_${this.aggregationField}`] = `"${source}"`;
            queryParams[`target_${this.aggregationField}`] = `"${target}"`;
            queryParams.db_table = this.globalConfig.databaseName;
            this.clearSelection();
            this.$router.push(`/search?${this.paramsToUrl(queryParams)}`);
        },

        viewEdgeTitles(source, target) {
            // Navigate to network view showing titles from both nodes
            const drillDownField = 'title';

            let queryParams = { ...this.$route.query };
            queryParams[`source_${this.aggregationField}`] = `"${source}"`;
            queryParams[`target_${this.aggregationField}`] = `"${target}"`;
            queryParams.db_table = this.globalConfig.databaseName;
            queryParams.aggregation_field = drillDownField; // Set the aggregation to title

            this.clearSelection();
            // Navigate to network view with title aggregation
            this.$router.push(`/network?${this.paramsToUrl(queryParams)}`);
        },

        filterToNode(nodeLabel) {
            // Reload network data to show only connections for this node
            this.loading = true;

            let params = { ...this.$route.query };
            params.db_table = this.globalConfig.databaseName;
            params.aggregation_field = this.aggregationField;
            params.min_threshold = this.minThreshold;
            params.max_nodes = 10000;

            // Special filter: same node for both source and target means "show all connections"
            params[`source_${this.aggregationField}`] = `"${nodeLabel}"`;
            params[`target_${this.aggregationField}`] = `"${nodeLabel}"`;

            this.$http
                .get(`${this.globalConfig.apiServer}/network_data/?${this.paramsToUrl(params)}`)
                .then((response) => {
                    if (response.data.error) {
                        this.error = response.data.error;
                        this.loading = false;
                        return;
                    }

                    this.rawData = response.data;
                    this.expandedNode = nodeLabel;  // Mark as filtered
                    this.initializeGraph();
                    this.loading = false;
                })
                .catch((error) => {
                    this.loading = false;
                    this.error = error.toString();
                    console.log(error);
                });
        },

        filterByNode(nodeId) {
            // Navigate to search results filtered by this author/title
            let queryParams = { ...this.$route.query };
            queryParams[`source_${this.aggregationField}`] = `"${nodeId}"`;
            queryParams.db_table = this.globalConfig.databaseName;
            this.$router.push(`/search?${this.paramsToUrl(queryParams)}`);
        }
    }
};
</script>

<style scoped>
.node-info-panel,
.edge-info-panel {
    position: absolute;
    top: 20px;
    right: 20px;
    width: 300px;
    z-index: 10;
}

.edge-info-panel {
    position: fixed;
    top: 50%;
    left: 50%;
    right: auto;
    bottom: auto;
    transform: translate(-50%, -50%);
    z-index: 1050;
}

.form-label {
    font-weight: 600;
    font-size: 0.9rem;
}
</style>
