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

                <!-- Network Controls -->
                <div class="card-body p-3 border-bottom">
                    <div class="row align-items-center">
                        <div class="col-md-2">
                            <label class="form-label mb-1">Aggregate by:</label>
                            <select class="form-select form-select-sm" v-model="aggregationField"
                                @change="reloadNetwork">
                                <option v-for="option in availableAggregations" :key="option.value"
                                    :value="option.value">
                                    {{ option.label }}
                                </option>
                            </select>
                        </div>
                        <div class="col-md-2">
                            <label class="form-label mb-1">Measure Importance By:</label>
                            <select class="form-select form-select-sm" v-model="centralityMode" @change="reloadNetwork">
                                <option value="degree">Total connections (degree)</option>
                                <option value="eigenvector">Influence (eigenvector)</option>
                                <option value="betweenness">Bridging Role (betweenness)</option>
                            </select>
                        </div>
                        <div class="col-md-2">
                            <label class="form-label mb-1">
                                Min. alignments: <strong>{{ minThreshold }}</strong>
                            </label>
                            <input type="range" class="form-range" v-model.number="minThreshold" min="1" max="100"
                                @change="applyThreshold">
                        </div>
                        <div class="col-md-2">
                            <label class="form-label mb-1">Arrange by:</label>
                            <select class="form-select form-select-sm" v-model="layoutType" @change="applyLayout">
                                <option value="communities">Community</option>
                                <option value="circular">Circular</option>
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
                            <span class="text-muted">
                                Showing {{ visibleNodes }} nodes
                                <span v-if="expandedNode"> | Expanded: <strong>{{ expandedNode }}</strong></span>
                            </span>
                        </div>
                    </div>
                </div>

                <!-- Sigma Graph Container with overlays -->
                <div style="position: relative;">
                    <div id="sigma-container" ref="sigmaContainer" class="vector-space-bg"
                        style="width: 100%; height: calc(100vh - 360px); min-height: 600px;"></div>

                    <!-- Node Info Panel -->
                    <div v-if="selectedNode" class="node-info-panel card shadow-lg">
                        <div class="card-header d-flex justify-content-between align-items-center">
                            <span class="node-label-text">{{ getNodeLabel(selectedNode) }}</span>
                        </div>
                        <div class="card-body p-2">
                            <div class="d-grid gap-2">
                                <button class="btn btn-sm btn-outline-secondary"
                                    @click="viewNodeAsSource(getNodeLabel(selectedNode))">
                                    <i class="bi bi-arrow-right"></i> View as Source
                                </button>
                                <button class="btn btn-sm btn-outline-secondary"
                                    @click="viewNodeAsTarget(getNodeLabel(selectedNode))">
                                    <i class="bi bi-arrow-left"></i> View as Target
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

        </div> <!-- Close container-fluid -->
    </div> <!-- Close mt-3 wrapper -->
</template>

<script>
import { createNodeBorderProgram } from "@sigma/node-border";
import Graph from "graphology";
import louvain from "graphology-communities-louvain";
import forceAtlas2 from "graphology-layout-forceatlas2";
import noverlap from "graphology-layout-noverlap";
import circular from "graphology-layout/circular";
import Sigma from "sigma";
import { EdgeLineProgram } from "sigma/rendering";
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
            loadingMessage: "Fetching data...",
            error: null,
            globalConfig: this.$globalConfig,

            // Network data
            graph: null,
            renderer: null,
            rawData: { nodes: [], edges: [] },
            nodeTypes: new Map(), // Track node types separately
            nodeColorMap: new Map(), // Track assigned colors
            communityColors: new Map(), // Colors for communities
            colorPalette: [
                '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
                '#6C5CE7', '#A29BFE', '#FD79A8', '#FDCB6E', '#00B894',
                '#0984E3', '#E17055', '#D63031', '#FF7675', '#74B9FF',
                '#55EFC4', '#81ECEC', '#A29BFE', '#FDA7DF', '#FAB1A0',
                '#00CEC9', '#FF6348', '#1E90FF', '#FF1493', '#32CD32',
                '#FF6347', '#4169E1', '#FF69B4', '#00FA9A', '#FFD700',
                '#E84393', '#00B8D4', '#6C5CE7', '#FDCB6E', '#00CEC9'
            ],
            colorIndex: 0,

            // UI state
            aggregationField: "author",
            availableAggregations: [],
            centralityMode: "degree",  // degree, eigenvector, or betweenness
            minThreshold: 10,
            layoutType: "communities",  // Default to community layout
            expandedNode: null,
            selectedNode: null,
            selectedEdge: null,
            visibleNodes: 0
        };
    },
    created() {
        // Non-reactive property - avoid Vue reactivity overhead for performance-critical lookup
        this.connectedNodes = new Set();

        this.initializeAggregationOptions();
        this.fetchNetworkData();
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
            this.loadingMessage = "Fetching data...";
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
            params.aggregation_field = this.aggregationField;
            params.centrality = this.centralityMode;
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

                    this.rawData = Object.freeze(response.data);
                    this.expandedNode = null;
                    this.loadingMessage = "Calculating layout...";

                    // Use setTimeout to allow the UI to update with the new message
                    setTimeout(() => {
                        this.initializeGraph();
                        this.loading = false;
                    }, 10);

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
            console.time('Total graph initialization');

            // Create new graph
            this.graph = new Graph();

            // Use centrality for node sizing
            const centralities = this.rawData.nodes.map(n => n.centrality || 0);
            const maxCentrality = Math.max(...centralities);
            const minCentrality = Math.min(...centralities);

            // Define size range (min and max pixel sizes)
            // For small graphs (< 1000 nodes), use larger min size so all nodes are visible
            const nodeCount = this.rawData.nodes.length;
            const minNodeSize = nodeCount < 1000 ? 3 : 2;
            const maxNodeSize = 30;

            // Add nodes with normalized sizes based on centrality
            console.time('Add nodes');
            this.rawData.nodes.forEach(node => {
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

                const nodeColor = this.getUniqueNodeColor(node.id);
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

                this.graph.addNode(node.id, {
                    label: displayLabel,
                    fullLabel: label,  // Store full label for reference
                    size: normalizedSize,
                    color: nodeColor,
                    originalColor: nodeColor, // Store original color
                    connections: node.total_alignments,
                    centrality: node.centrality,
                    hidden: false,  // Initialize as visible
                    zIndex: 0,  // Initialize with default zIndex
                    x: Math.random(),
                    y: Math.random()
                });
                this.nodeTypes.set(node.id, node.type); // Store type separately
            });
            console.timeEnd('Add nodes');

            // Add edges
            console.time('Add edges');
            this.rawData.edges.forEach((edge, index) => {
                if (this.graph.hasNode(edge.source) && this.graph.hasNode(edge.target)) {
                    this.graph.addEdge(edge.source, edge.target, {
                        weight: edge.weight,
                        size: Math.log(edge.weight + 1) * 2, // Log scale for edge thickness
                        color: '#5B7FDB', // Medium blue - distinguishable from desaturated nodes
                        hidden: true  // Initialize as hidden
                    });
                }
            });
            console.timeEnd('Add edges');
            console.timeEnd('Total graph initialization');

            // Apply layout and initialize renderer after layout is complete
            this.applyLayoutAndInitRenderer();
        },

        applyLayoutAndInitRenderer() {
            if (!this.graph) return;

            // Apply layout first
            this.applyLayoutOnly();

            // Wait for layout to complete, then initialize renderer
            setTimeout(() => {
                this.initRenderer();
            }, 100);
        },

        initRenderer() {
            console.time('Renderer initialization');

            // Initialize or update renderer
            if (this.renderer) {
                this.renderer.kill();
            }

            this.renderer = new Sigma(this.graph, this.$refs.sigmaContainer, {
                edgeProgramClasses: {
                    line: EdgeLineProgram,
                },
                nodeProgramClasses: {
                    // Use border program to create hollow circles with colored border
                    circle: createNodeBorderProgram({
                        borders: [
                            {
                                size: { value: 0.2 },  // Thin border (10% of node size)
                                color: { attribute: "color" }  // Use node's color for border
                            }
                        ]
                    })
                },
                renderEdgeLabels: false,
                defaultNodeColor: "#999",
                defaultEdgeColor: "#5B7FDB",
                labelSize: 12,
                labelWeight: "bold",
                labelColor: { attribute: "labelColor", color: "#ffffff" },  // Use node's labelColor attribute
                enableEdgeEvents: true,  // Make edges clickable
                labelDensity: 0.1,  // Show 10% of labels initially
                labelGridCellSize: 100,  // Grid cell size for label positioning
                labelRenderedSizeThreshold: 15,  // Only show labels for nodes with rendered size > 10px
                zIndex: true,  // Enable zIndex for node layering
                hideLabelsOnMove: true,
                hideEdgesOnMove: true,
            });

            // Set up reducers using setSetting (best practice from Sigma examples)
            this.renderer.setSetting('nodeReducer', (node, data) => {
                const res = { ...data };

                // Highlight on hover - use black label on white background
                if (data.highlighted) {
                    res.color = "#f00";
                    res.zIndex = 10;
                    res.labelColor = "#000000";  // Black label for hover
                    return res;
                }

                // Default: white label for dark background
                res.labelColor = "#ffffff";

                // When a node is selected
                if (this.selectedNode) {
                    if (this.connectedNodes.has(node)) {
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

            this.renderer.setSetting('edgeReducer', (edge, data) => {
                const res = { ...data };

                if (this.selectedNode) {
                    // Show edge only if it connects to the selected node
                    if (!this.graph.hasExtremity(edge, this.selectedNode)) {
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
            const camera = this.renderer.getCamera();
            camera.setState({ ratio: 1.0 });

            // Update counts
            this.updateCounts();

            // Setup interactions
            this.setupInteractions();

            console.timeEnd('Renderer initialization');
        },

        applyLayoutOnly() {
            if (!this.graph) return;

            console.time('Total layout');

            if (this.layoutType === "circular") {
                console.time('Circular layout');
                circular.assign(this.graph);
                console.timeEnd('Circular layout');
            } else if (this.layoutType === "communities") {
                // Detect communities using Louvain algorithm and assign to nodes
                console.time('Louvain community detection');
                louvain.assign(this.graph, {});
                console.timeEnd('Louvain community detection');

                // Assign colors based on community
                console.time('Assign community colors');
                this.communityColors.clear();

                this.graph.forEachNode((node) => {
                    const community = this.graph.getNodeAttribute(node, 'community');

                    // Assign color for this community if not yet assigned
                    if (!this.communityColors.has(community)) {
                        const colorIndex = this.communityColors.size % this.colorPalette.length;
                        this.communityColors.set(community, this.colorPalette[colorIndex]);
                    }

                    this.graph.setNodeAttribute(node, 'color', this.communityColors.get(community));
                });
                console.timeEnd('Assign community colors');

                // Apply force-directed layout with linLogMode for better community separation
                console.time('ForceAtlas2 layout');
                forceAtlas2.assign(this.graph, {
                    iterations: 100,
                    settings: {
                        gravity: 1,
                        scalingRatio: 10,
                        strongGravityMode: true,
                        barnesHutOptimize: true,
                        barnesHutTheta: 0.5,
                        edgeWeightInfluence: 1.0,  // Use edge weights for community structure
                        slowDown: 5,
                    }
                });
                console.timeEnd('ForceAtlas2 layout');

                // Apply noverlap to prevent node overlapping
                console.time('Noverlap layout');
                noverlap.assign(this.graph, {
                    maxIterations: 50,
                    settings: {
                        margin: 5,
                        ratio: 1.2,
                        expansion: 1.1
                    }
                });
                console.timeEnd('Noverlap layout');

            } else {
                // Circular layout (default/fallback)
                console.time('Circular layout (fallback)');
                circular.assign(this.graph);
                console.timeEnd('Circular layout (fallback)');
            }

            console.timeEnd('Total layout');
        },

        applyLayout() {
            if (!this.graph) return;

            this.loading = true;
            this.loadingMessage = "Applying layout...";

            // Kill the renderer to hide the graph during layout calculation
            if (this.renderer) {
                this.renderer.kill();
                this.renderer = null;
            }

            // Use setTimeout to allow UI to update before heavy computation
            setTimeout(() => {
                this.applyLayoutOnly();

                // Reinitialize the renderer
                this.initRenderer();

                // Give renderer time to complete rendering before hiding spinner
                setTimeout(() => {
                    this.loading = false;
                }, 50);
            }, 10);
        },

        setupInteractions() {
            // Click on node - show edges and desaturate unconnected nodes
            this.renderer.on("clickNode", ({ node }) => {
                this.selectedEdge = null;

                // Toggle selection
                if (this.selectedNode === node) {
                    // Deselect - restore everything
                    this.setSelectedNode(null);
                } else {
                    // Select new node
                    this.setSelectedNode(node);
                }
            });

            // Click on edge - show direction selector
            this.renderer.on("clickEdge", ({ edge }) => {
                this.setSelectedNode(null);
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
                this.setSelectedNode(null);
            });

            // Hover effects - use graph attributes, reducer handles rendering
            this.renderer.on("enterNode", ({ node }) => {
                this.graph.setNodeAttribute(node, 'highlighted', true);
            });

            this.renderer.on("leaveNode", ({ node }) => {
                this.graph.setNodeAttribute(node, 'highlighted', false);
            });
        },

        // Set selected node
        setSelectedNode(node) {
            if (node) {
                this.selectedNode = node;
                const neighbors = this.graph.neighbors(node);
                this.connectedNodes = new Set([node, ...neighbors]);
            } else {
                this.selectedNode = null;
                this.connectedNodes.clear();
            }

            // Refresh with skipIndexation for better performance
            this.updateCounts();
            this.renderer.refresh({
                skipIndexation: true
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
            // Count nodes that are actually visible based on selection state
            if (this.selectedNode) {
                // When a node is selected, only count connected nodes
                this.visibleNodes = this.connectedNodes.size;
            } else {
                // When no selection, count all non-hidden nodes
                this.visibleNodes = this.graph.filterNodes((node, attr) => !attr.hidden).length;
            }
        },

        reloadNetwork() {
            this.fetchNetworkData();
        },

        resetView() {
            this.expandedNode = null;
            this.selectedNode = null;
            this.selectedEdge = null;
            this.minThreshold = 10;
            this.fetchNetworkData();
        },

        clearSelection() {
            this.selectedNode = null;
            this.selectedEdge = null;
        },

        getNodeLabel(nodeId) {
            // Return fullLabel for complete text, fallback to label if fullLabel doesn't exist
            return this.graph.getNodeAttribute(nodeId, 'fullLabel') || this.graph.getNodeAttribute(nodeId, 'label');
        },

        viewNodeAsSource(nodeLabel) {
            let queryParams = { ...this.$route.query };
            queryParams[`source_${this.aggregationField}`] = `"${nodeLabel}"`;
            queryParams.db_table = this.globalConfig.databaseName;
            this.clearSelection();
            this.$router.push(`/network?${this.paramsToUrl(queryParams)}`);
        },

        viewNodeAsTarget(nodeLabel) {
            let queryParams = { ...this.$route.query };
            queryParams[`target_${this.aggregationField}`] = `"${nodeLabel}"`;
            queryParams.db_table = this.globalConfig.databaseName;
            this.clearSelection();
            this.$router.push(`/network?${this.paramsToUrl(queryParams)}`);
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
    background-color: rgba(256, 256, 256, 0.0);
    border-color: theme.$graph-btn-panel-color;
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
</style>
