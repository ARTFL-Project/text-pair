<template>
    <nav class="report-switcher" aria-label="Views of matches">
        <div class="btn-group segmented">
            <router-link v-for="view in views" :key="view.name" :to="`${view.path}?${query}`"
                class="btn btn-outline-secondary" :class="{ active: route.name == view.name }"
                :aria-current="route.name == view.name ? 'page' : undefined">
                <svg class="view-icon" viewBox="0 0 16 16" aria-hidden="true" focusable="false">
                    <path :d="view.icon" fill="none" stroke="currentColor" :stroke-width="view.stroke || 1.5"
                        stroke-linecap="round" stroke-linejoin="round" />
                </svg><span><span class="view-prefix">{{ view.prefix }}</span><span class="view-name">{{
                    view.label }}</span></span></router-link>
        </div>
    </nav>
</template>
<script setup>
import { computed } from "vue";
import { useRoute } from "vue-router";
import globalConfig from "../config.js";
import { paramsToUrl } from "../utils.js";

const route = useRoute();

// Icons: two facing columns, bars by length, bars in time, linked nodes, a scatter of points.
// Narrow screens show the label without its prefix, which stays in the accessible name.
const views = [
    {
        name: "searchResults",
        path: "/search",
        prefix: "View all ",
        label: "passage pairs",
        icon: "M2 3h5M2 6h5M2 9h5M9 3h5M9 6h5M9 9h5M2 12h12",
    },
    {
        name: "sortedResults",
        path: "/sorted-results/",
        prefix: "View passages by ",
        label: "frequency of reuse",
        icon: "M2 3h12M2 6.5h9M2 10h6M2 13.5h3",
    },
    {
        name: "timeSeries",
        path: "/time/",
        prefix: "View passages in ",
        label: "timeline",
        icon: "M2 14h12M4 14V9M7 14V4M10 14V7M13 14v-3",
    },
    globalConfig.showNetworkGraph && {
        name: "networkGraph",
        path: "/network/",
        prefix: "View ",
        label: "network graph",
        icon: "M4 4.5l7 2M4 4.5l2 7M11 6.5l-5 5M4 4.5a1.5 1.5 0 1 0 0-.01M11 6.5a1.5 1.5 0 1 0 0-.01M6 11.5a1.5 1.5 0 1 0 0-.01",
    },
    globalConfig.showSemanticGraph && {
        name: "semanticGraph",
        path: "/semantic/",
        prefix: "View ",
        label: "semantic graph",
        icon: "M3 5h.01M5.5 3h.01M5.5 6.5h.01M10 9h.01M12.5 10.5h.01M10.5 12.5h.01M13 7h.01M6.5 12.5h.01M3 10h.01",
        stroke: 2.5,
    },
].filter(Boolean);

const query = computed(() => {
    const queryParams = { ...route.query };
    delete queryParams.page;
    delete queryParams.id_anchor;
    queryParams.db_table = globalConfig.databaseName;
    return paramsToUrl(queryParams);
});
</script>
<style lang="scss" scoped>
@import "../assets/theme.module.scss";

.report-switcher {
    margin: 1rem 0 1rem;
}

// Segmented control: a quiet well, the current view filled maroon
.segmented {
    display: inline-flex;
    flex-wrap: wrap;
    gap: 2px;
    padding: 3px;
    border-radius: 4px;
    background-color: $well;
    box-shadow: inset 0 0 0 1px $rule-color;
}

.segmented > .btn {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    flex: 0 1 auto;
    margin: 0 !important;
    padding: 0.42rem 0.85rem;
    border: 0;
    border-radius: 3px !important;
    background-color: transparent;
    color: $ink-2 !important;
    font-size: 0.82rem;
    font-weight: 600;
    line-height: 1.25;
}

.segmented > .btn:hover,
.segmented > .btn:focus {
    background-color: rgba($link-color, 0.08);
    box-shadow: none;
    color: $source-deep !important;
}

.segmented > .btn.active {
    background-color: $button-color;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.12), 0 1px 2px rgba(64, 40, 24, 0.12);
    color: #fff !important;
}

.view-icon {
    flex: 0 0 auto;
    width: 1rem;
    height: 1rem;
    color: $muted;
}

.btn:hover .view-icon {
    color: $button-color;
}

.btn.active .view-icon {
    color: currentColor;
}

@media (max-width: 767.98px) {
    .segmented {
        display: grid;
        grid-template-columns: 1fr 1fr;
        width: 100%;
    }

    .segmented > .btn {
        justify-content: flex-start;
        text-align: left;
    }

    .view-prefix {
        position: absolute;
        width: 1px;
        height: 1px;
        overflow: hidden;
        clip: rect(0, 0, 0, 0);
        white-space: nowrap;
    }

    .view-name {
        display: inline-block;
    }

    .view-name::first-letter {
        text-transform: uppercase;
    }
}
</style>
