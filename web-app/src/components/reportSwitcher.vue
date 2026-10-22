<template>
    <nav class="mt-2" aria-label="Result views">
        <div class="btn-group">
            <router-link v-for="view in views" :key="view.name" :to="`${view.path}?${query}`"
                class="btn btn-outline-secondary" :class="{ active: route.name == view.name }"
                :aria-current="route.name == view.name ? 'page' : undefined">{{ view.label }}</router-link>
        </div>
    </nav>
</template>
<script setup>
import { computed } from "vue";
import { useRoute } from "vue-router";
import globalConfig from "../config.js";
import { paramsToUrl } from "../utils.js";

const route = useRoute();

const views = [
    { name: "searchResults", path: "/search", label: "View all passage pairs" },
    { name: "sortedResults", path: "/sorted-results/", label: "View passages by frequency of reuse" },
    { name: "timeSeries", path: "/time/", label: "View passages in timeline" },
    globalConfig.showNetworkGraph && { name: "networkGraph", path: "/network/", label: "View network graph" },
    globalConfig.showSemanticGraph && { name: "semanticGraph", path: "/semantic/", label: "View semantic graph" },
].filter(Boolean);

const query = computed(() => {
    const queryParams = { ...route.query };
    delete queryParams.page;
    delete queryParams.id_anchor;
    queryParams.db_table = globalConfig.databaseName;
    return paramsToUrl(queryParams);
});
</script>
<style scoped>
.btn-group .btn {
    border-bottom-width: 0;
}

.btn-group .btn:first-child {
    border-bottom-left-radius: 0;
}

.btn-group .btn:last-child {
    border-bottom-right-radius: 0;
}

@media (max-width: 767.98px) {
    .btn-group {
        flex-wrap: wrap;
    }
}
</style>
