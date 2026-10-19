<template>
    <div class="btn-group mt-2" role="group" aria-label="alt-results">
        <button type="button" class="btn btn-outline-secondary" :class="{ active: report == 'searchResults' }"
            @click="viewAllResults">View all
            passage pairs</button>
        <button type="button" class="btn btn-outline-secondary" :class="{ active: report == 'sortedResults' }"
            @click="sortResults">View passages by frequency of
            reuse</button>
        <button type="button" class="btn btn-outline-secondary" :class="{ active: report == 'timeSeries' }"
            @click="viewPassageInTimeline">View passages in
            timeline</button>
        <button type="button" class="btn btn-outline-secondary" :class="{ active: report == 'networkGraph' }"
            v-if="$globalConfig.showNetworkGraph" @click="viewNetworkGraph">View network
            graph</button>
        <button type="button" class="btn btn-outline-secondary" :class="{ active: report == 'semanticGraph' }"
            v-if="$globalConfig.showSemanticGraph" @click="viewSemanticGraph">View semantic
            graph</button>
    </div>
</template>
<script>
export default {
    name: 'reportSwitcher',
    data() {
        return {
            report: this.$route.name,
        }
    },
    methods: {
        viewAllResults() {
            let queryParams = { ...this.$route.query };
            delete queryParams.page;
            delete queryParams.id_anchor;
            queryParams.db_table = this.$globalConfig.databaseName;
            this.$router.push(`/search?${this.paramsToUrl(queryParams)}`);
        },
        sortResults() {
            let queryParams = { ...this.$route.query };
            delete queryParams.page;
            delete queryParams.id_anchor;
            queryParams.db_table = this.$globalConfig.databaseName;
            this.$router.push(`/sorted-results/?${this.paramsToUrl(queryParams)}`);
        },
        viewPassageInTimeline() {
            let queryParams = { ...this.$route.query };
            delete queryParams.page;
            delete queryParams.id_anchor;
            queryParams.db_table = this.$globalConfig.databaseName;
            this.$router.push(`/time/?${this.paramsToUrl(queryParams)}`);
        },
        viewNetworkGraph() {
            let queryParams = { ...this.$route.query };
            delete queryParams.page;
            delete queryParams.id_anchor;
            queryParams.db_table = this.$globalConfig.databaseName;
            this.$router.push(`/network/?${this.paramsToUrl(queryParams)}`);
        },
        viewSemanticGraph() {
            let queryParams = { ...this.$route.query };
            delete queryParams.page;
            delete queryParams.id_anchor;
            queryParams.db_table = this.$globalConfig.databaseName;
            this.$router.push(`/semantic/?${this.paramsToUrl(queryParams)}`);
        },
    }
}
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
</style>