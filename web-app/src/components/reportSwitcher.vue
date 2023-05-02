<template>
    <div class="btn-group" role="group" aria-label="alt-results">
        <button type="button" class="btn btn-outline-secondary" :class="{ active: report == 'searchResults' }"
            style="border-bottom-left-radius: 0;" @click="viewAllResults">View all
            passage pairs</button>
        <button type="button" class="btn btn-outline-secondary" :class="{ active: report == 'sortedResults' }"
            @click="sortResults">Sort passages by frequency of
            reuse</button>
        <button type="button" class="btn btn-outline-secondary" :class="{ active: report == 'timeSeries' }"
            style="border-bottom-right-radius: 0;" @click="viewPassageInTimeline">View passages in
            timeline</button>
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
    }
}
</script>
<style scoped>
.btn-group .btn {
    border-bottom-width: 0;
}
</style>