<template>
    <div class="btn-group mb-2" role="group" aria-label="alt-results">
        <button type="button" class="btn btn-outline-secondary" :class="{ active: report == 'searchResults' }"
            @click="viewAllResults">View all
            passage pairs</button>
        <button type="button" class="btn btn-outline-secondary" :class="{ active: report == 'sortedResults' }"
            @click="sortResults">Sort passages by frequency of
            reuse</button>
        <button type="button" class="btn btn-outline-secondary" :class="{ active: report == 'time' }"
            @click="viewPassageInTimeline">View passages in
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
