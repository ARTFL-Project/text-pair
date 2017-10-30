<template>
    <div id="global-stats-results" class="card mb-3 rounded-0 shadow-1 mt-5 p-4">
        <div class="loading" v-if="loading">
            Loading...
        </div>
        <div v-if="error" class="error">
            {{ error }}
        </div>
        <div id="stats" v-if="stats.results">
            <div class="row" v-for="(result, index) in stats.results" :key="result.label" v-bind:data-index="index">
                <div class="col card rounded-0 mb-3 shadow-1">
                    <div class="row">
                        <div class="col-md-4 col-sm-5 p-2 result-label">
                            {{ result.label }}
                        </div>
                        <div class="col show-btn">
                            <button type="button" class="btn btn-light rounded-0" @click="getReusedPassages(result.label, index)">
                                Show {{ result.count }} reused passages
                            </button>
                        </div>
                    </div>
                    <ul class="list-group mt-3 rounded-0" v-if="passages[index]">
                        <li class="list-group-item rounded-0" v-for="(passage, innerIndex) in passages[index]" :key="innerIndex">
                            {{ passage.metadata.source_passage }} ({{ passage.count }})
                        </li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
export default {
    name: "globalStats",
    data() {
        return {
            stats: null,
            passages: {},
            loading: null,
            error: null
        }
    },
    created() {
        // fetch the data when the view is created and the data is
        // already being observed
        this.fetchData()
    },
    watch: {
        // call again the method if the route changes
        '$route': 'fetchData'
    },
    methods: {
        fetchData() {
            this.stats = { results: [] } // clear stats with new search
            let params = { ...this.$route.query }
            params.db_table = this.$globalConfig.databaseName
            this.$http.get(`${this.$globalConfig.apiServer}/stats/?`, {
                params: params
            }).then(response => {
                this.stats = response.data
                for (let i=0; i < this.stats.length; i+=1) {
                    this.passages[i] = null
                }
                this.loading = false
            }).catch(error => {
                this.loading = false
                this.error = error.toString();
                console.log(error)
            });
        },
        getReusedPassages(fieldValue, index) {
            let params = { ...this.$route.query }
            params.db_table = this.$globalConfig.databaseName;
            params[this.stats.stats_field] = fieldValue
            console.log(params)
            this.$http.get(`${this.$globalConfig.apiServer}/get_most_reused_passages/?`, {
                params: params
            }).then(response => {
                this.$set(this.passages, index, response.data.results)
                // this.passages[index] = response.data
                console.log(this.passages[index])
            }).catch(error => {
                this.loading = false
                this.error = error.toString();
                console.log(error)
            });
        }
    }
}
</script>

<style>
.result-label {
    text-overflow: ellipsis;
    overflow-x: hidden;
    white-space: nowrap;
    height: 38px;
}

.result-count {
    border: 1px solid rgba(0, 0, 0, .125);
}

.show-btn {
    position: relative;
}

.show-btn button {
    position: absolute;
    /* top: -.55rem; */
    right: .45rem;
    border-right-width: 0;
    height: 38px;
}
</style>
