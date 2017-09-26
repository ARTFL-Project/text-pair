<template>
    <div id="graph-results" class="card mb-3 rounded-0 shadow-1 mt-5 p-4">
        <div class="loading" v-if="loading">
            Loading...
        </div>
        <div v-if="error" class="error">
            {{ error }}
        </div>
        <div id="graph"></div>
    </div>
</template>

<<script>
export default {
    name: "graphResults",
    data() {
        return {
            graphResults: null,
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
        '$route': 'fetchData',
        'graphResults': 'drawGraph'
    },
    methods: {
            fetchData() {
                this.graphResults = null // clear graph with new search
                let params = this.cloneObject(this.$route.query)
                params.db_table = this.$globalConfig.databaseName
                this.loading = true
                this.$http.get(`${this.$globalConfig.apiServer}/search_alignments_full/?`, {
                    params: params
                }).then(response => {
                    this.graphResults = response.data
                }).catch(error => {
                    this.loading = false
                    this.error = error.toString();
                    console.log(error)
                });
            },
            drawGraph() {
                let fieldToID = {}
                let count = 0
                let nodes = []
                let edges = []
                for (let fieldPair of this.graphResults.results.slice(0, 100)) {
                    let source = fieldPair[0]
                    let target = fieldPair[1]
                    if (!fieldToID.hasOwnProperty(source)) {
                        fieldToID[source] = count
                        nodes.push({
                            id: count,
                            label: source
                        })
                        count++
                    }

                    if (!fieldToID.hasOwnProperty(target)) {
                        fieldToID[target] = count
                        nodes.push({
                            id: count,
                            label: target
                        })
                        count++
                    }
                    edges.push({
                        from: fieldToID[source],
                        to: fieldToID[target]
                    })
                }
                console.log(count)
                console.log(nodes, edges)
                this.loading = false
            }
    }
}
</script>
