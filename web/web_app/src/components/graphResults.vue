<template>
    <div>
        {{ graphResults }}
    </div>
</template>

<<script>
export default {
    name: "graphResults",
    data() {
        return {
            graphResults: null
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
            this.graphResults = null // clear graph with new search
            let params = this.cloneObject(this.$route.query)
            params.db_table = this.$globalConfig.databaseName
            this.$http.get(`${this.$globalConfig.apiServer}/search_alignments_full/?`, {
                params: params
            }).then(response => {
                this.graphResults = response.data
                this.loading = false
                this.done = true
            }).catch(error => {
                this.loading = false
                this.error = error.toString();
                console.log(error)
            });
        }
    }
}
</script>
