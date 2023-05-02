<template>
    <div id="search-arguments" class="mt-2 mb-4 pt-2 shadow-1"
        style="background-clip: border-box; border: 1px solid rgba(0, 0, 0, 0.125)">
        <div class="mb-1 p-2" style="font-size: 1rem" v-if="counts && !error">
            {{ counts }} results for the following query:
        </div>
        <div class="m-2 pt-2 pb-1" v-if="banality">
            <div class="metadata-args rounded-pill">
                <span class="metadata-label"> Banality filter </span>
                <span class="metadata-value">
                    {{ banality }}
                </span>
                <span class="remove-metadata" @click="removeMetadata({ fieldName: 'banality' }, $event)">x </span>
            </div>
        </div>
        <div class="row pl-2" v-if="!error">
            <div class="col-6 rounded-0 pt-2 pb-3 mb-2 search-args-group" v-for="(paramGroup, groupIndex) in searchParams"
                :key="groupIndex">
                <h6 class="text-center text-capitalize">
                    <span v-html="paramGroup.direction"></span>
                    Parameters:
                </h6>
                <div v-if="paramGroup.params != null">
                    <div class="metadata-args" v-for="metadata in paramGroup.params" :key="metadata.field">
                        <span class="metadata-label">
                            {{ metadata.label }}
                        </span>
                        <span class="metadata-value">
                            {{ checkValue(metadata.value) }}
                        </span>
                        <span class="remove-metadata" @click="removeMetadata(metadata, $event)">x</span>
                    </div>
                </div>
                <div class="metadata-args none" v-if="paramGroup.params == null">None</div>
            </div>
        </div>
    </div>
</template>
<script>
export default {
    name: "searchArguments",
    created() {
        this.emitter.on("searchArgsUpdate", (params) => {
            this.counts = params.counts.toLocaleString();
            this.searchParams = this.processParams(params.searchParams);
            if ("banality" in params.searchParams && params.searchParams.banality.length > 0) {
                if (params.searchParams.banality == "false") {
                    this.banality = "Filter all";
                } else {
                    this.banality = "Only banalities";
                }
            } else {
                this.banality = false;
            }
        });
    },
    data() {
        return {
            globalConfig: this.$globalConfig,
            counts: "...",
            error: null,
            searchParams: [
                { direction: this.$globalConfig.sourceLabel, params: null },
                { direction: this.$globalConfig.targetLabel, params: null },
            ],
            banality: null,
        };
    },
    methods: {
        toggleSearchForm() {
            this.emitter.emit("toggleSearchForm");
        },
        processParams(params) {
            let searchParams = [];
            for (let direction of ["source", "target"]) {
                let paramGroup = [];
                for (let metadata of this.$globalConfig.metadataFields[direction]) {
                    if (metadata.value in params && params[metadata.value].length > 0) {
                        paramGroup.push({
                            label: metadata.label,
                            fieldName: metadata.value,
                            value: params[metadata.value],
                        });
                    }
                }
                if (paramGroup.length > 0) {
                    searchParams.push({ direction: this.globalConfig[`${direction}Label`], params: paramGroup });
                } else {
                    searchParams.push({ direction: this.globalConfig[`${direction}Label`], params: null });
                }
            }
            return searchParams;
        },
        removeMetadata(metadata, event) {
            event.target.parentNode.parentNode.style.display = "none";
            let queryParams = { ...this.$route.query };
            delete queryParams.page;
            delete queryParams.id_anchor;
            queryParams.db_table = this.$globalConfig.databaseName;
            queryParams[metadata.fieldName] = "";
            this.emitter.emit("urlUpdate", queryParams);
            this.facetResults = null;
            this.results = { alignments: [] };
            let route = this.$route.path;
            this.$router.push(`${route}?${this.paramsToUrl(queryParams)}`);
        },
        checkValue(value) {
            if (value == '""') {
                return "N/A";
            } else {
                return value;
            }
        },
    },
};
</script>
<style scoped>
#search-arguments {
    font-family: "Open-Sans", sans-serif;
}

.metadata-args {
    border: 1px solid #ddd;
    display: -webkit-inline-box !important;
    display: -ms-inline-flexbox !important;
    display: inline-flex !important;
    margin-right: 5px;
    border-radius: 50rem;
    width: -webkit-fit-content;
    width: -moz-fit-content;
    width: fit-content;
    line-height: 2;
    margin-bottom: 0.5rem;
}

.metadata-args.none {
    border-width: 0px !important;
}

.metadata-label {
    display: -webkit-box;
    display: -ms-flexbox;
    display: flex;
    -webkit-box-align: center;
    -ms-flex-align: center;
    align-items: center;
    border: solid #ddd;
    border-width: 0 1px 0 0;
    border-top-left-radius: 50rem;
    border-bottom-left-radius: 50rem;
    padding: 0 0.5rem;
}

.metadata-value {
    display: -webkit-box;
    display: -ms-flexbox;
    display: flex;
    -webkit-box-align: center;
    -ms-flex-align: center;
    align-items: center;
    -webkit-box-decoration-break: clone;
    box-decoration-break: clone;
    padding: 0 0.5rem;
}

.remove-metadata {
    display: -webkit-box;
    display: -ms-flexbox;
    display: flex;
    -webkit-box-align: center;
    -ms-flex-align: center;
    align-items: center;
    padding-right: 5px;
    padding-left: 5px;
    border-left: 1px solid #ddd;
    border-top-right-radius: 50rem;
    border-bottom-right-radius: 50rem;
    padding: 0 0.5rem;
}

.remove-metadata:hover {
    cursor: pointer;
}

.corner-btn.right {
    position: initial;
    line-height: 29px;
    padding: 5px;
    border-width: 0px 0px 1px 1px;
}

.search-args-group:last-of-type {
    border-left: 1px solid #ddd;
}
</style>