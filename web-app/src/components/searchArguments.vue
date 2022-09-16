<template>
    <div
        class="ml-3 mr-3 mt-2 mb-4 pr-2 pl-2 pt-2 shadow-1"
        style="background-clip: border-box; border: 1px solid rgba(0, 0, 0, 0.125); width: 100%"
    >
        <div class="mb-1 p-2" style="font-size: 1rem" v-if="counts && !error">
            {{ counts }} results for the following query:
        </div>
        <div class="m-2 pt-2 pb-1" v-if="banality">
            <div class="metadata-args">
                <span class="metadata-label"> Banality filter </span>
                <span class="remove-metadata">
                    <span class="corner-btn destroy right" @click="removeMetadata({ fieldName: 'banality' })">x</span>
                </span>
                <span class="metadata-value">
                    {{ banality }}
                </span>
            </div>
        </div>
        <div class="row pl-2" v-if="!error">
            <div
                class="col-6 rounded-0 pt-2 pb-3 mb-2 search-args-group"
                v-for="(paramGroup, groupIndex) in searchParams"
                :key="groupIndex"
            >
                <h6 class="text-center text-capitalize">
                    <span v-html="paramGroup.direction"></span>
                    {{ paramGroup.direction }} Parameters:
                </h6>
                <div v-if="paramGroup.params != null">
                    <div class="metadata-args" v-for="metadata in paramGroup.params" :key="metadata.field">
                        <span class="metadata-label">
                            {{ metadata.label }}
                        </span>
                        <span class="remove-metadata">
                            <span class="corner-btn destroy right" @click="removeMetadata(metadata)">x</span>
                        </span>
                        <span class="metadata-value">
                            {{ checkValue(metadata.value) }}
                        </span>
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
        var vm = this;
        this.emitter.on("searchArgsUpdate", (params) => {
            vm.counts = params.counts.toLocaleString();
            vm.searchParams = this.processParams(params.searchParams);
            if ("banality" in params.searchParams && params.searchParams.banality.length > 0) {
                if (params.searchParams.banality == "false") {
                    vm.banality = "Filter all";
                } else {
                    vm.banality = "Only banalities";
                }
            } else {
                vm.banality = false;
            }
        });
    },
    data() {
        return {
            globalConfig: this.$globalConfig,
            counts: null,
            error: null,
            searchParams: null,
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
        removeMetadata(metadata) {
            event.srcElement.parentNode.parentNode.style.display = "none";
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
.metadata-args {
    display: inline-block !important;
    margin-top: 20px;
    border: 1px solid #ddd;
    text-align: justify;
}

.metadata-args.none {
    border-width: 0px !important;
}

.metadata-label {
    border: 1px solid #ddd;
    border-width: 0px 1px 0px 0px;
    padding: 0px 5px;
    background-color: #e9ecef;
    float: left;
    line-height: 29px;
}

.metadata-value {
    padding: 6px 10px 10px 10px;
    line-height: 29px;
    /* Needs prefixing */
    box-decoration-break: clone;
    -webkit-box-decoration-break: clone;
}

.remove-metadata {
    float: right;
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