<template>
    <div class="ml-3 mr-3 mt-2 mb-4 pr-2 pl-2 pt-2 shadow-1" style="background-clip: border-box; border: 1px solid rgba(0,0,0,.125); width: 100%">
        <div class="mb-1 p-2" style="font-size: 1rem" v-if="counts && !error">
            {{ counts }} results for the following query:
        </div>
        <div class="m-2 pt-2 pb-1" v-if="banality">
            <span class="metadata-label">
                Banality filter
            </span>
            <span class="metadata-value">
                {{ banality }}
            </span>
            <span class="remove-metata">
                <span class="corner-btn destroy right" @click="removeMetadata({fieldName: 'banality'})">x</span>
            </span>
        </div>
        <div class="row pl-2" v-if="!error">
            <div class="col-6 rounded-0 pt-2 pb-3 mb-2 search-args-group" v-for="(paramGroup, groupIndex) in searchParams" :key="groupIndex">
                <h6 class="text-center text-capitalize">{{ paramGroup.direction }} Parameters:</h6>
                <div class="metadata-args" v-for="(metadata, index) in paramGroup.params" :key="index" v-if="paramGroup.params !=null">
                    <span class="metadata-label">
                        {{ metadata.label }}
                    </span>
                    <span class="metadata-value">
                        {{ checkValue(metadata.value) }}
                    </span>
                    <span class="remove-metata">
                        <span class="corner-btn destroy right" @click="removeMetadata(metadata)">x</span>
                    </span>
                </div>
                <div class="metadata-args" v-if="paramGroup.params == null">
                    None
                </div>
            </div>
        </div>
    </div>
</template>
<script>
import { EventBus } from '../main.js';

export default {
    name: "searchArguments",
        created() {
            var vm = this
            EventBus.$on("searchArgsUpdate", params => {
                vm.counts = params.counts.toLocaleString()
                vm.searchParams = this.processParams(params.searchParams)
                if ("banality" in params.searchParams && params.searchParams.banality.length > 0) {
                    vm.banality = params.searchParams.banality
                }
            })
        },
        data() {
            return {
               counts: null,
               error: null,
               searchParams: null,
               banality: null
            }
        },
        methods: {
            toggleSearchForm() {
                EventBus.$emit("toggleSearchForm")
            },
            processParams(params) {
                let searchParams = []
                for (let direction of ["source", "target"]) {
                    let paramGroup = []
                    for (let metadata of this.$globalConfig.metadataFields[direction]) {
                        if (metadata.value in params && params[metadata.value].length > 0) {
                            paramGroup.push({label: metadata.label, fieldName: metadata.value, value: params[metadata.value]})
                        }
                    }
                    if (paramGroup.length > 0) {
                        searchParams.push({direction: direction, params: paramGroup})
                    } else {
                        searchParams.push({direction: direction, params: null})
                    }
                }
                return searchParams
            },
            removeMetadata(metadata) {
                event.srcElement.parentNode.parentNode.style.display = "none"
                let queryParams = { ...this.$route.query }
                delete queryParams.page
                delete queryParams.id_anchor
                queryParams.db_table = this.$globalConfig.databaseName
                queryParams[metadata.fieldName] = ""
                EventBus.$emit("urlUpdate", queryParams)
                this.facetResults = null
                this.results = { alignments: [] }
                this.$router.push(`/search?${this.paramsToUrl(queryParams)}`)
            },
            checkValue(value) {
                if (value == '""') {
                    return "N/A"
                } else {
                    return value
                }
            }
        }
}
</script>
<style scoped>
.metadata-args {
    margin-top: 20px;
    white-space: nowrap;
}

.metadata-label {
    /* margin-left: 10px; */
    border: 1px solid #ddd;
    border-width: 1px 1px 1px 1px;
    padding: 5px;
    background-color: #e9ecef;
}

.metadata-value {
    border: 1px solid #ddd;
    border-width: 1px 1px 1px 0px;
    padding: 5px;
}

.remove-metata {
    padding-right: 20px;
    position: relative;
}

.corner-btn.right {
    right: inherit;
    margin-top: -6px;
    line-height: inherit;
    padding: 5px 6px 4px 6px;
    border-width: 1px 1px 1px 0px;
}

.search-args-group:last-of-type {
    border-left: 1px solid #ddd;
}
</style>