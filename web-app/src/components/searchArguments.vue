<template>
    <!-- Compact version when no search params on either side -->
    <div v-if="hasNoParams" class="compact-search-args p-2 mb-2 shadow-1"
        style="background-clip: border-box; border: 1px solid rgba(0, 0, 0, 0.125)">
        <div class="p-2" v-if="!counts || counts === '...'">
            <span class="spinner-border spinner-border-sm me-1" aria-hidden="true"></span>
            results across all documents
        </div>
        <div class="p-2" v-else>
            <strong>{{ counts }}</strong> results across all documents
        </div>
    </div>

    <!-- Regular version when search params exist -->
    <div v-else id="search-arguments" class="mt-1 mb-2 pt-2 shadow-1"
        style="background-clip: border-box; border: 1px solid rgba(0, 0, 0, 0.125);">
        <div class="mb-1 p-2" style="font-size: 1rem">
            <span v-if="!counts || counts === '...'">
                <span class="spinner-border spinner-border-sm me-1" aria-hidden="true"></span>
                results for the following query:
            </span>
            <span v-else>
                <strong>{{ counts }}</strong> results for the following query:
            </span>
        </div>
        <div class="row pl-2">
            <div class="col-12 col-md-6 rounded-0 pb-2 search-args-group" v-for="(paramGroup, groupIndex) in searchParams"
                :key="groupIndex">
                <h3 class="h6 text-center text-capitalize mb-2">
                    <span v-html="paramGroup.direction"></span>
                    Parameters:
                </h3>
                <ul class="list-unstyled mb-0" v-if="paramGroup.params != null">
                    <li class="metadata-args" v-for="metadata in paramGroup.params" :key="metadata.fieldName">
                        <span class="metadata-label">
                            {{ metadata.label }}
                        </span>
                        <span class="metadata-value">
                            {{ checkValue(metadata.value) }}
                        </span>
                        <button type="button" class="remove-metadata"
                            :aria-label="`Remove ${plainText(paramGroup.direction)} ${metadata.label} ${checkValue(metadata.value)}`"
                            @click="removeMetadata(metadata)">&times;</button>
                    </li>
                </ul>
                <div class="metadata-args none" v-if="paramGroup.params == null">None</div>
            </div>
        </div>
        <div class="my-2 pb-1 px-2" v-if="banality">
            <div class="metadata-args rounded-pill">
                <span class="metadata-label"> Banality filter </span>
                <span class="metadata-value">
                    {{ banality }}
                </span>
                <button type="button" class="remove-metadata" :aria-label="`Remove banality filter ${banality}`"
                    @click="removeMetadata({ fieldName: 'banality' })">&times;</button>
            </div>
        </div>
    </div>
</template>
<script setup>
import emitter from "tiny-emitter/instance";
import { computed, onUnmounted, ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import globalConfig from "../config.js";
import { paramsToUrl, plainText } from "../utils.js";

const route = useRoute();
const router = useRouter();
const counts = ref("...");
const searchParams = ref([
    { direction: globalConfig.sourceLabel, params: null },
    { direction: globalConfig.targetLabel, params: null },
]);
const banality = ref(null);

function update(params) {
    counts.value = params.counts.toLocaleString();
    searchParams.value = processParams(params.searchParams);
    if ("banality" in params.searchParams && params.searchParams.banality.length > 0) {
        banality.value = params.searchParams.banality == "false" ? "Filter all" : "Only banalities";
    } else {
        banality.value = false;
    }
    if (params.counts !== "") {
        emitter.emit("announce", `${counts.value} results`);
    }
}
emitter.on("searchArgsUpdate", update);
onUnmounted(() => emitter.off("searchArgsUpdate", update));

const hasNoParams = computed(() => {
    // Check if both source and target have no params, and no banality filter
    const noSourceParams = !searchParams.value[0].params || searchParams.value[0].params.length === 0;
    const noTargetParams = !searchParams.value[1].params || searchParams.value[1].params.length === 0;
    return noSourceParams && noTargetParams && !banality.value;
});

function processParams(params) {
    const processed = [];
    for (const direction of ["source", "target"]) {
        const paramGroup = [];
        for (const metadata of globalConfig.metadataFields[direction]) {
            if (metadata.value in params && params[metadata.value].length > 0) {
                paramGroup.push({
                    label: metadata.label,
                    fieldName: metadata.value,
                    value: params[metadata.value],
                });
            }
        }
        processed.push({
            direction: globalConfig[`${direction}Label`],
            params: paramGroup.length > 0 ? paramGroup : null,
        });
    }
    return processed;
}

function removeMetadata(metadata) {
    const queryParams = { ...route.query };
    delete queryParams.page;
    delete queryParams.id_anchor;
    queryParams.db_table = globalConfig.databaseName;
    queryParams[metadata.fieldName] = "";
    emitter.emit("urlUpdate", queryParams);
    router.push(`${route.path}?${paramsToUrl(queryParams)}`).then(() => emitter.emit("focusMain"));
}

function checkValue(value) {
    return value == '""' ? "N/A" : value;
}
</script>
<style scoped>
#search-arguments {
    font-family: "Open-Sans", sans-serif;
}

.compact-search-args {
    font-family: "Open-Sans", sans-serif;
    font-size: 1rem;
    line-height: 1.2;
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
    background-color: transparent;
    border-width: 0;
    color: inherit;
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

@media (min-width: 768px) {
    .search-args-group:last-of-type {
        border-left: 1px solid #ddd;
    }
}
</style>