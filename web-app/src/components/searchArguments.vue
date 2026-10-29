<template>
    <!-- Compact version when no search params on either side -->
    <div v-if="hasNoParams" class="compact-search-args query-summary">
        <div v-if="!counts || counts === '...'">
            <span class="spinner-border spinner-border-sm me-1" aria-hidden="true"></span>
            {{ noun }} across all documents
        </div>
        <div v-else>
            <strong class="result-count">{{ counts }}</strong> {{ noun }} across all documents
        </div>
        <slot name="note"></slot>
    </div>

    <!-- Regular version when search params exist -->
    <div v-else id="search-arguments" class="query-summary">
        <div class="query-count">
            <span v-if="!counts || counts === '...'">
                <span class="spinner-border spinner-border-sm me-1" aria-hidden="true"></span>
                {{ noun }}{{ queryTail }}
            </span>
            <span v-else>
                <strong class="result-count">{{ counts }}</strong> {{ noun }}{{ queryTail }}
            </span>
            <slot v-if="origin" name="note"></slot>
        </div>
        <div class="query-sides">
            <div class="search-args-group" v-for="(paramGroup, groupIndex) in searchParams" :key="groupIndex"
                :class="groupIndex ? 'side-target-color' : 'side-source-color'">
                <h3 class="query-side" :class="{ stacked: origin }">
                    <span class="side-mark" :class="groupIndex ? 'target' : 'source'" aria-hidden="true"></span>
                    <span v-if="origin" class="query-side-text">
                        <span>{{ foundLeads[groupIndex] }}</span><span class="visually-hidden">, </span>
                        <span class="query-side-label" v-html="paramGroup.direction"></span>
                    </span>
                    <template v-else><span v-html="paramGroup.direction"></span>&nbsp;Parameters:</template>
                </h3>
                <ul class="list-unstyled mb-0" v-if="paramGroup.params != null">
                    <li class="metadata-args" v-for="metadata in paramGroup.params" :key="metadata.fieldName">
                        <span class="metadata-label">
                            {{ metadata.label }}
                        </span>
                        <span class="metadata-value">
                            {{ shownValue(metadata.value) }}
                        </span>
                        <button type="button" class="remove-metadata"
                            :aria-label="`Remove ${plainText(paramGroup.direction)} ${metadata.label} ${spokenValue(metadata.value)}`"
                            @click="removeMetadata(metadata)">&times;</button>
                    </li>
                </ul>
                <div class="metadata-args none" v-if="paramGroup.params == null">{{ origin ? "No filter" : "None" }}</div>
            </div>
            <span class="query-arrow" aria-hidden="true"></span>
        </div>
        <p class="query-note" v-if="scope == 'any'">
            This database cannot read this search by where a passage was first found, so every reused passage that
            appears in a pair of texts meeting these filters is listed, wherever its text falls in the history of the
            passage.
        </p>
        <p class="query-note" v-if="origin && hasLength">
            A passage length is that of the passage in one pair of texts; given on both sides, one pair must meet both.
        </p>
        <div class="query-banality" v-if="banality">
            <div class="metadata-args rounded-pill">
                <span class="metadata-label"> Banality filter </span>
                <span class="metadata-value">
                    {{ banality }}
                </span>
                <button type="button" class="remove-metadata" :aria-label="`Remove banality filter ${banality}`"
                    @click="removeMetadata({ fieldName: 'banality' })">&times;</button>
            </div>
        </div>
        <slot v-if="!origin" name="note"></slot>
    </div>
</template>
<script setup>
import emitter from "tiny-emitter/instance";
import { computed, onUnmounted, ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import globalConfig from "../config.js";
import { paramsToUrl, plainText, unquote } from "../utils.js";

// What the view counts: shared passages, the pairs of texts, or reused passages, the families
const props = defineProps({ unit: { type: String, default: "shared" } });

const route = useRoute();
const router = useRouter();
const counts = ref("...");
const single = ref(false);
const nouns = { shared: ["shared passage", "shared passages"], reused: ["reused passage", "reused passages"] };
const noun = computed(() => nouns[props.unit][single.value ? 0 : 1]);
const searchParams = ref([
    { direction: globalConfig.sourceLabel, params: null },
    { direction: globalConfig.targetLabel, params: null },
]);
const banality = ref(null);
// The rule a sorted view applied: origin, or any where the database cannot read the search by it
const scope = ref(null);

function update(params) {
    counts.value = params.counts.toLocaleString();
    single.value = Number(params.counts) === 1;
    scope.value = params.scope || null;
    searchParams.value = processParams(params.searchParams);
    if ("banality" in params.searchParams && params.searchParams.banality.length > 0) {
        banality.value = params.searchParams.banality == "false" ? "Filter all" : "Only banalities";
    } else {
        banality.value = false;
    }
    if (params.counts !== "") {
        emitter.emit("announce", `${counts.value} ${noun.value}`);
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

// A sorted view reads the search by where each passage was first found; scope "any" is the fallback
const origin = computed(() => scope.value == "origin");
const queryTail = computed(() => (origin.value ? "" : " for the following query:"));
const foundLeads = ["First found in", "Found again in"];
const hasLength = computed(() =>
    searchParams.value.flatMap((group) => group.params || []).some((metadata) => metadata.fieldName.endsWith("passage_length"))
);

function processParams(params) {
    const processed = [];
    for (const direction of ["source", "target"]) {
        const paramGroup = [];
        for (const metadata of globalConfig.metadataFields[direction]) {
            if (metadata.value in params && params[metadata.value].length > 0) {
                paramGroup.push({ label: metadata.label, fieldName: metadata.value, value: params[metadata.value] });
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

// The value on a chip: N/A for none, curly quotes for an exact match, as typed otherwise
function shownValue(value) {
    if (value == '""') return "N/A";
    const bare = unquote(value);
    return bare == value ? value : `\u201c${bare}\u201d`;
}

// ... and as it is read out
function spokenValue(value) {
    return value == '""' ? "N/A" : unquote(value);
}
</script>
<style lang="scss" scoped>
@import "../assets/theme.module.scss";

.query-summary {
    width: 100%;
    margin-top: 0.25rem;
    padding: 0.25rem 0 0;
    font-family: $font-ui;
    font-size: 0.95rem;
    color: $muted;
}

.result-count {
    margin-right: 0.15rem;
    color: $header-color;
    @include display-type;
    font-size: 1.9rem;
    font-weight: 600;
    line-height: 1;
}

.query-count {
    margin-bottom: 0.6rem;
}

.query-note {
    margin: 0.6rem 0 0;
    font-size: 0.85rem;
    font-style: italic;
}

// Both sides of the query, joined like the sides of a pair
.query-sides {
    position: relative;
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 0 4rem;
    padding: 0.85rem 1.1rem 0.35rem;
    border: 1px solid $rule-color;
    border-radius: 3px;
    background-color: #fff;
}

.query-arrow {
    position: absolute;
    top: 1.45rem;
    left: calc(50% - 1.5rem);
    width: 3rem;
    height: 1.5px;
    background: linear-gradient(90deg, $source-color, $target-color);
}

.query-arrow::after {
    content: "";
    position: absolute;
    right: 0;
    top: -3.5px;
    width: 8px;
    height: 8px;
    border-top: 1.5px solid $target-color;
    border-right: 1.5px solid $target-color;
    transform: rotate(45deg);
}

.query-side {
    @include kicker;
    display: flex;
    align-items: center;
    margin: 0 0 0.6rem;
    color: inherit;
}

// Where the passage was, over the side's own name, as in the facet panel
.query-side.stacked {
    align-items: flex-start;
}

.query-side.stacked .side-mark {
    margin-top: 0.22rem;
}

.query-side-text {
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
}

.query-side-label {
    color: $muted;
    font-size: 0.75rem;
    font-weight: 400;
    letter-spacing: 0;
    text-transform: none;
}

.metadata-args {
    display: inline-flex !important;
    margin: 0 0.4rem 0.5rem 0;
    border: 1px solid currentColor !important;
    border-radius: 2rem;
    width: fit-content;
    line-height: 1.9;
    background-color: #fff;
    font-size: 0.85rem;
}

.side-source-color .metadata-args {
    border-color: rgba($source-color, 0.5) !important;
}

.side-target-color .metadata-args {
    border-color: rgba($target-color, 0.5) !important;
}

.metadata-args.none {
    border-width: 0px !important;
    color: $muted;
    font-style: italic;
}

.metadata-label {
    display: flex;
    align-items: center;
    padding: 0 0.4rem 0 0.7rem;
    border-radius: 2rem 0 0 2rem;
    background-color: transparent !important;
    color: inherit !important;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.metadata-value {
    display: flex;
    align-items: center;
    padding: 0 0.4rem;
    color: $ink;
    box-decoration-break: clone;
}

.remove-metadata {
    display: flex;
    align-items: center;
    padding: 0 0.6rem 0 0.45rem;
    border: 0;
    border-radius: 0 2rem 2rem 0;
    background-color: transparent;
    color: inherit;
    font-size: 1rem;
}

.remove-metadata:hover {
    cursor: pointer;
    background-color: $source-tint !important;
    color: $source-deep !important;
}

.side-target-color .remove-metadata:hover {
    background-color: $target-tint !important;
    color: $target-deep !important;
}

.side-target-color .remove-metadata:focus-visible {
    outline-color: $target-color !important;
}

.query-banality {
    margin-top: 0.75rem;
    color: $source-color;
}

.compact-search-args {
    padding-bottom: 0.25rem;
}

@media (max-width: 767.98px) {
    .query-sides {
        grid-template-columns: minmax(0, 1fr);
        gap: 0.75rem;
    }

    .query-arrow {
        display: none;
    }

    .result-count {
        font-size: 1.6rem;
    }
}
</style>
