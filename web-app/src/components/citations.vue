<template>
    <component :is="tag" class="pt-2 mb-2 px-3" :class="{ 'd-block': tag != 'p' }" :lang="language">
        <span class="citation" v-for="(cite, citeIndex) in filled" :key="citeIndex"
            :data-field="cite.field.replace(/^(source|target)_/, '')">
            <span class="separator px-2" aria-hidden="true" v-if="citeIndex > 0">&#9679;</span>
            <router-link :to="href" :style="cite.style"
                v-if="linkToDoc == cite.field && textAvailable">{{ cite.text }}</router-link>
            <span :style="cite.style" v-else>{{ cite.text }}</span>
        </span>
        <span class="citation unattributed" lang="en" v-if="!filled.length">
            <router-link :to="href" v-if="linkToDoc && textAvailable">{{ UNATTRIBUTED }}</router-link>
            <span v-else>{{ UNATTRIBUTED }}</span>
        </span>
    </component>
</template>
<script setup>
import { computed } from "vue";
import globalConfig from "../config.js";
import { UNATTRIBUTED } from "../utils.js";

const props = defineProps({
    citation: Array,
    alignment: Object,
    linkToDoc: { type: String, default: "" },
    tag: { type: String, default: "p" },
});

// Only the fields the document has: a link or button with no text has no name
const filled = computed(() =>
    (props.citation || [])
        .map((cite) => ({ ...cite, text: String(props.alignment[cite.field] ?? "").trim() }))
        .filter((cite) => cite.text)
);
const direction = computed(() => (props.linkToDoc.startsWith("source_") ? "source" : "target"));
const language = computed(() => {
    const field = props.citation && props.citation.length ? props.citation[0].field : "";
    const side = field.startsWith("target_") ? "target" : "source";
    return globalConfig[`${side}Language`] || undefined;
});
const textAvailable = computed(() => globalConfig[`${direction.value}TextAvailable`]);
const href = computed(() => {
    if (props.linkToDoc.length == 0) return;
    return `/text-view/?db_table=${globalConfig.databaseName}&doc_id=${props.alignment[`${direction.value}_doc_id`]}&directionSelected=${direction.value}`;
});
</script>
<style lang="scss" scoped>
@import "../assets/theme.module.scss";

.separator {
    font-size: 0.4rem;
    vertical-align: 0.2rem;
    opacity: 0.55;
}

.citation {
    font-weight: 400;
    line-height: 1.45;
    white-space: nowrap;
}

// Only the separator is held to the item after it
.citation > :not(.separator) {
    white-space: normal;
}

// Like "None" in the search summary
.unattributed {
    font-style: italic;
}

.unattributed > span {
    color: $muted;
}

.citation[data-field="author"] {
    color: $ink;
    font-weight: 600;
}

.citation[data-field="year"] {
    font-weight: 600;
    font-variant-numeric: lining-nums tabular-nums;
}
</style>
