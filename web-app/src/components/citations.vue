<template>
    <component :is="tag" class="pt-2 mb-2 px-3" :class="{ 'd-block': tag != 'p' }" :lang="language">
        <span class="citation" v-for="(cite, citeIndex) in filled" :key="citeIndex"
            :data-field="cite.field.replace(/^(source|target)_/, '')">
            <span class="separator px-2" aria-hidden="true" v-if="citeIndex > 0">&#9679;</span>
            <router-link :to="href" :style="cite.style"
                v-if="linkField == cite.field && textAvailable">{{ cite.text }}</router-link>
            <span :style="cite.style" v-else>{{ cite.text }}</span>
        </span>
        <span class="citation unattributed" lang="en" v-if="!filled.length">
            <router-link :to="href" v-if="linkField && textAvailable">{{ UNATTRIBUTED }}</router-link>
            <span v-else>{{ UNATTRIBUTED }}</span>
        </span>
    </component>
</template>
<script setup>
import { computed } from "vue";
import globalConfig from "../config.js";
import { documentCollection, UNATTRIBUTED } from "../utils.js";

const props = defineProps({
    citation: Array,
    alignment: Object,
    linkToDoc: { type: String, default: "" },
    tag: { type: String, default: "p" },
    // Or the side of `alignment` whose fields the document has: cited as its own collection cites its texts,
    // which a passage family may hold on either side, and linked to its text if `linked`
    side: { type: String, default: "" },
    linked: Boolean,
});

const collection = computed(() => (props.side ? documentCollection(props.alignment, props.side) : ""));
const cites = computed(() => (props.side ? globalConfig[`${collection.value}Citation`] : props.citation) || []);
const linkField = computed(() =>
    props.side ? (props.linked && globalConfig[`${collection.value}LinkToDocMetadata`]) || "" : props.linkToDoc
);
// A cited field's value, from the side's own keys where the document stands on a side
const valueOf = (field) => props.alignment[props.side ? `${props.side}_${field.replace(/^(source|target)_/, "")}` : field];

// Only the fields the document has: a link or button with no text has no name
const filled = computed(() =>
    cites.value
        .map((cite) => ({ ...cite, text: String(valueOf(cite.field) ?? "").trim() }))
        .filter((cite) => cite.text)
);
const direction = computed(() => collection.value || (linkField.value.startsWith("source_") ? "source" : "target"));
const language = computed(() => {
    const field = cites.value.length ? cites.value[0].field : "";
    const side = collection.value || (field.startsWith("target_") ? "target" : "source");
    return globalConfig[`${side}Language`] || undefined;
});
const textAvailable = computed(() => globalConfig[`${direction.value}TextAvailable`]);
const href = computed(() => {
    if (linkField.value.length == 0) return;
    const docId = props.alignment[`${props.side || direction.value}_doc_id`];
    return `/text-view/?db_table=${globalConfig.databaseName}&doc_id=${docId}&directionSelected=${direction.value}`;
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
