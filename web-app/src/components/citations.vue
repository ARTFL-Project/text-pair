<template>
    <component :is="tag" class="pt-2 mb-2 px-3" :class="{ 'd-block': tag != 'p' }" :lang="language">
        <span class="citation" v-for="(cite, citeIndex) in citation" :key="citeIndex">
            <router-link :to="href" :style="cite.style"
                v-if="linkToDoc == cite.field && textAvailable">{{ alignment[cite.field] }}</router-link>
            <span :style="cite.style" v-else-if="alignment[cite.field]">{{ alignment[cite.field] }}</span>
            <span class="separator px-2" aria-hidden="true"
                v-if="alignment[cite.field] && citeIndex != citation.length - 1">&#9679;</span>
        </span>
    </component>
</template>
<script setup>
import { computed } from "vue";
import globalConfig from "../config.js";

const props = defineProps({
    citation: Array,
    alignment: Object,
    linkToDoc: { type: String, default: "" },
    tag: { type: String, default: "p" },
});

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
<style scoped>
.separator {
    font-size: 0.75rem;
    vertical-align: 0.05rem;
}

.citation {
    font-weight: 600;
}

</style>