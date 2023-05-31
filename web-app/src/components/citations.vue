<template>
    <p class="pt-2 mb-2 px-3">
        <span class="citation" v-for="(cite, citeIndex) in citation" :key="citeIndex">
            <!-- <router-link :to="cite.href" :style="cite.style" v-if="cite.href">{{ alignment[citation.field] }}</router-link> -->
            <span :style="cite.style" v-if="alignment[cite.field]">{{ alignment[cite.field] }}</span>
            <span class="separator px-2" v-if="alignment[cite.field] && citeIndex != citation.length - 1">&#9679;</span>
        </span>
    </p>
</template>
<script>
export default {
    name: "citations",
    props: {
        "citation": Object, "alignment": Object
    },
};
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