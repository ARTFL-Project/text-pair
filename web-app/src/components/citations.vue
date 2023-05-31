<template>
    <p class="pt-2 mb-2 px-3">
        <span class="citation" v-for="(cite, citeIndex) in citation" :key="citeIndex">
            <router-link :to="href" :style="cite.style" v-if="linkToDoc == cite.field">{{ alignment[cite.field]
            }}</router-link>
            <span :style="cite.style" v-else v-if="alignment[cite.field]">{{ alignment[cite.field] }}</span>
            <span class="separator px-2" v-if="alignment[cite.field] && citeIndex != citation.length - 1">&#9679;</span>
        </span>
    </p>
</template>
<script>
export default {
    name: "citations",
    props: {
        "citation": Object, "alignment": Object, "linkToDoc": { type: String, default: "" }
    },
    data() {
        return {
            globalConfig: this.$globalConfig
        }
    },
    computed: {

        href() {
            if (this.linkToDoc.length == 0) return;
            let philoUrl, philoPath, philo_id;
            if (this.linkToDoc.indexOf("source_")) {
                philoPath = this.globalConfig.sourcePhiloDBPath;
                philoUrl = this.globalConfig.sourcePhiloDBUrl;
                philo_id = this.alignment.source_philo_id;
            } else {
                philoPath = this.globalConfig.targetPhiloDBPath
                philoUrl = this.globalConfig.targetPhiloDBUrl;
                philo_id = this.alignment.target_philo_id;
            }
            return `/text-view/?db_table=${this.globalConfig.databaseName}&philo_url=${philoUrl}&philo_path=${philoPath}&philo_id=${philo_id}`
        }

    }
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