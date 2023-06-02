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
            if (this.linkToDoc.indexOf("source_") != -1) {
                return `/text-view/?db_table=${this.globalConfig.databaseName}&philo_url=${this.globalConfig.sourcePhiloDBLink}&philo_path=${this.globalConfig.sourcePhiloDBPath}&philo_id=${this.alignment.source_philo_id}&directionSelected=source`
            }
            return `/text-view/?db_table=${this.globalConfig.databaseName}&philo_url=${this.globalConfig.targetPhiloDBLink}&philo_path=${this.globalConfig.targetPhiloDBPath}&philo_id=${this.alignment.target_philo_id}&directionSelected=target`
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