<template>
    <p class="pt-2 mb-2 px-3">
        <span class="citation" v-for="(cite, citeIndex) in citation" :key="citeIndex">
            <router-link :to="href" :style="cite.style" v-if="linkToDoc == cite.field && getFieldValue(cite.field)">{{ getFieldValue(cite.field)
            }}</router-link>
            <span :style="cite.style" v-else v-if="getFieldValue(cite.field)">{{ getFieldValue(cite.field) }}</span>
            <span class="separator px-2" v-if="getFieldValue(cite.field) && citeIndex != citation.length - 1">&#9679;</span>
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
    methods: {
        getFieldValue(field) {
            // If field exists, return it
            if (this.alignment[field]) {
                return this.alignment[field];
            }
            // If target field doesn't exist (source_against_source case), try source equivalent
            if (field.startsWith('target_')) {
                const sourceField = field.replace('target_', 'source_');
                return this.alignment[sourceField];
            }
            return null;
        }
    },
    computed: {
        href() {
            if (this.linkToDoc.length == 0) return;
            if (this.linkToDoc.indexOf("source_") != -1) {
                return `/text-view/?db_table=${this.globalConfig.databaseName}&philo_path=${this.globalConfig.sourcePhiloDBPath}&philo_id=${this.alignment.source_philo_id}&directionSelected=source`
            }
            return `/text-view/?db_table=${this.globalConfig.databaseName}&philo_path=${this.globalConfig.targetPhiloDBPath}&philo_id=${this.alignment.target_philo_id}&directionSelected=target`
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