<template>
    <div class="local-pair">
        <div class="row">
            <div class="col mt-4">
                <component :is="`h${headingLevel}`" class="h6 paired-passage-label text-center pb-2"
                    v-html="globalConfig.sourceLabel"></component>
                <citations :citation="globalConfig.sourceCitation" :alignment="alignment"
                    :link-to-doc="globalConfig.sourceLinkToDocMetadata"></citations>
            </div>
            <div class="col mt-4 border border-top-0 border-right-0 border-bottom-0 target-passage-container">
                <component :is="`h${headingLevel}`" class="h6 paired-passage-label text-center pb-2"
                    v-html="globalConfig.targetLabel"></component>
                <citations :citation="globalConfig.targetCitation" :alignment="alignment"
                    :link-to-doc="globalConfig.targetLinkToDocMetadata"></citations>

                <div class="classification-tags ms-3" style="margin-top: -0.75rem"
                    v-if="alignment.target_first_class || alignment.target_second_class || alignment.target_third_class">
                    <span v-if="alignment.target_first_class" class="classification-tag tag-primary">
                        {{ alignment.target_first_class.replace(/_/g, ' ') }}
                    </span>
                    <span v-if="alignment.target_second_class" class="classification-tag tag-secondary">
                        {{ alignment.target_second_class.replace(/_/g, ' ') }}
                    </span>
                    <span v-if="alignment.target_third_class" class="classification-tag tag-tertiary">
                        {{ alignment.target_third_class.replace(/_/g, ' ') }}
                    </span>
                </div>
            </div>
        </div>

        <div class="row passages">
            <div class="col mb-2">
                <p class="card-text text-justify px-3 pt-2 mb-2" :lang="globalConfig.sourceLanguage || undefined">
                    {{ alignment.source_context_before }}
                    <span class="source-passage" v-if="sourceHtml === null">{{ alignment.source_passage }}</span>
                    <span class="source-passage" v-else v-html="sourceHtml"></span>
                    {{ alignment.source_context_after }}
                </p>
                <button type="button" class="btn btn-outline-secondary position-absolute rounded-0"
                    style="bottom: 0; left: 0" v-if="globalConfig.sourceTextAvailable"
                    @click="goToContext(alignment, 'source')">
                    View passage in context<span class="visually-hidden">: {{ sourceLabel }}</span>
                </button>
            </div>
            <div class="col mb-2 border border-top-0 border-right-0 border-bottom-0 target-passage-container">
                <p class="card-text text-justify px-3 mb-2" :lang="globalConfig.targetLanguage || undefined">
                    {{ alignment.target_context_before }}
                    <span class="target-passage" v-if="targetHtml === null">{{ alignment.target_passage }}</span>
                    <span class="target-passage" v-else v-html="targetHtml"></span>
                    {{ alignment.target_context_after }}
                </p>
                <button type="button" class="btn btn-outline-secondary position-absolute rounded-0"
                    style="bottom: 0; right: 0" v-if="globalConfig.targetTextAvailable"
                    @click="goToContext(alignment, 'target')">
                    View passage in context<span class="visually-hidden">: {{ targetLabel }}</span>
                </button>
            </div>
        </div>
        <div class="mb-2 ms-3" style="margin-top: -0.5rem"
            v-if="globalConfig.matchingAlgorithm == 'sa' && alignment.count > 1">
            &rarr;
            <router-link :to="`/group/${alignment.group_id}`">
                Passage reused in {{ alignment.count }} different titles
            </router-link>
        </div>

        <div class="text-muted text-center mb-2">
            <div v-if="globalConfig.matchingAlgorithm == 'vsa'">
                <div class="d-flex justify-content-center align-items-center gap-2 mb-1">
                    <span v-if="alignment.llm_stance && alignment.llm_stance !== 'Unknown'"
                        :class="stanceClass(alignment.llm_stance)" class="stance-badge">
                        {{ alignment.llm_stance }}
                    </span>
                    <div>{{ alignment.similarity.toFixed(2) * 100 }} % similar</div>
                </div>
                <div v-if="alignment.llm_reasoning && !alignment.llm_reasoning.startsWith('Error:')"
                    class="reasoning-text small px-4 pb-1">
                    {{ alignment.llm_reasoning }}
                </div>
                <button type="button" class="diff-btn" @click="showMatches">
                    {{ showingDiff ? "Hide matching words" : "Show matching words" }}
                </button>
            </div>
            <div v-if="globalConfig.matchingAlgorithm == 'sa'">
                <button type="button" class="diff-btn" @click="showDifferences">
                    {{ showingDiff ? "Hide differences" : "Show differences" }}
                </button>
                <div class="loading position-absolute" style="left: 50%; transform: translateX(-50%)" v-if="diffLoading">
                    <div class="spinner-border"
                        style="width: 1.4rem; height: 1.4rem; position: absolute; z-index: 50; top: 5px; left: -10px;"
                        role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup>
import emitter from "tiny-emitter/instance";
import { onMounted, ref } from "vue";
import { useRouter } from "vue-router";
import globalConfig from "../config.js";
import { plainText } from "../utils.js";
import citations from "./citations";
import Worker from "./diffStrings?worker";

const props = defineProps({
    alignment: Object,
    index: Number,
    diffed: Boolean,
    headingLevel: { type: Number, default: 4 },
    startByteIndex: Number,
    endByteIndex: Number,
});

const router = useRouter();
const sourceLabel = plainText(globalConfig.sourceLabel);
const targetLabel = plainText(globalConfig.targetLabel);
// Passages as HTML while differences or matches are shown, null otherwise
const sourceHtml = ref(null);
const targetHtml = ref(null);
const showingDiff = ref(false);
const diffLoading = ref(false);

onMounted(() => {
    if (props.diffed) {
        showDifferences();
    }
});

function escapeHtml(text) {
    return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function showDifferences() {
    if (showingDiff.value) {
        sourceHtml.value = null;
        targetHtml.value = null;
        showingDiff.value = false;
        emitter.emit("announce", "Differences hidden");
        return;
    }
    const sourceText = props.alignment.source_passage.replace(/\s\s+/g, " ");
    const targetText = props.alignment.target_passage.replace(/\s\s+/g, " ");
    if (props.alignment.source_passage_length > 10000 || props.alignment.target_passage.length > 10000) {
        alert("Passage of 10000 words or more may take up a long time to compare");
    }
    diffLoading.value = true;
    const worker = new Worker();
    const startByteIndex = props.startByteIndex;
    const endByteIndex = props.endByteIndex;
    // works around issue with diff-match-patch where no matches are found if the match is far after the beginning of the string
    const slicing = startByteIndex != undefined && startByteIndex > 0;
    if (slicing) {
        worker.postMessage([sourceText.slice(startByteIndex, endByteIndex), targetText]);
    } else {
        worker.postMessage([sourceText, targetText]);
    }
    worker.onmessage = (response) => {
        let newSourceString = "";
        let newTargetString = "";
        for (const [diffCode, text] of response.data) {
            if (diffCode === 0) {
                newSourceString += escapeHtml(text);
                newTargetString += escapeHtml(text);
            } else if (diffCode === -1) {
                newSourceString += `<del class="removed">${escapeHtml(text)}</del>`;
            } else if (diffCode === 1) {
                newTargetString += `<ins class="added">${escapeHtml(text)}</ins>`;
            }
        }
        if (slicing) {
            newSourceString = `<del class="removed">${escapeHtml(sourceText.slice(0, startByteIndex))}</del>${newSourceString}<del class="removed">${escapeHtml(sourceText.slice(endByteIndex))}</del>`;
        }
        sourceHtml.value = newSourceString;
        targetHtml.value = newTargetString;
        showingDiff.value = true;
        diffLoading.value = false;
        worker.terminate();
        emitter.emit("announce", "Differences shown");
    };
}

function stanceClass(stance) {
    const map = {
        Agree: "stance-agree",
        Disagree: "stance-disagree",
        Neutral: "stance-neutral",
        Unrelated: "stance-unrelated",
    };
    return map[stance] || "";
}

function showMatches() {
    if (showingDiff.value) {
        sourceHtml.value = null;
        targetHtml.value = null;
        showingDiff.value = false;
        emitter.emit("announce", "Matching words hidden");
        return;
    }
    sourceHtml.value = props.alignment.source_passage_with_matches.replace(/&gt;/g, ">").replace(/&lt;/g, "<");
    targetHtml.value = props.alignment.target_passage_with_matches.replace(/&gt;/g, ">").replace(/&lt;/g, "<");
    showingDiff.value = true;
    emitter.emit("announce", "Matching words shown");
}

function goToContext(alignment, direction) {
    router.push({
        path: "/text-view/",
        query: {
            db_table: globalConfig.databaseName,
            doc_id: alignment[`${direction}_doc_id`],
            start_byte: alignment[`${direction}_start_byte`],
            end_byte: alignment[`${direction}_end_byte`],
            directionSelected: direction,
        },
    });
}
</script>

<style lang="scss" scoped>
@import "../assets/theme.module.scss";

.source-passage,
.target-passage {
    color: $passage-color;
    font-weight: 700;
}

:deep(.added) {
    color: $added-color;
    font-weight: 700;
    text-decoration: underline;
}

:deep(.removed) {
    color: $removed-color;
    font-weight: 700;
    text-decoration: line-through;
}

:deep(.token-match) {
    color: darkblue;
    font-weight: 700;
    text-decoration: underline;
}

:deep(.filtered-token) {
    color: #6c757d;
    font-weight: 400;
}


.target-passage-container {
    border-right-width: 0 !important;
}

.stance-badge {
    display: inline-block;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 2px 9px;
    border-radius: 4px;
    color: #fff;
    text-transform: capitalize;
}

.stance-agree { background-color: #1e8449; }
.stance-disagree { background-color: #c0392b; }
.stance-neutral { background-color: #5f6b6d; }
.stance-unrelated { background-color: #707070; }

.reasoning-text {
    font-style: italic;
    line-height: 1.4;
    max-width: 90%;
    margin: 0 auto;
    color: #6c757d;
}

/* Classification tag styling */
.classification-tags {
    padding-top: 8px;
    padding-bottom: 8px;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.classification-tag {
    display: inline-block;
    font-size: 0.85rem;
    font-weight: 500;
    line-height: 1;
    padding: 6px 10px;
    border-radius: 4px;
    color: #fff;
    text-transform: capitalize;
}

.tag-primary {
    background-color: #2c3e50;
}

.tag-secondary {
    background-color: #2471a3;
}

.tag-tertiary {
    background-color: #8e44ad;
}
</style>