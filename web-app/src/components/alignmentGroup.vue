<template>
    <div id="alignment-group" class="mt-4">
        <h2 class="visually-hidden">Reuses of a passage</h2>
        <div class="h5" style="text-align: center">
            <i :lang="globalConfig.sourceLanguage || undefined">"{{ sourcePassage.source_passage }}"</i>
            <div class="reuse-title mt-1">
                <citations :citation="globalConfig.sourceCitation" :alignment="sourcePassage"></citations>
            </div>
        </div>
        <div id="timeline-container" class="px-2 pb-2">
            <div id="vertical-line"></div>
            <section class="timeline-dates" v-for="(date, index) in timeline" :key="index">
                <h3 class="year btn btn-secondary">{{ date.year }}</h3>
                <div class="timeline-events card shadow-1 px-2 pt-2 mt-3" v-for="(reuse, reuseIndex) in date.result"
                    :key="reuseIndex">
                    <h4 class="h5 reuse-title">
                        <button type="button" class="reuse-toggle" :aria-expanded="String(isOpen(index, reuseIndex))"
                            :aria-controls="`reuse-${index}-${reuseIndex}`" @click="togglePassage(index, reuseIndex)">
                            <citations :citation="globalConfig[`${reuse.direction}Citation`]" :alignment="reuse"
                                tag="span"></citations>
                        </button>
                    </h4>
                    <div class="timeline-text-content m-0" :class="{ show: isOpen(index, reuseIndex) }"
                        :id="`reuse-${index}-${reuseIndex}`">
                        <p class="text-content m-0" :lang="globalConfig[`${reuse.direction}Language`] || undefined">
                            {{ reuse[`${reuse.direction}_context_before`] }}
                            <span class="highlight">{{ reuse[`${reuse.direction}_passage`] }}</span> {{
                                reuse[`${reuse.direction}_context_after`] }}
                        </p>
                        <span class="text-muted text-center d-block mt-1">
                            <button type="button" class="group-diff-btn" @click="showDifferences(reuse)">Show
                                differences</button>
                        </span>
                    </div>
                </div>
            </section>
        </div>
        <div id="passage-diff" ref="modalElement" class="modal fade" tabindex="-1" aria-labelledby="passage-diff-title">
            <div class="modal-dialog modal-xl">
                <div class="modal-content">
                    <div class="modal-header">
                        <h2 class="modal-title h5" id="passage-diff-title">Passage Pair</h2>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <passage-pair v-if="showPair" :alignment="localAlignment" :diffed="true" :index="0"
                            :heading-level="3" :startByteIndex="startByteIndex" :endByteIndex="endByteIndex"
                            :key="render"></passage-pair>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup>
import axios from "axios";
import { Modal } from "bootstrap";
import { onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useRoute } from "vue-router";
import globalConfig from "../config.js";
import { paramsToUrl } from "../utils.js";
import citations from "./citations.vue";
import passagePair from "./passagePair.vue";

const route = useRoute();
const timeline = ref({});
const sourcePassage = ref({});
const openPassages = ref({});
const showPair = ref(false);
const localAlignment = ref({});
const startByteIndex = ref(0);
const endByteIndex = ref(0);
const render = ref(0); // used to force re-rendering of passagePair
const modalElement = ref(null);
let modal = null;
let modalOpener = null;

function returnFocus() {
    if (modalOpener) modalOpener.focus();
}

onMounted(() => {
    modal = new Modal(modalElement.value, { backdrop: "static" });
    modalElement.value.addEventListener("hidden.bs.modal", returnFocus);
});

// Hidden rather than disposed, so the next page doesn't inherit the modal's scroll lock
onBeforeUnmount(() => {
    modalElement.value.removeEventListener("hidden.bs.modal", returnFocus);
    modal.hide();
});

function fetchData() {
    const params = { db_table: globalConfig.databaseName };
    openPassages.value = {};
    axios
        .get(`${globalConfig.apiServer}/group/${route.params.groupId}/?${paramsToUrl(params)}`)
        .then((response) => {
            timeline.value = response.data.passageList;
            sourcePassage.value = response.data.original_passage;
        })
        .catch((error) => {
            console.log(error);
        });
}
fetchData();
watch(() => route.fullPath, fetchData);

function isOpen(index, reuseIndex) {
    return Boolean(openPassages.value[`${index}-${reuseIndex}`]);
}

function togglePassage(index, reuseIndex) {
    const key = `${index}-${reuseIndex}`;
    openPassages.value[key] = !openPassages.value[key];
}

function showDifferences(reuse) {
    modalOpener = document.activeElement;
    showPair.value = false;
    localAlignment.value = { ...sourcePassage.value };
    // The reuse's own side only: it holds both sides of its alignment
    const side = `${reuse.direction}_`;
    for (const key in reuse) {
        if (key.startsWith(side)) {
            localAlignment.value[`target_${key.slice(side.length)}`] = reuse[key];
        }
    }
    localAlignment.value.count = 1;
    reuse.source_passage = reuse.source_passage.replace(/\s\s+/g, " ");
    sourcePassage.value.source_passage = sourcePassage.value.source_passage.replace(/\s\s+/g, " ");
    startByteIndex.value = sourcePassage.value.source_passage.indexOf(reuse.source_passage);
    let remove = reuse.source_passage.length;
    while (startByteIndex.value == -1) { // if the passage is not found, remove one character from the end and try again
        startByteIndex.value = sourcePassage.value.source_passage.indexOf(reuse.source_passage.slice(0, remove));
        remove -= 1;
    }
    endByteIndex.value = startByteIndex.value + reuse.source_passage.length;
    render.value += 1;
    showPair.value = true;
    modal.show();
}
</script>

<style  lang="scss" scoped>
@import "../assets/theme.module.scss";


#timeline-container,
.timeline-dates {
    text-align: center;
    position: relative;
}

#vertical-line {
    position: absolute;
    left: 50%;
    border-left: 2px dotted $button-color;
    top: 0;
    height: 105%;
    z-index: -2;
}

.year,
.reuse-title,
.timeline-events {
    position: relative;
    z-index: 1;
}

.timeline-events {
    width: 70%;
    margin-left: auto;
    margin-right: auto;
}

.highlight,
.reuse-title {
    color: $passage-color;
}

.reuse-toggle {
    width: 100%;
    padding: 0;
    border-width: 0;
    background-color: transparent;
    color: inherit;
    font: inherit;
}

.timeline-text-content {
    visibility: hidden;
    font-size: 1rem;
    overflow: hidden;
    z-index: -1;
    position: relative;
    line-height: 0;
    padding: 0;
    transform: translateY(-25%);
    opacity: 0;
    transition: all 200ms ease-out;
}

.timeline-text-content.show {
    visibility: visible;
    opacity: 1;
    padding-top: 0.25rem;
    padding-bottom: 0.5rem;
    line-height: 1.5;
    transform: translateY(0);
}

.year {
    margin-top: 2rem;
    margin-bottom: .25rem;
    cursor: initial;
}

.group-diff-btn {
    font-size: smaller;
}
</style>