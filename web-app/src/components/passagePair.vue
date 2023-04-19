<template>
    <div class="local-pair">
        <div class="row">
            <div class="col mt-4">
                <h6 class="passage-label text-center pb-2" v-html="globalConfig.sourceLabel"></h6>
                <p class="pt-3 px-3">
                    <span v-for="(citation, citationIndex) in globalConfig.sourceCitation" :key="citation.field">
                        <span v-if="alignment[citation.field]">
                            <span :style="citation.style">{{ alignment[citation.field] }}</span>
                            <span class="separator"
                                v-if="citationIndex != globalConfig.sourceCitation.length - 1">&#9679;&nbsp;</span>
                        </span>
                    </span>
                </p>
            </div>
            <div class="col mt-4 border border-top-0 border-right-0 border-bottom-0 target-passage-container">
                <h6 class="passage-label text-center pb-2" v-html="globalConfig.targetLabel"></h6>
                <p class="pt-3 px-3">
                    <span v-for="(citation, citationIndex) in globalConfig.targetCitation" :key="citation.field">
                        <span v-if="alignment[citation.field]">
                            <span :style="citation.style">{{ alignment[citation.field] }}</span>
                            <span class="separator"
                                v-if="citationIndex != globalConfig.targetCitation.length - 1">&#9679;&nbsp;</span>
                        </span>
                    </span>
                </p>
            </div>
        </div>
        <div class="row passages">
            <div class="col mb-2">
                <p class="card-text text-justify px-3 pt-2 mb-2">
                    {{ alignment.source_context_before }}
                    <span class="source-passage">{{ alignment.source_passage }}</span>
                    {{ alignment.source_context_after }}
                </p>
                <button type="button" class="btn btn-outline-secondary position-absolute rounded-0"
                    style="bottom: 0; left: 0" v-if="globalConfig.sourcePhiloDBLink"
                    @click="goToContext(alignment, 'source')">
                    View passage in context
                </button>
            </div>
            <div class="col mb-2 border border-top-0 border-right-0 border-bottom-0 target-passage-container">
                <p class="card-text text-justify px-3 mb-2">
                    {{ alignment.target_context_before }}
                    <span class="target-passage">{{ alignment.target_passage }}</span>
                    {{ alignment.target_context_after }}
                </p>
                <button type="button" class="btn btn-outline-secondary position-absolute rounded-0"
                    style="bottom: 0; right: 0" v-if="globalConfig.targetPhiloDBLink"
                    @click="goToContext(alignment, 'target')">
                    View passage in context
                </button>
            </div>
        </div>
        <div class="mb-2 ms-3" style="margin-top: -0.5rem"
            v-if="globalConfig.matchingAlgorithm == 'sa' && alignment.count > 1">
            &rarr;
            <router-link class="" :to="`group/${alignment.group_id}`">
                Passage reused in {{ alignment.count }} different titles
            </router-link>
        </div>

        <div class="text-muted text-center mb-2">
            <div v-if="globalConfig.matchingAlgorithm == 'vsa'">
                <div>{{ alignment.similarity.toFixed(2) * 100 }} % similar</div>
                <a class="diff-btn" diffed="false" @click="showMatches(alignment)">Show matching words</a>
                <div class="loading position-absolute" style="display: none; left: 50%; transform: translateX(-50%)">
                    <div class="spinner-border"
                        style="width: 4rem; height: 4rem; position: absolute; z-index: 50; top: 30px" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                </div>
            </div>
            <div v-if="globalConfig.matchingAlgorithm == 'sa'">
                <a class="diff-btn" diffed="false" @click="
                    showDifferences(
                        alignment.source_passage,
                        alignment.target_passage,
                        alignment.source_passage_length,
                        alignment.target_passage.length
                    )
                ">Show differences</a>
                <div class="loading position-absolute" style="display: none; left: 50%; transform: translateX(-50%)">
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

<script>
import Worker from "./diffStrings?worker";


export default {
    name: "passagePair",
    inject: ["$http"],
    props: {
        alignment: Object,
        index: Number,
        diffed: Boolean,
        startByteIndex: Number,
        endByteIndex: Number,
    },
    data() {
        return {
            loading: true,
            globalConfig: this.$globalConfig,
            done: false,
            timeline: {},
            sourcePassage: {},
            passages: [],
        };
    },
    mounted() {
        if (this.diffed) {
            let element = document.getElementsByClassName("diff-btn")[this.index];
            this.showDifferences(
                this.alignment.source_passage,
                this.alignment.target_passage,
                this.alignment.source_passage_length,
                this.alignment.target_passage.length,
                element
            );
        }
    },
    methods: {
        showDifferences(sourceText, targetText, sourcePassageLength, targetPassageLength, diffBtn) {
            sourceText = sourceText.replace(/\s\s+/g, ' ')
            targetText = targetText.replace(/\s\s+/g, ' ')
            if (sourcePassageLength > 10000 || targetPassageLength > 10000) {
                alert("Passage of 10000 words or more may take up a long time to compare");
            }
            if (diffBtn == undefined) {
                diffBtn = document.getElementsByClassName("diff-btn")[this.index];
            }
            let parent = diffBtn.parentNode.parentNode.parentNode;
            let loading = parent.querySelector(".loading");
            let sourceElement = parent.querySelector(".source-passage");
            let targetElement = parent.querySelector(".target-passage");
            if (diffBtn.getAttribute("diffed") == "false") {
                loading.style.display = "initial";
                const worker = new Worker();
                let slicing = false;
                let startByteIndex = this.startByteIndex
                let endByteIndex = this.endByteIndex
                if (this.startByteIndex != undefined && this.startByteIndex > 0) {
                    // works around issue with diff-match-patch where no matches are found if the match is far after the beginning of the string
                    slicing = true

                    worker.postMessage([sourceText.slice(this.startByteIndex, this.endByteIndex), targetText]);
                } else {
                    worker.postMessage([sourceText, targetText]);
                }
                worker.onmessage = function (response) {
                    let differences = response.data;
                    let newSourceString = "";

                    let newTargetString = "";
                    for (let diffObj of differences) {
                        let [diffCode, text] = diffObj;
                        if (diffCode === 0) {
                            newSourceString += text;
                            newTargetString += text;
                        } else if (diffCode === -1) {
                            newSourceString += `<span class="removed">${text}</span>`;
                        } else if (diffCode === 1) {
                            newTargetString += `<span class="added">${text}</span>`;
                        }
                    }
                    if (slicing) {
                        newSourceString = `<span class="removed">${sourceText.slice(0, startByteIndex)}</span>${newSourceString}<span class="removed">${sourceText.slice(endByteIndex)}</span>`;
                    }
                    sourceElement.innerHTML = newSourceString;
                    targetElement.innerHTML = newTargetString;
                    diffBtn.setAttribute("diffed", "true");
                    loading.style.display = "none";
                    diffBtn.textContent = "Hide differences";
                };
            } else {
                sourceElement.innerHTML = sourceText;
                targetElement.innerHTML = targetText;
                diffBtn.setAttribute("diffed", "false");
                diffBtn.textContent = "Show differences";
            }
        },
        showMatches: function (alignment, diffBtn) {
            if (diffBtn == undefined) {
                diffBtn = document.getElementsByClassName("diff-btn")[this.index];
            }
            let parent = diffBtn.parentNode.parentNode.parentNode;
            let sourceElement = parent.querySelector(".source-passage");
            let targetElement = parent.querySelector(".target-passage");
            if (diffBtn.getAttribute("diffed") == "false") {
                let source = alignment.source_passage_with_matches.replace(/&gt;/g, ">").replace(/&lt;/g, "<");
                sourceElement.innerHTML = source;
                let target = alignment.target_passage_with_matches.replace(/&gt;/g, ">").replace(/&lt;/g, "<");
                targetElement.innerHTML = target;
                diffBtn.setAttribute("diffed", "true");
                diffBtn.textContent = "Hide matching words";
            } else {
                sourceElement.innerHTML = alignment.source_passage;
                targetElement.innerHTML = alignment.target_passage;
                diffBtn.setAttribute("diffed", "false");
                diffBtn.textContent = "Show matching words";
            }
        },
        goToContext(alignment, direction) {
            let rootURL = "";
            let params = {};
            if (direction == "source") {
                rootURL = this.globalConfig.sourcePhiloDBLink.replace(/\/$/, "");
                params = {
                    filename: alignment.source_filename.substr(alignment.source_filename.lastIndexOf("/") + 1),
                    start_byte: alignment.source_start_byte,
                    end_byte: alignment.source_end_byte,
                };
            } else {
                rootURL = this.globalConfig.targetPhiloDBLink.replace(/\/$/, "");
                params = {
                    filename: alignment.target_filename.substr(alignment.target_filename.lastIndexOf("/") + 1),
                    start_byte: alignment.target_start_byte,
                    end_byte: alignment.target_end_byte,
                };
            }
            this.$http
                .get(`${rootURL}/scripts/alignment_to_text.py?`, {
                    params: params,
                })
                .then((response) => {
                    window.open(`${rootURL}${response.data.link}`, "_blank");
                })
                .catch((error) => {
                    alert(error);
                });
        },
    },
};
</script>

<style  lang="scss" scoped>
@import "../assets/theme.module.scss";

.source-passage,
.target-passage {
    color: $passage-color;
}

:deep(.added) {
    color: $added-color;
    font-weight: 700;
}

:deep(.removed) {
    color: $removed-color;
    font-weight: 700;
    text-decoration: line-through;
}

:deep(.token-match) {
    color: darkblue;
    font-weight: 700;
    letter-spacing: -0.5px;
}

:deep(.filtered-token) {
    opacity: 0.25;
}

.separator {
    padding: 5px;
}

.target-passage-container {
    border-right-width: 0 !important;
}
</style>