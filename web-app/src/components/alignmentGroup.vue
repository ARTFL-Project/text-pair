<template>
    <div id="alignment-group" class="mt-4">
        <h5 style="text-align: center">
            <i>"{{ sourcePassage.source_passage }}"</i>
            <div class="reuse-title mt-1">
                <citations :citation="globalConfig.sourceCitation" :alignment="sourcePassage"></citations>
            </div>
        </h5>
        <div id="timeline-container" class="px-2 pb-2">
            <div id="vertical-line"></div>
            <div class="timeline-dates" v-for="(date, index) in timeline" :key="index">
                <div class="year btn btn-secondary">{{ date.year }}</div>
                <div class="timeline-events card shadow-1 px-2 pt-2 mt-3" v-for="(reuse, reuseIndex) in date.result"
                    :key="reuseIndex">
                    <h5 class="reuse-title" @click="showPassage">
                        <citations :citation="globalConfig[`${reuse.direction}Citation`]" :alignment="reuse"></citations>
                    </h5>
                    <p class="timeline-text-content m-0">
                        <span class="text-content">
                            {{ reuse[`${reuse.direction}_context_before`] }}
                            <span class="highlight">{{ reuse[`${reuse.direction}_passage`] }}</span> {{
                                reuse[`${reuse.direction}_context_after`] }}
                        </span>
                        <span class="text-muted text-center d-block mt-1">
                            <button class="group-diff-btn" @click="showDifferences(reuse)">Show differences</button>
                        </span>
                    </p>
                </div>
            </div>
        </div>
        <div id="passage-diff" class="modal fade" tabindex="-1">
            <div class="modal-dialog" style="min-width: 1024px">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">Passage Pair</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <passage-pair v-if="showPair" :alignment="localAlignment" :diffed="true" :index="0"
                            :startByteIndex="startByteIndex" :endByteIndex="this.endByteIndex" :key="render"></passage-pair>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
import passagePair from "./passagePair.vue";
import citations from "./citations.vue";
import { Modal } from "bootstrap";

export default {
    name: "alignmentGroup",
    components: { passagePair, citations },
    inject: ["$http"],
    data() {
        return {
            loading: true,
            globalConfig: this.$globalConfig,
            done: false,
            timeline: {},
            sourcePassage: {},
            passages: [],
            showPair: false,
            localAlignment: {},
            modal: null,
            startByteIndex: 0,
            endByteIndex: 0,
            render: 0, // used to force re-rendering of passagePair
        };
    },
    created() {
        this.fetchData();
    },
    mounted() {
        this.modal = new Modal(document.getElementById("passage-diff"), {
            keyboard: false,
            backdrop: "static",
        });
    },
    watch: {
        $route: "fetchData",
    },
    methods: {
        fetchData() {
            this.results = {};
            let params = { db_table: this.$globalConfig.databaseName };
            this.loading = true;
            this.$http
                .get(`${this.$globalConfig.apiServer}/group/${this.$route.params.groupId}/?${this.paramsToUrl(params)}`, {
                    metadata: this.$globalConfig.metadataTypes,
                })
                .then((response) => {
                    this.loading = false;
                    this.done = true;
                    this.timeline = response.data.passageList
                    this.sourcePassage = response.data.original_passage
                })
                .catch((error) => {
                    this.loading = false;
                    this.error = error.toString();
                    console.log(error);
                });
        },
        formatTitle(title) {
            if (title.length > 300) {
                let titleSplit = title.slice(0, 300).split(' ');
                title = titleSplit.slice(0, titleSplit.length - 1).join(" ") + " [...]";
            }
            return title;
        },
        showPassage(event) {
            let textPassage = event.srcElement.closest(".timeline-events").querySelector(".timeline-text-content")
            if (!textPassage.classList.contains("show")) {
                textPassage.classList.add("show")
            } else {
                textPassage.classList.remove("show")
            }

        },
        showDifferences(reuse) {
            this.showPair = false;
            this.localAlignment = { ...this.sourcePassage }
            for (let key in reuse) {
                let newKey = key.replace(/source_/, "target_")
                if (newKey.startsWith("target_")) {
                    this.localAlignment[newKey] = reuse[key]
                }
            }
            this.localAlignment.count = 1;
            reuse.source_passage = reuse.source_passage.replace(/\s\s+/g, ' ')
            this.sourcePassage.source_passage = this.sourcePassage.source_passage.replace(/\s\s+/g, ' ')
            this.startByteIndex = this.sourcePassage.source_passage.indexOf(reuse.source_passage)
            let remove = reuse.source_passage.length
            while (this.startByteIndex == -1) { // if the passage is not found, remove one character from the end and try again
                this.startByteIndex = this.sourcePassage.source_passage.indexOf(reuse.source_passage.slice(0, remove))
                remove -= 1
            }
            this.endByteIndex = this.startByteIndex + reuse.source_passage.length
            this.render += 1;
            this.showPair = true;
            this.modal.show()
        },
    },
};
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

.reuse-title {
    cursor: pointer;
}

.timeline-text-content {
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