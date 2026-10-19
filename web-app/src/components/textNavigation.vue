<template>
    <div class="row mt-1" id="all-content">
        <div class="position-relative" v-scroll="handleScroll">
            <button type="button" class="btn btn-secondary btn-sm" id="back-to-top" @click="backToTop()"
                v-scroll="handleScroll">
                <span class="d-sm-inline-block">Back to top</span>
            </button>
        </div>
        <div class="d-flex justify-content-center position-relative" v-if="loading">
            <div class="spinner-border" style="width: 8rem; height: 8rem; position: absolute; z-index: 50; top: 30px"
                role="status">
            </div>
        </div>
        <div class="m-4" style="font-size: 120%" v-if="error">{{ error }}</div>
        <div class="col-12 col-sm-10 offset-sm-1 col-lg-8 offset-lg-2" id="center-content" v-if="textObject && !error">
            <div class="h4 text-title text-center mt-3 mb-2" v-if="textObject.metadata">
                <citations :citation="globalConfig[`${direction}Citation`]" :alignment="textObject.metadata"></citations>
            </div>
            <div class="d-flex justify-content-center mb-3" v-if="bothDirections">
                <div class="btn-group" role="group" aria-label="Direction of reuse">
                    <button type="button" class="btn btn-outline-secondary" :class="{ active: direction == 'target' }"
                        :disabled="!passageCounts.target" @click="changeDirection('target')">
                        Reuses of earlier texts ({{ passageCounts.target }})
                    </button>
                    <button type="button" class="btn btn-outline-secondary" :class="{ active: direction == 'source' }"
                        :disabled="!passageCounts.source" @click="changeDirection('source')">
                        Reused in later texts ({{ passageCounts.source }})
                    </button>
                </div>
            </div>
            <div class="card shadow-sm mb-3" id="reuse-summary" v-if="textObject.text">
                <div class="card-header d-flex justify-content-between align-items-center">
                    <span>{{ summaryHeading }}</span>
                    <button type="button" class="btn btn-sm btn-outline-light py-0" v-if="otherTexts.length"
                        :aria-expanded="String(!summaryHidden)" aria-controls="reuse-summary-list" @click="toggleSummary()">
                        {{ summaryHidden ? "Show" : "Hide" }}
                    </button>
                </div>
                <template v-if="!summaryHidden && otherTexts.length">
                    <ul class="list-group list-group-flush" id="reuse-summary-list">
                        <li class="list-group-item" v-for="other in otherTexts" :key="other.id">
                            <div class="summary-citation" :title="citationText(other.metadata)">
                                <citations :citation="globalConfig[`${otherDirection}Citation`]" :alignment="other.metadata"
                                    :link-to-doc="`${otherDirection}_title`"></citations>
                            </div>
                            <div class="passage-links px-3 pb-1">
                                <button type="button" class="btn btn-sm btn-outline-secondary"
                                    v-for="passage in shownPassages(other)" :key="passage" :title="passageSnippets[passage]"
                                    @click="goToPassage(passage)">{{ passage + 1 }}</button>
                                <button type="button" class="btn btn-sm btn-link" v-if="other.passages.length > passageLimit"
                                    @click="toggleExpanded(other.id)">
                                    {{ expanded[other.id] ? "fewer" : `${other.passages.length - passageLimit} more` }}
                                </button>
                            </div>
                        </li>
                    </ul>
                    <div class="card-footer text-muted small">
                        Numbers are passages in the order of the text: click one to go to it.
                    </div>
                </template>
            </div>
            <div class="card mb-4 pb-4 shadow">
                <div id="book-page" class="text-view px-4">
                    <div id="text-obj-content" class="text-content-area" v-html="textObject.text" tabindex="0"></div>
                </div>
            </div>
        </div>
        <div class="modal modal-xl modal-dialog-scrollable fade" id="passage-pairs" tabindex="-1" v-if="passagePairs"
            aria-labelledby="passage-pairs" aria-hidden="true">
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header">
                        <h1 class="modal-title fs-5">Shared passages
                        </h1>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <ul class="list-group">
                        <li class="list-group-item" v-for="(passage, passageIndex) in passagePairs" :index="passageIndex">
                            <passage-pair :alignment="passage" :index="passageIndex" :diffed="false"></passage-pair>
                        </li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
</template>
<script>
import citations from "./citations.vue";
import passagePair from "./passagePair.vue";
import { Popover, Modal } from "bootstrap";
import vueScrollTo from 'vue-scrollto'

export default {
    name: "textNavigation",
    components: { citations, passagePair },
    inject: ["$http"],
    data() {
        return {
            globalConfig: this.$globalConfig,
            textObject: {},
            bookPagePosition: 0,
            topButtonVisible: false,
            direction: "target",
            popoverList: [],
            passagePairs: [],
            modal: null,
            loading: false,
            error: null,
            passageLimit: 24,
            summaryHidden: localStorage.getItem("textpair.hideReuseSummary") == "true",
            expanded: {},
            passageSnippets: [],
        }
    },
    computed: {
        otherDirection() {
            return this.direction == "source" ? "target" : "source";
        },
        passageCounts() {
            return this.textObject.passage_counts || {};
        },
        // Both sides are the same texts, so this one has passages in either direction
        bothDirections() {
            return "source" in this.passageCounts && "target" in this.passageCounts;
        },
        // The texts sharing passages with this one, most shared first
        otherTexts() {
            const texts = new Map();
            (this.textObject.other_metadata || []).forEach((group, passage) => {
                for (const metadata of group) {
                    const id = metadata[`${this.otherDirection}_doc_id`];
                    if (!texts.has(id)) {
                        texts.set(id, { id, metadata, passages: [] });
                    }
                    texts.get(id).passages.push(passage);
                }
            });
            return [...texts.values()].sort((a, b) => b.passages.length - a.passages.length);
        },
        summaryHeading() {
            const passages = this.plural(this.passageCounts[this.direction] || 0, "passage");
            const texts = this.otherTexts.length;
            if (!this.bothDirections) {
                return `${passages} shared with ${this.plural(texts, "other text")}`;
            }
            if (this.direction == "target") {
                return `${passages} reusing ${this.plural(texts, "earlier text")}`;
            }
            return `${passages} reused in ${this.plural(texts, "later text")}`;
        },
    },
    created() {
        this.fetchText();
    },
    destroyed() {
        this.$off("scroll", this.handleScroll);

        // Remove all event listeners
        for (let popover of this.popoverList) {
            popover.dispose();
        }
        let passages = document.querySelectorAll('[class^="passage-"]')
        for (let passage of passages) {
            passage.removeEventListener("click", this.getAlignments);
        }
        if (this.modal != null) {
            this.modal.hide();
        }
    },
    mounted() {
        let bookPage = document.querySelector("#book-page");
        if (bookPage) {
            this.bookPagePosition = bookPage.getBoundingClientRect().top;
        }
    },
    watch: {
        // call again the method if the route changes
        $route: "fetchText",
    },
    methods: {
        fetchText() {
            this.textObject = {};
            this.loading = true;
            this.error = null;
            this.expanded = {};
            this.passageSnippets = [];
            if (this.modal != null) { // TODO: broken
                this.modal.hide();
            }
            this.passagePairs = [];
            this.$http
                .get(`${this.$globalConfig.apiServer}/text_view/`, {
                    params: this.$route.query,
                })
                .then((response) => {
                    this.loading = false;
                    if (response.data.error) {
                        this.error = response.data.error;
                        return;
                    }
                    this.textObject = response.data;
                    this.direction = this.textObject.direction;
                    this.$nextTick(() => {
                        this.bookPagePosition = document.querySelector("#book-page").getBoundingClientRect().top + window.scrollY;
                        this.passageSnippets = this.textObject.other_metadata.map((_, passage) => {
                            const words = [...document.getElementsByClassName(`passage-${passage}`)].map((el) => el.textContent).join(" ");
                            return words.replace(/\s+/g, " ").trim().slice(0, 100);
                        });
                        if (this.$route.query.start_byte && this.$route.query.end_byte) {
                            this.goToPassage(response.data.passage_number);
                        }
                    });
                    let otherDirection = ""
                    if (this.direction == "source") {
                        otherDirection = "target"
                    } else {
                        otherDirection = "source"
                    }
                    this.$nextTick(() => {
                        let passageMarkers = document.getElementsByClassName("passage-marker");
                        for (let index = 0; passageMarkers.length > index; index += 1) {
                            let metadataGroup = this.textObject.other_metadata[index];
                            let citation = this.buildCitation(metadataGroup, otherDirection)
                            let passages = document.getElementsByClassName(`passage-${index}`);
                            for (let el of passages) {
                                let popover = new Popover(el, {
                                    content: citation,
                                    sanitize: false,
                                    html: true,
                                    trigger: "hover focus",
                                    placement: "top",
                                })
                                this.popoverList.push(popover)
                                el.addEventListener("click", this.getAlignments, false);
                            }
                        }
                    })
                })
                .catch((error) => {
                    alert(error);
                });
        },
        backToTop() {
            window.scrollTo({
                top: 0,
                behavior: "smooth",
            });
        },
        handleScroll() {
            if (!this.topButtonVisible) {
                if (window.scrollY > this.bookPagePosition) {
                    this.topButtonVisible = true;
                    let backToTop = document.getElementById("back-to-top");
                    backToTop.classList.add("visible");
                    backToTop.style.top = 0;
                    backToTop.style.position = "fixed";
                }
            } else if (window.scrollY < this.bookPagePosition) {
                this.topButtonVisible = false;
                let backToTop = document.getElementById("back-to-top");
                backToTop.classList.remove("visible");
                backToTop.style.position = "";
                backToTop.style.top = "";
            }
        },
        changeDirection(direction) {
            this.direction = direction;
            this.$router.push(`/text-view/?db_table=${this.$route.query.db_table}&doc_id=${this.textObject.doc_id}&directionSelected=${direction}`)
        },
        plural(count, noun) {
            return `${count} ${noun}${count == 1 ? "" : "s"}`;
        },
        shownPassages(other) {
            return this.expanded[other.id] ? other.passages : other.passages.slice(0, this.passageLimit);
        },
        toggleSummary() {
            this.summaryHidden = !this.summaryHidden;
            localStorage.setItem("textpair.hideReuseSummary", this.summaryHidden);
        },
        citationText(metadata) {
            return this.globalConfig[`${this.otherDirection}Citation`]
                .map((cite) => metadata[cite.field])
                .filter((value) => value)
                .join(" • ");
        },
        toggleExpanded(id) {
            this.expanded = { ...this.expanded, [id]: !this.expanded[id] };
        },
        goToPassage(passage) {
            let marker = document.querySelector(`.passage-marker[n="${passage}"]`);
            vueScrollTo.scrollTo(marker, 250, {
                easing: "ease-out",
                offset: -150,
                onDone: () => this.focusPassage(passage),
            });
        },
        focusPassage(passage) {
            for (let el of document.querySelectorAll("#book-page .focus")) {
                el.classList.remove("focus");
            }
            for (let el of document.getElementsByClassName(`passage-${passage}`)) {
                el.classList.add("focus");
            }
        },
        buildCitation(metadataFields, direction) {
            let citation = '<ul class="list-group" style="list-style-type: none; margin: 0; padding: 0;">'
            for (let metadata of metadataFields) {
                citation += '<li class="list-group-item">';
                let labels = []
                for (let cite of this.globalConfig[`${direction}Citation`]) {
                    let style = Object.entries(cite.style).map(([k, v]) => `${k}:${v}`).join(';')
                    if (metadata[cite.field] != "" && metadata[cite.field] != null) {
                        labels.push(`<span style="${style}">${metadata[cite.field]}</span>`)
                    }
                }
                citation += labels.join('<span class="separator">&nbsp;&#9679;&nbsp;</span>') + "</li>";
            }
            citation += '</ul><div class="p-2"><b>Click on passage to see reuse.</b></div>'
            return citation;
        },
        getAlignments(event) {
            let passageNumber = event.target.getAttribute("n")
            let passageMarker = document.querySelector(`.passage-marker[n="${passageNumber}"]`)
            let offsets = passageMarker.getAttribute("data-offsets").split("-")
            let startByte = offsets[0]
            let endByte = offsets[1]
            this.$http.get(`${this.$globalConfig.apiServer}/get_passages/`, {
                params: {
                    start_byte: startByte,
                    end_byte: endByte,
                    db_table: this.$route.query.db_table,
                    directionSelected: this.$route.query.directionSelected,
                    filename: this.textObject.metadata[`${this.$route.query.directionSelected}_filename`]
                }
            }).then((response) => {
                this.passagePairs = response.data.passages
                this.modal = new Modal(document.getElementById('passage-pairs'))
                this.modal.show()

            }).catch((error) => {
                alert(error);
            });
        }
    },
}

</script>
<style lang="scss" scoped>
@import "../assets/theme.module.scss";

#reuse-summary .list-group-item {
    padding: 0;
}

/* Long lists scroll within the card, rather than push the text down */
#reuse-summary-list {
    max-height: 16rem;
    overflow-y: auto;
}

.summary-citation :deep(p) {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.passage-links .btn-outline-secondary {
    min-width: 2.25rem;
    margin: 0 0.25rem 0.375rem 0;
    padding: 0 0.375rem;
}

.separator {
    padding: 5px;
    font-size: 60%;
    display: inline-block;
    vertical-align: middle;
}


#back-to-top {
    position: absolute;
    left: 0;
    opacity: 0;
    transition: opacity 0.25s;
    pointer-events: none;
}

#report-error {
    position: absolute;
    right: 0;
    opacity: 0;
    transition: opacity 0.25s;
    pointer-events: none;
}

#back-to-top.visible,
#report-error.visible {
    opacity: 0.95;
    pointer-events: all;
}

#nav-buttons {
    position: absolute;
    opacity: 0.9;
    width: 100%;
}

#toc-nav-bar {
    background-color: #ddd;
    opacity: 0.95;
    backdrop-filter: blur(5px) contrast(0.8);
}

a.current-obj,
#toc-container a:hover {
    background: #e8e8e8;
}

#book-page {
    text-align: justify;
    white-space: v-bind("whiteSpace");
}

:deep(.xml-pb) {
    display: block;
    text-align: center;
    margin: 10px;
}

:deep(.xml-pb::before) {
    content: "-" attr(n) "-";
    white-space: pre;
}

:deep(p) {
    margin-bottom: 0.5rem;
}

:deep(.highlight) {
    background-color: red;
    color: #fff;
}

:deep(.xml-div1::after),
/* clear floats from inline images */
:deep(.xml-div2::after),
:deep(.xml-div3::after) {
    content: "";
    display: block;
    clear: right;
}

/* Styling for theater */

:deep(.xml-castitem::after) {
    content: "\A";
    white-space: pre;
}

:deep(.xml-castlist > .xml-castitem:first-of-type::before) {
    content: "\A";
    white-space: pre;
}

:deep(.xml-castgroup::before) {
    content: "\A";
    white-space: pre;
}

:deep(b.headword) {
    font-weight: 700 !important;
    font-size: 130%;
    font-variant: small-caps;
    display: block;
    margin-top: 20px;
}

:deep(b.headword::before) {
    content: "\A";
    white-space: pre;
}

:deep(#bibliographic-results b.headword) {
    font-weight: 400 !important;
    font-size: 100%;
    display: inline;
}

:deep(.xml-lb),
:deep(.xml-l) {
    text-align: justify;
    display: block;
}

:deep(.xml-sp .xml-lb:first-of-type) {
    content: "";
    white-space: normal;
}

:deep(.xml-lb[type="hyphenInWord"]) {
    display: inline;
}

#book-page .xml-sp {
    display: block;
}

:deep(.xml-sp::before) {
    content: "\A";
    white-space: pre;
}

:deep(.xml-stage + .xml-sp:nth-of-type(n + 2)::before) {
    content: "";
}

:deep(.xml-fw, .xml-join) {
    display: none;
}

:deep(.xml-speaker + .xml-stage::before) {
    content: "";
    white-space: normal;
}

:deep(.xml-stage) {
    font-style: italic;
}

:deep(.xml-stage::after) {
    content: "\A";
    white-space: pre;
}

:deep(div1 div2::before) {
    content: "\A";
    white-space: pre;
}

:deep(.xml-speaker) {
    font-weight: 700;
}

:deep(.xml-pb) {
    display: block;
    text-align: center;
    margin: 10px;
}

:deep(.xml-pb::before) {
    content: "-" attr(n) "-";
    white-space: pre;
}

:deep(.xml-lg) {
    display: block;
}

:deep(.xml-lg::after) {
    content: "\A";
    white-space: pre;
}

:deep(.xml-lg:first-of-type::before) {
    content: "\A";
    white-space: pre;
}

:deep(.xml-castList) :deep(.xml-front),
:deep(.xml-castItem),
:deep(.xml-docTitle),
:deep(.xml-docImprint),
:deep(.xml-performance),
:deep(.xml-docAuthor),
:deep(.xml-docDate),
:deep(.xml-premiere),
:deep(.xml-casting),
:deep(.xml-recette),
:deep(.xml-nombre) {
    display: block;
}

:deep(.xml-docTitle) {
    font-style: italic;
    font-weight: bold;
}

:deep(.xml-docAuthor),
:deep(.xml-docTitle),
:deep(.xml-docDate) {
    text-align: center;
}

:deep(.xml-docTitle span[type="main"]) {
    font-size: 150%;
    display: block;
}

:deep(.xml-docTitle span[type="sub"]) {
    font-size: 120%;
    display: block;
}

:deep(.xml-performance),
:deep(.xml-docImprint) {
    margin-top: 10px;
}

:deep(.xml-set) {
    display: block;
    font-style: italic;
    margin-top: 10px;
}

/*Dictionary formatting*/

body {
    counter-reset: section;
    /* Set the section counter to 0 */
}

:deep(.xml-prononciation::before) {
    content: "(";
}

:deep(.xml-prononciation::after) {
    content: ")\A";
}

:deep(.xml-nature) {
    font-style: italic;
}

:deep(.xml-indent),
:deep(.xml-variante) {
    display: block;
}

:deep(.xml-variante) {
    padding-top: 10px;
    padding-bottom: 10px;
    text-indent: -1.3em;
    padding-left: 1.3em;
}

:deep(.xml-variante::before) {
    counter-increment: section;
    content: counter(section) ")\00a0";
    font-weight: 700;
}

:deep(:not(.xml-rubrique) + .xml-indent) {
    padding-top: 10px;
}

:deep(.xml-indent) {
    padding-left: 1.3em;
}

:deep(.xml-cit) {
    padding-left: 2.3em;
    display: block;
    text-indent: -1.3em;
}

:deep(.xml-indent > .xml-cit) {
    padding-left: 1em;
}

:deep(.xml-cit::before) {
    content: "\2012\00a0\00ab\00a0";
}

:deep(.xml-cit::after) {
    content: "\00a0\00bb\00a0(" attr(aut) "\00a0" attr(ref) ")";
    font-variant: small-caps;
}

:deep(.xml-rubrique) {
    display: block;
    margin-top: 20px;
}

:deep(.xml-rubrique::before) {
    content: attr(nom);
    font-variant: small-caps;
    font-weight: 700;
}

:deep(.xml-corps + .xml-rubrique) {
    margin-top: 10px;
}

/*Methodique styling*/

:deep(div[type="article"] .headword) {
    display: inline-block;
    margin-bottom: 10px;
}

:deep(.headword + p) {
    display: inline;
}

:deep(.headword + p + p) {
    margin-top: 10px;
}

/*Note handling*/
:deep(.popover-content .xml-p:not(:first-of-type)) {
    display: block;
    margin-top: 1em;
    margin-bottom: 1em;
}

:deep(.note-content) {
    display: none;
}

:deep(.note),
:deep(.note-ref) {
    vertical-align: 0.3em;
    font-size: 0.7em;
    background-color: $button-color;
    color: #fff !important;
    padding: 0 0.2rem;
    border-radius: 50%;
}

:deep(.note:hover),
:deep(.note-ref:hover) {
    cursor: pointer;
    text-decoration: none;
}

:deep(div[type="notes"] .xml-note) {
    margin: 15px 0px;
    display: block;
}

:deep(.xml-note::before) {
    content: "note\00a0" attr(n) "\00a0:\00a0";
    font-weight: 700;
}

/*Page images*/

:deep(.xml-pb-image) {
    display: block;
    text-align: center;
    margin: 10px;
}

:deep(.page-image-link) {
    margin-top: 10px;
    /*display: block;*/
    text-align: center;
}

/*Inline images*/
:deep(.inline-img) {
    max-width: 40%;
    float: right;
    height: auto;
    padding-left: 15px;
    padding-top: 15px;
}

:deep(.inline-img:hover) {
    cursor: pointer;
}

:deep(.link-back) {
    margin-left: 10px;
    line-height: initial;
}

:deep(.xml-add) {
    color: #ef4500;
}

:deep(.xml-seg) {
    display: block;
}

/*Table display*/

:deep(b.headword[rend="center"]) {
    margin-bottom: 30px;
    text-align: center;
}

:deep(.xml-table) {
    display: table;
    position: relative;
    text-align: center;
    border-collapse: collapse;
}

:deep(.xml-table .xml-pb-image) {
    position: absolute;
    width: 100%;
    margin-top: 15px;
}

:deep(.xml-row) {
    display: table-row;
    font-weight: 700;
    text-align: left;
    min-height: 50px;
    font-variant: small-caps;
    padding-top: 10px;
    padding-bottom: 10px;
    padding-right: 20px;
    border-bottom: #ddd 1px solid;
}

:deep(.xml-row ~ .xml-row) {
    font-weight: inherit;
    text-align: justify;
    font-variant: inherit;
}

:deep(.xml-pb-image + .xml-row) {
    padding-top: 50px;
    padding-bottom: 10px;
    border-top-width: 0px;
}

:deep(.xml-cell) {
    display: table-cell;
    padding-top: inherit;
    /*inherit padding when image is above */
    padding-bottom: inherit;
}

:deep(s) {
    text-decoration: none;
}

:deep(h5 .text-view) {
    font-size: inherit !important;
}

.slide-fade-enter-active {
    transition: all 0.3s ease-out;
}

.slide-fade-leave-active {
    transition: all 0.3s ease-out;
}

.slide-fade-enter-from,
.slide-fade-leave-to {
    transform: translateY(-30px);
    opacity: 0;
}

/* Image button styling */
.img-buttons {
    font-size: 45px !important;
    color: #fff !important;
}

:deep(#full-size-image) {
    right: 90px;
    font-weight: 700 !important;
    font-size: 20px !important;
    left: auto;
    margin: -15px;
    text-decoration: none;
    cursor: pointer;
    position: absolute;
    top: 28px;
    color: #fff;
    opacity: 0.8;
    border: 3px solid;
    padding: 0 0.25rem;
}

#full-size-image:hover {
    opacity: 1;
}

:deep([class^="passage-"]) {
    color: $passage-color;
    font-weight: 700;
    cursor: pointer;
}

:deep([class^="passage-"].focus) {
    background-color: $passage-color;
    color: #fff;
}

:deep(.xml-titlePage) {
    display: none;
}

:deep(.link-back) {
    display: none;
}

:deep(.xml-ref) {
    vertical-align: 0.3em;
    font-size: 0.7em;
    padding: 0 0.2rem;
    border-radius: 50%;
}

.popover-body {
    padding: 0 !important;
}
</style>