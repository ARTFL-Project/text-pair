<template>
    <div class="row mt-1" id="all-content">
        <div class="col-12 col-sm-10 offset-sm-1 col-lg-8 offset-lg-2" id="center-content">
            <h2 class="h4 text-title text-center mt-3 mb-2" :class="{ 'visually-hidden': !textObject.metadata }">
                <span class="visually-hidden">Full text<template v-if="textObject.metadata">: </template></span>
                <citations :citation="globalConfig[`${direction}Citation`]" :alignment="textObject.metadata" tag="span"
                    v-if="textObject.metadata"></citations>
            </h2>
            <div class="d-flex justify-content-center position-relative" v-if="loading">
                <div class="spinner-border" style="width: 8rem; height: 8rem; position: absolute; z-index: 50; top: 30px"
                    role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            </div>
            <div class="m-4" style="font-size: 120%" v-if="error">{{ error }}</div>
            <template v-else>
                <div class="d-flex justify-content-center mb-3" v-if="bothDirections">
                    <div class="btn-group" role="group" aria-label="Direction of reuse">
                        <button type="button" class="btn btn-outline-secondary" id="direction-target"
                            :class="{ active: direction == 'target' }" :aria-pressed="String(direction == 'target')"
                            :disabled="!passageCounts.target" @click="changeDirection('target')">
                            Reuses of earlier texts ({{ passageCounts.target }})
                        </button>
                        <button type="button" class="btn btn-outline-secondary" id="direction-source"
                            :class="{ active: direction == 'source' }" :aria-pressed="String(direction == 'source')"
                            :disabled="!passageCounts.source" @click="changeDirection('source')">
                            Reused in later texts ({{ passageCounts.source }})
                        </button>
                    </div>
                </div>
                <div class="card shadow-sm mb-3" id="reuse-summary" v-if="textObject.text">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h3 class="fs-6 fw-normal lh-base mb-0">{{ summaryHeading }}</h3>
                        <button type="button" class="btn btn-sm btn-outline-light py-0" v-if="otherTexts.length"
                            :aria-expanded="String(!summaryHidden)" aria-controls="reuse-summary-list"
                            @click="toggleSummary()">
                            {{ summaryHidden ? "Show" : "Hide" }}<span class="visually-hidden"> texts sharing
                                passages</span>
                        </button>
                    </div>
                    <template v-if="!summaryHidden && otherTexts.length">
                        <ul class="list-group list-group-flush" id="reuse-summary-list">
                            <li class="list-group-item" v-for="other in otherTexts" :key="other.id">
                                <div class="summary-citation" :title="citationText(other.metadata)">
                                    <citations :citation="globalConfig[`${otherDirection}Citation`]"
                                        :alignment="other.metadata" :link-to-doc="`${otherDirection}_title`"></citations>
                                </div>
                                <div class="passage-links px-3 pb-1">
                                    <button type="button" class="btn btn-sm btn-outline-secondary"
                                        v-for="passage in shownPassages(other)" :key="passage"
                                        :title="passageSnippets[passage]" @click="goToPassage(passage)">
                                        <span class="visually-hidden">Passage </span>{{ passage + 1 }}
                                    </button>
                                    <button type="button" class="btn btn-sm btn-link"
                                        v-if="other.passages.length > passageLimit"
                                        :aria-expanded="String(!!expanded[other.id])" @click="toggleExpanded(other.id)">
                                        {{ expanded[other.id] ? "fewer" : `${other.passages.length - passageLimit} more`
                                        }}<span class="visually-hidden"> passages</span>
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
                    <div id="book-page" class="text-view px-4" ref="bookPage" role="region" aria-label="Document text">
                        <div id="text-obj-content" class="text-content-area" ref="textContent" v-html="textObject.text"
                            :lang="globalConfig[`${direction}Language`] || undefined" @click="textClick"
                            @keydown="textKeydown" @mouseover="textMouseover" @mouseout="textMouseout"
                            @focusin="textFocusin" @focusout="textFocusout"></div>
                    </div>
                </div>
            </template>
        </div>
        <div>
            <button type="button" class="btn btn-secondary btn-sm" id="back-to-top" :class="{ visible: topButtonVisible }"
                aria-label="Back to top" @click="backToTop()">
                <span aria-hidden="true">&uarr;</span><span class="d-none d-sm-inline ms-1">Back to top</span>
            </button>
        </div>
        <div class="modal modal-xl modal-dialog-scrollable fade" id="passage-pairs" tabindex="-1" ref="modalElement"
            aria-labelledby="passage-pairs-title" aria-hidden="true">
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header">
                        <h2 class="modal-title fs-5" id="passage-pairs-title">Shared passages</h2>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <ul class="list-group">
                        <li class="list-group-item" v-for="(passage, passageIndex) in passagePairs" :key="passageIndex">
                            <passage-pair :alignment="passage" :index="passageIndex" :diffed="false"
                                :heading-level="3"></passage-pair>
                        </li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
</template>
<script setup>
import axios from "axios";
import { Modal, Popover } from "bootstrap";
import emitter from "tiny-emitter/instance";
import vueScrollTo from "vue-scrollto";
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import globalConfig from "../config.js";
import { prefersReducedMotion } from "../utils.js";
import citations from "./citations.vue";
import passagePair from "./passagePair.vue";

const route = useRoute();
const router = useRouter();

// A reused passage in the text: its number button, or any of its fragments
const PASSAGE = '.reuse-number, [class^="passage-"]:not(.passage-marker)';
const pageLanguage = document.documentElement.lang || "en";
const passageLimit = 24;

const textObject = ref({});
const topButtonVisible = ref(false);
const direction = ref("target");
const passagePairs = ref([]);
const loading = ref(false);
const error = ref(null);
const summaryHidden = ref(localStorage.getItem("textpair.hideReuseSummary") == "true");
const expanded = ref({});
const passageSnippets = ref([]);
const bookPage = ref(null);
const textContent = ref(null);
const modalElement = ref(null);

let bookPagePosition = 0;
let modal = null;
let modalOpener = null;
let popover = null;
let popoverTrigger = null;
let hideTimer = null;
let quietFocus = false;
let focusAfterLoad = null;
// Responses to earlier requests, or after leaving the view, are dropped
let textRequest = 0;

const otherDirection = computed(() => (direction.value == "source" ? "target" : "source"));
const passageCounts = computed(() => textObject.value.passage_counts || {});
// Both sides are the same texts, so this one has passages in either direction
const bothDirections = computed(() => "source" in passageCounts.value && "target" in passageCounts.value);
// The texts sharing passages with this one, most shared first
const otherTexts = computed(() => {
    const texts = new Map();
    (textObject.value.other_metadata || []).forEach((group, passage) => {
        for (const metadata of group) {
            const id = metadata[`${otherDirection.value}_doc_id`];
            if (!texts.has(id)) {
                texts.set(id, { id, metadata, passages: [] });
            }
            texts.get(id).passages.push(passage);
        }
    });
    return [...texts.values()].sort((a, b) => b.passages.length - a.passages.length);
});
const summaryHeading = computed(() => {
    const passages = plural(passageCounts.value[direction.value] || 0, "passage");
    const texts = otherTexts.value.length;
    if (!bothDirections.value) {
        return `${passages} shared with ${plural(texts, "other text")}`;
    }
    if (direction.value == "target") {
        return `${passages} reusing ${plural(texts, "earlier text")}`;
    }
    return `${passages} reused in ${plural(texts, "later text")}`;
});

fetchText();

watch(
    () => route.fullPath,
    () => {
        if (route.name == "textNavigation") {
            fetchText(true);
        }
    }
);

onMounted(() => {
    if (bookPage.value) {
        bookPagePosition = bookPage.value.getBoundingClientRect().top;
    }
    window.addEventListener("scroll", handleScroll, { passive: true });
    document.addEventListener("keydown", documentKeydown);
    modalElement.value.addEventListener("hidden.bs.modal", restoreFocus);
});

onBeforeUnmount(() => {
    textRequest += 1;
    window.removeEventListener("scroll", handleScroll);
    document.removeEventListener("keydown", documentKeydown);
    hidePopover();
    if (modal != null) {
        modal.hide();
    }
});

function fetchText(sameView = false) {
    textObject.value = {};
    loading.value = true;
    error.value = null;
    expanded.value = {};
    passageSnippets.value = [];
    hidePopover();
    if (modal != null) {
        modal.hide();
    }
    passagePairs.value = [];
    const request = ++textRequest;
    axios
        .get(`${globalConfig.apiServer}/text_view/`, {
            params: route.query,
        })
        .then((response) => {
            if (request != textRequest) return;
            loading.value = false;
            if (response.data.error) {
                showError(response.data.error, sameView);
                return;
            }
            textObject.value = response.data;
            direction.value = textObject.value.direction;
            nextTick(() => {
                bookPagePosition = bookPage.value.getBoundingClientRect().top + window.scrollY;
                prepareText();
                passageSnippets.value = textObject.value.other_metadata.map((_, passage) => {
                    const words = [...textContent.value.getElementsByClassName(`passage-${passage}`)]
                        .map((el) => el.textContent)
                        .join(" ");
                    return words.replace(/\s+/g, " ").trim().slice(0, 100);
                });
                emitter.emit("announce", `Text loaded: ${summaryHeading.value}`);
                if (route.query.start_byte && route.query.end_byte) {
                    goToPassage(response.data.passage_number);
                } else if (focusAfterLoad) {
                    focusElement(document.getElementById(focusAfterLoad));
                } else if (sameView && focusLost()) {
                    emitter.emit("focusMain");
                }
                focusAfterLoad = null;
            });
        })
        .catch((err) => {
            if (request != textRequest) return;
            loading.value = false;
            showError(`The text could not be loaded (${err.message}).`, sameView);
        });
}

function showError(message, sameView) {
    error.value = message;
    emitter.emit("announce", message);
    if (sameView && focusLost()) {
        emitter.emit("focusMain");
    }
    focusAfterLoad = null;
}

// Focus went with the content it was on, or is in the closing modal
function focusLost() {
    const el = document.activeElement;
    return !el || el == document.body || modalElement.value.contains(el);
}

// Accessible structure for the HTML PhiloLogic returns, and a button opening each reused passage
function prepareText() {
    const container = textContent.value;
    // Headings go under the view's h2 by division number (div1, div2...), as divisions may be left unclosed
    const headings = [];
    for (const heading of container.querySelectorAll('[role="heading"]')) {
        if (heading.textContent.trim()) {
            const division = heading.parentElement.closest('[class*="xml-div"]');
            const depth = division?.className.match(/\bxml-div(\d)\b/);
            headings.push([heading, depth ? Number(depth[1]) : null]);
        } else {
            heading.removeAttribute("role");
        }
    }
    const top = headings.reduce((min, [, depth]) => (depth === null ? min : Math.min(min, depth)), Infinity);
    for (const [heading, depth] of headings) {
        const level = depth === null ? 3 : 3 + depth - top;
        heading.setAttribute("aria-level", String(Math.min(level, 6)));
    }
    for (const image of container.querySelectorAll("img:not([alt])")) {
        image.setAttribute("alt", "illustrative image");
    }
    // Inline notes open below their marker; references to notes elsewhere lead nowhere here
    for (const note of container.querySelectorAll("a.note")) {
        note.setAttribute("role", "button");
        note.setAttribute("aria-expanded", "false");
    }
    for (const reference of container.querySelectorAll(".note-ref")) {
        reference.removeAttribute("tabindex");
    }
    for (const marker of container.getElementsByClassName("passage-marker")) {
        const passage = Number(marker.getAttribute("n"));
        const button = document.createElement("button");
        button.type = "button";
        button.className = "reuse-number";
        button.dataset.passage = passage;
        button.lang = pageLanguage;
        button.innerHTML = `<span class="visually-hidden">Reused passage </span>${passage + 1}`;
        marker.append(button);
        const end = document.getElementById(`end-passage-${passage}`);
        if (end) {
            end.lang = pageLanguage;
            end.innerHTML = `<span class="visually-hidden">End of reused passage ${passage + 1}</span>`;
        }
    }
}

function passageElement(target) {
    return target instanceof Element ? target.closest(PASSAGE) : null;
}

function passageNumber(el) {
    return Number(el.dataset.passage ?? el.getAttribute("n"));
}

function textClick(event) {
    const note = event.target.closest("a.note");
    if (note) {
        toggleNote(note);
        return;
    }
    const el = passageElement(event.target);
    if (el) {
        getAlignments(passageNumber(el));
    }
}

function textKeydown(event) {
    const note = event.target.closest("a.note");
    if (note && (event.key == "Enter" || event.key == " ")) {
        event.preventDefault();
        toggleNote(note);
    }
}

function toggleNote(note) {
    const open = note.getAttribute("aria-expanded") != "true";
    note.setAttribute("aria-expanded", String(open));
    note.nextElementSibling?.classList.toggle("open", open);
}

function textMouseover(event) {
    const el = passageElement(event.target);
    if (el) {
        showPopover(el);
    }
}

function textMouseout(event) {
    const el = passageElement(event.target);
    if (el && el == popoverTrigger && !el.contains(event.relatedTarget)) {
        scheduleHide();
    }
}

function textFocusin(event) {
    if (event.target.classList.contains("reuse-number") && !quietFocus) {
        showPopover(event.target);
    }
}

function textFocusout(event) {
    if (event.target == popoverTrigger) {
        hidePopover();
    }
}

function documentKeydown(event) {
    if (event.key == "Escape" && popover) {
        hidePopover();
    }
}

// Popovers stay while the pointer is over them or their passage, or while its button has focus
function showPopover(el) {
    clearTimeout(hideTimer);
    if (el == popoverTrigger) return;
    hidePopover();
    const passage = passageNumber(el);
    popoverTrigger = el;
    popover = new Popover(el, {
        content: buildCitation(textObject.value.other_metadata[passage], otherDirection.value),
        sanitize: false,
        html: true,
        trigger: "manual",
        placement: "top",
        customClass: "passage-popover",
    });
    popover.show();
    if (popover.tip) {
        popover.tip.addEventListener("mouseenter", () => clearTimeout(hideTimer));
        popover.tip.addEventListener("mouseleave", scheduleHide);
    }
}

function scheduleHide() {
    clearTimeout(hideTimer);
    hideTimer = setTimeout(() => {
        if (popoverTrigger && popoverTrigger != document.activeElement) {
            hidePopover();
        }
    }, 300);
}

function hidePopover() {
    clearTimeout(hideTimer);
    if (popover) {
        popover.dispose();
        popoverTrigger.removeAttribute("aria-describedby");
        popover = null;
        popoverTrigger = null;
    }
}

function focusElement(el) {
    if (el) {
        quietFocus = true;
        el.focus({ preventScroll: true });
        quietFocus = false;
    } else {
        emitter.emit("focusMain");
    }
}

function backToTop() {
    window.scrollTo({
        top: 0,
        behavior: prefersReducedMotion() ? "auto" : "smooth",
    });
    emitter.emit("focusMain");
}

function handleScroll() {
    topButtonVisible.value = window.scrollY > bookPagePosition;
}

function changeDirection(newDirection) {
    if (newDirection == direction.value) return;
    direction.value = newDirection;
    focusAfterLoad = `direction-${newDirection}`;
    router.push(`/text-view/?db_table=${route.query.db_table}&doc_id=${textObject.value.doc_id}&directionSelected=${newDirection}`);
}

function plural(count, noun) {
    return `${count} ${noun}${count == 1 ? "" : "s"}`;
}

function shownPassages(other) {
    return expanded.value[other.id] ? other.passages : other.passages.slice(0, passageLimit);
}

function toggleSummary() {
    summaryHidden.value = !summaryHidden.value;
    localStorage.setItem("textpair.hideReuseSummary", summaryHidden.value);
}

function citationText(metadata) {
    return globalConfig[`${otherDirection.value}Citation`]
        .map((cite) => metadata[cite.field])
        .filter((value) => value)
        .join(" • ");
}

function toggleExpanded(id) {
    expanded.value = { ...expanded.value, [id]: !expanded.value[id] };
}

function goToPassage(passage) {
    const marker = textContent.value.querySelector(`.passage-marker[n="${passage}"]`);
    vueScrollTo.scrollTo(marker, prefersReducedMotion() ? 0 : 250, {
        easing: "ease-out",
        offset: -150,
        cancelable: false,
        onDone: () => focusPassage(passage),
    });
}

// Highlights the passage and moves focus to its button
function focusPassage(passage) {
    for (const el of textContent.value.querySelectorAll(".focus")) {
        el.classList.remove("focus");
    }
    for (const el of textContent.value.getElementsByClassName(`passage-${passage}`)) {
        el.classList.add("focus");
    }
    focusElement(textContent.value.querySelector(`.reuse-number[data-passage="${passage}"]`));
}

function buildCitation(metadataFields, side) {
    const language = globalConfig[`${side}Language`];
    let citation = `<ul class="list-group"${language ? ` lang="${language}"` : ""} style="list-style-type: none; margin: 0; padding: 0;">`;
    for (const metadata of metadataFields) {
        citation += '<li class="list-group-item">';
        const labels = [];
        for (const cite of globalConfig[`${side}Citation`]) {
            const style = Object.entries(cite.style).map(([k, v]) => `${k}:${v}`).join(";");
            if (metadata[cite.field] != "" && metadata[cite.field] != null) {
                labels.push(`<span style="${style}">${metadata[cite.field]}</span>`);
            }
        }
        citation += labels.join('<span class="separator" aria-hidden="true">&nbsp;&#9679;&nbsp;</span>') + "</li>";
    }
    citation += `</ul><div class="p-2" lang="${pageLanguage}"><b>Click on passage to see reuse.</b></div>`;
    return citation;
}

function getAlignments(passage) {
    hidePopover();
    const marker = textContent.value.querySelector(`.passage-marker[n="${passage}"]`);
    const [startByte, endByte] = marker.getAttribute("data-offsets").split("-");
    modalOpener = marker.querySelector(".reuse-number");
    axios
        .get(`${globalConfig.apiServer}/get_passages/`, {
            params: {
                start_byte: startByte,
                end_byte: endByte,
                db_table: route.query.db_table,
                directionSelected: route.query.directionSelected,
                filename: textObject.value.metadata[`${route.query.directionSelected}_filename`],
            },
        })
        .then((response) => {
            if (!modalElement.value) return;
            passagePairs.value = response.data.passages;
            emitter.emit("announce", `Showing ${plural(passagePairs.value.length, "shared passage")}`);
            modal = Modal.getOrCreateInstance(modalElement.value);
            modal.show();
        })
        .catch((err) => {
            alert(err);
        });
}

// Back to the passage that opened the modal, unless the text changed meanwhile
function restoreFocus() {
    if (modalOpener && modalOpener.isConnected) {
        focusElement(modalOpener);
    }
    modalOpener = null;
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

/* Narrow screens wrap citations rather than cut them */
@media (max-width: 575.98px) {
    .summary-citation :deep(p) {
        white-space: normal;
    }
}

#reuse-summary .card-header :focus-visible {
    outline-color: #fff !important;
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


/* Floats over the corner of the page, where it stays clear of the text on wide screens */
#back-to-top {
    position: fixed;
    right: 1rem;
    bottom: 1rem;
    z-index: 1030;
    min-width: 2.5rem;
    min-height: 2.5rem;
    border-radius: 1.25rem;
    box-shadow: 0 0.25rem 0.75rem rgba(0, 0, 0, 0.25);
    opacity: 0;
    /* Hidden, it leaves the tab order once faded out */
    visibility: hidden;
    transition: opacity 0.25s, visibility 0s 0.25s;
}

#back-to-top.visible {
    opacity: 0.95;
    visibility: visible;
    transition: opacity 0.25s;
}

#report-error {
    position: absolute;
    right: 0;
    opacity: 0;
    transition: opacity 0.25s;
    pointer-events: none;
}

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
    background-color: #c00;
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

:deep(.note-content.open) {
    display: block;
    margin: 0.5rem 0;
    padding-left: 0.75rem;
    border-left: 3px solid $button-color;
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
    color: #b83500;
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

/* Number opening a reused passage, at its start */
:deep(.reuse-number) {
    vertical-align: 0.3em;
    margin-right: 0.2em;
    padding: 0.1em 0.4em;
    border: 0;
    border-radius: 0.6em;
    background-color: $passage-color;
    color: #fff;
    font-size: 0.7em;
    font-weight: 700;
    line-height: 1.2;
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