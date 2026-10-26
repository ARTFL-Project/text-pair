<template>
    <div class="row mt-1" id="all-content" :class="`text-side-${direction}`">
        <div class="col-12 col-sm-10 offset-sm-1 col-lg-8 offset-lg-2" id="center-content">
            <h2 class="text-title" :class="{ 'visually-hidden': !textObject.metadata }">
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
                    <div class="btn-group direction-toggle" role="group" aria-label="Direction of reuse">
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
                <div class="card shadow-1 mb-4" id="reuse-summary" v-if="textObject.text"
                    :class="`side-${otherDirection}`">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h3 class="summary-heading">{{ summaryHeading }}</h3>
                        <button type="button" class="summary-toggle" v-if="otherTexts.length"
                            :aria-expanded="String(!summaryHidden)" aria-controls="reuse-summary-list"
                            @click="toggleSummary()">
                            {{ summaryHidden ? "Show" : "Hide" }}<span class="visually-hidden"> texts sharing
                                passages</span>
                        </button>
                    </div>
                    <template v-if="!summaryHidden && otherTexts.length">
                        <ul class="list-group list-group-flush" id="reuse-summary-list">
                            <li class="list-group-item" v-for="other in otherTexts" :key="other.id">
                                <div class="summary-row">
                                    <div class="summary-citation" :title="citationText(other.metadata)">
                                        <citations :citation="globalConfig[`${otherDirection}Citation`]"
                                            :alignment="other.metadata" :link-to-doc="`${otherDirection}_title`">
                                        </citations>
                                    </div>
                                    <span class="summary-gap" v-if="yearGap(other) !== null">{{ yearGap(other)
                                    }}</span>
                                </div>
                                <div class="passage-links px-3 pb-1">
                                    <button type="button" class="passage-chip"
                                        :class="{ elsewhere: !inSection(passage) }"
                                        v-for="passage in shownPassages(other)" :key="passage"
                                        :title="passageSnippets[passage] || passageSectionLabel(passage)"
                                        @click="openPassage(passage)">
                                        <span class="visually-hidden">Passage </span>{{ passage + 1 }}<span
                                            class="visually-hidden" v-if="!inSection(passage)">, in {{
                                                passageSectionLabel(passage) }}</span>
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
                            <template v-if="section">Dashed numbers are in other sections.</template>
                        </div>
                    </template>
                </div>
                <div class="section-bar" :class="`side-${direction}`" v-if="section">
                    <nav aria-label="Sections of this text" @keydown.esc="tocOpen && closeToc()">
                        <div class="section-nav">
                            <button type="button" class="section-button" id="section-previous"
                                :disabled="!section.previous" @click="goToSection(section.previous, 'section-previous')">
                                <span aria-hidden="true">&larr;</span> Previous<span class="visually-hidden">
                                    section</span>
                            </button>
                            <p class="section-label" id="section-label" tabindex="-1" :title="sectionLabel(section)">
                                <span class="section-position">Section {{ section.index + 1 }} of {{ sections.length
                                }}</span>
                                <span class="section-head">{{ sectionLabel(section) }}</span>
                            </p>
                            <button type="button" class="section-button" id="section-toc-toggle"
                                :aria-expanded="String(tocOpen)" aria-controls="section-toc" @click="toggleToc()">
                                Contents
                            </button>
                            <button type="button" class="section-button" id="section-next" :disabled="!section.next"
                                @click="goToSection(section.next, 'section-next')">
                                Next<span class="visually-hidden"> section</span> <span aria-hidden="true">&rarr;</span>
                            </button>
                        </div>
                        <ul class="section-toc" id="section-toc" ref="tocList" v-if="tocOpen">
                            <li v-for="(item, index) in sections" :key="item.philo_id">
                                <button type="button" class="toc-item"
                                    :class="{ current: index == section.index, reused: item.passages }"
                                    :aria-current="index == section.index ? 'true' : null"
                                    @click="goToSection(item.philo_id, 'section-label')">
                                    <span class="toc-head">{{ sectionLabel(item, index) }}</span>
                                    <span class="toc-count" v-if="item.passages">{{ item.passages }}<span
                                            class="visually-hidden"> reused {{ item.passages == 1 ? "passage starts" :
                                                "passages start" }} here</span></span>
                                </button>
                            </li>
                        </ul>
                    </nav>
                </div>
                <div class="card book-card mb-4 shadow-1">
                    <div id="book-page" class="text-view" ref="bookPage" role="region" aria-label="Document text">
                        <div id="text-obj-content" class="text-content-area" ref="textContent" v-html="textObject.text"
                            :lang="globalConfig[`${direction}Language`] || undefined" @click="textClick"
                            @keydown="textKeydown" @mouseover="textMouseover" @mouseout="textMouseout"
                            @focusin="textFocusin" @focusout="textFocusout"></div>
                    </div>
                </div>
                <nav class="section-nav section-nav-end mb-4" aria-label="Sections of this text, end of section"
                    v-if="section">
                    <button type="button" class="section-button" :disabled="!section.previous"
                        @click="goToSection(section.previous, 'section-label')">
                        <span aria-hidden="true">&larr;</span> Previous<span class="visually-hidden"> section</span>
                    </button>
                    <button type="button" class="section-button" :disabled="!section.next"
                        @click="goToSection(section.next, 'section-label')">
                        Next<span class="visually-hidden"> section</span> <span aria-hidden="true">&rarr;</span>
                    </button>
                </nav>
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
const tocList = ref(null);
const tocOpen = ref(false);

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
let sideSwitchedFor = null;

const section = computed(() => textObject.value.section);
const sections = computed(() => textObject.value.sections || []);
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
        if (route.name != "textNavigation") return;
        if (inLoadedSection()) {
            goToPassage(Number(route.query.passage));
        } else {
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
    tocOpen.value = false;
    hidePopover();
    if (modal != null) {
        modal.hide();
    }
    passagePairs.value = [];
    const request = ++textRequest;
    axios
        .get(`${globalConfig.apiServer}/text_view/`, {
            params: { ...route.query, by_section: true },
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
                if (route.query.start_byte && route.query.end_byte && switchSide()) return;
                emitter.emit("announce", loadedMessage());
                if (route.query.passage !== undefined) {
                    goToPassage(Number(route.query.passage));
                } else if (route.query.start_byte && route.query.end_byte) {
                    goToPassage(response.data.passage_number);
                } else if (focusAfterLoad) {
                    const el = document.getElementById(focusAfterLoad);
                    focusElement(el?.disabled ? document.getElementById("section-label") : el);
                    if (section.value) {
                        scrollToText();
                    }
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

function loadedMessage() {
    if (!section.value) {
        return `Text loaded: ${summaryHeading.value}`;
    }
    const shown = plural(section.value.passages.length, "reused passage");
    return `${sectionPosition()}, ${sectionLabel(section.value)}: ${shown}. ${summaryHeading.value}`;
}

function sectionPosition() {
    return `Section ${section.value.index + 1} of ${sections.value.length}`;
}

// Headless divisions, such as PhiloLogic's placeholders, are named by their number
function sectionLabel(item, index = item.index) {
    return item.head || `Section ${index + 1}`;
}

function passageSectionLabel(passage) {
    const id = textObject.value.passage_sections?.[passage];
    const index = sections.value.findIndex((item) => item.philo_id == id);
    return index < 0 ? "" : sectionLabel(sections.value[index], index);
}

function inSection(passage) {
    return !section.value || section.value.passages.includes(passage);
}

// The route asks for a passage of the section on show, so there is nothing to fetch
function inLoadedSection() {
    const query = route.query;
    return (
        section.value &&
        query.passage !== undefined &&
        query.div1_id == section.value.philo_id &&
        query.doc_id == textObject.value.doc_id &&
        query.directionSelected == direction.value &&
        !loading.value
    );
}

function sectionQuery(extra) {
    return {
        db_table: route.query.db_table,
        doc_id: textObject.value.doc_id,
        directionSelected: direction.value,
        ...(section.value ? { div1_id: section.value.philo_id } : {}),
        ...extra,
    };
}

function goToSection(id, focusId) {
    tocOpen.value = false;
    if (id == section.value.philo_id) {
        focusElement(document.getElementById(focusId));
        return;
    }
    focusAfterLoad = focusId;
    router.push({ query: sectionQuery({ div1_id: id }) });
}

function openPassage(passage) {
    if (!section.value) {
        goToPassage(passage);
        return;
    }
    const div1Id = inSection(passage) ? section.value.philo_id : textObject.value.passage_sections[passage];
    router.push({ query: sectionQuery({ div1_id: div1Id, passage }) });
}

function toggleToc() {
    tocOpen.value = !tocOpen.value;
    if (tocOpen.value) {
        nextTick(() => {
            const current = tocList.value?.querySelector(".current");
            if (current) {
                tocList.value.scrollTop = current.offsetTop - tocList.value.clientHeight / 2;
            }
        });
    }
}

function closeToc() {
    tocOpen.value = false;
    focusElement(document.getElementById("section-toc-toggle"));
}

function scrollToText() {
    window.scrollTo({
        top: Math.max(bookPagePosition - 200, 0),
        behavior: prefersReducedMotion() ? "auto" : "smooth",
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
    router.push({ query: sectionQuery({ directionSelected: newDirection }) });
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

// Years between this text and another, as in the bridge of a pair
function yearGap(other) {
    const own = parseInt(textObject.value.metadata[`${direction.value}_year`]);
    const theirs = parseInt(other.metadata[`${otherDirection.value}_year`]);
    if (isNaN(own) || isNaN(theirs)) return null;
    const gap = Math.abs(theirs - own);
    if (gap == 0) return "same year";
    const years = `${gap} ${gap == 1 ? "year" : "years"}`;
    return theirs > own ? `${years} later` : `${years} earlier`;
}

function toggleExpanded(id) {
    expanded.value = { ...expanded.value, [id]: !expanded.value[id] };
}

// A link can name the other side of the passage's alignments, as for a reuse the group view shows facing its passage
function switchSide() {
    const start = Number(route.query.start_byte);
    const found = [...textContent.value.getElementsByClassName("passage-marker")].some((marker) => {
        const [first, last] = marker.getAttribute("data-offsets").split("-").map(Number);
        return first <= start && start <= last;
    });
    const link = `${route.query.doc_id}:${route.query.start_byte}`;
    if (found || sideSwitchedFor == link || !textObject.value.passage_counts[otherDirection.value]) return false;
    sideSwitchedFor = link;
    router.replace({ query: { ...route.query, directionSelected: otherDirection.value } });
    return true;
}

function goToPassage(passage) {
    const marker = textContent.value?.querySelector(`.passage-marker[n="${passage}"]`);
    if (!marker) return;
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
    let citation = `<ul class="list-group popover-side-${side}"${language ? ` lang="${language}"` : ""} style="list-style-type: none; margin: 0; padding: 0;">`;
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
    citation += `</ul><div class="p-2 popover-hint" lang="${pageLanguage}"><b>Click on passage to see reuse.</b></div>`;
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

.text-title {
    margin: 2rem auto 1.5rem;
    max-width: 44rem;
    font-size: 1.55rem;
    font-weight: 400;
    line-height: 1.35;
    text-align: center;
}

.text-title :deep([data-field="author"]) {
    display: block;
    margin-bottom: 0.35rem;
    font-family: $font-ui;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.text-title :deep([data-field="author"] .separator) {
    display: none;
}

.text-title :deep([data-field="title"]) {
    color: $header-color;
}

.text-title :deep([data-field="year"]) {
    @include display-type;
    color: $ink;
}

.direction-toggle .btn {
    font-weight: 600;
}

// The texts on the other side, in that side's ink
#reuse-summary {
    border-top: 3px solid $target-color;
    border-radius: 3px;
}

#reuse-summary.side-source {
    border-top-color: $source-color;
}

#reuse-summary .card-header {
    padding: 0.7rem 1rem;
}

.summary-heading {
    margin: 0;
    color: $ink;
    font-family: $font-text;
    font-size: 1rem;
    font-weight: 600;
    line-height: 1.4;
}

.summary-toggle {
    flex: 0 0 auto;
    padding: 0.15rem 0.7rem;
    border: 1px solid rgba($button-color, 0.55);
    border-radius: 2rem;
    background-color: #fff;
    color: $link-color;
    font-family: $font-ui;
    font-size: 0.75rem;
    font-weight: 600;
}

.summary-toggle:hover {
    background-color: $source-tint;
}

#reuse-summary .list-group-item {
    padding: 0.6rem 1rem 0.5rem;
    border-color: $rule-color;
}

/* Long lists scroll within the card, rather than push the text down */
#reuse-summary-list {
    max-height: 16rem;
    overflow-y: auto;
}

.summary-row {
    display: flex;
    align-items: baseline;
    gap: 1rem;
}

.summary-citation {
    flex: 1 1 auto;
    min-width: 0;
}

.summary-citation :deep(p) {
    margin: 0 !important;
    padding: 0 !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.summary-gap {
    flex: 0 0 auto;
    color: $muted;
    font-family: $font-ui;
    font-size: 0.72rem;
    font-weight: 600;
    white-space: nowrap;
}

/* Narrow screens wrap citations rather than cut them */
@media (max-width: 575.98px) {
    .summary-citation :deep(p) {
        white-space: normal;
    }

    .summary-row {
        flex-direction: column;
        gap: 0.15rem;
    }
}

#reuse-summary .card-footer {
    background-color: $paper;
    border-color: $rule-color;
    color: $muted !important;
}

.passage-links {
    padding: 0.45rem 0 0 !important;
}

// Passage numbers: small chips leading to each passage in the text
.passage-chip {
    min-width: 2rem;
    margin: 0 0.25rem 0.3rem 0;
    padding: 0.05rem 0.4rem;
    border: 1px solid rgba($button-color, 0.5);
    border-radius: 2rem;
    background-color: #fff;
    color: $link-color;
    font-family: $font-ui;
    font-size: 0.75rem;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
}

.passage-chip:hover {
    border-color: $button-color;
    background-color: $source-tint;
}

.text-side-target .passage-chip {
    border-color: rgba($target-color, 0.5);
    color: $target-color;
}

.text-side-target .passage-chip:hover {
    border-color: $target-color;
    background-color: $target-tint;
}

.text-side-target .passage-chip:focus-visible {
    outline-color: $target-color !important;
}

// Passages in other sections: lighter, dashed, and not filled
.passage-chip.elsewhere {
    border-style: dashed;
    background-color: transparent;
    font-weight: 400;
}

.passage-chip.elsewhere:hover {
    background-color: $source-tint;
}

.text-side-target .passage-chip.elsewhere:hover {
    background-color: $target-tint;
}

.section-bar {
    margin-bottom: 0.75rem;
    border: 1px solid $rule-color;
    border-top: 3px solid $target-color;
    border-radius: 3px;
    background-color: #fff;
}

.section-bar.side-source {
    border-top-color: $source-color;
}

.section-nav {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0.75rem;
}

.section-nav-end {
    justify-content: space-between;
}

.section-label {
    flex: 1 1 auto;
    min-width: 0;
    margin: 0;
    line-height: 1.3;
}

.section-position {
    display: block;
    color: $muted;
    font-family: $font-ui;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.section-head {
    display: block;
    overflow: hidden;
    color: $ink;
    font-family: $font-text;
    font-size: 0.95rem;
    font-weight: 600;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.section-button {
    flex: 0 0 auto;
    padding: 0.2rem 0.75rem;
    border: 1px solid rgba($button-color, 0.55);
    border-radius: 2rem;
    background-color: #fff;
    color: $link-color;
    font-family: $font-ui;
    font-size: 0.8rem;
    font-weight: 600;
}

.section-button:hover:not(:disabled),
.section-button[aria-expanded="true"] {
    background-color: $source-tint;
}

.section-bar.side-target .section-button {
    border-color: rgba($target-color, 0.5);
    color: $target-color;
}

.section-bar.side-target .section-button:hover:not(:disabled),
.section-bar.side-target .section-button[aria-expanded="true"] {
    background-color: $target-tint;
}

.section-button:disabled {
    border-color: $rule-color;
    color: $muted;
    opacity: 0.7;
}

// The table of contents scrolls within its panel, as the reuse summary does
.section-toc {
    position: relative;
    max-height: 18rem;
    margin: 0;
    padding: 0;
    overflow-y: auto;
    border-top: 1px solid $rule-color;
    list-style: none;
}

.toc-item {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 1rem;
    width: 100%;
    padding: 0.35rem 0.75rem;
    border: 0;
    border-bottom: 1px solid $rule-color;
    background-color: transparent;
    color: $link-color;
    font-family: $font-text;
    font-size: 0.9rem;
    text-align: left;
}

.toc-item:hover {
    background-color: $source-tint;
}

.section-bar.side-target .toc-item {
    color: $target-color;
}

.section-bar.side-target .toc-item:hover {
    background-color: $target-tint;
}

.toc-item.current {
    background-color: $well;
    font-weight: 700;
}

.toc-item.reused .toc-head {
    font-weight: 600;
}

.toc-count {
    flex: 0 0 auto;
    min-width: 1.6rem;
    padding: 0 0.4rem;
    border-radius: 2rem;
    background-color: $source-tint-strong;
    font-family: $font-ui;
    font-size: 0.72rem;
    font-weight: 700;
    text-align: center;
}

.section-bar.side-target .toc-count {
    background-color: $target-tint-strong;
}

@media (max-width: 575.98px) {
    .section-nav {
        flex-wrap: wrap;
    }

    .section-label {
        flex-basis: 100%;
        order: -1;
    }
}

.passage-links .btn-link {
    font-family: $font-ui;
    font-size: 0.75rem;
    font-weight: 600;
    vertical-align: baseline;
}

.separator {
    padding: 5px;
    font-size: 60%;
    display: inline-block;
    vertical-align: middle;
}

.book-card {
    border-radius: 3px;
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

// The page itself: a reading measure, generous leading
#book-page {
    max-width: 44rem;
    margin: 0 auto;
    padding: 2.5rem 2rem 3rem;
    color: $ink;
    font-size: 1.08rem;
    line-height: 1.72;
    text-align: justify;
    hyphens: auto;
}

// Narrow pages read ragged right, without the gaps of justified lines
@media (max-width: 575.98px) {
    #book-page {
        padding: 1.75rem 1.1rem 2rem;
        text-align: left;
    }
}

:deep(b.headword[role="heading"]) {
    @include display-type;
    color: $header-color;
    letter-spacing: 0.02em;
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

// Page breaks as a quiet folio between rules
:deep(.xml-pb-image) {
    display: flex !important;
    align-items: center;
    gap: 1rem;
    margin: 1.25rem 0 !important;
    color: $muted;
    font-family: $font-ui;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    line-height: 1;
}

:deep(.xml-pb-image::before),
:deep(.xml-pb-image::after) {
    content: "";
    flex: 1 1 auto;
    height: 1px;
    background-color: $rule-color;
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

// Reused passages in this text's ink, over a light wash of it
#text-obj-content :deep([class^="passage-"]) {
    color: $source-deep;
    font-weight: 600;
    background-color: rgba($source-color, 0.07);
    box-shadow: 0 1px 0 rgba($source-color, 0.35);
    cursor: pointer;
}

.text-side-target #text-obj-content :deep([class^="passage-"]) {
    color: $target-deep;
    background-color: rgba($target-color, 0.07);
    box-shadow: 0 1px 0 rgba($target-color, 0.35);
}

#text-obj-content :deep([class^="passage-"].focus) {
    background-color: $source-color;
    box-shadow: none;
    color: #fff;
}

.text-side-target #text-obj-content :deep([class^="passage-"].focus) {
    background-color: $target-color;
}

/* Number opening a reused passage, at its start */
:deep(.reuse-number) {
    vertical-align: 0.3em;
    margin-right: 0.25em;
    padding: 0.12em 0.45em;
    border: 0;
    border-radius: 0.7em;
    background-color: $source-color;
    color: #fff;
    font-family: $font-ui;
    font-size: 0.66em;
    font-weight: 700;
    line-height: 1.2;
    text-indent: 0;
}

.text-side-target :deep(.reuse-number) {
    background-color: $target-color;
}

.text-side-target :deep(.reuse-number:focus-visible) {
    outline-color: $target-color !important;
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