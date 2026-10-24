<template>
    <div id="app">
        <a class="skip-link" href="#main-content" @click.prevent="focusMain">Skip to main content</a>
        <main-header></main-header>
        <div class="container-fluid">
            <search-form></search-form>
            <main id="main-content" tabindex="-1">
                <router-view></router-view>
            </main>
        </div>
        <div class="visually-hidden" role="status" aria-live="polite" aria-atomic="true">{{ announcement }}</div>
    </div>
</template>

<script setup>
import emitter from "tiny-emitter/instance";
import { nextTick, onUnmounted, ref, watch } from "vue";
import { useRoute } from "vue-router";
import mainHeader from "./components/header";
import searchForm from "./components/searchForm";
import globalConfig from "./config.js";
import { plainText } from "./utils.js";

const route = useRoute();
const announcement = ref("");

// Messages wait their turn, so that one sent right after another is still read
const queue = [];
let speaking = false;

function announce(message) {
    queue.push(message);
    if (!speaking) announceNext();
}

function announceNext() {
    if (!queue.length) {
        speaking = false;
        return;
    }
    speaking = true;
    // Cleared first so that the same message is read again
    announcement.value = "";
    setTimeout(() => {
        announcement.value = queue.shift();
        setTimeout(announceNext, 1000);
    }, 100);
}

function focusMain() {
    const main = document.getElementById("main-content");
    const heading = main.querySelector("h2");
    if (heading) {
        heading.setAttribute("tabindex", "-1");
        heading.focus({ preventScroll: true });
    } else {
        main.focus({ preventScroll: true });
    }
}

function focusMainLater() {
    nextTick(focusMain);
}

emitter.on("announce", announce);
emitter.on("focusMain", focusMainLater);
onUnmounted(() => {
    emitter.off("announce", announce);
    emitter.off("focusMain", focusMainLater);
});

watch(
    () => route.fullPath,
    (fullPath, previousPath) => {
        const label = plainText(globalConfig.databaseLabel);
        document.title = route.meta.title ? `${route.meta.title} | ${label}` : label;
        if (previousPath !== undefined && route.path != previousPath.split("?")[0]) {
            focusMainLater();
        }
    },
    { immediate: true }
);
</script>

<style lang="scss">
@import "./assets/theme.module.scss";
@import "../node_modules/bootstrap/scss/bootstrap.scss";

body {
    background-color: $paper;
}

body,
.btn {
    font-size: 0.9rem;
}

.passages {
    font-size: 1rem;
}

#app {
    font-family: $font-text;
    font-optical-sizing: auto;
}

// Controls and labels in the sans, texts and citations in the serif
.btn,
.form-control,
.form-select,
.input-group-text,
.form-label,
.form-check-label,
.badge,
.page-link,
.dropdown-header,
.modal-title,
summary {
    font-family: $font-ui;
}

// Links as in PhiloLogic5: no underline, a tint on hover and focus
a {
    text-decoration: none;
    transition: all 0.2s ease;
    border-radius: 2px;
}

a:hover,
a:focus {
    background-color: rgba($link-color, 0.08);
    box-shadow: 0 0 0 3px rgba($link-color, 0.1);
}

// The later side's links in its own ink, over the scoped copies of the link rule
#app .side-target a:not(.btn) {
    color: $target-color !important;
}

#app .side-target a:not(.btn):hover,
#app .side-target a:not(.btn):focus {
    background-color: rgba($target-color, 0.08);
    box-shadow: 0 0 0 3px rgba($target-color, 0.1);
}

#app .side-target a:not(.btn):focus-visible {
    outline-color: $target-color !important;
}

// Focus indicator for every control, over Bootstrap's faint focus shadows
:focus-visible {
    outline: 2px solid $focus-color !important;
    outline-offset: 2px;
}

// Toggle buttons: the checkbox or radio is hidden, so its label shows the focus
.btn-check:focus-visible + .btn {
    outline: 2px solid $focus-color !important;
    outline-offset: 2px;
}

.btn-link {
    text-decoration: none;
}

#main-content:focus,
#main-content h2:focus {
    outline: none !important;
}

.skip-link {
    position: absolute;
    top: 0;
    left: 0;
    z-index: 2000;
    padding: 0.5rem 1rem;
    background-color: #fff;
    transform: translateY(-120%);
}

.skip-link:focus {
    transform: none;
}

@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition: none !important;
        scroll-behavior: auto !important;
    }
}

// Paper cards: a hairline and a soft lift
.card {
    border-color: $rule-color;
}

.shadow-1 {
    box-shadow: 0 1px 2px rgba(64, 40, 24, 0.06), 0 4px 14px -6px rgba(64, 40, 24, 0.14);
}

.card-header {
    background-color: #fff;
    border-bottom: 1px solid $rule-color;
    color: $header-color;
    @include display-type;
    font-weight: 600;
}

// Section label: small caps over a hairline
.kicker {
    @include kicker;
    color: $muted;
}

// Side marks: a disc for the earlier text, a ring for the later one
.side-mark {
    display: inline-block;
    flex: 0 0 auto;
    width: 0.62em;
    height: 0.62em;
    margin-right: 0.45em;
    border: 2px solid currentColor;
    border-radius: 50%;
    vertical-align: 0.02em;
}

.side-mark.source {
    background-color: currentColor;
}

.side-source-color {
    color: $source-color;
}

.side-target-color {
    color: $target-color;
}

.btn-secondary {
    --bs-btn-hover-bg: #{$button-color-active};
    --bs-btn-hover-border-color: #{$button-color-active};
}

.btn-outline-secondary {
    --bs-btn-bg: #fff;
}

.btn-light {
    --bs-btn-bg: #fff;
    border-color: $rule-strong;
}

.form-control,
.form-select {
    border-color: $input-border-color;
}

.pagination {
    --bs-pagination-color: #{$link-color};
    --bs-pagination-hover-color: #{$source-deep};
    --bs-pagination-hover-bg: #{$source-tint};
    --bs-pagination-active-bg: #{$header-color};
    --bs-pagination-active-border-color: #{$header-color};
}

.spinner-border {
    color: $button-color;
    border-width: 0.18em;
}

.modal-header {
    background-color: $paper;
}

.modal-header .modal-title {
    @include kicker;
    font-size: 0.78rem !important;
    color: $header-color;
}

.popover {
    --bs-popover-border-color: #{$rule-strong};
    --bs-popover-box-shadow: 0 6px 24px -8px rgba(64, 40, 24, 0.35);
    box-shadow: var(--bs-popover-box-shadow);
}

// Texts sharing a passage, over the full text
.passage-popover {
    --bs-popover-max-width: 24rem;
    font-family: $font-text;
}

.passage-popover .list-group-item {
    padding: 0.55rem 0.8rem;
    border-color: $rule-color;
    font-size: 0.88rem;
    line-height: 1.4;
}

.passage-popover .popover-side-source .list-group-item {
    border-left: 3px solid $source-color;
}

.passage-popover .popover-side-target .list-group-item {
    border-left: 3px solid $target-color;
}

.passage-popover .popover-hint {
    color: $muted;
    font-family: $font-ui;
    font-size: 0.75rem;
}

.corner-btn {
    position: absolute;
    transition: all 0.1s ease-out;
    padding: 5px;
    line-height: 1;
    top: 0;
    overflow: hidden;
    border-color: $rule-color;
    border-style: solid;
    background-color: #fff;
    color: $header-color;
}

.corner-btn.right {
    right: 0;
    border-width: 0px 0px 1px 1px;
}

.corner-btn.left {
    left: 0;
    border-width: 0px 1px 1px 0px;
}

.corner-btn.destroy:hover {
    background-color: $source-tint;
    color: $source-deep;
    cursor: pointer;
}

// Quiet toggles that compare the two texts of a pair
.diff-btn,
.group-diff-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.4em;
    padding: 0.3125rem 0.625rem;
    border: 1px solid $rule-strong;
    border-radius: 3px;
    background-color: #fff;
    color: $link-color;
    cursor: pointer;
    font-family: $font-ui;
    font-size: 0.8rem;
    font-weight: 600;
    line-height: 1.25;
    transition: background-color 0.12s ease, border-color 0.12s ease, color 0.12s ease;
}

.diff-btn:hover,
.group-diff-btn:hover {
    border-color: rgba($button-color, 0.55);
    background-color: $source-tint;
    color: $source-deep;
}

.diff-btn.active {
    border-color: rgba($button-color, 0.55);
    background-color: $source-tint-strong;
    color: $source-deep;
}

.my-dropdown {
    position: relative;
}

.my-dropdown-menu {
    position: absolute;
    display: none;
    left: 0;
    top: 38px;
    background-color: #fff;
    border: 1px solid $rule-color;
    width: auto;
    line-height: 200%;
    z-index: 5;
    list-style-type: none;
    margin: 0;
    padding: 0.25rem 0;
}

.my-dropdown-item {
    cursor: pointer;
    padding: 0 0.75rem;
}

.my-dropdown-item:hover {
    background: $source-tint;
}
</style>
