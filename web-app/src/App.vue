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

body,
.btn {
    font-size: 0.9rem;
}

.passages {
    font-size: 1rem;
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

// Focus indicator for every control, over Bootstrap's faint focus shadows
:focus-visible {
    outline: 2px solid $focus-color !important;
    outline-offset: 2px;
}

.site-header :focus-visible {
    outline-color: #fff !important;
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

#app {
    font-family: "Source Serif Pro", serif;
}

.shadow-1 {
    box-shadow: 0 2px 5px 0 rgba(0, 0, 0, 0.16), 0 2px 10px 0 rgba(0, 0, 0, 0.12);
}

.btn-light {
    border-color: #ddd;
}

.corner-btn {
    position: absolute;
    transition: all 0.1s ease-out;
    padding: 5px;
    line-height: 1;
    top: 0;
    overflow: hidden;
    border-color: #ddd;
    border-style: solid;
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
    background-color: #565656;
    color: #f8f8f8;
    cursor: pointer;
}

.diff-btn,
.group-diff-btn {
    display: inline-block;
    padding: 0.2rem;
    margin-bottom: 2px;
    border: solid 1px #ddd;
    background-color: #fff;
    color: $link-color;
    cursor: pointer;
    font-family: "Open-Sans", sans-serif;
}

.diff-btn:hover,
.group-diff-btn:hover {
    color: #565656 !important;
    background-color: #f8f8f8;
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
    width: auto;
    line-height: 200%;
    z-index: 5;
    list-style-type: none;
    margin: 0;
    padding: 0;
}

.my-dropdown-item {
    cursor: pointer;
    padding: 0 0.75rem;
}

.my-dropdown-item:hover {
    background: #ddd;
}
</style>
