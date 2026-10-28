<template>
    <div class="scope-tabs" role="tablist" aria-label="Which later reuse counts" v-if="filtered" @keydown="onKeydown">
        <button v-for="tab in tabs" :key="tab.value" type="button" role="tab" class="scope-tab"
            :id="`scope-tab-${tab.value}`" :ref="(element) => (buttons[tab.value] = element)"
            :aria-selected="String(scope === tab.value)" :aria-controls="panelId"
            :tabindex="isStop(tab.value) ? 0 : -1" @click="select(tab.value)">
            <span class="scope-tab-name">{{ tab.name }}</span>
            <span class="scope-tab-hint">{{ tab.hint }}</span>
        </button>
    </div>
</template>
<script setup>
import { computed } from "vue";
import { useRoute, useRouter } from "vue-router";
import { hasSearch, paramsToUrl } from "../utils.js";

defineProps({ panelId: { type: String, required: true } });

const route = useRoute();
const router = useRouter();
const buttons = {};

// Both read the earlier side's fields as where the passage was first found
const tabs = [
    { value: "origin", name: "Any later reuse", hint: "Found again anywhere after its first appearance" },
    { value: "direct", name: "Direct reuse", hint: "Taken straight from the text it was first found in" },
];

// Without a search, every passage is its own first appearance and the tabs would change nothing
const filtered = computed(() => hasSearch(route.query));
const scope = computed(() => route.query.scope || "origin");
// With ?scope=any no tab is selected, and the first one is where the keyboard enters
const isStop = (value) =>
    tabs.some((tab) => tab.value === scope.value) ? scope.value === value : value === tabs[0].value;

function select(value) {
    if (value === scope.value) return;
    const query = { ...route.query, scope: value };
    delete query.page;
    delete query.id_anchor;
    router.push(`${route.path}?${paramsToUrl(query)}`);
}

// The arrow keys move between the tabs and select the one they land on, as the switch is one request
function onKeydown(event) {
    const current = tabs.findIndex((tab) => `scope-tab-${tab.value}` === document.activeElement.id);
    let next;
    if (event.key === "ArrowRight") next = (current + 1) % tabs.length;
    else if (event.key === "ArrowLeft") next = (current - 1 + tabs.length) % tabs.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = tabs.length - 1;
    else return;
    event.preventDefault();
    buttons[tabs[next].value].focus();
    select(tabs[next].value);
}
</script>
<style lang="scss" scoped>
@import "../assets/theme.module.scss";

// A strip of two tabs on a hairline; the selected one carries the maroon rule
.scope-tabs {
    display: flex;
    gap: 0.25rem;
    margin: 0 0 1.1rem;
    border-bottom: 1px solid $rule-color;
    font-family: $font-ui;
}

.scope-tab {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 0.1rem;
    margin-bottom: -1px;
    padding: 0.6rem 1.1rem 0.55rem;
    border: 0;
    border-bottom: 3px solid transparent;
    border-radius: 3px 3px 0 0;
    background-color: transparent;
    color: $ink-2;
    text-align: left;
    transition: background-color 0.2s ease, box-shadow 0.2s ease;
}

.scope-tab:hover,
.scope-tab:focus {
    background-color: rgba($link-color, 0.08);
    box-shadow: 0 0 0 3px rgba($link-color, 0.1);
}

.scope-tab[aria-selected="true"] {
    border-bottom-color: $button-color;
    color: $header-color;
}

.scope-tab-name {
    font-size: 0.9rem;
    font-weight: 600;
    line-height: 1.3;
}

.scope-tab-hint {
    color: $muted;
    font-size: 0.75rem;
    line-height: 1.3;
}

@media (max-width: 575.98px) {
    .scope-tabs {
        gap: 0;
    }

    .scope-tab {
        flex: 1 1 0;
        min-width: 0;
        padding-inline: 0.6rem;
    }
}
</style>
