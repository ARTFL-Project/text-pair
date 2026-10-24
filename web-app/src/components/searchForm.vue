<template>
    <section id="search-form" class="card mt-3 shadow-1" :class="{ open: formOpen }"
        aria-labelledby="search-form-title">
        <div class="card-body" style="position: relative">
            <h2 id="search-form-title" class="visually-hidden">Search</h2>
            <button type="button" id="show-form" :class="{ hide: formOpen }" ref="showFormButton"
                :aria-expanded="String(formOpen)" aria-controls="search-alignments" @click="toggleSearchForm(true)">
                <svg class="show-form-icon" viewBox="0 0 20 20" aria-hidden="true" focusable="false">
                    <circle cx="8.5" cy="8.5" r="5.25" fill="none" stroke="currentColor" stroke-width="1.75" />
                    <path d="M12.5 12.5l4.25 4.25" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" />
                </svg>
                <span class="show-form-label">Show search form</span>
                <svg class="show-form-chevron" viewBox="0 0 20 20" aria-hidden="true" focusable="false">
                    <path d="M5.5 8l4.5 4.5L14.5 8" fill="none" stroke="currentColor" stroke-width="1.75"
                        stroke-linecap="round" stroke-linejoin="round" />
                </svg>
            </button>
            <form id="search-alignments" ref="form" role="search" aria-labelledby="search-form-title"
                @submit.prevent="search()">
                <!-- Metadata Fields -->
                <div class="form-sides">
                    <fieldset class="form-side side-source-color">
                        <legend class="section-title"><span class="side-mark source" aria-hidden="true"></span><span
                                v-html="globalConfig.sourceLabel"></span></legend>
                        <div class="input-group pb-3" v-for="field in globalConfig.metadataFields.source"
                            :key="field.label">
                            <label class="input-group-text" :for="`search-${field.value}`">{{ field.label
                            }}</label>
                            <input type="text" class="form-control" :id="`search-${field.value}`"
                                :name="field.value" v-model="formValues[field.value]" />
                        </div>
                    </fieldset>
                    <span class="form-arrow" aria-hidden="true"></span>
                    <fieldset class="form-side side-target-color">
                        <legend class="section-title"><span class="side-mark target" aria-hidden="true"></span><span
                                v-html="globalConfig.targetLabel"></span></legend>
                        <div class="input-group pb-3" v-for="field in globalConfig.metadataFields.target"
                            :key="field.label">
                            <label class="input-group-text" :for="`search-${field.value}`">{{ field.label
                            }}</label>
                            <input type="text" class="form-control" :id="`search-${field.value}`"
                                :name="field.value" v-model="formValues[field.value]" />
                        </div>
                    </fieldset>
                </div>
                <!-- Passage Classification -->
                <fieldset id="passage-classification" class="mb-3" v-if="globalConfig.passageClassification.length">
                    <legend class="section-title">Passage Classification</legend>
                    <div class="row">
                        <div class="col-12 col-md-4">
                            <div class="input-group pb-3">
                                <label class="input-group-text" for="classification-filter-1">First Criterion</label>
                                <select id="classification-filter-1" class="form-control select-with-arrow"
                                    v-model="classificationFilters[0]">
                                    <option value="">Any classification</option>
                                    <option v-for="classItem in globalConfig.passageClassification"
                                        :key="classItem.label" :value="classItem.label" :title="classItem.description">
                                        {{ classItem.label.replace(/_/g, ' ') }}
                                    </option>
                                </select>
                            </div>
                        </div>
                        <div class="col-12 col-md-4">
                            <div class="input-group pb-3">
                                <label class="input-group-text" for="classification-filter-2">Second Criterion</label>
                                <select id="classification-filter-2" class="form-control select-with-arrow"
                                    v-model="classificationFilters[1]">
                                    <option value="">Any classification</option>
                                    <option v-for="classItem in globalConfig.passageClassification"
                                        :key="classItem.label" :value="classItem.label" :title="classItem.description">
                                        {{ classItem.label.replace(/_/g, ' ') }}
                                    </option>
                                </select>
                            </div>
                        </div>
                        <div class="col-12 col-md-4">
                            <div class="input-group pb-3">
                                <label class="input-group-text" for="classification-filter-3">Third Criterion</label>
                                <select id="classification-filter-3" class="form-control select-with-arrow"
                                    v-model="classificationFilters[2]">
                                    <option value="">Any classification</option>
                                    <option v-for="classItem in globalConfig.passageClassification"
                                        :key="classItem.label" :value="classItem.label" :title="classItem.description">
                                        {{ classItem.label.replace(/_/g, ' ') }}
                                    </option>
                                </select>
                            </div>
                        </div>
                    </div>
                    <div class="row mt-3" v-if="banalitiesStored">
                        <div class="col-12">
                            <div id="banality-filter" class="pb-3">
                                <div class="input-group">
                                    <label class="input-group-text" for="banality-select">Banality
                                        Filter</label>
                                    <select id="banality-select" class="form-control select-with-arrow"
                                        v-model="banalitySelected" @change="updateBanalityValue">
                                        <option v-for="(option, optionIndex) in formBanalityValues" :key="optionIndex"
                                            :value="option.label">
                                            {{ option.label }}
                                        </option>
                                    </select>
                                </div>
                            </div>
                        </div>
                    </div>
                </fieldset>

                <div id="search-buttons">
                    <button class="btn btn-secondary" type="submit">Search</button>
                    <button type="button" class="btn btn-outline-secondary" @click="clearForm()">
                        Reset
                    </button>
                </div>
            </form>
        </div>
    </section>
</template>

<script setup>
import emitter from "tiny-emitter/instance";
import Velocity from "velocity-animate";
import { nextTick, onUnmounted, reactive, ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import globalConfig from "../config.js";
import { prefersReducedMotion } from "../utils.js";

const route = useRoute();
const router = useRouter();
const form = ref(null);
const showFormButton = ref(null);
const formOpen = ref(true);
const banalitiesStored = globalConfig.banalitiesStored;
const formBanalityValues = [
    { label: "Don't filter banalities", value: "" },
    { label: "Filter all banalities", value: false },
    { label: "Search only banalities", value: true },
];
const formValues = reactive(populateSearchForm());
const classificationFilters = ref([
    route.query.classification_filter_1 || "",
    route.query.classification_filter_2 || "",
    route.query.classification_filter_3 || "",
]);
const banalitySelected = ref("Don't filter banalities");
if ("banality" in route.query) {
    banalitySelected.value = route.query.banality == "true" ? "Search only banalities" : "Filter all banalities";
}

function updateFromUrl(updatedParams) {
    for (const key in updatedParams) {
        if (key in formValues) {
            formValues[key] = updatedParams[key];
        }
    }
}
emitter.on("urlUpdate", updateFromUrl);
emitter.on("toggleSearchForm", toggleSearchForm);
onUnmounted(() => {
    emitter.off("urlUpdate", updateFromUrl);
    emitter.off("toggleSearchForm", toggleSearchForm);
});

if (route.path != "/") {
    toggleSearchForm();
}

function populateSearchForm() {
    const values = {};
    for (const direction of ["source", "target"]) {
        for (const key of globalConfig.metadataFields[direction]) {
            values[key.value] = key.value in route.query ? route.query[key.value] : "";
        }
    }
    values.banality = "";
    values.timeSeriesInterval = globalConfig.timeSeriesIntervals[0].value;
    values.directionSelected = "source";
    return values;
}

function search() {
    toggleSearchForm();

    // Create URLSearchParams to properly handle multiple parameters with the same name
    const urlParams = new URLSearchParams();

    // Add all regular form values
    for (const [key, value] of Object.entries(formValues)) {
        if (value !== "" && value !== null && value !== undefined) {
            urlParams.append(key, value);
        }
    }

    for (const [index, filter] of classificationFilters.value.entries()) {
        if (filter) {
            urlParams.append(`classification_filter_${index + 1}`, filter);
        }
    }

    router.push(`/search?${urlParams.toString()}`).then(() => emitter.emit("focusMain"));
}

function clearForm() {
    for (const key in formValues) {
        formValues[key] = "";
    }
    // Reset classification filters
    classificationFilters.value = ["", "", ""];
}

function toggleSearchForm(moveFocus = false) {
    nextTick(() => {
        const duration = prefersReducedMotion() ? 0 : 200;
        if (form.value.style.display == "none") {
            formOpen.value = true;
            Velocity(form.value, "slideDown", {
                duration,
                easing: "ease-out",
                complete: () => {
                    if (moveFocus === true) form.value.querySelector("input, select").focus();
                },
            });
        } else {
            Velocity(form.value, "slideUp", {
                duration,
                easing: "ease-out",
                complete: () => {
                    formOpen.value = false;
                },
            });
        }
    });
}

function updateBanalityValue() {
    // Find the matching option and set its value in formValues
    const selectedOption = formBanalityValues.find((option) => option.label === banalitySelected.value);
    if (selectedOption) {
        formValues.banality = selectedOption.value;
    }
}
</script>

<style lang="scss" scoped>
@import "../assets/theme.module.scss";

#search-form {
    font-family: $font-ui;
    border-radius: 3px;
    transition: background-color 0.2s ease-out, border-color 0.2s ease-out, box-shadow 0.2s ease-out;
}

// Closed, the card is the toggle: a bar that lifts on hover
#search-form:not(.open):hover {
    border-color: rgba($button-color, 0.45);
    box-shadow: $lift;
}

#search-form .card-body {
    padding: 0;
    transition: padding 0.2s ease-out;
}

#search-form.open .card-body {
    padding: 1.4rem 1.5rem 1.25rem;
}

.input-group-text,
.form-control {
    font-size: 0.85rem;
}

// Fields of each side: a label cell tinted in the side's color
.input-group-text {
    flex: 0 0 8.25rem;
    justify-content: flex-start;
    font-weight: 600;
    border-color: $input-border-color !important;
}

.side-source-color .input-group-text {
    color: $source-deep !important;
    background-color: $source-tint !important;
}

.side-target-color .input-group-text {
    color: $target-deep !important;
    background-color: $target-tint !important;
}

#passage-classification .input-group-text {
    flex-basis: auto;
    color: $ink !important;
    background-color: $paper !important;
}

.input-group > :not(:first-child):not(.dropdown-menu):not(.valid-tooltip):not(.valid-feedback):not(.invalid-tooltip):not(.invalid-feedback) {
    border-left-color: $input-border-color;
}

.side-target-color .form-control:focus {
    border-color: $target-color !important;
    box-shadow: 0 0 0 0.05rem $target-color !important;
}

.side-target-color :focus-visible {
    outline-color: $target-color !important;
}

.form-sides {
    position: relative;
    display: grid;
    grid-template-columns: minmax(0, 1fr) 5rem minmax(0, 1fr);
}

.form-side {
    min-width: 0;
}

.side-source-color.form-side {
    grid-column: 1;
}

.side-target-color.form-side {
    grid-column: 3;
}

// The search reads like a pair: earlier, then later
.form-arrow {
    grid-column: 2;
    grid-row: 1;
    position: relative;
    align-self: start;
    justify-self: center;
    width: 3rem;
    height: 1.5px;
    margin-top: 0.62rem;
    background: linear-gradient(90deg, $source-color, $target-color);
}

.form-arrow::after {
    content: "";
    position: absolute;
    right: 0;
    top: -3.5px;
    width: 8px;
    height: 8px;
    border-top: 1.5px solid $target-color;
    border-right: 1.5px solid $target-color;
    transform: rotate(45deg);
}

.hide {
    opacity: 0;
    max-height: 0 !important;
    margin: 0;
    padding: 0;
    visibility: hidden;
}

#show-form {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    width: 100%;
    min-height: 2.875rem;
    padding: 0.5rem 1rem;
    border: 0;
    border-radius: 3px;
    background-color: transparent;
    color: $ink-2;
    font-size: 0.95rem;
    font-weight: 600;
    text-align: left;
    cursor: pointer;
    transition: color 0.12s ease;
}

#show-form:hover {
    color: $source-deep;
}

#show-form.hide {
    position: absolute;
    min-height: 0;
    height: 0;
    overflow: hidden;
}

#show-form svg {
    flex: 0 0 auto;
    width: 1.125rem;
    height: 1.125rem;
}

.show-form-label {
    flex: 1 1 auto;
}

.show-form-chevron {
    color: $muted;
}

.show-form-icon {
    color: $button-color;
}

fieldset legend + * {
    clear: left;
}

#reuse-classification {
    select {
        option[title] {
            cursor: help;
        }
    }

    .select-with-arrow {
        position: relative;
        appearance: none;
        -webkit-appearance: none;
        -moz-appearance: none;
        padding-right: 25px;
        /* Space for the arrow */
    }

    .select-with-arrow::after {
        content: '\25BC';
        /* Unicode down triangle */
        position: absolute;
        right: 10px;
        top: 50%;
        transform: translateY(-50%);
        pointer-events: none;
        /* Ensures clicks pass through to the select */
    }
}

.section-title {
    @include kicker;
    display: flex;
    align-items: center;
    float: none;
    width: 100%;
    margin-bottom: 0.9rem;
    padding: 0;
    color: inherit;
    font-size: 0.74rem;
}

#passage-classification {
    margin-top: 0.75rem;
    padding-top: 1rem;
    border-top: 1px solid $rule-color;
}

#passage-classification .section-title {
    color: $muted;
}

#banality-filter {
    text-align: left;
    width: fit-content;

    label {
        font-weight: 500;
        font-size: 0.9rem;
        color: #666;
    }

    button {
        width: fit-content;
        text-align: left;
    }

    .my-dropdown-menu {
        width: fit-content;
    }
}

#search-buttons {
    display: flex;
    justify-content: center;
    gap: 0.6rem;
    margin-top: 0.4rem;
}

#search-buttons .btn {
    min-width: 6.5rem;
    font-weight: 600;
}

/* Select styling enhancements */
.select-with-arrow {
    appearance: none;
    -webkit-appearance: none;
    -moz-appearance: none;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='12' height='12' fill='%23495057' viewBox='0 0 16 16'><path d='M7.247 11.14L2.451 5.658C1.885 5.013 2.345 4 3.204 4h9.592a1 1 0 0 1 .753 1.659l-4.796 5.48a1 1 0 0 1-1.506 0z'/></svg>");
    background-repeat: no-repeat;
    background-position: calc(100% - 10px) center;
    padding-right: 28px;
    /* Space for the arrow */
}

@media (max-width: 767.98px) {
    #search-form.open .card-body {
        padding: 1rem;
    }

    .form-sides {
        grid-template-columns: minmax(0, 1fr);
    }

    .side-source-color.form-side,
    .side-target-color.form-side {
        grid-column: 1;
    }

    .form-arrow {
        grid-column: 1;
        grid-row: 2;
        width: 1.5px;
        height: 1.75rem;
        margin: -0.5rem 0 0.75rem;
        justify-self: start;
        left: 0.25rem;
        background: linear-gradient($source-color, $target-color);
    }

    .form-arrow::after {
        top: auto;
        bottom: 0;
        right: -3.5px;
        transform: rotate(135deg);
    }
}
</style>
