<template>
    <section id="search-form" class="card rounded-0 mt-3 shadow-1" aria-labelledby="search-form-title">
        <div class="card-body rounded-0" style="position: relative">
            <h2 id="search-form-title" class="visually-hidden">Search</h2>
            <button type="button" id="show-form" class="p-2" :class="{ hide: formOpen }" ref="showFormButton"
                :aria-expanded="String(formOpen)" aria-controls="search-alignments" @click="toggleSearchForm(true)">Show
                search form</button>
            <form id="search-alignments" ref="form" role="search" aria-labelledby="search-form-title"
                @submit.prevent="search()">
                <!-- Metadata Fields -->
                <div class="row">
                    <fieldset class="col-12 col-md-6">
                        <legend class="section-title text-center pb-2" v-html="globalConfig.sourceLabel"></legend>
                        <div class="input-group pb-3" v-for="field in globalConfig.metadataFields.source"
                            :key="field.label">
                            <label class="input-group-text rounded-0" :for="`search-${field.value}`">{{ field.label
                            }}</label>
                            <input type="text" class="form-control rounded-0" :id="`search-${field.value}`"
                                :name="field.value" v-model="formValues[field.value]" />
                        </div>
                    </fieldset>
                    <fieldset class="col-12 col-md-6 target-fields">
                        <legend class="section-title text-center pb-2" v-html="globalConfig.targetLabel"></legend>
                        <div class="input-group pb-3" v-for="field in globalConfig.metadataFields.target"
                            :key="field.label">
                            <label class="input-group-text rounded-0" :for="`search-${field.value}`">{{ field.label
                            }}</label>
                            <input type="text" class="form-control rounded-0" :id="`search-${field.value}`"
                                :name="field.value" v-model="formValues[field.value]" />
                        </div>
                    </fieldset>
                </div>
                <!-- Passage Classification -->
                <fieldset id="passage-classification" class="mb-3" v-if="globalConfig.passageClassification.length">
                    <legend class="section-title text-center pb-2">Passage Classification</legend>
                    <div class="row">
                        <div class="col-12 col-md-4">
                            <div class="input-group pb-3">
                                <label class="input-group-text rounded-0" for="classification-filter-1">First Criterion</label>
                                <select id="classification-filter-1" class="form-control rounded-0 select-with-arrow"
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
                                <label class="input-group-text rounded-0" for="classification-filter-2">Second Criterion</label>
                                <select id="classification-filter-2" class="form-control rounded-0 select-with-arrow"
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
                                <label class="input-group-text rounded-0" for="classification-filter-3">Third Criterion</label>
                                <select id="classification-filter-3" class="form-control rounded-0 select-with-arrow"
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
                                    <label class="input-group-text rounded-0" for="banality-select">Banality
                                        Filter</label>
                                    <select id="banality-select" class="form-control rounded-0 select-with-arrow"
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
                    <button class="btn btn-secondary rounded-0" type="submit">Search</button>
                    <button type="button" class="btn btn-outline-secondary rounded-0" @click="clearForm()">
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
    font-family: "Open-Sans", sans-serif;
}

.input-group>span {
    font-size: 0.85rem !important;
}

.input-group> :not(:first-child):not(.dropdown-menu):not(.valid-tooltip):not(.valid-feedback):not(.invalid-tooltip):not(.invalid-feedback) {
    border-left-color: $link-color;
}

.input-group-text,
.form-control {
    font-size: inherit;
}

.hide {
    opacity: 0;
    max-height: 0 !important;
    margin: 0;
    padding: 0;
    visibility: hidden;
}

#show-form {
    position: absolute;
    border-width: 0;
    background-color: transparent;
    color: inherit;
    font-size: 1rem;
    font-weight: 500;
    transition: all 0.2s ease-out;
    cursor: pointer;
    top: 0;
    left: 50%;
    transform: translateX(-50%);
}

#show-form:hover {
    color: #565656;
}

@media (min-width: 768px) {
    .target-fields {
        border-left: 1px solid var(--bs-border-color);
    }
}

fieldset legend + * {
    clear: left;
}

.tab-line {
    position: relative;
    width: 100%;
    top: -16.9px;
    color: $button-color;
    opacity: 0.4;
    z-index: 0;
}

.report {
    position: relative;
    z-index: 1;
}

.selected {
    border-bottom-color: #fff;
}

.report:hover {
    background: #fff !important;
    color: $button-color !important;
    border-bottom-color: #fff;
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

/* Enhanced section styling */
.section-title {
    font-weight: 600;
    color: #343a40;
    font-size: 1.1rem;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

.section-divider {
    margin: 1.5rem auto;
    width: 50%;
    border-top: 2px solid #e9ecef;
}

#reuse-classification {
    padding: 1.25rem;
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
</style>
