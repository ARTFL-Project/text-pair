<template>
    <div id="search-form" class="card rounded-0 mt-3 shadow-1">
        <div class="card-body rounded-0" style="position: relative">
            <h6 id="show-form" class="p-2 hide" @click="toggleSearchForm()">Show search form</h6>
            <form @submit.prevent @keyup.enter="searchSelected()">
                <div class="row">
                    <div class="col">
                        <h5 class="section-title text-center pb-2" v-html="globalConfig.sourceLabel"></h5>
                    </div>
                    <div class="col border border-top-0 border-right-0 border-bottom-0">
                        <h5 class="section-title text-center pb-2" v-html="globalConfig.targetLabel"></h5>
                    </div>
                </div>

                <!-- Metadata Fields -->
                <div class="row">
                    <div class="col">
                        <div class="input-group pb-3" v-for="field in globalConfig.metadataFields.source"
                            :key="field.label">
                            <span class="input-group-prepend">
                                <span class="input-group-text rounded-0">{{ field.label }}</span>
                            </span>
                            <input type="text" class="form-control rounded-0" :name="field.value"
                                v-model="formValues[field.value]" />
                        </div>
                    </div>
                    <div class="col border border-top-0 border-right-0 border-bottom-0">
                        <div class="input-group pb-3" v-for="field in globalConfig.metadataFields.target"
                            :key="field.label">
                            <span class="input-group-prepend">
                                <span class="input-group-text rounded-0">{{ field.label }}</span>
                            </span>
                            <input type="text" class="form-control rounded-0" :name="field.value"
                                v-model="formValues[field.value]" />
                        </div>
                    </div>
                </div>
                <!-- Reuse Classification -->
                <div id="reuse-classification" class="mb-3">
                    <div class="row">
                        <div class="col-12">
                            <h5 class="section-title text-center pb-2">Context of Reuse Classification</h5>
                        </div>
                    </div>
                    <div class="row">
                        <div class="col">
                            <div class="input-group pb-3">
                                <span class="input-group-prepend">
                                    <span class="input-group-text rounded-0">First Criterion</span>
                                </span>
                                <select class="form-control rounded-0 select-with-arrow"
                                    v-model="classificationFilters[0]">
                                    <option value="">Any classification</option>
                                    <option v-for="classItem in globalConfig.reuseClassification.classes"
                                        :key="classItem.label" :value="classItem.label" :title="classItem.description">
                                        {{ classItem.label.replace(/_/g, ' ') }}
                                    </option>
                                </select>
                            </div>
                        </div>
                        <div class="col">
                            <div class="input-group pb-3">
                                <span class="input-group-prepend">
                                    <span class="input-group-text rounded-0">Second Criterion</span>
                                </span>
                                <select class="form-control rounded-0 select-with-arrow"
                                    v-model="classificationFilters[1]">
                                    <option value="">Any classification</option>
                                    <option v-for="classItem in globalConfig.reuseClassification.classes"
                                        :key="classItem.label" :value="classItem.label" :title="classItem.description">
                                        {{ classItem.label.replace(/_/g, ' ') }}
                                    </option>
                                </select>
                            </div>
                        </div>
                        <div class="col">
                            <div class="input-group pb-3">
                                <span class="input-group-prepend">
                                    <span class="input-group-text rounded-0">Third Criterion</span>
                                </span>
                                <select class="form-control rounded-0 select-with-arrow"
                                    v-model="classificationFilters[2]">
                                    <option value="">Any classification</option>
                                    <option v-for="classItem in globalConfig.reuseClassification.classes"
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
                                    <span class="input-group-prepend">
                                        <span class="input-group-text rounded-0">Banality Filter</span>
                                    </span>
                                    <select class="form-control rounded-0 select-with-arrow" v-model="banalitySelected"
                                        @change="updateBanalityValue">
                                        <option v-for="(option, optionIndex) in formBanalityValues" :key="optionIndex"
                                            :value="option.label">
                                            {{ option.label }}
                                        </option>
                                    </select>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div id="search-alignments">
                    <button class="btn btn-secondary rounded-0" type="button" @click="search()">Search</button>
                    <button type="button" class="btn btn-outline-secondary rounded-0" @click="clearForm()">
                        Reset
                    </button>
                </div>
            </form>
        </div>
    </div>
</template>

<script>
import Velocity from "velocity-animate";

export default {
    name: "search-form",
    data: function () {
        return {
            globalConfig: this.$globalConfig,
            formValues: this.populateSearchForm(),
            // Add this new array for classification filters
            classificationFilters: ["", "", ""],
            formBanalityValues: [
                {
                    label: "Don't filter banalities",
                    value: "",
                },
                {
                    label: "Filter all banalities",
                    value: false,
                },
                {
                    label: "Search only banalities",
                    value: true,
                },
            ],
            banalitySelected: "Don't filter banalities",
            directionSelected: {
                label: this.$globalConfig.sourceLabel,
                value: "source",
            },
            searchSelected: this.search,
            banalitiesStored: this.$globalConfig.banalitiesStored,
        };
    },
    created() {
        var vm = this;
        this.emitter.on("urlUpdate", (updatedParams) => {
            for (let key in updatedParams) {
                if (key in vm.formValues) {
                    vm.formValues[key] = updatedParams[key];
                }
            }
        });
        this.emitter.on("toggleSearchForm", () => {
            this.toggleSearchForm();
        });
        if ("banality" in this.$route.query) {
            if (this.$route.query.banality == "true") {
                this.banalitySelected = "Search only banalities";
            } else {
                this.banalitySelected = "Filter all banalities";
            }
        } else {
            this.banalitySelected = "Don't filter banalities";
        }
        if (this.$route.path != "/") {
            if (this.$route.path == "/search" || this.$route.path.startsWith("/group")) {
                this.searchActive = true;
                this.timeActive = false;
                this.toggleSearchForm();
            } else {
                this.searchActive = false;
                this.timeActive = true;
                this.toggleSearchForm();
            }
        }
        if (this.$route.query.directionSelected == "source") {
            vm.directionSelected = {
                label: this.$globalConfig.sourceLabel,
                value: "source",
            };
        } else if (this.$route.query.directionSelected == "target") {
            vm.directionSelected = {
                label: this.$globalConfig.targetLabel,
                value: "target",
            };
        } else {
            vm.directionSelected = {
                label: this.$globalConfig.sourceLabel,
                value: "source",
            };
        }
        if ("timeSeriesInterval" in this.$route.query) {
            for (let interval of this.$globalConfig.timeSeriesIntervals) {
                if (this.$route.query.timeSeriesInterval == interval.value) {
                    vm.timeSeriesInterval = interval;
                    break;
                }
            }
        }

        // Initialize classification filters from URL
        if (this.$route.query.classification_filter_1) {
            this.classificationFilters[0] = this.$route.query.classification_filter_1;
        }
        if (this.$route.query.classification_filter_2) {
            this.classificationFilters[1] = this.$route.query.classification_filter_2;
        }
        if (this.$route.query.classification_filter_3) {
            this.classificationFilters[2] = this.$route.query.classification_filter_3;
        }
    },
    methods: {
        populateSearchForm() {
            let formValues = {};
            if (!this.$route.query) {
                formValues.page = 1;
            } else {
                for (const key of this.$globalConfig.metadataFields.source) {
                    if (key.value in this.$route.query) {
                        formValues[key.value] = this.$route.query[key.value];
                    } else {
                        formValues[key.value] = "";
                    }
                }
                for (const key of this.$globalConfig.metadataFields.target) {
                    if (key.value in this.$route.query) {
                        formValues[key.value] = this.$route.query[key.value];
                    } else {
                        formValues[key.value] = "";
                    }
                }
            }
            formValues.banality = "";
            formValues.timeSeriesInterval = this.$globalConfig.timeSeriesIntervals[0].value;
            formValues.directionSelected = "source";
            return formValues;
        },
        banalitySelect(index) {
            this.formValues.banality = this.formBanalityValues[index].value;
            this.banalitySelected = this.formBanalityValues[index].label;
        },
        search() {
            this.toggleSearchForm();

            // Create URLSearchParams to properly handle multiple parameters with the same name
            const urlParams = new URLSearchParams();

            // Add all regular form values
            for (const [key, value] of Object.entries(this.formValues)) {
                if (value !== "" && value !== null && value !== undefined) {
                    urlParams.append(key, value);
                }
            }

            for (const [index, filter] of this.classificationFilters.entries()) {
                if (filter) {
                    urlParams.append(`classification_filter_${index + 1}`, filter);
                }
            }

            this.$router.push(`/search?${urlParams.toString()}`);
        },
        clearForm() {
            for (const key in this.formValues) {
                this.formValues[key] = "";
            }
            // Reset classification filters
            this.classificationFilters = ["", "", ""];
        },
        toggleSearchForm() {
            this.$nextTick(() => {
                let form = document.querySelector("form");
                if (form.style.display == "none") {
                    document.querySelector("#show-form").classList.toggle("hide");
                    Velocity(form, "slideDown", {
                        duration: 200,
                        easing: "ease-out",
                    });
                } else {
                    Velocity(form, "slideUp", {
                        duration: 200,
                        easing: "ease-out",
                        complete: function () {
                            document.querySelector("#show-form").classList.toggle("hide");
                        },
                    });
                }
            });
        },
        toggleDropdown() {
            let element = event.srcElement.closest(".my-dropdown").querySelector("ul");
            if (element.style.display != "inline-block") {
                element.style.display = "inline-block";
            } else {
                element.style.display = "none";
            }
        },
        updateBanalityValue() {
            // Find the matching option and set its value in formValues
            const selectedOption = this.formBanalityValues.find(option => option.label === this.banalitySelected);
            if (selectedOption) {
                this.formValues.banality = selectedOption.value;
            }
        }
    },
};
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
    border-top-left-radius: 0 !important;
    border-bottom-left-radius: 0 !important;
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
}

#show-form {
    position: absolute;
    transition: all 0.2s ease-out;
    cursor: pointer;
    top: 0;
    left: 50%;
    transform: translateX(-50%);
}

#show-form:hover {
    color: #565656;
}

#search-form>div>form>div:nth-child(1)>div.col.border.border-top-0.border-right-0.border-bottom-0,
#search-form>div>form>div:nth-child(2)>div.col.border.border-top-0.border-right-0.border-bottom-0 {
    border-right-width: 0 !important;
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
