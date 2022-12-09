<template>
    <div id="search-form" class="card rounded-0 mt-3 shadow-1">
        <div class="card-body rounded-0" style="position: relative">
            <h6 id="show-form" class="p-2 hide" @click="toggleSearchForm()">Show search form</h6>
            <form @submit.prevent @keyup.enter="searchSelected()">
                <div class="row">
                    <div class="col">
                        <h6 class="text-center pb-2" v-html="globalConfig.sourceLabel"></h6>
                    </div>
                    <div class="col border border-top-0 border-right-0 border-bottom-0">
                        <h6 class="text-center pb-2" v-html="globalConfig.targetLabel"></h6>
                    </div>
                </div>

                <!-- Metadata Fields -->
                <div class="row">
                    <div class="col">
                        <div
                            class="input-group pb-3"
                            v-for="field in globalConfig.metadataFields.source"
                            :key="field.label"
                        >
                            <span class="input-group-prepend">
                                <span class="input-group-text rounded-0">{{ field.label }}</span>
                            </span>
                            <input
                                type="text"
                                class="form-control rounded-0"
                                :name="field.value"
                                v-model="formValues[field.value]"
                            />
                        </div>
                    </div>
                    <div class="col border border-top-0 border-right-0 border-bottom-0">
                        <div
                            class="input-group pb-3"
                            v-for="field in globalConfig.metadataFields.target"
                            :key="field.label"
                        >
                            <span class="input-group-prepend">
                                <span class="input-group-text rounded-0">{{ field.label }}</span>
                            </span>
                            <input
                                type="text"
                                class="form-control rounded-0"
                                :name="field.value"
                                v-model="formValues[field.value]"
                            />
                        </div>
                    </div>
                </div>

                <!-- Banality filter -->
                <div id="banality-filter" class="my-dropdown mb-3" v-if="banalitiesStored">
                    <button type="button" class="btn btn-outline-secondary rounded-0" @click="toggleDropdown()">
                        {{ banalitySelected }}&nbsp;&nbsp;&#9662;
                    </button>
                    <ul class="my-dropdown-menu shadow-1">
                        <li
                            class="my-dropdown-item"
                            v-for="(option, optionIndex) in formBanalityValues"
                            :key="optionIndex"
                            @click="banalitySelect(optionIndex)"
                        >
                            {{ option.label }}
                        </li>
                    </ul>
                </div>

                <!-- Search reports -->
                <div class="position-relative mb-3" style="width: 100%" aria-label="Search reports">
                    <button
                        class="report btn rounded-0 d-inline"
                        :class="{ 'btn-secondary': timeActive, 'btn-outline-secondary selected': searchActive }"
                        type="button"
                        @click="changeReport('search')"
                    >
                        Search Alignments
                    </button>
                    <button
                        class="report btn rounded-0 d-inline"
                        :class="{
                            'btn-secondary selected': searchActive,
                            'btn-outline-secondary selected': timeActive,
                        }"
                        style="border-left-width: 0"
                        type="button"
                        @click="changeReport('time-series')"
                    >
                        Display Time Series
                    </button>
                    <hr class="tab-line" />
                </div>

                <div id="search-alignments" v-if="searchActive">
                    <button class="btn btn-secondary rounded-0" type="button" @click="search()">Search</button>
                    <button type="button" class="btn btn-outline-secondary rounded-0" @click="clearForm()">
                        Reset
                    </button>
                </div>
                <div id="time-series" v-if="timeActive">
                    Group
                    <div class="my-dropdown" style="display: inline-block">
                        <button type="button" class="btn btn-light rounded-0" @click="toggleDropdown()">
                            {{ directionSelected.label }} &#9662;
                        </button>
                        <ul class="my-dropdown-menu shadow-1">
                            <li
                                class="my-dropdown-item"
                                v-for="direction in directions"
                                :key="direction.label"
                                @click="selectItem('directionSelected', direction)"
                            >
                                {{ direction.label }}
                            </li>
                        </ul>
                    </div>
                    results by
                    <div class="my-dropdown" style="display: inline-block">
                        <button type="button" class="btn btn-light rounded-0" @click="toggleDropdown()">
                            {{ timeSeriesInterval.label }} &#9662;
                        </button>
                        <ul class="my-dropdown-menu shadow-1">
                            <li
                                class="my-dropdown-item text-nowrap"
                                v-for="interval in globalConfig.timeSeriesIntervals"
                                :key="interval.label"
                                @click="selectItem('timeSeriesInterval', interval)"
                            >
                                {{ interval.label }}
                            </li>
                        </ul>
                    </div>
                    <div class="mt-3">
                        <button class="btn btn-secondary rounded-0" type="button" @click="displayTimeSeries()">
                            Display Time Series
                        </button>
                        <button type="button" class="btn btn-outline-secondary rounded-0" @click="clearForm()">
                            Reset
                        </button>
                    </div>
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
            timeSeriesInterval: this.$globalConfig.timeSeriesIntervals[0],
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
            directions: [
                {
                    label: this.$globalConfig.sourceLabel,
                    value: "source",
                },
                {
                    label: this.$globalConfig.targetLabel,
                    value: "target",
                },
            ],
            directionSelected: {
                label: this.$globalConfig.sourceLabel,
                value: "source",
            },
            searchSelected: this.search,
            searchActive: true,
            timeActive: false,
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
            if (this.$route.path == "/search") {
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
            this.toggleDropdown();
        },
        search() {
            this.toggleSearchForm();
            this.$router.push(`/search?${this.paramsToUrl(this.formValues)}`);
        },
        displayTimeSeries() {
            this.toggleSearchForm();
            this.$router.push(`/time?${this.paramsToUrl(this.formValues)}`);
        },
        clearForm() {
            for (const key in this.formValues) {
                this.formValues[key] = "";
            }
        },
        toggleDropdown() {
            let element = event.srcElement.closest(".my-dropdown").querySelector("ul");
            if (element.style.display != "inline-block") {
                element.style.display = "inline-block";
            } else {
                element.style.display = "none";
            }
        },
        selectItem(key, item) {
            this[key] = item;
            this.formValues[key] = item.value;
            this.toggleDropdown();
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
        changeReport(report) {
            if (report == "search") {
                this.searchActive = true;
                this.timeActive = false;
            } else {
                this.searchActive = false;
                this.timeActive = true;
            }
        },
    },
};
</script>

<style lang="scss" scoped>
@import "../assets/theme.module.scss";

#search-form {
    font-family: "Open-Sans", sans-serif;
}
.input-group > span {
    font-size: 0.85rem !important;
}

.input-group
    > :not(:first-child):not(.dropdown-menu):not(.valid-tooltip):not(.valid-feedback):not(.invalid-tooltip):not(.invalid-feedback) {
    border-top-left-radius: 0 !important;
    border-bottom-left-radius: 0 !important;
    border-left-color: $link-color;
}

.my-dropdown {
    position: relative;
}

.my-dropdown .btn:focus,
my-dropdown .btn:active {
    outline: none !important;
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

#search-form > div > form > div:nth-child(1) > div.col.border.border-top-0.border-right-0.border-bottom-0,
#search-form > div > form > div:nth-child(2) > div.col.border.border-top-0.border-right-0.border-bottom-0 {
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
</style>

