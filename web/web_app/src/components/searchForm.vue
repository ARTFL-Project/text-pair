<template>
    <div id="search-form" class="card rounded-0 mt-4 shadow-1">
        <h4 class="card-header text-center">
            Search {{ globalConfig.dbName }}
        </h4>
        <div class="card-body rounded-0">
            <form @submit.prevent="submitForm">
                <div class="row">
                    <div class="col">
                        <h6 class="text-center pb-2">
                            Source
                        </h6>
                    </div>
                    <div class="col border border-top-0 border-right-0 border-bottom-0">
                        <h6 class="text-center pb-2">
                            Target
                        </h6>
                    </div>
                </div>
                <div class="row">
                    <div class="col">
                        <div class="input-group rounded-0 pb-3" v-for="field in globalConfig.metadataFields.source" :key="field.label">
                            <span class="input-group-addon rounded-0">{{ field.label }}</span>
                            <input type="text" class="form-control" :name="field.value" v-model="formValues[field.value]">
                        </div>
                    </div>
                    <div class="col border border-top-0 border-right-0 border-bottom-0">
                        <div class="input-group pb-3" v-for="field in globalConfig.metadataFields.target" :key="field.label">
                            <span class="input-group-addon rounded-0">{{ field.label }}</span>
                            <input type="text" class="form-control" :name="field.value" v-model="formValues[field.value]">
                        </div>
                    </div>
                </div>
                <div class="my-dropdown mb-3">
                    <button type="button" class="btn btn-light rounded-0" @click="toggleDropdown()">{{ banalitySelected }}&nbsp;&nbsp;&#9662;</button>
                    <ul class="my-dropdown-menu shadow-1">
                        <li class="my-dropdown-item" v-for="(option, optionIndex) in formBanalityValues" :key="optionIndex" @click="banalitySelect(optionIndex)">{{ option.label }}</li>
                    </ul>
                </div>
                <ul class="nav nav-tabs" id="myTab" role="tablist">
                    <li class="nav-item">
                        <a class="nav-link active" id="search-alignments-tab" data-toggle="tab" href="#search-alignments" role="tab" aria-controls="search-alignments" aria-expanded="true">Search Alignments</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" id="time-series-tab" data-toggle="tab" href="#time-series" role="tab" aria-controls="time-series" aria-expanded="true">Display Time Series</a>
                    </li>
                </ul>
                <div class="tab-content mt-3" id="myTabContent">
                    <div class="tab-pane fade show active" id="search-alignments" role="tabpanel" aria-labelledby="search-alignments-tab">
                        <button class="btn btn-primary rounded-0" type="button" @click="search()">Search</button>
                        <button type="button" class="btn btn-secondary rounded-0" @click="clearForm()">Reset</button>
                    </div>
                    <div class="tab-pane fade" id="time-series" role="tabpanel" aria-labelledby="time-series-tab">
                        Group
                        <div class="my-dropdown" style="display: inline-block">
                            <button type="button" class="btn btn-light rounded-0" @click="toggleDropdown()">{{ directionSelected.label }} &#9662;</button>
                            <ul class="my-dropdown-menu shadow-1">
                                <li class="my-dropdown-item" v-for="direction in directions" :key="direction.label" @click="selectItem('directionSelected', direction)">{{ direction.label }}</li>
                            </ul>
                        </div>
                        results by:
                        <div class="my-dropdown" style="display: inline-block">
                            <button type="button" class="btn btn-light rounded-0" @click="toggleDropdown()">{{ timeSeriesInterval.label }} &#9662;</button>
                            <ul class="my-dropdown-menu shadow-1">
                                <li class="my-dropdown-item" v-for="interval in globalConfig.timeSeriesIntervals" :key="interval.label" @click="selectItem('timeSeriesInterval', interval)">{{ interval.label }}</li>
                            </ul>
                        </div>
                        <div class="mt-3">
                            <button class="btn btn-primary rounded-0" type="button" @click="displayTimeSeries()">Display Time Series</button>
                            <button type="button" class="btn btn-secondary rounded-0" @click="clearForm()">Reset</button>
                        </div>
                    </div>
                </div>
            </form>
        </div>
    </div>
</template>

<script>
import { EventBus } from "../main.js"

export default {
    name: "search-form",
    data: function() {
        return {
            globalConfig: this.$globalConfig,
            formValues: this.populateSearchForm(),
            timeSeriesInterval: this.$globalConfig.timeSeriesIntervals[0],
            formBanalityValues: [
                {
                    label: "Filter all banalities",
                    value: false
                },
                {
                    label: "Don't filter banalities",
                    value: ""
                },
                {
                    label: "Search only banalities",
                    value: true
                }
            ],
            banalitySelected: "Filter all banalities",
            directions: [
                {
                    label: "Source",
                    value: "source"
                },
                {
                    label: "Target",
                    value: "target"
                }
            ],
            directionSelected: {
                    label: "Source",
                    value: "source"
                }
        }
    },
    created() {
        var vm = this
        EventBus.$on("urlUpdate", updatedParams => {
            for (let key in updatedParams) {
                if (key in vm.formValues) {
                    vm.formValues[key] = updatedParams[key]
                }
            }
        })
    },
    methods: {
        populateSearchForm() {
            let formValues = {}
            if (!this.$route.query) {
                formValues.page = 1
            } else {
                for (const key of this.$globalConfig.metadataFields.source) {
                    if (key.value in this.$route.query) {
                        formValues[key.value] = this.$route.query[key.value]
                    } else {
                        formValues[key.value] = ""
                    }
                }
                for (const key of this.$globalConfig.metadataFields.target) {
                    if (key.value in this.$route.query) {
                        formValues[key.value] = this.$route.query[key.value]
                    } else {
                        formValues[key.value] = ""
                    }
                }
            }
            formValues.banality = false
            return formValues
        },
        banalitySelect(index) {
            this.formValues.banality = this.formBanalityValues[index].value
            this.banalitySelected = this.formBanalityValues[index].label
            this.toggleDropdown()
        },
        search() {
            this.$router.push(`/search?${this.paramsToUrl(this.formValues)}`)
        },
        submitForm() {
            var element = document.querySelector("myTabContent")
            console.log(element, "NOT WORKING")
        },
        displayTimeSeries() {
            this.$router.push(`/time?${this.paramsToUrl(this.formValues)}`)
        },
        clearForm() {
            for (const key in this.formValues) {
                this.formValues[key] = ""
            }
        },
        toggleDropdown() {
            let element = event.srcElement
                .closest(".my-dropdown")
                .querySelector("ul")
            if (element.style.display != "inline-block") {
                element.style.display = "inline-block"
            } else {
                element.style.display = "none"
            }
        },
        selectItem(key, item) {
            this[key] = item
            this.formValues[key] = item.value
            this.toggleDropdown()
        }
    }
}
</script>

<style>
.input-group > span {
    font-size: 0.85rem !important;
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
</style>

