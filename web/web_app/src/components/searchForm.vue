<template>
    <div id="search-form" class="card rounded-0 mt-3 shadow-1">
        <div class="card-body rounded-0" style="position:relative">
            <h5 id="show-form" class="p-2 hide" @click="toggleSearchForm()">
                Show search form
            </h5>
            <form @submit.prevent @keyup.enter="searchSelected()">
                <div class="row">
                    <div class="col">
                        <h6 class="text-center pb-2" v-html="globalConfig.sourceLabel"></h6>
                    </div>
                    <div class="col border border-top-0 border-right-0 border-bottom-0">
                        <h6 class="text-center pb-2" v-html="globalConfig.targetLabel"></h6>
                    </div>
                </div>
                <div class="row">
                    <div class="col">
                        <div class="input-group pb-3" v-for="field in globalConfig.metadataFields.source" :key="field.label">
                            <span class="input-group-prepend">
                                <span class="input-group-text rounded-0">{{ field.label }}</span>
                            </span>
                            <input type="text" class="form-control rounded-0" :name="field.value" v-model="formValues[field.value]">
                        </div>
                    </div>
                    <div class="col border border-top-0 border-right-0 border-bottom-0">
                        <div class="input-group pb-3" v-for="field in globalConfig.metadataFields.target" :key="field.label">
                            <span class="input-group-prepend">
                                <span class="input-group-text rounded-0">{{ field.label }}</span>
                            </span>
                            <input type="text" class="form-control rounded-0" :name="field.value" v-model="formValues[field.value]">
                        </div>
                    </div>
                </div>
                <div id="banality-filter" class="my-dropdown mb-3">
                    <button type="button" class="btn btn-light rounded-0" @click="toggleDropdown()">{{ banalitySelected }}&nbsp;&nbsp;&#9662;</button>
                    <ul class="my-dropdown-menu shadow-1">
                        <li class="my-dropdown-item" v-for="(option, optionIndex) in formBanalityValues" :key="optionIndex" @click="banalitySelect(optionIndex)">{{ option.label }}</li>
                    </ul>
                </div>
                <ul class="nav nav-tabs" id="myTab" role="tablist">
                    <li class="nav-item">
                        <a class="nav-link active" id="search-alignments-tab" data-toggle="tab" href="#search-alignments" role="tab" aria-controls="search-alignments" aria-expanded="true" @click="searchSelected = search">Search Alignments</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" id="time-series-tab" data-toggle="tab" href="#time-series" role="tab" aria-controls="time-series" aria-expanded="true" @click="searchSelected = displayTimeSeries">Display Time Series</a>
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
                        results by
                        <div class="my-dropdown" style="display: inline-block">
                            <button type="button" class="btn btn-light rounded-0" @click="toggleDropdown()">{{ timeSeriesInterval.label }} &#9662;</button>
                            <ul class="my-dropdown-menu shadow-1">
                                <li class="my-dropdown-item text-nowrap" v-for="interval in globalConfig.timeSeriesIntervals" :key="interval.label" @click="selectItem('timeSeriesInterval', interval)">{{ interval.label }}</li>
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
                    label: "Don't filter banalities",
                    value: ""
                },
                {
                    label: "Filter all banalities",
                    value: false
                },
                {
                    label: "Search only banalities",
                    value: true
                }
            ],
            banalitySelected: "Don't filter banalities",
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
                },
            searchSelected: this.search
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
            console.log(updatedParams)
        })
        EventBus.$on("toggleSearchForm", function() {
            vm.toggleSearchForm()
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
            formValues.banality = ""
            formValues.timeSeriesInterval = this.$globalConfig.timeSeriesIntervals[0].value
            formValues.directionSelected = "source"
            return formValues
        },
        banalitySelect(index) {
            this.formValues.banality = this.formBanalityValues[index].value
            this.banalitySelected = this.formBanalityValues[index].label
            this.toggleDropdown()
        },
        search() {
            this.toggleSearchForm()
            this.$router.push(`/search?${this.paramsToUrl(this.formValues)}`)
        },
        displayTimeSeries() {
            this.toggleSearchForm()
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
        },
        toggleSearchForm() {
            let form = document.querySelector("form")
            if (form.style.display == 'none') {
                document.querySelector("#show-form").classList.toggle("hide")
                Velocity(form, "slideDown", {
                duration: 200,
                easing: "ease-out",
            })
            } else {
                Velocity(form, "slideUp", {
                duration: 200,
                easing: "ease-out",
                complete: function() {
                    document.querySelector("#show-form").classList.toggle("hide")
                }
            })
            }
        }
    }
}
</script>

<style scoped>
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

.input-group-text, .form-control {
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
    transition: all .2s ease-out;
    cursor: pointer;
    top: 0;
    left: 50%;
    transform: translateX(-50%);
}

#show-form:hover {
    color: #565656;
}
</style>

