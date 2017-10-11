<template>
    <div class="card rounded-0 mt-4 shadow-1">
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
                <ul class="nav nav-tabs" id="myTab" role="tablist">
                    <li class="nav-item">
                        <a class="nav-link active" id="search-alignments-tab" data-toggle="tab" href="#search-alignments" role="tab" aria-controls="search-alignments" aria-expanded="true">Search Alignments</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" id="graph-tab" data-toggle="tab" href="#graph" role="tab" aria-controls="graph" aria-expanded="true">Display Network Graph</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" id="stats-tab" data-toggle="tab" href="#global-stats" role="tab" aria-controls="global-stats" aria-expanded="true">Display Global Stats</a>
                    </li>
                </ul>
                <div class="tab-content mt-3" id="myTabContent">
                    <div class="tab-pane fade show active" id="search-alignments" role="tabpanel" aria-labelledby="search-alignments-tab">
                        <button class="btn btn-primary rounded-0" type="submit">Search</button>
                        <button type="button" class="btn btn-secondary rounded-0" @click="clearForm()">Reset</button>
                    </div>
                    <div class="tab-pane fade" id="graph" role="tabpanel" aria-labelledby="graph-tab">
                        <div class="row">
                            <div class="col-1">
                                Source metadata:
                            </div>
                            <div class="col-2">
                                <div class="my-dropdown">
                                    <button type="button" class="btn btn-light rounded-0" @click="toggleDropdown()">{{ formGraphValues.source.label }} &#9662;</button>
                                    <ul class="my-dropdown-menu shadow-1">
                                        <li class="my-dropdown-item" v-for="field in globalConfig.metadataFields.source" :key="field.label" @click="selectItem('source', field)">{{ field.label }}</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                        <div class="row mt-3">
                            <div class="col-1">
                                Target metadata:
                            </div>
                            <div class="col-2">
                                <div class="my-dropdown">
                                    <button type="button" class="btn btn-light rounded-0" @click="toggleDropdown()">{{ formGraphValues.target.label }} &#9662;</button>
                                    <ul class="my-dropdown-menu shadow-1">
                                        <li class="my-dropdown-item" v-for="field in globalConfig.metadataFields.target" :key="field.label" @click="selectItem('target', field)">{{ field.label }}</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                        <div class="mt-3">
                            <button class="btn btn-primary rounded-0" type="button" @click="getGraphData()">Display network graph</button>
                            <button type="button" class="btn btn-secondary rounded-0" @click="clearForm()">Reset</button>
                        </div>
                    </div>
                    <div class="tab-pane fade show" id="global-stats" role="tabpanel" aria-labelledby="global-stats-tab">
                        <div class="row">
                            <div class="col-1">
                                Display stats for:
                            </div>
                            <div class="col-2">
                                <div class="my-dropdown">
                                    <button type="button" class="btn btn-light rounded-0" @click="toggleDropdown()">{{ formStatsValues.selected.label }} &#9662;</button>
                                    <ul class="my-dropdown-menu shadow-1">
                                        <h6 class="dropdown-header">Source</h6>
                                        <li class="my-dropdown-item" v-for="field in globalConfig.metadataFields.source" :key="field.label" @click="selectStatsItem('Source', field)">{{ field.label }}</li>
                                        <div class="dropdown-divider"></div>
                                        <h6 class="dropdown-header">Target</h6>
                                        <li class="my-dropdown-item" v-for="field in globalConfig.metadataFields.target" :key="field.label" @click="selectStatsItem('Target', field)">{{ field.label }}</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                        <div class="mt-3">
                            <button class="btn btn-primary rounded-0" type="button" @click="getStats()">Display stats</button>
                            <button type="button" class="btn btn-secondary rounded-0" @click="clearForm()">Reset</button>
                        </div>
                    </div>
                </div>
            </form>
        </div>
    </div>
</template>

<script>
import { EventBus } from '../main.js';

export default {
    name: "search-form",
    data: function() {
        return {
            globalConfig: this.$globalConfig,
            formValues: this.populateSearchForm(),
            formGraphValues: {
                source: this.$globalConfig.metadataFields.source[0],
                target: this.$globalConfig.metadataFields.target[0]
            },
            formStatsValues: {
                values: [...this.$globalConfig.metadataFields.source, ...this.$globalConfig.metadataFields.target],
                selected: {
                    label: `Source ${this.$globalConfig.metadataFields.source[0].label}`,
                    value: this.$globalConfig.metadataFields.source[0].value,
                    direction: "source"
                }
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
            return formValues
        },
        submitForm() {
            this.$router.push(`/search?${this.paramsToUrl(this.formValues)}`)
        },
        clearForm() {
            for (const key in this.formValues) {
                this.formValues[key] = ""
            }
        },
        toggleDropdown() {
            let element = event.srcElement.closest(".my-dropdown").querySelector("ul")
            if (element.style.display != "table") {
                element.style.display = "table"
            } else {
                element.style.display = "none"
            }
        },
        selectGraphItem(key, item) {
            this.formGraphValues[key] = item
            this.toggleDropdown()
        },
        getGraphData() {
            let params = { ...this.formValues, source: this.formGraphValues.source.value, target: this.formGraphValues.target.value }
            this.$router.push(`/graph?${this.paramsToUrl(params)}`)
        },
        selectStatsItem(direction, item) {
            this.formStatsValues.selected.label = `${direction} ${item.label}`
            this.formStatsValues.selected.value = item.value
            this.formStatsValues.direction = direction.toLowerCase()
            this.toggleDropdown()
        },
        getStats() {
            let params = { ...this.formValues, stats_field: this.formStatsValues.selected.value, direction: this.formStatsValues.selected.direction }
            this.$router.push(`/stats?${this.paramsToUrl(params)}`)
        },
    }
}
</script>

<style>
.input-group>span {
    font-size: 0.85rem !important;
}

.my-dropdown {
    display: inline-block;
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
    width: 100%;
    line-height: 200%;
    z-index: 5;
    list-style-type: none;
    margin: 0;
    padding: 0;
}

.my-dropdown-item {
    cursor: pointer;
    padding-left: .75rem;
}

.my-dropdown-item:hover {
    background: #ddd;
}
</style>

