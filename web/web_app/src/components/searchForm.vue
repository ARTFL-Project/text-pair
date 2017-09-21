<template>
    <div class="card rounded-0 mt-4 shadow-1">
        <h4 class="card-header text-center">
            Search {{ globalConfig.dbName }}
        </h4>
        <div class="card-body rounded-0">
            <form v-on:submit.prevent="submitForm">
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
                        <a class="nav-link" id="graph-tab" data-toggle="tab" href="#graph" role="tab" aria-controls="graph" aria-expanded="true">Display Network</a>
                    </li>
                </ul>
                <div class="tab-content mt-3" id="myTabContent">
                    <div class="tab-pane fade show active" id="search-alignments" role="tabpanel" aria-labelledby="search-alignments-tab">
                        <button class="btn btn-primary rounded-0" type="submit">Search</button>
                        <button type="button" class="btn btn-secondary rounded-0" v-on:click="clearForm()">Reset</button>
                    </div>
                    <div class="tab-pane fade" id="graph" role="tabpanel" aria-labelledby="graph-tab">
                        <div class="row">
                            <div class="col-1">
                                Source metadata:
                            </div>
                            <div class="col-2">
                                <div class="my-dropdown">
                                    <button class="btn btn-light rounded-0" v-on:click="toggleDropdown('source')">{{ formGraphValues.source.label }} &#9662;</button>
                                    <ul class="my-dropdown-menu shadow-1" v-if="dropdownShow.source">
                                        <li class="my-dropdown-item" v-for="field in globalConfig.metadataFields.source" :key="field.label" v-on:click="selectItem('source', field)">{{ field.label }}</li>
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
                                    <button class="btn btn-light rounded-0" v-on:click="toggleDropdown('target')">{{ formGraphValues.target.label }} &#9662;</button>
                                    <ul class="my-dropdown-menu shadow-1" v-if="dropdownShow.target">
                                        <li class="my-dropdown-item" v-for="field in globalConfig.metadataFields.target" :key="field.label" v-on:click="selectItem('target', field)">{{ field.label }}</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                        <div class="mt-3">
                            <button class="btn btn-primary rounded-0" type="button" v-on:click="getGraphData()">Display network graph</button>
                            <button type="button" class="btn btn-secondary rounded-0" v-on:click="clearForm()">Reset</button>
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
            dropdownShow: { "source": false, "target": false }
        }
    },
    created() {
        var vm = this
        EventBus.$on("urlUpdate", updatedParams => {
            for (let key in updatedParams) {
                if (key in vm.formValues && updatedParams[key] != vm.formValues[key]) {
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
                    }
                }
                for (const key of this.$globalConfig.metadataFields.target) {
                    if (key.value in this.$route.query) {
                        formValues[key.value] = this.$route.query[key.value]
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
        toggleDropdown(direction) {
            if (this.dropdownShow[direction]) {
                this.dropdownShow[direction] = false
            } else {
                this.dropdownShow[direction] = true
            }
        },
        selectItem(direction, item) {
            this.formGraphValues[direction] = item
            this.toggleDropdown(direction)
        },
        getGraphData() {
            this.$router.push(`/graph?${this.paramsToUrl({source: this.formGraphValues.source.value, target: this.formGraphValues.target.value})}`)
        }
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

