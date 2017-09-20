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
                <div class="row">
                    <div class="col">
                        <button class="btn btn-primary rounded-0" type="submit">Submit</button>
                        <button type="button" class="btn btn-secondary rounded-0" v-on:click="clearForm()">Reset</button>
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
    data() {
        return {
            globalConfig: this.$globalConfig,
            formValues: this.populateSearchForm()
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
        }
    }
}
</script>

<style>
.input-group>span {
    font-size: 0.85rem !important;
}
</style>

