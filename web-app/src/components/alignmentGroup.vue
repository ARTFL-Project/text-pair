<template>
    <div id="alignment-group" class="mt-4">
        <h5 style="text-align: center">
            <i>"{{ sourcePassage.passage }}"</i>

        </h5>
        <h6 style="text-align: center">
            <div><a href=""><span v-if="sourcePassage.author">{{
                sourcePassage.author }}, </span><i>{{
        sourcePassage.title }}</i></a></div>
        </h6>
        <div id="timeline-container" class="p-2">
            <div id="vertical-line"></div>
            <div class="timeline-dates" v-for="(date, index) in timeline" :key="index">
                <div class="year btn btn-secondary">{{ date.year }}</div>
                <div class="timeline-events card shadow-1 px-2 pt-2" v-for="(reuse, reuseIndex) in date.result"
                    :key="reuseIndex">
                    <h5 class="reuse-title" @click="showPassage">
                        <span v-if="reuse.author">{{ reuse.author }}<br /></span><i>{{
                            formatTitle(reuse.title) }}</i>
                    </h5>
                    <p class="timeline-text-content m-0 pb-2">
                        <span class="text-content">
                            {{ reuse.context_before }}
                            <span class="highlight">{{ reuse.passage }}</span> {{ reuse.context_after }}
                        </span>
                    </p>
                </div>
            </div>
        </div>
    </div>
</template>

<script>

export default {
    name: "alignmentGroup",
    inject: ["$http"],
    data() {
        return {
            loading: true,
            done: false,
            timeline: {},
            sourcePassage: {},
            passages: []
        };
    },
    created() {
        // fetch the data when the view is created and the data is
        // already being observed
        this.fetchData();
    },
    watch: {
        // call again the method if the route changes
        $route: "fetchData",
    },
    methods: {
        fetchData() {
            this.results = {};
            let params = { db_table: this.$globalConfig.databaseName };
            this.loading = true;
            this.$http
                .get(`${this.$globalConfig.apiServer}/group/${this.$route.params.groupId}/?${this.paramsToUrl(params)}`, {
                    metadata: this.$globalConfig.metadataTypes,
                })
                .then((response) => {
                    this.loading = false;
                    this.done = true;
                    this.emitter.emit("searchArgsUpdate", {});
                    this.timeline = response.data.titleList
                    this.sourcePassage = response.data.original_passage
                })
                .catch((error) => {
                    this.loading = false;
                    this.error = error.toString();
                    console.log(error);
                });
        },
        formatTitle(title) {
            if (title.length > 300) {
                let titleSplit = title.slice(0, 300).split(' ');
                title = titleSplit.slice(0, titleSplit.length - 1).join(" ") + " [...]";
            }
            return title;
        },
        showPassage(event) {
            let textPassage = event.srcElement.parentNode.parentNode.querySelector(".timeline-text-content")
            if (!textPassage.classList.contains("show")) {
                textPassage.classList.add("show")
            } else {
                textPassage.classList.remove("show")
            }

        }
    },
};
</script>

<style  lang="scss" scoped>
@import "../assets/theme.module.scss";


#timeline-container,
.timeline-dates {
    text-align: center;
    position: relative;
}

#vertical-line {
    position: absolute;
    left: 50%;
    border-left: 2px dotted $button-color;
    top: 0;
    height: 100%;
    z-index: 0;
}

.year,
.reuse-title,
.timeline-events {
    position: relative;
    z-index: 1;
}

.timeline-events {
    width: 70%;
    margin-left: auto;
    margin-right: auto;
}

.highlight,
.reuse-title {
    color: $passage-color;
    cursor: pointer;
}

.timeline-text-content {
    height: 0;
    transform: translateY(-10%);
    opacity: 0;
    transition: all 300ms ease-out;
}

.timeline-text-content.show {
    height: 100%;
    opacity: 1;
    transform: translateY(0);
}

.year {
    margin-top: 2.5rem;
    margin-bottom: 1rem;
    cursor: initial;
}
</style>