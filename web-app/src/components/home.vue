<template>
    <section class="card mt-4 shadow-1 stats" v-if="stats !== false" aria-labelledby="stats-title">
        <div class="card-body">
            <h2 id="stats-title" class="stats-title">In this collection</h2>
            <p class="stats-counting" v-if="stats === null">Counting...</p>
            <template v-else-if="families">
                <ul class="stats-figures">
                    <li>
                        <router-link class="figure" :to="views.reused"
                            aria-describedby="stats-reused">{{ stats.group_count.toLocaleString() }}</router-link>
                        <span class="figure-caption" id="stats-reused">reused passages found in two or more documents</span>
                    </li>
                    <li v-if="stats.group_count_3plus != null">
                        <span class="figure">{{ stats.group_count_3plus.toLocaleString() }}</span>
                        <span class="figure-caption">of them found in three or more documents</span>
                    </li>
                    <li v-if="stats.document_count != null">
                        <span class="figure">{{ stats.document_count.toLocaleString() }}</span>
                        <span class="figure-caption">documents share at least one passage</span>
                    </li>
                    <li v-if="stats.author_count != null">
                        <span class="figure">{{ stats.author_count.toLocaleString() }}</span>
                        <span class="figure-caption">authors wrote these documents</span>
                    </li>
                </ul>
                <p class="stats-source">
                    Built from
                    <router-link :to="views.shared">{{ stats.pairs_count.toLocaleString() }} shared passages</router-link>:
                    pairs of documents with a passage in common.
                </p>
            </template>
            <ul class="stats-figures" v-else>
                <li>
                    <router-link class="figure" :to="views.shared"
                        aria-describedby="stats-shared">{{ stats.pairs_count.toLocaleString() }}</router-link>
                    <span class="figure-caption" id="stats-shared">shared passages in this database</span>
                </li>
            </ul>
        </div>
    </section>
</template>
<script setup>
import axios from "axios";
import { computed, ref } from "vue";
import globalConfig from "../config.js";
import { paramsToUrl } from "../utils.js";

const stats = ref(null);
// Where alignments form passage families, the figures are about the reused passages
const families = computed(() => globalConfig.hasPassageFamilies && stats.value?.group_count != null);
// As the report switcher links to the two views
const query = paramsToUrl({ db_table: globalConfig.databaseName });
const views = { reused: `/sorted-results/?${query}`, shared: `/search?${query}` };

axios
    .get(`${globalConfig.apiServer}/database_stats/`, {
        params: { db_table: globalConfig.databaseName },
    })
    .then((response) => {
        stats.value = response.data.error ? false : response.data;
    })
    .catch((error) => {
        stats.value = false;
        console.log(error);
    });
</script>
<style lang="scss" scoped>
@import "../assets/theme.module.scss";

.stats {
    border-radius: 3px;
}

.stats .card-body {
    padding: 1.25rem 1.5rem 1.5rem;
}

// Title between hairlines, in synthesised small caps: the web font has none of its own
.stats-title {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    margin: 0 0 1.25rem;
    color: $link-color;
    font-family: $font-text;
    font-size: 1.2rem;
    font-weight: 600;
    font-variant-caps: small-caps;
    letter-spacing: 0.06em;
    line-height: 1.3;
    text-align: center;
}

.stats-title::before,
.stats-title::after {
    content: "";
    flex: 1 1 0;
    border-top: 1px solid $rule-color;
}

.stats-counting {
    margin: 0;
    color: $muted;
    font-style: italic;
    text-align: center;
}

// The database's figures, as on a statistical plate
.stats-figures {
    display: grid;
    grid-auto-columns: minmax(0, 1fr);
    grid-auto-flow: column;
    margin: 0;
    padding: 0;
    list-style: none;
    text-align: center;
}

.stats-figures li {
    padding: 0.25rem 1rem;
}

.stats-figures li + li {
    border-left: 1px solid $rule-color;
}

.figure {
    display: block;
    color: $link-color;
    @include display-type;
    font-size: 2.6rem;
    font-weight: 500;
    line-height: 1.1;
}

// A figure that opens its view, the link's tint around the number only
a.figure {
    display: inline-block;
    padding: 0 0.35rem;
}

.figure-caption {
    display: block;
    margin-top: 0.25rem;
    color: $ink-2;
    font-family: $font-text;
    font-size: 0.95rem;
    font-style: italic;
    line-height: 1.3;
}

// What the figures are made from, its link set apart by weight as well as color
.stats-source {
    margin: 1.25rem 0 0;
    color: $ink-2;
    font-family: $font-text;
    font-size: 0.95rem;
    line-height: 1.4;
    text-align: center;
}

.stats-source a {
    color: $link-color;
    font-weight: 600;
}

@media (max-width: 767.98px) {
    .stats-title {
        font-size: 1.05rem;
    }

    .stats-figures {
        grid-auto-flow: row;
    }

    .stats-figures li + li {
        margin-top: 0.4rem;
        padding-top: 0.6rem;
        border-left: 0;
        border-top: 1px solid $rule-color;
    }
}
</style>
