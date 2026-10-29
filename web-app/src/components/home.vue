<template>
    <section class="card mt-4 shadow-1 stats" v-if="stats !== false" aria-labelledby="stats-title">
        <div class="card-body">
            <h2 id="stats-title" class="stats-title">Some basic statistics on current alignment:</h2>
            <p class="stats-counting" v-if="stats === null">Counting...</p>
            <ul class="stats-figures" v-else-if="stats">
                <li>
                    <span class="figure">{{ stats.pairs_count.toLocaleString() }}</span>
                    <span class="figure-caption">shared passages in this database</span>
                </li>
                <li v-if="stats.group_count != null">
                    <span class="figure">{{ stats.group_count.toLocaleString() }}</span>
                    <span class="figure-caption">reused passages they form</span>
                </li>
                <li v-if="stats.group_count != null && stats.author_group_count != null">
                    <span class="figure">{{ stats.author_group_count.toLocaleString() }}</span>
                    <span class="figure-caption">authors these passages are borrowed from</span>
                </li>
                <li v-if="stats.group_count != null && stats.title_group_count != null">
                    <span class="figure">{{ stats.title_group_count.toLocaleString() }}</span>
                    <span class="figure-caption">titles they come from</span>
                </li>
            </ul>
        </div>
    </section>
</template>
<script setup>
import axios from "axios";
import { ref } from "vue";
import globalConfig from "../config.js";

const stats = ref(null);

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

.figure-caption {
    display: block;
    margin-top: 0.25rem;
    color: $ink-2;
    font-family: $font-text;
    font-size: 0.95rem;
    font-style: italic;
    line-height: 1.3;
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
