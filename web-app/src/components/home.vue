<template>
    <div class="card mt-4 shadow-1" style="font-size: 1rem;" v-if="stats !== false">

        <div class="card-body">
            <h2 class="h5 card-title">Some basic statistics on current alignment:</h2>
            <p class="card-text text-muted" v-if="stats === null">Counting...</p>
            <ul class="card-text" v-else-if="stats">
                <li>This database contains {{ stats.pairs_count.toLocaleString() }} passage pairs.</li>
                <li v-if="stats.group_count != null">These pairs were generated from {{
                    stats.group_count.toLocaleString() }} source passages<span v-if="stats.author_group_count != null">
                        borrowed from {{ stats.author_group_count.toLocaleString() }} authors</span><span
                        v-if="stats.title_group_count != null"> in {{ stats.title_group_count.toLocaleString() }}
                        titles</span>.
                </li>
            </ul>
        </div>

    </div>
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
