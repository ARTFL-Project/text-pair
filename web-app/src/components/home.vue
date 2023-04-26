<template>
    <div class="card mt-4 shadow-1" style="font-size: 1rem;">

        <div class="card-body">
            <h5 class="card-title">Some basic statistics on current alignment:</h5>
            <p class="card-text">
            <ul>
                <li>This database contains {{ stats.pairs_count.toLocaleString() }} passage pairs.</li>
                <li v-if="globalConfig.algorithm == 'sa'">These pairs were generated from {{
                    stats.group_count.toLocaleString() }} source passages
                    borrowed from
                    {{
                        stats.author_group_count.toLocaleString() }}
                    authors in {{
                        stats.title_group_count.toLocaleString() }} titles.
                </li>

            </ul>
            </p>
        </div>

    </div>
</template>
<script>
import stats from '../../stats.json'

export default {
    name: 'home',
    data() {
        return {
            stats: stats,
            globalConfig: this.$globalConfig,
        }
    }
}
</script>