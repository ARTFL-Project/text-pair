<template>
    <nav class="navbar navbar-expand-lg navbar-light bg-light">
        <a class="navbar-brand" :href="`/${globalConfig.appPath}`">{{ globalConfig.databaseLabel }}</a>
        <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
            <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="navbarNav">
            <ul class="navbar-nav">
                <li class="nav-item">
                    <a class="nav-link" href="https://textual-optics-lab.uchicago.edu">Textual Optics Lab</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="https://artfl-project.uchicago.edu">ARTFL Project</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="http://obvil.paris-sorbonne.fr/">OBVIL</a>
                </li>
            </ul>
        </div>
    </nav>
</template>

<script>
export default {
    name: "main-header",
    data: function() {
        return {
            globalConfig: this.$globalConfig,
            home: this.$route.fullPath.replace(this.$route.query.path, "")
        }
    }
}
</script>

    <style>

</style>
