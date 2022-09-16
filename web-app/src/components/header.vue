<template>
    <nav class="navbar navbar-expand-lg navbar-light bg-light" style="height: 53px" v-once>
        <div class="collapse navbar-collapse top-links">
            <ul class="navbar-nav me-auto mb-2 mb-lg-0">
                <li class="nav-item">
                    <a class="nav-link" href="https://artfl-project.uchicago.edu">ARTFL Project</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="https://textual-optics-lab.uchicago.edu">Textual Optics Lab</a>
                </li>
            </ul>
        </div>
        <router-link class="navbar-brand" to="/" v-html="globalConfig.databaseLabel"></router-link>
        <button
            class="navbar-toggler"
            type="button"
            data-toggle="collapse"
            data-target="#navbarNav"
            aria-controls="navbarNav"
            aria-expanded="false"
            aria-label="Toggle navigation"
        >
            <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="navbarNav">
            <ul class="navbar-nav">
                <li class="nav-item" v-for="brand in globalConfig.branding" :key="brand.label">
                    <a class="nav-link" :href="brand.link">{{ brand.label }}</a>
                </li>
            </ul>
        </div>
    </nav>
</template>

<script>
export default {
    name: "main-header",
    data: function () {
        return {
            globalConfig: this.$globalConfig,
            home: this.$route.fullPath.replace(this.$route.query.path, ""),
        };
    },
};
</script>

<style>
.top-links {
    margin-left: -0.25rem;
    font-size: 80%;
    margin-top: -2rem;
    font-variant: small-caps;
}
.navbar-brand {
    font-weight: 700;
    font-size: 1.6rem !important;
    font-variant: small-caps;
    position: absolute;
    left: 50%;
    transform: translateX(-50%);
    line-height: 80%;
}
</style>
