<template>
    <header class="site-header navbar shadow-sm" v-once>
        <nav class="top-links" aria-label="Project links" v-if="globalConfig.branding.length">
            <ul class="navbar-nav flex-row flex-wrap">
                <li class="nav-item" v-for="brand in globalConfig.branding" :key="brand.label">
                    <a class="nav-link" :href="brand.link">{{ brand.label }}</a>
                </li>
            </ul>
        </nav>
        <h1 class="site-title">
            <router-link class="navbar-brand" to="/" v-html="globalConfig.databaseLabel"></router-link>
        </h1>
    </header>
</template>

<script setup>
import globalConfig from "../config.js";
</script>

<style>
.site-header.navbar {
    display: grid;
    grid-template-columns: 1fr minmax(auto, 60%) 1fr;
    align-items: center;
    min-height: 53px;
    padding: 0.25rem 1rem;
}

.top-links {
    grid-column: 1;
    align-self: start;
    margin-left: -0.25rem;
    font-size: 80%;
    font-variant: small-caps;
}

.top-links .nav-link {
    padding: 0 0.5rem;
}

.site-title {
    grid-column: 2;
    margin: 0;
    text-align: center;
    font-weight: 700;
    font-size: 1.6rem;
    font-variant: small-caps;
    line-height: 1.1;
}

.site-title .navbar-brand {
    margin: 0;
    padding: 0;
    font-size: inherit;
    white-space: normal;
}

@media (max-width: 767.98px) {
    .site-header.navbar {
        grid-template-columns: 1fr;
        justify-items: center;
        row-gap: 0.25rem;
    }

    .top-links,
    .site-title {
        grid-column: 1;
    }
}
</style>
