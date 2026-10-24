<template>
    <header class="site-header navbar" v-once>
        <nav class="top-links" aria-label="Project links" v-if="globalConfig.branding.length">
            <ul class="navbar-nav flex-row flex-wrap">
                <li class="nav-item" v-for="brand in globalConfig.branding" :key="brand.label">
                    <a class="nav-link" :href="brand.link">{{ brand.label }}</a>
                </li>
            </ul>
        </nav>
        <div class="site-title-block">
            <h1 class="site-title">
                <router-link class="navbar-brand" to="/" v-html="globalConfig.databaseLabel"></router-link>
            </h1>
            <!-- An earlier text joined to a later one, in light inks for the band -->
            <svg class="site-thread" viewBox="0 0 120 10" aria-hidden="true" focusable="false">
                <defs>
                    <linearGradient id="site-thread-gradient" gradientUnits="userSpaceOnUse" x1="12" x2="106" y1="0" y2="0">
                        <stop offset="0" stop-color="#e4bab3" />
                        <stop offset="1" stop-color="#b9cde3" />
                    </linearGradient>
                </defs>
                <circle cx="5" cy="5" r="3.5" fill="#e4bab3" />
                <path d="M11 5H106" stroke="url(#site-thread-gradient)" stroke-width="1.25" />
                <path d="M101.5 2.25l4.5 2.75-4.5 2.75" fill="none" stroke="#b9cde3" stroke-width="1.25" />
                <circle cx="114" cy="5" r="3" fill="none" stroke="#b9cde3" stroke-width="1.25" />
            </svg>
        </div>
    </header>
</template>

<script setup>
import globalConfig from "../config.js";
</script>

<style lang="scss">
@use "../assets/theme.module.scss" as theme;

// The maroon band: project links at the left, the database title centred
.site-header.navbar {
    display: grid;
    grid-template-columns: 1fr minmax(auto, 60%) 1fr;
    align-items: center;
    min-height: 4.5rem;
    padding: 0.6rem 1.5rem 0.55rem;
    background-color: theme.$header-color;
    box-shadow: inset 0 -1px 0 #420101;
}

.site-header :focus-visible {
    outline-color: #fff !important;
}

.top-links {
    grid-column: 1;
    align-self: center;
}

.top-links .navbar-nav {
    gap: 0.25rem;
}

.site-header .top-links .nav-link {
    @include theme.kicker;
    padding: 0.25rem 0.5rem;
    border-radius: 2px;
    color: rgba(255, 255, 255, 0.86) !important;
    font-size: 0.66rem;
}

.site-header .top-links .nav-link:hover,
.site-header .top-links .nav-link:focus {
    background-color: rgba(255, 255, 255, 0.1);
    box-shadow: none;
    color: #fff !important;
}

.site-title-block {
    grid-column: 2;
    display: flex;
    flex-direction: column;
    align-items: center;
}

.site-title {
    margin: 0;
    text-align: center;
    @include theme.display-type;
    font-weight: 500;
    font-size: 2rem;
    line-height: 1.05;
    letter-spacing: 0.01em;
    text-wrap: balance;
}

.site-title .navbar-brand {
    margin: 0;
    padding: 0 0.15em;
    font-size: inherit;
    white-space: normal;
    color: #fff !important;
}

.site-title .navbar-brand:hover,
.site-title .navbar-brand:focus {
    background-color: rgba(255, 255, 255, 0.08);
    box-shadow: none;
}

.site-thread {
    display: block;
    width: 6.5rem;
    height: 0.55rem;
    margin-top: 0.35rem;
}

@media (max-width: 767.98px) {
    .site-header.navbar {
        grid-template-columns: 1fr;
        justify-items: center;
        row-gap: 0.2rem;
        min-height: 0;
        padding: 0.45rem 0.75rem 0.6rem;
    }

    .top-links,
    .site-title-block {
        grid-column: 1;
    }

    .site-title {
        font-size: 1.6rem;
    }
}
</style>
