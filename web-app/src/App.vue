<template>
    <div id="app">
        <main-header></main-header>
        <div class="container-fluid">
            <search-form></search-form>
            <router-view></router-view>
        </div>
    </div>
</template>

<script>
import mainHeader from "./components/header";
import searchForm from "./components/searchForm";

export default {
    name: "app",
    components: {
        mainHeader,
        searchForm,
    },
    data: function () {
        return {
            globalConfig: this.$globalConfig,
        };
    },
};
</script>

<style lang="scss">
@import "./assets/theme.module.scss";
@import "../node_modules/bootstrap/scss/bootstrap.scss";

body,
.btn {
    font-size: 0.9rem;
}

.passages {
    font-size: 1rem;
}

a {
    text-decoration: none;
}

#app {
    font-family: "Source Serif Pro", serif;
}

.shadow-1 {
    box-shadow: 0 2px 5px 0 rgba(0, 0, 0, 0.16), 0 2px 10px 0 rgba(0, 0, 0, 0.12);
}

.btn-light {
    border-color: #ddd;
}

.corner-btn {
    position: absolute;
    transition: all 0.1s ease-out;
    padding: 5px;
    line-height: 1;
    top: 0;
    overflow: hidden;
    border-color: #ddd;
    border-style: solid;
}

.corner-btn.right {
    right: 0;
    border-width: 0px 0px 1px 1px;
}

.corner-btn.left {
    left: 0;
    border-width: 0px 1px 1px 0px;
}

.corner-btn.destroy:hover {
    background-color: #565656;
    color: #f8f8f8;
    cursor: pointer;
}

.diff-btn,
.group-diff-btn {
    display: inline-block;
    padding: 0.2rem;
    margin-bottom: 2px;
    border: solid 1px #ddd;
    cursor: pointer;
    font-family: "Open-Sans", sans-serif;
}

.diff-btn:hover,
.group-diff-btn:hover {
    color: #565656 !important;
    background-color: #f8f8f8;
}

.my-dropdown {
    position: relative;
}

.my-dropdown .btn:focus,
my-dropdown .btn:active {
    outline: none !important;
}

.my-dropdown-menu {
    position: absolute;
    display: none;
    left: 0;
    top: 38px;
    background-color: #fff;
    width: auto;
    line-height: 200%;
    z-index: 5;
    list-style-type: none;
    margin: 0;
    padding: 0;
}

.my-dropdown-item {
    cursor: pointer;
    padding: 0 0.75rem;
}

.my-dropdown-item:hover {
    background: #ddd;
}
</style>
