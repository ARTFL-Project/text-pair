import Vue from 'vue';
import VueRouter from "vue-router";
import searchResults from "../components/searchResults";
import timeSeries from "../components/timeSeries";

import globalConfig from "../../appConfig.json";

Vue.use(VueRouter);

export default new VueRouter({
    mode: "history",
    base: globalConfig.appPath,
    routes: [
        {
            path: "/search",
            name: "searchResults",
            component: searchResults
        },
        {
            path: "/time",
            name: "timeSeries",
            component: timeSeries
        }
    ],
    scrollBehavior(to, from, savedPosition) {
        if (savedPosition) {
            return savedPosition;
        } else {
            return { x: 0, y: 0 };
        }
    }
});
