import Vue from "vue";
import Router from "vue-router";
import searchResults from "../components/searchResults";
import graphResults from "../components/graphResults";
import globalStats from "../components/globalStats";

import globalConfig from "../../appConfig.json";

Vue.use(Router);

export default new Router({
    mode: "history",
    base: globalConfig.appPath,
    routes: [
        {
            path: "/search",
            name: "searchResults",
            component: searchResults
        },
        {
            path: "/graph",
            name: "graphResults",
            component: graphResults
        },
        {
            path: "/stats",
            name: "globalStats",
            component: globalStats
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
