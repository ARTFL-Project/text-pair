import Vue from "vue";
import Router from "vue-router";
import searchResults from "../components/searchResults";
import graphResults from "../components/graphResults";

Vue.use(Router);

export default new Router({
    mode: "history",
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
