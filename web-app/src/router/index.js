import { createRouter, createWebHistory } from "vue-router";
import globalConfig from "../config.js";
import home from "../components/home";

const searchResults = () => import("../components/searchResults");
const timeSeries = () => import("../components/timeSeries");
const sortedResults = () => import("../components/sortedResults");
const alignmentGroup = () => import("../components/alignmentGroup");
const textNavigation = () => import("../components/textNavigation");
const networkGraph = () => import("../components/networkGraph");
const semanticGraph = () => import("../components/semanticGraph");

const router = createRouter({
    history: createWebHistory(globalConfig.appPath),
    routes: [
        { path: "/", name: "home", meta: { title: "Search" }, component: home },
        {
            path: "/search",
            name: "searchResults",
            meta: { title: "Shared passages" },
            component: searchResults,
        },
        {
            path: "/time",
            name: "timeSeries",
            meta: { title: "Passages in timeline" },
            component: timeSeries,
        },
        {
            path: "/sorted-results",
            name: "sortedResults",
            meta: { title: "Reused passages" },
            component: sortedResults,
        },
        {
            path: "/network",
            name: "networkGraph",
            meta: { title: "Network graph" },
            component: networkGraph,
        },
        {
            path: "/semantic",
            name: "semanticGraph",
            meta: { title: "Semantic graph" },
            component: semanticGraph,
        },
        {
            path: "/group/:groupId",
            name: "alignmentGroup",
            meta: { title: "Reuses of a passage" },
            component: alignmentGroup,
        },
        {
            path: "/text-view/",
            name: "textNavigation",
            meta: { title: "Full text" },
            component: textNavigation,
        },
    ],
    scrollBehavior(to, from, savedPosition) {
        if (savedPosition) {
            return savedPosition;
        } else if (to.name == "textNavigation" && from.name == "textNavigation" && to.query.passage !== undefined) {
            return false;
        } else {
            return {
                left: 0,
                top: 0,
            };
        }
    },
});
router.beforeEach((to) => {
    const hidden =
        (to.name === "networkGraph" && !globalConfig.showNetworkGraph) ||
        (to.name === "semanticGraph" && !globalConfig.showSemanticGraph);
    if (hidden) {
        return { name: "searchResults", query: to.query };
    }
});
export default router;
