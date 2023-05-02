import { createRouter, createWebHistory } from "vue-router";
import globalConfig from "../../appConfig.json";
import home from "../components/home";

const searchResults = () => import("../components/searchResults");
const timeSeries = () => import("../components/timeSeries");
const sortedResults = () => import("../components/sortedResults");
const alignmentGroup = () => import("../components/alignmentGroup");

const router = createRouter({
    history: createWebHistory(globalConfig.appPath),
    routes: [
        { path: "/", name: "home", component: home },
        {
            path: "/search",
            name: "searchResults",
            component: searchResults,
        },
        {
            path: "/time",
            name: "timeSeries",
            component: timeSeries,
        },
        {
            path: "/sorted-results",
            name: "sortedResults",
            component: sortedResults,
        },
        {
            path: "/group/:groupId",
            name: "alignmentGroup",
            component: alignmentGroup,
        },
    ],
    scrollBehavior(to, from, savedPosition) {
        if (savedPosition) {
            return savedPosition;
        } else {
            return {
                left: 0,
                top: 0,
            };
        }
    },
});
export default router;
