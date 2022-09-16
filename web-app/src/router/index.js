import { createRouter, createWebHistory } from 'vue-router'
import globalConfig from "../../appConfig.json";

const searchResults = () =>
    import ('../components/searchResults');
const timeSeries = () =>
    import ('../components/timeSeries');



const router = createRouter({
    history: createWebHistory(globalConfig.appPath),
    routes: [{
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
            return savedPosition
        } else {
            return {
                left: 0,
                top: 0
            }
        }
    }
})
export default router