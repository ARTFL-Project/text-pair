import { createApp } from "vue";
import App from "./App.vue";
import router from "./router";
import axios from "axios";
import emitter from "tiny-emitter/instance";
import globalConfig from "../appConfig.json";

const app = createApp(App);
app.config.globalProperties.$globalConfig = globalConfig;
app.config.globalProperties.emitter = emitter;
app.config.unwrapInjectedRef = true;
app.provide("$http", axios);

app.mixin({
    methods: {
        paramsToUrl: function (formValues) {
            var queryParams = [];
            for (var param in formValues) {
                queryParams.push(
                    `${param}=${encodeURIComponent(formValues[param])}`
                );
            }
            return queryParams.join("&");
        },
    },
});
app.directive("scroll", {
    mounted: function (el, binding) {
        el.scrollHandler = function (evt) {
            if (binding.value(evt, el)) {
                window.removeEventListener("scroll", el.scrollHandler);
            }
        };
        window.addEventListener("scroll", el.scrollHandler);
    },
    unmounted: function (el) {
        window.removeEventListener("scroll", el.scrollHandler);
    },
});
app.use(router);

router.isReady().then(() => app.mount("#app"));
