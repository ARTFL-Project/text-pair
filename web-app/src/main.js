import axios from "axios";
import emitter from "tiny-emitter/instance";
import { createApp } from "vue";
import globalConfig from "../appConfig.json";
import App from "./App.vue";
import router from "./router";

const app = createApp(App);
app.config.globalProperties.$globalConfig = globalConfig;
app.config.globalProperties.emitter = emitter;
app.provide("$http", axios);

app.mixin({
    methods: {
        paramsToUrl: function (params) {
            const urlParams = [];
            for (const [key, value] of Object.entries(params)) {
                if (value !== "" && value !== null && value !== undefined) {
                    if (Array.isArray(value)) {
                        // Handle array values (multiple parameters with same name)
                        value.forEach((val) => {
                            if (val) {
                                urlParams.push(
                                    `${encodeURIComponent(
                                        key
                                    )}=${encodeURIComponent(val)}`
                                );
                            }
                        });
                    } else {
                        urlParams.push(
                            `${encodeURIComponent(key)}=${encodeURIComponent(
                                value
                            )}`
                        );
                    }
                }
            }
            return urlParams.join("&");
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
