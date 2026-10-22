import axios from "axios";
import emitter from "tiny-emitter/instance";
import { createApp } from "vue";
import App from "./App.vue";
import globalConfig from "./config.js";
import router from "./router";
import { paramsToUrl } from "./utils.js";

const app = createApp(App);
// For networkGraph.vue, the one component still on the Options API
app.config.globalProperties.$globalConfig = globalConfig;
app.config.globalProperties.emitter = emitter;
app.provide("$http", axios);
app.mixin({ methods: { paramsToUrl } });
app.use(router);

router.isReady().then(() => app.mount("#app"));
