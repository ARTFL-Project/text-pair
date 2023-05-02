import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import { fileURLToPath, URL } from "node:url";

export default defineConfig({
    plugins: [vue()],
    resolve: {
        alias: {
            "@": fileURLToPath(new URL("./src", import.meta.url)),
        },
        // TODO: Remove by explicitely adding extension in imports
        extensions: [".js", ".json", ".vue"],
    },
    base: process.env.NODE_ENV === "production" ? getAppPath() : "/",
    server: {
        cors: true,
    },
});

function getAppPath() {
    const globalConfig = require("./appConfig.json");
    console.log(globalConfig.appPath);
    return globalConfig.appPath;
}
