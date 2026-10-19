import builtConfig from "../appConfig.json";

// The API serves the app with its current config; the built-in copy covers the dev server.
export default {
    showNetworkGraph: true,
    showSemanticGraph: true,
    ...(window.textpairConfig || builtConfig),
};
