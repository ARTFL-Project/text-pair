import {
    diff_match_patch
} from "./diffMatchPatch.js";

addEventListener("message", (event) => {
    let diff = new diff_match_patch();
    diff.Diff_Timeout = 0;
    let diffs = diff.diff_main(event.data[0], event.data[1]);
    diff.diff_cleanupSemantic(diffs);
    postMessage(diffs);
});