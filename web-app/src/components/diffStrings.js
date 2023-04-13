import DiffMatchPatch from "diff-match-patch";
addEventListener("message", (event) => {
    let diff = new DiffMatchPatch();
    diff.Diff_Timeout = 0;
    let diffs = diff.diff_main(event.data[0], event.data[1]);
    diff.diff_cleanupSemantic(diffs);
    postMessage(diffs);
});
