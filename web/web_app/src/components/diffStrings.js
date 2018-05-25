import diff_match_patch from './diffMatchPatch.js'

onmessage = function(e) {
  let diff = new diff_match_patch()
  diff.Diff_Timeout = 0
  let diffs = diff.diff_main(e.data[0], e.data[1])
  diff.diff_cleanupSemantic(diffs)
  postMessage(diffs)
}
