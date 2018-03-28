import * as differ from 'diff';

onmessage = function(e) {
    let diffedResult = differ.diffChars(e.data[0], e.data[1], { ignoreCase: true })
    postMessage(diffedResult);
  }