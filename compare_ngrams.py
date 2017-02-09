#!/usr/bin/env python3
"""Sequence alignment module"""

import argparse
import json
import os
import pickle
import re
import sys
import timeit
from ast import literal_eval
from collections import Counter, defaultdict, deque, namedtuple
from glob import glob
from operator import itemgetter
from xml.dom.minidom import parseString

from dicttoxml import dicttoxml
from multiprocess import Pool
try:
    from philologic.DB import DB
except ImportError:
    pass

IndexedNgram = namedtuple("IndexedNgram", "index, position")
NgramMatch = namedtuple("NgramMatch", "source, target, ngram_index")
DocObject = namedtuple("DocObject", "doc_id, ngram_object")

REMOVE_ALL_TAGS = re.compile(r'<[^>]*?>')
BROKEN_BEGIN_TAG = re.compile(r'^[^<]*?>')
BROKEN_END_TAG = re.compile(r'<[^>]*?$')

MATCHING_DEFAULTS = {
    "matching_window_size": 20,
    "max_gap": 10,
    "minimum_matching_ngrams":4,
    "minimum_matching_ngrams_in_window":4,
    "common_ngrams_limit": 50,
    "percent_matching": 10
}


class SequenceAligner(object):
    """Base class for running sequence alignment tasks from ngram
    representation of text."""

    def __init__(self, source_files, source_ngram_index, target_files=None, target_ngram_index=None, filtered_ngrams=50,
                 minimum_matching_ngrams_in_docs=4, context=300, output="tab", workers=4, debug=False, matching_algorithm="default",
                 source_db_path="", target_db_path="", cached=True, **matching_args):

        # Set global variables
        self.minimum_matching_ngrams_in_docs = minimum_matching_ngrams_in_docs
        self.context = context
        self.matching_algorithm = matching_algorithm
        self.cached = cached
        self.banal_ngrams = filtered_ngrams

        # Set matching args
        for matching_arg, value in matching_args.items():
            setattr(self, matching_arg, value)
        for default_matching_arg, value in MATCHING_DEFAULTS.items():
            if default_matching_arg not in self:
                setattr(self, default_matching_arg, value)
        self.common_ngrams_limit /= 100

        # Setting PhiloLogic DB paths
        if not source_db_path:
            print("Error: No source db path provided")
            exit()
        self.source_path = source_db_path
        if not target_db_path:
            if target_files is None:
                self.target_path = source_db_path
            else:
                print("Error: No target db path provided")
                exit()
        else:
            self.target_path = target_db_path

        self.source_files = []
        self.target_files = []

        self.output = output.lower()

        self.workers = workers

        self.debug = debug

        os.system("rm -rf tmp/source && mkdir -p tmp/source")
        os.system("rm -rf tmp/target && mkdir -p tmp/target")

        # Build data representation of index
        print("\n## Loading ngram index ##")
        self.ngram_index = self.__load_ngram_index(source_ngram_index, target_ngram_index)
        self.common_ngrams = set(range(0, filtered_ngrams))
        if self.debug:
            self.index_to_ngram = {value: key for key, value in self.ngram_index.items()}

        # Build document representation
        source_files = file_arg_to_files(source_files)
        self.global_doc_index = {"source": {}, "target": {}}
        self.docs_to_compare = {os.path.basename(file).replace("_ngrams.json", "").strip(): set([]) for file in source_files}
        print("\n## Indexing source docs ##")
        self.source_files = [self.__build_doc_object(f, "source") for f in source_files]
        if target_files is None:
            self.target_files = self.source_files
            for source_doc_id in self.docs_to_compare:
                for target_doc_id, ngrams in self.target_files:
                    if target_doc_id != source_doc_id:
                        if (
                                len(self.global_doc_index["source"][source_doc_id].intersection(self.global_doc_index["source"][target_doc_id]))
                                >=
                                self.minimum_matching_ngrams_in_docs
                        ):
                            self.docs_to_compare[source_doc_id].add(target_doc_id)
        else:
            print("\n## Indexing target docs ##")
            target_files = file_arg_to_files(target_files)
            self.target_files = [self.__build_doc_object(f, "target") for f in target_files]
            for source_doc_id in self.docs_to_compare:
                for target_doc_id, ngrams in self.target_files:
                    if (
                            len(self.global_doc_index["source"][source_doc_id].intersection(self.global_doc_index["target"][target_doc_id]))
                            >=
                            self.minimum_matching_ngrams_in_docs
                    ):
                        self.docs_to_compare[source_doc_id].add(target_doc_id)

        # release memory since we no longer need those
        self.ngram_index = {}
        self.global_doc_index = {}

    def __setattr__(self, attr, value):
        self.__dict__[attr] = value

    def __contains__(self, value):
        if hasattr(self, value):
            return True
        else:
            return False

    def __load_ngram_index(self, source_index, target_index):
        """Load ngram index"""
        print("Loading ngram index...")
        try:
            all_ngrams = json.load(open(source_index))
        except TypeError:
            print("Error: No JSON file (or an invalid JSON file) was provided for the ngram index.", file=sys.stderr)
            exit()
        if target_index is not None:
            target_ngrams = json.load(open(target_index))
        else:
            target_ngrams = []
        for target_ngram in target_ngrams:
            if target_ngram in all_ngrams:
                all_ngrams[target_ngram] += target_ngrams[target_ngram]
            else:
                all_ngrams[target_ngram] = target_ngrams[target_ngram]
        # Sort and eliminate hapax ngrams
        sorted_ngrams = [literal_eval(i[0]) for i in sorted(all_ngrams.items(), key=itemgetter(1), reverse=True) if i[1] > 1]
        ngram_index = {tuple(ngram): n for n, ngram in enumerate(sorted_ngrams)}
        return ngram_index

    def __build_doc_object(self, file, direction):
        """Build a file representation used for ngram comparisons"""
        doc_id = os.path.basename(file).replace("_ngrams.json", "").strip()
        print("Processing doc %s..." % doc_id)
        with open(file) as filehandle:
            ngrams = json.load(filehandle)
            doc_index = defaultdict(list)
            index_pos = 0
            for ngram_obj in ngrams:
                ngram = tuple(i[0] for i in ngram_obj)
                philo_ids = tuple(i[1] for i in ngram_obj)
                try:
                    ngram_int = self.ngram_index[ngram]
                    doc_index[ngram_int].append(IndexedNgram(index_pos, philo_ids))
                    index_pos += 1
                except KeyError:
                    pass
        self.global_doc_index[direction][doc_id] = set(doc_index)
        with open("tmp/%s/%s.pickle" % (direction, doc_id), "wb") as file_to_pickle:
            pickle.dump(doc_index, file_to_pickle, pickle.HIGHEST_PROTOCOL)
        def unpickle_file():
            """Unpickle file"""
            with open("tmp/%s/%s.pickle" % (direction, doc_id), "rb") as pickled_file:
                return pickle.load(pickled_file)
        return DocObject(doc_id, unpickle_file)

    def compare(self):
        """Run comparison between source and target files.
        If no target is defined, it will compare source files against themselves."""
        print("\n## Running sequence alignment ##")
        start_time = timeit.default_timer()
        pool = Pool(self.workers)
        count = list(pool.map(self.__find_matches_in_docs, self.source_files))
        print("\n## Results ##")
        print("Found a total of %d" % sum(count))
        print("Comparison took %f" % (timeit.default_timer() - start_time))

    def __find_matches_in_docs(self, source_file):
        """Default matching algorithm"""
        source_doc_id, source_pickle = source_file
        source_ngrams = source_pickle()
        source_set = set(source_ngrams)
        alignment_count = Counter()
        print("Comparing source doc %s to all target docs..." % source_file.doc_id)
        for target_doc_id, target_ngram_object in self.target_files:
            if target_doc_id not in self.docs_to_compare[source_doc_id]:
                if self.debug:
                    print("Skipping source doc %s to target doc %s comparison: not enough matching ngrams" % (source_doc_id, target_doc_id))
                continue
            target_ngrams = target_ngram_object()
            matches = []
            ngram_intersection = {ngram: source_ngrams[ngram] + target_ngrams[ngram] for ngram in source_set.intersection(target_ngrams)}
            common_ngrams = sorted(ngram_intersection.items(), key=lambda x: x[0], reverse=True)[:self.banal_ngrams]
            for ngram in ngram_intersection:
                for source_obj in source_ngrams[ngram]:
                    for target_obj in target_ngrams[ngram]:
                        matches.append(NgramMatch(source_obj, target_obj, ngram))
            if self.matching_algorithm == "default":
                alignments = self.__default_algorithm(source_doc_id, target_doc_id, matches, common_ngrams)
            else:
                alignments = self.__out_of_order_algorithm(source_doc_id, target_doc_id, matches, common_ngrams)
            alignment_count[source_doc_id] += len(alignments)
            if alignment_count[source_doc_id] > 0:
                self.__write_alignments(source_doc_id, target_doc_id, alignments)
        return alignment_count[source_doc_id]

    def __default_algorithm(self, source_doc_id, target_doc_id, matches, common_ngrams):
        """Default matching algorithm. Algorithm modeled after the following example
        # ex: [(5, 12), (5, 78), (7, 36), (7, 67), (9, 14)]"""
        if self.debug:
            debug_output = open("aligner_debug_%s-%s.log" % (source_doc_id, target_doc_id), "w")
        matches.sort(key=lambda x: x[0][0])
        matches = deque(matches)
        alignments = []
        last_source_position = 0
        break_out = False
        in_alignment = False
        last_match = tuple()
        while matches:
            current_anchor = matches.popleft()
            while current_anchor[0].index < last_source_position:
                try:
                    current_anchor = matches.popleft()
                except IndexError:
                    break_out = True
                    break
            if break_out:
                break
            source_anchor = current_anchor.source.index
            last_source_position = source_anchor
            target_anchor = current_anchor.target.index
            last_target_position = target_anchor
            in_alignment = True
            previous_source_index = source_anchor
            current_alignment = {"source": [current_anchor.source.position[0]], "target": [current_anchor.target.position[0]]}
            matches_in_current_alignment = 1
            matches_in_current_window = 1
            common_ngram_matches = 0
            if current_anchor.ngram_index in common_ngrams:
                common_ngram_matches += 1
            # This holds the last match and will be added as the last element in the alignment
            last_match = (current_anchor.source, current_anchor.target)
            if self.debug:
                debug_alignment = [(current_anchor.source.index, current_anchor.target.index, self.index_to_ngram[current_anchor.ngram_index])]
            for source, target, ngram in matches:
                if source.index == previous_source_index: # we skip source_match if the same as before
                    continue
                if target.index <= last_target_position: # we only want targets that are after last target match
                    continue
                if source.index > last_source_position+self.max_gap or target.index > last_target_position+self.max_gap:
                    in_alignment = False
                if source.index > source_anchor+self.matching_window_size or target.index > target_anchor+self.matching_window_size:
                    # should we have different numbers for source window and target window?
                    if matches_in_current_window < self.minimum_matching_ngrams_in_window:
                        in_alignment = False
                    else:
                        # Check size of gap before opening new window
                        if source.index > last_source_position+self.max_gap or target.index > last_target_position+self.max_gap:
                            in_alignment = False
                        else:  # open new window
                            source_anchor = source.index
                            target_anchor = target.index
                            matches_in_current_window = 0
                if not in_alignment:
                    if common_ngram_matches/matches_in_current_alignment < self.common_ngrams_limit:
                        if matches_in_current_alignment >= self.minimum_matching_ngrams_in_window:
                            current_alignment["source"].append(last_match[0].position[-1])
                            current_alignment["target"].append(last_match[1].position[-1])
                            alignments.append(current_alignment)
                        # Looking for small match within max_gap
                        elif (last_match[0].index - current_anchor[0].index) <= self.max_gap and matches_in_current_alignment >= self.minimum_matching_ngrams:
                            current_alignment["source"].append(last_match[0].position[-1])
                            current_alignment["target"].append(last_match[1].position[-1])
                            alignments.append(current_alignment)
                    last_source_position = last_match[0].index + 1 # Make sure we start the next match at index that follows last source match
                    if self.debug:
                        self.__debug_output(debug_alignment, debug_output, matches_in_current_alignment, last_match, current_anchor, common_ngram_matches)
                    break
                last_source_position = source.index
                last_target_position = target.index
                previous_source_index = source.index
                matches_in_current_window += 1
                matches_in_current_alignment += 1
                last_match = (source, target) # save last matching ngrams in case it ends the match
                if ngram in common_ngrams:
                    common_ngram_matches += 1
                if self.debug:
                    debug_alignment.append((source.index, target.index, self.index_to_ngram[ngram]))

        # Add current alignment if not already done
        if in_alignment and matches_in_current_alignment >= self.minimum_matching_ngrams:
            current_alignment["source"].append(last_match[0].position[-1])
            current_alignment["target"].append(last_match[1].position[-1])
            alignments.append(current_alignment)
            if self.debug:
                self.__debug_output(debug_alignment, debug_output, matches_in_current_alignment, last_match, current_anchor, common_ngram_matches)
        if self.debug:
            debug_output.close()
        return alignments

    def __out_of_order_algorithm(self, source_doc_id, target_doc_id, matches, common_ngrams):
        """Out of order matching algorithm does not take order of matching ngrams into account.
        Algorithm modeled after the following example
        # ex: [(5, 12), (5, 78), (7, 36), (7, 67), (9, 14)]"""
        if self.debug:
            debug_output = open("aligner_debug_%s-%s.log" % (source_doc_id, target_doc_id), "w")
        matches.sort(key=lambda x: x[0][0])
        matches = deque(matches)
        alignments = []
        last_source_position = 0
        break_out = False
        in_alignment = False
        last_match = tuple()
        current_match = []
        while matches:
            current_anchor = matches.popleft()
            while current_anchor[0].index < last_source_position:
                try:
                    current_anchor = matches.popleft()
                except IndexError:
                    break_out = True
                    break
            if break_out:
                break
            source_anchor = current_anchor.source.index
            last_source_position = source_anchor
            target_anchor = current_anchor.target.index
            last_target_position = target_anchor
            in_alignment = True
            previous_source_index = source_anchor
            matches_in_current_alignment = 1
            matches_in_current_window = 1
            last_match = current_anchor.source
            common_ngram_matches = 0
            if current_anchor.ngram_index in self.common_ngrams:
                common_ngram_matches += 1
            current_match = []
            if self.debug:
                debug_alignment = [(current_anchor.source.index, current_anchor.target.index, self.index_to_ngram[current_anchor.ngram_index])]
            for source, target, ngram in matches:
                if source.index > last_source_position+self.max_gap:
                    in_alignment = False
                if source.index > source_anchor+self.matching_window_size:
                    if (matches_in_current_window/self.matching_window_size*100) < self.percent_matching:
                        in_alignment = False
                    else:
                        # Check size of gap before opening new window
                        if source.index > last_source_position+self.max_gap:
                            in_alignment = False
                        else:  # open new window
                            source_anchor = source.index
                            target_anchor = target.index
                            matches_in_current_window = 0
                if not in_alignment:
                    current_match.sort(key=lambda x: x[1][0])
                    current_match = deque(current_match)
                    target_start = 0
                    target_end = target_start
                    source_matches = []
                    target_matches = []
                    while current_match:
                        local_anchor_match = current_match.popleft()
                        while local_anchor_match[1].index < last_target_position:
                            try:
                                local_anchor_match = current_match.popleft()
                            except IndexError:
                                break_out = True
                                break
                        if break_out:
                            break
                        target_start = local_anchor_match[1].index
                        target_end = target_start
                        local_matches = 1
                        common_ngram_matches = 0
                        if local_anchor_match[2] in self.common_ngrams:
                            common_ngram_matches += 1
                        for source, target, ngram in current_match:
                            if target.index <= (target_end+self.max_gap):
                                target_end = target.index
                                local_matches += 1
                                source_matches.append(source)
                                target_matches.append(target)
                                if ngram in self.common_ngrams:
                                    common_ngram_matches += 1
                                continue
                            break
                        if common_ngram_matches/local_matches < self.common_ngrams_limit:
                            try:
                                if (local_matches/(target_end-target_start)*100) > self.percent_matching:
                                    sorted_source = sorted(source_matches, key=lambda x: x.index)
                                    sorted_target = sorted(target_matches, key=lambda x: x.index)
                                    current_alignment = {
                                        "source": [sorted_source[0].position[0], sorted_source[-1].position[-1]],
                                        "target": [sorted_target[0].position[0], sorted_target[-1].position[-1]]
                                    }
                                    alignments.append(current_alignment)
                            except ZeroDivisionError:
                                pass
                            local_matches = 0
                            last_target_position = target_end
                    last_source_position = last_match.index + 1 # Make sure we start the next match at index that follows last source match
                    if self.debug:
                        self.__debug_output(debug_alignment, debug_output, matches_in_current_alignment, last_match, current_anchor, common_ngram_matches)
                    break
                last_source_position = source.index
                last_target_position = target.index
                previous_source_index = source.index
                matches_in_current_window += 1
                matches_in_current_alignment += 1
                last_match = source
                current_match.append((source, target, ngram))
                if ngram in self.common_ngrams:
                    common_ngram_matches += 1
                if self.debug:
                    debug_alignment.append((source.index, target.index, self.index_to_ngram[ngram]))

        # Add current alignment if not already done
        if in_alignment:
            current_match.sort(key=lambda x: x[1][0])
            current_match = deque(current_match)
            target_start = 0
            target_end = target_start
            source_matches = []
            target_matches = []
            while current_match:
                local_anchor_match = current_match.popleft()
                target_start = local_anchor_match[1].index
                target_end = target_start
                local_matches = 1
                common_ngram_matches = 0
                if local_anchor_match[2] in self.common_ngrams:
                    common_ngram_matches += 1
                for source, target, ngram in current_match:
                    if target.index <= (target_end+self.max_gap):
                        target_end = target.index
                        local_matches += 1
                        source_matches.append(source)
                        target_matches.append(target)
                        if ngram in self.common_ngrams:
                            common_ngram_matches += 1
                        continue
                    break
                if common_ngram_matches/local_matches < self.common_ngrams_limit:
                    try:
                        if (local_matches/(target_end-target_start)*100) > self.percent_matching:
                            sorted_source = sorted(source_matches, key=lambda x: x.index)
                            sorted_target = sorted(target_matches, key=lambda x: x.index)
                            current_alignment = {
                                "source": [sorted_source[0].position[0], sorted_source[-1].position[-1]],
                                "target": [sorted_target[0].position[0], sorted_target[-1].position[-1]]
                            }
                            alignments.append(current_alignment)
                    except ZeroDivisionError:
                        pass
            if self.debug:
                self.__debug_output(debug_alignment, debug_output, matches_in_current_alignment, last_match, current_anchor, common_ngram_matches)
        if self.debug:
            debug_output.close()
        return alignments

    def __debug_output(self, debug_alignment, debug_output, matches_in_current_alignment, last_match, current_anchor, common_ngram_matches):
        if common_ngram_matches/matches_in_current_alignment >= self.common_ngrams_limit:
            debug_output.write("\n\n## Failed passage ##\n")
            debug_output.write('Too many common ngrams: %d out of %d matches\n' % (common_ngram_matches, matches_in_current_alignment))
        elif matches_in_current_alignment >= self.minimum_matching_ngrams:
            debug_output.write("\n\n## Matching passage ##\n")
        elif (last_match[0].index - current_anchor[0].index) <= self.max_gap and matches_in_current_alignment >= self.minimum_matching_ngrams:
            debug_output.write("\n\n## Matching passage ##\n")
        else:
            debug_output.write("\n\n## Failed passage ##\n")
        for match in debug_alignment:
            print("%s: %s => %s" % (" ".join(match[2]), str(match[0]), str(match[1])), file=debug_output)

    def __write_alignments(self, source_doc_id, target_doc_id, alignments):
        """Write results to file"""
        source_metadata = get_metadata_from_position(source_doc_id, self.source_path)
        target_metadata = get_metadata_from_position(target_doc_id, self.target_path)
        if self.output == "html":
            self.__html_output(source_metadata, target_metadata, alignments, source_doc_id, target_doc_id)
        elif self.output == "json":
            self.__json_output(source_metadata, target_metadata, alignments, source_doc_id, target_doc_id)
        elif self.output == "tab":
            self.__tab_output(source_metadata, target_metadata, alignments, source_doc_id, target_doc_id)
        else:
            self.__xml_output(source_metadata, target_metadata, alignments, source_doc_id, target_doc_id)

    def __html_output(self, source_metadata, target_metadata, alignments, source_doc_id, target_doc_id):
        """HTML output"""
        output = open("%s-%s.html" % (source_doc_id, target_doc_id), "w")
        for alignment in alignments:
            output.write("<h1>===================</h1>")
            output.write("""<div><button type="button">Diff alignments</button>""")
            source_context_before, source_passage, source_context_after = \
                self.__alignment_to_text(alignment["source"], source_metadata["filename"], self.source_path)
            target_context_before, target_passage, target_context_after = \
                self.__alignment_to_text(alignment["target"], target_metadata["filename"], self.target_path)
            source_print = '<p>%s <span style="color:red">%s</span> %s</p>' % (source_context_before, source_passage, source_context_after)
            target_print = '<p>%s <span style="color:red">%s</span> %s</p>' % (target_context_before, target_passage, target_context_after)
            output.write('<h4>====== Source ======</h4>')
            output.write('<h5>%s, (%s) &gt; %s</h5>' % (source_metadata["title"], source_metadata["author"], source_metadata["head"]))
            output.write(source_print)
            output.write('<h4>====== Target ======</h4>')
            output.write('<h5>%s, (%s) &gt; %s</h5>' % (target_metadata["title"], target_metadata["author"], target_metadata["head"]))
            output.write(target_print)
            output.write("</div>")
        output.close()

    def __json_output(self, source_metadata, target_metadata, alignments, source_doc_id, target_doc_id):
        """JSON output"""
        output = open("%s-%s.json" % (source_doc_id, target_doc_id), "w")
        all_alignments = []
        for alignment in alignments:
            source_context_before, source_passage, source_context_after = self.__alignment_to_text(alignment["source"],
                                                                                                   source_metadata["filename"],
                                                                                                   self.source_path)
            source = {"metadata": source_metadata, "context_before": source_context_before, "context_after": source_context_after,
                      "matching_passage": source_passage}
            target_context_before, target_passage, target_context_after = self.__alignment_to_text(alignment["target"],
                                                                                                   target_metadata["filename"],
                                                                                                   self.target_path)
            target = {"metadata": target_metadata, "context_before": target_context_before, "context_after": target_context_after,
                      "matching_passage": target_passage}
            all_alignments.append({"source": source, "target": target})
        json.dump(all_alignments, output)
        output.close()

    def __tab_output(self, source_metadata, target_metadata, alignments, source_doc_id, target_doc_id):
        """Tab delimited output."""
        output = open("%s-%s.tab" % (source_doc_id, target_doc_id), "w")
        first_line = list(source_metadata.keys()) + ["source_context_before", "source_passage", "source_context_after"]
        first_line += list(target_metadata.keys()) + ["target_context_before", "target_passage", "target_context_after"]
        print("\t".join(first_line), file=output)
        for alignment in alignments:
            fields = list(source_metadata.values())
            fields.extend(self.__alignment_to_text(alignment["source"], source_metadata["filename"], self.source_path))
            fields.extend(list(target_metadata.values()))
            fields.extend(self.__alignment_to_text(alignment["target"], target_metadata["filename"], self.target_path))
            print("\t".join(str(i) for i in fields), file=output)
        output.close()

    def __xml_output(self, source_metadata, target_metadata, alignments, source_doc_id, target_doc_id):
        """XML output"""
        output = open("%s-%s.xml" % (source_doc_id, target_doc_id), "w")
        all_alignments = []
        for alignment in alignments:
            source_context_before, source_passage, source_context_after = self.__alignment_to_text(alignment["source"],
                                                                                                   source_metadata["filename"],
                                                                                                   self.source_path)
            source = {"metadata": source_metadata, "context_before": source_context_before, "context_after": source_context_after,
                      "matching_passage": source_passage}
            target_context_before, target_passage, target_context_after = self.__alignment_to_text(alignment["target"],
                                                                                                   target_metadata["filename"],
                                                                                                   self.target_path)
            target = {"metadata": target_metadata, "context_before": target_context_before, "context_after": target_context_after,
                      "matching_passage": target_passage}
            all_alignments.append({"source": source, "target": target})
        xml = dicttoxml(all_alignments)
        dom = parseString(xml)
        output.write(dom.toprettyxml())
        output.close()

    def __alignment_to_text(self, alignment, filename, path):
        """Fetches aligned passages using philo IDs.
        Returns context before match, matching passage, context after match."""
        start_byte = int(alignment[0].split()[7])
        end_byte = self.__get_end_byte(alignment[1], path)
        matching_passage = self.__get_text(filename, start_byte, end_byte, path)
        context_before = self.__get_text(filename, start_byte-self.context, start_byte, path)
        context_after = self.__get_text(filename, end_byte, end_byte+self.context, path)
        return context_before, matching_passage, context_after

    def __get_text(self, filename, start_byte, end_byte, path):
        """Get text"""
        file_path = os.path.join(path, "data/TEXT", filename)
        text_file = open(file_path, 'rb')
        text_file.seek(start_byte)
        text = text_file.read(end_byte-start_byte)
        text = text.decode("utf8", errors="ignore")
        text = REMOVE_ALL_TAGS.sub('', text)
        text = BROKEN_BEGIN_TAG.sub('', text)
        text = BROKEN_END_TAG.sub('', text)
        text = ' '.join(text.split()).strip() # remove all tabs, newlines, and replace with spaces
        return text

    def __get_end_byte(self, philo_id, path):
        """Get end byte of word given a philo ID."""
        conn = DB(os.path.join(path, "data"))
        cursor = conn.dbh.cursor()
        cursor.execute("SELECT end_byte from words where philo_id=?", (" ".join(philo_id.split()[:7]),))
        result = int(cursor.fetchone()[0])
        return result


def get_metadata_from_position(passage_id, path):
    """Pull metadata from PhiloLogic DB based on position of ngrams in file"""
    metadata = {}
    philo_db = DB(os.path.join(path, "data"), cached=False)
    text_object = philo_db[passage_id.split('_')]
    for field in philo_db.locals["metadata_fields"]:
        metadata[field] = text_object[field]
    return metadata

def file_arg_to_files(file_arg):
    """Interpret file argument on command line"""
    if isinstance(file_arg, list):
        return file_arg
    if file_arg.endswith('/') or os.path.isdir(file_arg):
        files = glob(file_arg + '/*')
    elif file_arg.endswith('*'):
        files = glob(file_arg)
    else:
        files = file_arg[:]
    return files

def parse_command_line():
    """Command line parsing function"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_files", help="path to source files from which to compare",
                        type=str)
    parser.add_argument("--target_files", help="path to target files to compare to source files",
                        type=str)
    parser.add_argument("--source_ngram_index", help="path to ngram index built from the source files",
                        type=str)
    parser.add_argument("--target_ngram_index", help="path to ngram index built from the target files",
                        type=str)
    parser.add_argument("--source_db_path", help="path to PhiloLogic4 DB built from the source files",
                        type=str, default="")
    parser.add_argument("--target_db_path", help="path to PhiloLogic4 DB built from the target files",
                        type=str, default="")
    parser.add_argument("--output", help="output format: html, json (see docs for proper decoding), xml, or tab",
                        type=str, default="html")
    parser.add_argument("--debug", help="add debugging", action='store_true', default=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    ARGS = parse_command_line()
    ALIGNER = SequenceAligner(ARGS.source_files, ARGS.source_ngram_index, target_files=ARGS.target_files,
                              target_ngram_index=ARGS.target_ngram_index, output=ARGS.output, debug=ARGS.debug,
                              source_db_path=ARGS.source_db_path, target_db_path=ARGS.target_db_path)
    ALIGNER.compare()
