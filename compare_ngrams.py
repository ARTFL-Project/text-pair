#!/usr/bin/env python3
"""Sequence alignment module"""

import argparse
import json
import os
import re
import timeit
from collections import deque, namedtuple, OrderedDict, defaultdict
from glob import glob
from xml.dom.minidom import parseString

from dicttoxml import dicttoxml
from multiprocess import Pool


IndexedNgram = namedtuple("IndexedNgram", "index, start_byte, end_byte")
NgramMatch = namedtuple("NgramMatch", "source, target, ngram")
Alignment = namedtuple("alignment", "start_byte end_byte start_index end_index")

REMOVE_ALL_TAGS = re.compile(r'<[^>]*?>')
BROKEN_BEGIN_TAG = re.compile(r'^[^<]*?>')
BROKEN_END_TAG = re.compile(r'<[^>]*?$')

MATCHING_DEFAULTS = {
    "matching_window_size": 20,
    "max_gap": 10,
    "minimum_matching_ngrams":4,
    "minimum_matching_ngrams_in_window":4,
    "common_ngrams_limit": 75,
    "percent_matching": 10
}


class SequenceAligner(object):
    """Base class for running sequence alignment tasks from ngram
    representation of text."""

    def __init__(self, source_files, target_files=None, banal_ngrams=25, minimum_matching_ngrams_in_docs=4,
                 context=300, output="tab", workers=6, debug=False, matching_algorithm="default",
                 source_metadata=None, target_metadata=None, output_path="./", in_memory=False, **matching_args):

        # Set global variables
        self.minimum_matching_ngrams_in_docs = minimum_matching_ngrams_in_docs
        self.context = context
        self.matching_algorithm = matching_algorithm
        self.in_memory = in_memory
        self.banal_ngrams = banal_ngrams

        # Set matching args
        for matching_arg, value in matching_args.items():
            setattr(self, matching_arg, value)
        for default_matching_arg, value in MATCHING_DEFAULTS.items():
            if default_matching_arg not in self:
                setattr(self, default_matching_arg, value)
        self.common_ngrams_limit /= 100

        self.source_files = []
        self.target_files = []
        self.output = output.lower()
        self.workers = workers
        self.debug = debug

        # Loading metadata
        print("Loading metadata...", end=" ")
        if source_metadata is None:
            print("Error: No source metadata file provided")
            exit()
        self.source_metadata = decode_json(source_metadata)
        for text_id, metadata in self.source_metadata.items():
            self.source_metadata[text_id] = OrderedDict(metadata)
        if target_metadata is None:
            if target_files is None:
                self.target_metadata = self.source_metadata
            else:
                print("Error: No target metadata file provided")
                exit()
        else:
            self.target_metadata = decode_json(target_metadata)
            for text_id, metadata in self.target_metadata.items():
                self.target_metadata[text_id] = OrderedDict(metadata)
        print("done")

        # Get all paths
        self.output_path = output_path
        source_files = file_arg_to_files(source_files)
        self.source_files_path = os.path.dirname(source_files[0])
        if target_files is not None:
            target_files = file_arg_to_files(target_files)
            self.target_files_path = os.path.dirname(target_files[0])
        else:
            self.target_files_path = self.source_files_path

        # Load indexes and document representations
        print("Loading document indexes...", end=" ")
        self.source_files = [os.path.basename(i).replace('.json', '') for i in source_files]
        if target_files is None:
            source_against_source = True
            target_files = self.source_files
        else:
            source_against_source = False
            target_files = [os.path.basename(i).replace('.json', '') for i in target_files]
        already_compared = set([])
        self.files_to_compare = defaultdict(list)
        for source_doc_id in self.source_files:
            for target_doc_id in target_files:
                if source_against_source:
                    if source_doc_id != target_doc_id:
                        pair = tuple(sorted([source_doc_id, target_doc_id]))
                        if pair not in already_compared:
                            self.files_to_compare[source_doc_id].append(target_doc_id)
                            already_compared.add(pair)
                else:
                    self.files_to_compare[source_doc_id].append(target_doc_id)

        if in_memory:
            self.target_files = {i: decode_json("%s/%s.json" % (self.target_files_path, i)) for i in target_files}
        else:
            self.target_files = {target_file: None for target_file in target_files}
        print("done")

    def __setattr__(self, attr, value):
        self.__dict__[attr] = value

    def __contains__(self, value):
        if hasattr(self, value):
            return True
        else:
            return False

    def compare(self):
        """Run comparison between source and target files.
        If no target is defined, it will compare source files against themselves."""
        print("\n## Running sequence alignment ##")
        start_time = timeit.default_timer()
        if self.in_memory:
            count = list(map(self.__find_matches_in_docs, self.source_files))
        else:
            pool = Pool(self.workers)
            count = list(pool.map(self.__find_matches_in_docs, self.source_files))
        print("\n## Results ##")
        print("Found a total of %d" % sum(count))
        print("Comparison took %f" % (timeit.default_timer() - start_time))

    def __find_matches_in_docs(self, source_doc_id):
        """Default matching algorithm"""
        source_json = "%s/%s.json" % (self.source_files_path, source_doc_id)
        source_ngrams = decode_json(source_json)
        source_set = set(source_ngrams)
        combined_alignments = []
        count = 0
        print("Comparing source doc %s to all target docs..." % source_doc_id)
        for target_doc_id in self.files_to_compare[source_doc_id]:
            if self.in_memory:
                target_ngrams = self.target_files[target_doc_id]
            if self.in_memory is False:
                target_ngrams = decode_json("%s/%s.json" % (self.target_files_path, target_doc_id))
            matches = []
            ngram_intersection = {ngram: len(source_ngrams[ngram] + target_ngrams[ngram]) for ngram in source_set.intersection(target_ngrams)}
            ngram_intersection = [a for a, b in sorted(ngram_intersection.items(), key=lambda x: (x[1], x[0]), reverse=True)]
            if len(ngram_intersection) < self.minimum_matching_ngrams_in_docs:
                continue
            common_ngrams = set(ngram_intersection[:self.banal_ngrams])
            append_to_matches = matches.append
            for ngram in ngram_intersection:
                for source_obj in source_ngrams[ngram]:
                    for target_obj in target_ngrams[ngram]:
                        append_to_matches(NgramMatch(source_obj, target_obj, ngram))
            if self.matching_algorithm == "default":
                alignments = self.__default_algorithm(source_doc_id, target_doc_id, matches, common_ngrams)
            else:
                alignments = self.__out_of_order_algorithm(source_doc_id, target_doc_id, matches, common_ngrams)
            combined_alignments.append((target_doc_id, alignments))
            count += len(alignments)
        if combined_alignments:
            self.__write_alignments(combined_alignments, source_doc_id)
        return count

    def __default_algorithm(self, source_doc_id, target_doc_id, matches, common_ngrams):
        """Default matching algorithm. Algorithm modeled after the following example
        # ex: [(5, 12), (5, 78), (7, 36), (7, 67), (9, 14)]"""
        if self.debug:
            debug_output = open("%s/aligner_debug_%s-%s.log" % (self.output_path, source_doc_id, target_doc_id), "w")
        matches.sort(key=lambda x: (x[0][0], x[1][0]))
        matches = deque(matches)
        alignments = []
        last_source_position = 0
        break_out = False
        in_alignment = False
        last_match = tuple()
        first_match = tuple()
        current_alignment = {}
        while matches:
            current_anchor = matches.popleft()
            while current_anchor[0][0] < last_source_position:
                try:
                    current_anchor = matches.popleft()
                except IndexError:
                    break_out = True
                    break
            if break_out:
                break
            source_anchor = current_anchor.source[0]
            last_source_position = source_anchor
            target_anchor = current_anchor.target[0]
            last_target_position = target_anchor
            in_alignment = True
            previous_source_index = source_anchor
            first_match = (current_anchor.source[1], current_anchor.target[1])
            matches_in_current_alignment = 1
            matches_in_current_window = 1
            common_ngram_matches = 0
            current_alignment = {}
            if current_anchor.ngram in common_ngrams:
                common_ngram_matches += 1
            # This holds the last match and will be added as the last element in the alignment
            last_match = (current_anchor.source, current_anchor.target)
            if self.debug:
                debug_alignment = [(current_anchor.source[0], current_anchor.target[0], current_anchor.ngram)]
            for source, target, ngram in matches:
                if source[0] == previous_source_index: # we skip source_match if the same as before
                    continue
                if target[0] <= last_target_position: # we only want targets that are after last target match
                    continue
                if source[0] > last_source_position+self.max_gap or target[0] > last_target_position+self.max_gap:
                    # print("Failed", source[0], last_source_position+self.max_gap, target[0], last_target_position+self.max_gap)
                    # print(firstep, secondstep)
                    in_alignment = False
                if source[0] > source_anchor+self.matching_window_size or target[0] > target_anchor+self.matching_window_size:
                    # should we have different numbers for source window and target window?
                    if matches_in_current_window < self.minimum_matching_ngrams_in_window:
                        in_alignment = False
                    else:
                        # Check size of gap before opening new window
                        if source[0] > last_source_position+self.max_gap or target[0] > last_target_position+self.max_gap:
                            in_alignment = False
                        else:  # open new window
                            source_anchor = source[0]
                            target_anchor = target[0]
                            matches_in_current_window = 0
                if not in_alignment:
                    if common_ngram_matches/matches_in_current_alignment < self.common_ngrams_limit:
                        if matches_in_current_alignment >= self.minimum_matching_ngrams_in_window:
                            current_alignment["source"] = Alignment(first_match[0], last_match[0][2], current_anchor.source[0], last_match[0][0])
                            current_alignment["target"] = Alignment(first_match[1], last_match[1][2], current_anchor.target[0], last_match[1][0])
                            alignments.append(current_alignment)
                        # Looking for small match within max_gap
                        elif (last_match[0][0] - current_anchor[0][0]) <= self.max_gap and matches_in_current_alignment >= self.minimum_matching_ngrams:
                            current_alignment["source"] = Alignment(first_match[0], last_match[0][2])
                            current_alignment["target"] = Alignment(first_match[1], last_match[1][2])
                            alignments.append(current_alignment)
                    last_source_position = last_match[0][0] + 1 # Make sure we start the next match at index that follows last source match
                    if self.debug:
                        self.__debug_output(debug_alignment, debug_output, matches_in_current_alignment, last_match, current_anchor, common_ngram_matches)
                    break
                last_source_position = source[0]
                last_target_position = target[0]
                previous_source_index = source[0]
                matches_in_current_window += 1
                matches_in_current_alignment += 1
                last_match = (source, target) # save last matching ngrams in case it ends the match
                if ngram in common_ngrams:
                    common_ngram_matches += 1
                if self.debug:
                    debug_alignment.append((source[0], target[0], ngram))

        # Add current alignment if not already done
        if in_alignment and matches_in_current_alignment >= self.minimum_matching_ngrams:
            current_alignment["source"] = Alignment(first_match[0], last_match[0][2])
            current_alignment["target"] = Alignment(first_match[1], last_match[1][2])
            alignments.append(current_alignment)
            if self.debug:
                self.__debug_output(debug_alignment, debug_output, matches_in_current_alignment, last_match, current_anchor, common_ngram_matches)
        if self.debug:
            debug_output.close()
        return alignments

    def __debug_output(self, debug_alignment, debug_output, matches_in_current_alignment, last_match, current_anchor, common_ngram_matches):
        if matches_in_current_alignment >= self.minimum_matching_ngrams and common_ngram_matches/matches_in_current_alignment >= self.common_ngrams_limit:
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

    def __write_alignments(self, combined_alignments, source_doc_id):
        """Write results to file"""
        with open("%s/%s_to_all.%s" % (self.output_path, source_doc_id, self.output), 'w') as output_file:
            combined_output = []
            if self.output == "tab":
                columns = list(self.source_metadata[source_doc_id].keys()) + ["source_context_before", "source_passage", "source_context_after"]
                columns.extend(list(self.target_metadata[combined_alignments[0][0]].keys()) + ["target_context_before", "target_passage", "target_context_after"])
                combined_output.append("\t".join(columns))
            for target_doc_id, alignments in combined_alignments:
                if self.output == "html":
                    combined_output.append(self.__html_output(alignments, source_doc_id, target_doc_id))
                elif self.output == "json":
                    combined_output.extend(self.__json_output(alignments, source_doc_id, target_doc_id))
                elif self.output == "tab":
                    combined_output.extend(self.__tab_output(alignments, source_doc_id, target_doc_id))
                else:
                    combined_output.extend(self.__xml_output(alignments, source_doc_id, target_doc_id))
            if self.output == "html":
                output_file.write("".join(combined_output))
            elif self.output == "tab":
                output_file.write("\n".join(combined_output))
            elif self.output == "json":
                json.dump(combined_output, output_file)
            elif self.output == "xml":
                xml = dicttoxml(combined_output)
                dom = parseString(xml)
                output_file.write(dom)

    def __html_output(self, alignments, source_doc_id, target_doc_id):
        """HTML output"""
        output_string = ""
        for alignment in alignments:
            output_string += "<h1>===================</h1>"
            output_string += """<div><button type="button">Diff alignments</button>"""
            source_context_before, source_passage, source_context_after = \
                self.__alignment_to_text(alignment["source"], self.source_metadata[source_doc_id]["filename"])
            target_context_before, target_passage, target_context_after = \
                self.__alignment_to_text(alignment["target"], self.target_metadata[target_doc_id]["filename"])
            source_print = '<p>%s <span style="color:red">%s</span> %s</p>' % (source_context_before, source_passage, source_context_after)
            target_print = '<p>%s <span style="color:red">%s</span> %s</p>' % (target_context_before, target_passage, target_context_after)
            output_string += '<h4>====== Source ======</h4>'
            output_string += '<h5>%s, (%s &gt; %s</h5>' % (self.source_metadata[source_doc_id]["title"], self.source_metadata[source_doc_id]["author"],
                                                           self.source_metadata[source_doc_id]["head"])
            output_string += source_print
            output_string += '<h4>====== Target ======</h4>'
            output_string += '<h5>%s, (%s &gt; %s</h5>' % (self.target_metadata[target_doc_id]["title"], self.target_metadata[target_doc_id]["author"],
                                                           self.target_metadata[target_doc_id]["head"])
            output_string += target_print
            output_string += "</div>"
        return output_string

    def __json_output(self, alignments, source_doc_id, target_doc_id):
        """JSON output"""
        all_alignments = []
        for alignment in alignments:
            source_context_before, source_passage, source_context_after = self.__alignment_to_text(alignment["source"],
                                                                                                   self.source_metadata[source_doc_id]["filename"])
            source = {"metadata": self.source_metadata[source_doc_id], "context_before": source_context_before, "context_after": source_context_after,
                      "matching_passage": source_passage}
            target_context_before, target_passage, target_context_after = self.__alignment_to_text(alignment["target"],
                                                                                                   self.target_metadata[target_doc_id]["filename"])
            target = {"metadata": self.target_metadata[target_doc_id], "context_before": target_context_before, "context_after": target_context_after,
                      "matching_passage": target_passage}
            all_alignments.append({"source": source, "target": target})
        return all_alignments

    def __tab_output(self, alignments, source_doc_id, target_doc_id):
        """Tab delimited output."""
        output_string = []
        for alignment in alignments:
            # print("source", alignment["source"].start_byte, alignment["source"].end_byte, alignment["source"].start_index, alignment["source"].end_index)
            # print("target", alignment["target"].start_byte, alignment["target"].end_byte, alignment["target"].start_index, alignment["target"].end_index)
            fields = list(self.source_metadata[source_doc_id].values())
            fields.extend(self.__alignment_to_text(alignment["source"], self.source_metadata[source_doc_id]["filename"]))
            fields.extend(list(self.target_metadata[target_doc_id].values()))
            fields.extend(self.__alignment_to_text(alignment["target"], self.target_metadata[target_doc_id]["filename"]))
            output_string.append("\t".join(str(i) for i in fields))
        return output_string

    def __xml_output(self, alignments, source_doc_id, target_doc_id):
        """XML output"""
        all_alignments = []
        for alignment in alignments:
            source_context_before, source_passage, source_context_after = self.__alignment_to_text(alignment["source"],
                                                                                                   self.source_metadata[source_doc_id]["filename"])
            source = {"metadata": self.source_metadata[source_doc_id], "context_before": source_context_before, "context_after": source_context_after,
                      "matching_passage": source_passage}
            target_context_before, target_passage, target_context_after = self.__alignment_to_text(alignment["target"],
                                                                                                   self.target_metadata[target_doc_id]["filename"])
            target = {"metadata": self.target_metadata[target_doc_id], "context_before": target_context_before, "context_after": target_context_after,
                      "matching_passage": target_passage}
            all_alignments.append({"source": source, "target": target})
        return all_alignments

    def __alignment_to_text(self, alignment, filename):
        """Fetches aligned passages using philo IDs.
        Returns context before match, matching passage, context after match."""
        matching_passage = self.__get_text(filename, alignment.start_byte, alignment.end_byte)
        context_before = self.__get_text(filename, alignment.start_byte-self.context, alignment.start_byte)
        context_after = self.__get_text(filename, alignment.end_byte, alignment.end_byte+self.context)
        return context_before, matching_passage, context_after

    def __get_text(self, filename, start_byte, end_byte):
        """Get text"""
        text_file = open(filename, 'rb')
        text_file.seek(start_byte)
        text = text_file.read(end_byte-start_byte)
        text = text.decode("utf8", errors="ignore")
        text = REMOVE_ALL_TAGS.sub('', text)
        text = BROKEN_BEGIN_TAG.sub('', text)
        text = BROKEN_END_TAG.sub('', text)
        text = ' '.join(text.split()).strip() # remove all tabs, newlines, and replace with spaces
        return text

def decode_json(path):
    """Unjson file"""
    with open(path, "r") as jsond_file:
        return json.load(jsond_file)

def file_arg_to_files(file_arg):
    """Interpret file argument on command line"""
    if isinstance(file_arg, list):
        return file_arg
    if file_arg.endswith('/') or os.path.isdir(file_arg):
        files = glob(file_arg + '/*')
    elif "*" in file_arg:
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
                        type=None)
    parser.add_argument("--in_memory", help="load all target files into RAM for faster processing at the cost of RAM usage",
                        action='store_true', default=False)
    parser.add_argument("--source_metadata", help="path to metadata for source files",
                        type=str, default="")
    parser.add_argument("--target_metadata", help="path to metadata for target files",
                        type=str, default=None)
    parser.add_argument("--output", help="output format: html, json (see docs for proper decoding), xml, or tab",
                        type=str, default="html")
    parser.add_argument("--output_path", help="path of output",
                        type=str, default="./")
    parser.add_argument("--debug", help="add debugging", action='store_true', default=False)
    parser.add_argument("--cores", help="define number of cores for pairwise comparisons", type=int, default=4)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    ARGS = parse_command_line()
    ALIGNER = SequenceAligner(ARGS.source_files, target_files=ARGS.target_files,
                              output=ARGS.output, debug=ARGS.debug, in_memory=ARGS.in_memory,
                              source_metadata=ARGS.source_metadata, target_metadata=ARGS.target_metadata,
                              output_path=ARGS.output_path, workers=ARGS.cores)
    ALIGNER.compare()
