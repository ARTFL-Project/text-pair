#!/usr/bin/env python3
"""Sequence alignment module"""

import argparse
import json
import os
import sqlite3
import msgpack
import re
from ast import literal_eval
from collections import deque, namedtuple, defaultdict
from glob import glob
from operator import itemgetter

ngramObject = namedtuple("ngramObject", "ngram, position")
indexedNgram = namedtuple("indexedNgram", "index, position")
ngramMatch = namedtuple("ngramMatch", "source, target, ngram_index")
docObject = namedtuple("docObject", "doc_id, ngrams")

SourceDB = sqlite3.connect("/var/www/html/philologic/racine/data/toms.db")
SourceDB.row_factory = sqlite3.Row
TargetDB = sqlite3.connect("/var/www/html/philologic/littre/data/toms.db")
TargetDB.row_factory = sqlite3.Row
SourcePath = "/var/www/html/philologic/racine/data/"
TargetPath = "/var/www/html/philologic/littre/data/"

RemoveAllTags = re.compile(r'<[^>]*?>')
# BrokenBeginTag = re.compile(r'^[^>]*?>')
# BrokenEndTag = re.compile(r'<[^>]*?$')


class SequenceAligner(object):
    """Base class for running sequence alignment tasks from ngram
    representation of text."""

    def __init__(self, source_ngram_index, target_ngram_index=None, filtered_ngrams=100,
                 minimum_matching_ngrams_in_docs=2, matching_window=30, max_gap=15,
                 minimum_matching_ngrams=2, minimum_matching_ngrams_in_window=3, context=300,
                 output="html", debug=False):
        # Set global variables
        self.filtered_ngrams = filtered_ngrams
        self.minimum_matching_ngrams_in_docs = minimum_matching_ngrams_in_docs
        self.minimum_matching_ngrams_in_window = minimum_matching_ngrams_in_window
        self.minimum_matching_ngrams = minimum_matching_ngrams
        self.matching_window = matching_window
        self.max_grap = max_gap
        self.context = context
        self.source_path = SourcePath
        self.target_path = TargetPath

        self.source_files = []
        self.target_files = []

        self.output = output.lower()

        self.alignment_count = defaultdict(int)

        self.debug = debug

        # Build data representations of index
        self.ngram_index, self.index_to_ngram = self.__load_ngram_index(source_ngram_index, target_ngram_index)
        if not self.ngram_index:
            self.filtered_ngrams = 0

    def compare(self, source_files, target_files=None, matching_algorithm="default"):
        """Run comparison between source and target files.
        If no target is defined, it will compare source files against themselves."""
        # We only build the target files as we compare source files to target files
        # one by one.
        if target_files is None:
            source_files = file_arg_to_files(source_files)
            self.target_files = self.build_doc_object(source_files)
            for source_file in self.target_files:
                self.__find_matches_in_docs(source_file)
        else:
            target_files = file_arg_to_files(target_files)
            self.target_files = self.build_doc_object(target_files)
            source_files = file_arg_to_files(source_files)
            for source_file in source_files:
                source_file = self.build_doc_object(source_file)
                self.__find_matches_in_docs(source_file[0])
        print("Found a total of %d" % sum([i for i in self.alignment_count.values()]))

    def build_doc_object(self, files):
        """Build a file representation used for ngram comparisons"""
        if isinstance(files, str):
            files = [files]
        files_with_ngrams = []
        for file in files:
            with open(file) as fh:
                ngrams = json.load(fh)
                doc_index = defaultdict(list)
                index_pos = 0
                for ngram_obj in ngrams:
                    ngram = tuple(i[0] for i in ngram_obj)
                    philo_ids = tuple(i[1] for i in ngram_obj)
                    if ngram in self.ngram_index:
                        ngram_int = self.ngram_index[ngram]
                        doc_index[ngram_int].append(indexedNgram(index_pos, philo_ids))
                        index_pos += 1
                print("%s:" % file, index_pos+1, "ngrams remaining out of", len(ngrams))
            doc_id = os.path.basename(file).replace("_ngrams.json", "").strip()
            files_with_ngrams.append(docObject(doc_id, doc_index))
        return files_with_ngrams

    def __load_ngram_index(self, source_index, target_index):
        """Load ngram index"""
        print("Loading ngram index...")
        try:
            all_ngrams = json.load(open(source_index))
        except TypeError:
            print("Error: No JSON file (or an invalid JSON file) was provided for the ngram index.")
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
        sorted_ngrams = [literal_eval(i[0]) for i in sorted(all_ngrams.items(), key=itemgetter(1), reverse=True)]
        ngram_index = {tuple(ngram): n for n, ngram in enumerate(sorted_ngrams[self.filtered_ngrams:])}
        index_to_ngram = {value: key for key, value in ngram_index.items()}
        return ngram_index, index_to_ngram

    def __find_matches_in_docs(self, source_file):
        """Default matching algorithm"""
        source_set = set(source_file.ngrams)
        print("Comparing doc ID %s to all" % source_file.doc_id)
        for target_doc_id, target_ngrams in self.target_files:
            matches = []
            target_set = set(target_ngrams)
            ngram_intersection = source_set.intersection(target_set)
            if len(ngram_intersection) < self.minimum_matching_ngrams_in_docs:
                continue
            if self.debug:
                debug_output = open("aligner_debug_%s-%s.log" % (source_file.doc_id, target_doc_id), "w")
            for ngram in ngram_intersection:
                for source_obj in source_file.ngrams[ngram]:
                    for target_obj in target_ngrams[ngram]:
                        matches.append(ngramMatch(source_obj, target_obj, ngram))

            matches.sort(key=lambda x: x[0][0])
            matches = deque(matches)
            # I need to model the algorithm after the following example
            # ex: [(5, 12), (5, 78), (7, 36), (7, 67), (9, 14)]
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
                # This holds the last match and will be added as the last element in the alignment
                last_match = (current_anchor.source, current_anchor.target)
                if self.debug:
                    debug_alignment = [(current_anchor.source.index, current_anchor.target.index, self.index_to_ngram[current_anchor.ngram_index])]
                for source, target, ngram in matches:
                    if source.index == previous_source_index: # we skip source_match if the same as before
                        continue
                    if target.index <= last_target_position: # we only want targets that are after last target match
                        continue
                    if source.index > last_source_position+self.max_grap or target.index > last_target_position+self.max_grap:
                        in_alignment = False
                    if source.index > source_anchor+self.matching_window or target.index > target_anchor+self.matching_window:
                        # should we have different numbers for source window and target window?
                        if matches_in_current_window < self.minimum_matching_ngrams_in_window:
                            in_alignment = False
                        else:
                            # Check size of gap before opening new window
                            if source.index > last_source_position+self.max_grap or target.index > last_target_position+self.max_grap:
                                in_alignment = False
                            else:  # open new window
                                source_anchor = source.index
                                target_anchor = target.index
                                matches_in_current_window = 0
                    if not in_alignment:
                        if matches_in_current_alignment >= self.minimum_matching_ngrams_in_window:
                            current_alignment["source"].append(last_match[0].position[-1])
                            current_alignment["target"].append(last_match[1].position[-1])
                            alignments.append(current_alignment)
                        # Looking for small match within max_gap
                        elif (last_match[0].index - current_anchor[0].index) <= self.max_grap and matches_in_current_alignment >= self.minimum_matching_ngrams:
                            current_alignment["source"].append(last_match[0].position[-1])
                            current_alignment["target"].append(last_match[1].position[-1])
                            alignments.append(current_alignment)
                        last_source_position = last_match[0].index + 1 # Make sure we start the next match at index that follows last source match
                        if self.debug:
                            if matches_in_current_alignment >= self.minimum_matching_ngrams:
                                debug_output.write("\n\n## Matching passage ##\n")
                            else:
                                debug_output.write("\n\n## Failed passage ##\n")
                            for match in debug_alignment:
                                print("%s: %s => %s" % (" ".join(match[2]), str(match[0]), str(match[1])), file=debug_output)
                        break
                    last_source_position = source.index
                    last_target_position = target.index
                    previous_source_index = source.index
                    matches_in_current_window += 1
                    matches_in_current_alignment += 1
                    last_match = (source, target) # save last matching ngrams in case it ends the match
                    if self.debug:
                        debug_alignment.append((source.index, target.index, self.index_to_ngram[ngram]))

            # Add current alignment if not already done
            if in_alignment and matches_in_current_alignment >= self.minimum_matching_ngrams:
                current_alignment["source"].append(last_match[0].position[-1])
                current_alignment["target"].append(last_match[1].position[-1])
                alignments.append(current_alignment)
                if self.debug:
                    debug_output.write("\n\n## Matching passage ##\n")
                    for match in debug_alignment:
                        print("%s: %s => %s" % (" ".join(match[2]), str(match[0]), str(match[1])), file=debug_output)
            if self.debug:
                debug_output.close()
            self.alignment_count[source_file.doc_id] += len(alignments)
            self.__write_alignments(source_file.doc_id, target_doc_id, alignments)
        print("Found %d alignments" % self.alignment_count[source_file.doc_id])

    def __write_alignments(self, source_doc_id, target_doc_id, alignments):
        """Write results to file"""
        metadata = {}
        metadata["source"] = get_metadata_from_position(source_doc_id, provenance="source")
        metadata["target"] = get_metadata_from_position(target_doc_id, provenance="target")
        if self.output == "html":
            self.__html_output(metadata, alignments, source_doc_id, target_doc_id)
        elif self.output == "json":
            self.__json_output(metadata, alignments, source_doc_id, target_doc_id)
        elif self.output == "tab":
            self.__tab_output(metadata, alignments, source_doc_id, target_doc_id)

    def __html_output(self, metadata, alignments, source_doc_id, target_doc_id):
        """HTML output"""
        output = open("%s-%s.html" % (source_doc_id, target_doc_id), "w")
        for alignment in alignments:
            output.write("<div><h1>===================</h1>")
            source_context_before, source_passage, source_context_after = self.__alignment_to_text(alignment["source"], metadata["source"]["filename"], self.source_path)
            target_context_before, target_passage, target_context_after = self.__alignment_to_text(alignment["target"], metadata["target"]["filename"], self.target_path)
            source_print = '<p>%s <span style="color:red">%s</span> %s</p>' % (source_context_before, source_passage, source_context_after)
            target_print = '<p>%s <span style="color:red">%s</span> %s</p>' % (target_context_before, target_passage, target_context_after)
            output.write('<h4>====== Source ======</h4>')
            output.write('<h5>%s, (%s)</h5>' % (metadata["source"]["title"], metadata["source"]["author"]))
            output.write(source_print)
            output.write('<h4>====== Target ======</h4>')
            output.write('<h5>%s, (%s)</h5>' % (metadata["target"]["title"], metadata["target"]["author"]))
            output.write(target_print)
            output.write("</div>")
        output.close()

    def __json_output(self, metadata, alignments, source_doc_id, target_doc_id):
        """JSON output"""
        output = open("%s-%s.json" % (source_doc_id, target_doc_id), "w")
        all_alignments = []
        for alignment in alignments:
            source_context_before, source_passage, source_context_after = self.__alignment_to_text(alignment["source"],
                                                                                                   metadata["source"]["filename"],
                                                                                                   self.source_path)
            source = {"metadata": metadata["source"], "context_before": source_context_before, "context_after": source_context_after,
                      "matching_passage": source_passage}
            target_context_before, target_passage, target_context_after = self.__alignment_to_text(alignment["target"],
                                                                                                   metadata["target"]["filename"],
                                                                                                   self.target_path)
            target = {"metadata": metadata["target"], "context_before": target_context_before, "context_after": target_context_after,
                      "matching_passage": target_passage}
            all_alignments.append({"source": source, "target": target})
        json.dump(all_alignments, output)
        output.close()

    def __tab_output(self, metadata, alignments, source_doc_id, target_doc_id):
        """Tab delimited output."""
        output = open("%s-%s.tab" % (source_doc_id, target_doc_id), "w")
        first_line = list(metadata["source"].keys()) + ["source_context_before", "source_passage", "source_context_after"]
        first_line += list(metadata["target"].keys()) + ["target_context_before", "target_passage", "target_context_after"]
        print("\t".join(first_line), file=output)
        for alignment in alignments:
            fields = list(metadata["source"].values())
            fields.extend(self.__alignment_to_text(alignment["source"], metadata["source"]["filename"], self.source_path))
            fields.extend(list(metadata["target"].values()))
            fields.extend(self.__alignment_to_text(alignment["target"], metadata["target"]["filename"], self.target_path))
            print("\t".join(str(i) for i in fields), file=output)
        output.close()

    def __xml_output(self, metadata, alignments, source_doc_id, target_doc_id):
        """XML output"""
        pass

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
        file_path = os.path.join(path, "TEXT", filename)
        text_file = open(file_path, encoding="utf8", errors="ignore")
        text_file.seek(start_byte)
        text = text_file.read(end_byte-start_byte)
        # text = BrokenBeginTag.sub('', text)
        # text = BrokenEndTag.sub('', text)
        if self.output != "json":
            text = RemoveAllTags.sub('', text)
            text = ' '.join(text.split()) # remove all tabs, newlines, and replace with spaces
        return text

    def __get_end_byte(self, philo_id, path):
        """Get end byte of word given a philo ID."""
        conn = sqlite3.connect(os.path.join(path, "toms.db"))
        cursor = conn.cursor()
        cursor.execute("SELECT end_byte from words where philo_id=?", (" ".join(philo_id.split()[:7]),))
        result = int(cursor.fetchone()[0])
        return result


def get_metadata_from_position(position, provenance="source"):
    """Pull metadata from PhiloLogic DB based on position of ngrams in file"""
    position = position.split()[0] + ' 0 0 0 0 0 0'
    if provenance == "source":
        cursor = SourceDB.cursor()
        cursor.execute("select * from toms where philo_id=? limit 1", (position,))
    else:
        cursor = TargetDB.cursor()
        cursor.execute("select * from toms where philo_id=? limit 1", (position,))
    results = cursor.fetchone()
    metadata = {}
    for field in results.keys():
        metadata[field] = results[field] or ""
    return metadata

def file_arg_to_files(file_arg):
    """Interpret file argument on command line"""
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
    parser.add_argument("--output", help="output format: html, json (see docs for proper decoding), xml, or tab", default="html",
                        type=str)
    parser.add_argument("--debug", help="add debugging", action='store_true', default=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_command_line()
    aligner = SequenceAligner(source_ngram_index=args.source_ngram_index, target_ngram_index=args.target_ngram_index,
                              output=args.output, debug=args.debug)
    aligner.compare(args.source_files, args.target_files)
