#!/usr/bin env python3


import os
from collections import defaultdict
from math import floor

import rapidjson as json
from multiprocess import Pool
from namedlist import namedlist

docIndex = namedlist("docIndex", "doc_id, ngrams, ngram_length")
indexedNgram = namedlist("indexedNgram", "n_index, start_byte, end_byte")
ngramMatch = namedlist("ngramMatch", "source target ngram")
matchingParams = namedlist(
    "matchingParams",
    """matching_window_size, max_gap, flex_gap, minimum_matching_ngrams, minimum_matching_ngrams_in_window,
    common_ngrams_limit, minimum_matching_ngrams_in_docs, context_size, banal_ngrams, merge_on_byte_distance,
    merge_on_ngram_distance, passage_distance_multiplier, duplicate_threshold, source_batch, target_batch,
    output_path, num_workers, sorting_field, debug""",
)
position = namedlist("position", [("start_byte", 0), ("end_byte", 0), ("start_ngram_index", 0), ("end_ngram_index", 0)])
Alignment = namedlist(
    "Alignment", [("source", position()), ("target", position()), ("total_matching_ngrams", 0), ("banality", False)]
)
matchValues = namedlist(
    "matchValues",
    [
        ("in_alignment", False),
        ("matches_in_current_alignment", 0),
        ("matches_in_current_window", 0),
        ("source_anchor", 0),
        ("last_source_position", 0),
        ("target_anchor", 0),
        ("last_target_position", 0),
        ("previous_source_index", 0),
        ("common_ngram_matches", 0),
        ("max_source_gap", 0),
        ("max_target_gap", 0),
        ("source_window_boundary", 0),
        ("target_window_boundary", 0),
        ("current_alignment", Alignment()),
        ("previous_alignment", None),
        ("first_match", None),
        ("last_match", None),
        ("debug", False),
    ],
)
alignmentsPerDoc = namedlist("alignmentsPerDoc", "doc_id, matches, duplicates")
CombinedAlignments = namedlist("CombinedAlignments", "source_id, alignments")


class compareNgrams:
    def __init__(
        self,
        source_files,
        source_metadata,
        target_files=None,
        target_metadata=None,
        output_path="./output",
        workers=4,
        sort_field="year",
        source_batch=1,
        target_batch=1,
        source_common_ngrams=None,
        target_common_ngrams=None,
        most_common_ngram_threshold=5000,
        common_ngrams_limit=20,
        matching_window_size=30,
        max_gap=15,
        flex_gap=True,
        minimum_matching_ngrams=4,
        minimum_matching_ngrams_in_window=4,
        minimum_matching_ngrams_in_doc=4,
        context_size=300,
        banal_ngrams=25,
        duplicate_threshold=50,
        merge_on_byte_distance=True,
        merge_on_ngram_distance=True,
        passage_distance=0.5,
        ngram_index=None,
        debug=False,
    ):
        self.debug = debug
        self.config = matchingParams(
            matching_window_size,
            max_gap,
            flex_gap,
            minimum_matching_ngrams,
            minimum_matching_ngrams_in_window,
            common_ngrams_limit / 100,
            minimum_matching_ngrams_in_doc,
            context_size,
            banal_ngrams,
            merge_on_byte_distance,
            merge_on_ngram_distance,
            passage_distance,
            duplicate_threshold,
            source_batch,
            target_batch,
            output_path,
            workers,
            sort_field,
            debug,
        )
        if self.debug is True and ngram_index is not None:
            self.ngram_index = ngram_index  # TODO
        self.source_metadata = self.__load_metadata(source_metadata)
        self.source_files = self.__load_file_paths(source_files, self.source_metadata)
        if target_files is not None:
            self.target_metadata = self.__load_metadata(target_metadata)
            self.target_files = self.__load_file_paths(target_files, self.target_metadata)
        else:
            self.target_files = []
            self.target_metadata = {}
        self.most_common_ngrams = self.__compile_most_common_ngrams(
            source_common_ngrams, target_common_ngrams, most_common_ngram_threshold
        )

    def __load_metadata(self, file_location):
        if not os.path.exists(file_location):
            print(f"No metadata file found at this location {file_location}")
            exit()
        with open(file_location) as metadata_file:
            metadata = json.load(metadata_file)
        return metadata

    def __load_file_paths(self, file_path, metadata):
        if not os.path.isdir(file_path):
            print(f"{file_path} is not a directory")
            exit()

        def sort_by(file):
            doc = os.path.basename(file).replace(".json", "")
            return (metadata[doc][self.config.sorting_field], file)

        file_paths = sorted([file.path for file in os.scandir(file_path)], key=sort_by)
        return file_paths

    def __compile_most_common_ngrams(self, source_common_ngrams, target_common_ngrams, most_common_ngram_threshold):
        unique_ngrams = set()
        for filename in [source_common_ngrams, target_common_ngrams]:
            if filename is None:
                continue
            with open(filename) as input_file:
                for pos, line in enumerate(input_file):
                    if pos == most_common_ngram_threshold:
                        break
                    unique_ngrams.add(int(line.strip()))
        return unique_ngrams

    def __get_json_docs(self, files, prefix_string):
        json_files = []
        total = len(files)

        def get_file(filepath):
            with open(filepath) as input_file:
                json_file = json.load(input_file)
                doc = defaultdict(list)
                for key, values in json_file.items():
                    for value in values:
                        doc[key].append(indexedNgram(value[0], value[1], value[2]))
                doc_id = os.path.basename(filepath).replace(".json", "")
            return docIndex(doc_id, doc, len(doc))

        with Pool(self.config.num_workers) as pool:
            running_total = 0
            for file_index in pool.imap_unordered(get_file, files):
                json_files.append(file_index)
                running_total += 1
                print(f"\r{prefix_string}... {running_total}/{total}", end="")
        return json_files

    def __load_ngram_index(self, filepath):
        ngram_index = {}
        with open(filepath) as input_file:
            for line in input_file:
                try:
                    ngram, ngram_int = line.strip().split()
                    ngram_index[ngram] = int(ngram_int)
                except Exception as e:
                    pass
        return ngram_index

    def align_passages(self):
        source_against_source = False
        if self.config.source_batch > len(self.source_files):
            self.config.source_batch = len(self.source_files)
        source_file_batches = self.__create_batches(self.source_files, self.config.source_batch)
        target_file_batches = []
        if len(self.target_files) == 0:
            self.target_metadata = self.source_metadata
            source_against_source = True
            target_file_batches = source_file_batches
            self.config.target_batch = self.config.source_batch
        else:
            if self.config.target_batch > len(self.target_files):
                self.config.target_batch = len(self.target_files)
            target_file_batches = self.__create_batches(self.target_files, self.config.target_batch)

        self.__save_alignment_params()

        duplicate_file_output = open(os.path.join(self.config.output_path, "duplicate_files.txt"), "w")
        print("## Duplicates of source files in target files ##", file=duplicate_file_output)

        counts = 0
        with open(os.path.join(self.config.output_path, "alignment.results"), "w") as alignment_output:
            for source_batch_number in range(0, self.config.source_batch):
                prefix_string = "Loading source files"
                if self.config.source_batch > 1:
                    prefix_string += f" from source batch {source_batch_number + 1}"
                    print(f"\n### Comparing source batch {source_batch_number + 1} against all... ###\n")
                source_file_indexes = self.__get_json_docs(source_file_batches[source_batch_number], prefix_string)
                for target_batch_number in range(0, self.config.target_batch):
                    if source_against_source is True and source_batch_number > target_batch_number:
                        continue  # we've already done these comparisons in the other direction
                    if source_against_source is True and target_batch_number == source_batch_number:
                        target_file_indexes = source_file_indexes
                    else:
                        target_prefix = "Loading target files"
                        if self.config.target_batch > 1:
                            target_prefix += f" from target batch {target_batch_number + 1}"
                        target_file_indexes = self.__get_json_docs(
                            target_file_batches[target_batch_number], target_prefix
                        )
                    local_source_files_done = {}
                    percent_steps = self.__build_percent_map(len(source_file_indexes))
                    print("\n\rComparing files... 0%", end="")
                    for pos, source_file in enumerate(source_file_indexes):
                        if self.config.debug is True:
                            pass
                        if pos in percent_steps:
                            print(f"\rComparing files... {percent_steps[pos]}%", end="")
                        local_alignments = []
                        for target_file in target_file_indexes:
                            if source_against_source is True:
                                if source_file.doc_id == target_file.doc_id:
                                    continue
                                elif target_file.doc_id in local_source_files_done:
                                    continue
                            source_target_intersection, total_common_ngrams = self.__get_intersection(
                                source_file, target_file
                            )
                            if len(source_target_intersection) < self.config.minimum_matching_ngrams_in_docs:
                                continue
                            elif total_common_ngrams / source_file.ngram_length * 100 > self.config.duplicate_threshold:
                                continue  # TODO
                            most_common_ngrams = self.__get_most_common_ngrams(source_target_intersection)
                            matches = []
                            for ngram in source_target_intersection:
                                for source_ngram_index in source_file.ngrams[ngram]:
                                    for target_ngram_index in target_file.ngrams[ngram]:
                                        matches.append(ngramMatch(source_ngram_index, target_ngram_index, ngram))
                            matches.sort(key=lambda x: (x.source.n_index, x.target.n_index))
                            alignments = self.__match_passages(matches, most_common_ngrams)
                            if (
                                self.config.merge_on_byte_distance is True
                                or self.config.merge_on_ngram_distance is True
                            ):
                                alignments = self.__merge_with_previous(alignments)
                            if alignments:
                                local_alignments.append(alignmentsPerDoc(target_file.doc_id, alignments, []))
                        if local_alignments:
                            for alignments in local_alignments:
                                full_alignment = {}
                                for key, value in self.source_metadata[source_file.doc_id].items():
                                    full_alignment[f"source_{key}"] = value
                                for key, value in self.target_metadata[alignments.doc_id].items():
                                    full_alignment[f"target_{key}"] = value
                                full_alignment["source_doc_id"] = source_file.doc_id
                                full_alignment["target_doc_id"] = alignments.doc_id
                                for alignment in alignments.matches:
                                    local_alignment = full_alignment
                                    local_alignment["source_start_byte"] = str(alignment.source.start_byte)
                                    local_alignment["source_end_byte"] = str(alignment.source.end_byte)
                                    source_passages = self.__alignment_to_text(
                                        alignment.source, self.source_metadata[source_file.doc_id]["filename"]
                                    )
                                    local_alignment["source_context_before"] = source_passages[0]
                                    local_alignment["source_passage"] = source_passages[1]
                                    local_alignment["source_context_after"] = source_passages[2]
                                    local_alignment["target_start_byte"] = str(alignment.target.start_byte)
                                    local_alignment["target_end_byte"] = str(alignment.target.end_byte)
                                    target_passages = self.__alignment_to_text(
                                        alignment.target, self.target_metadata[alignments.doc_id]["filename"]
                                    )
                                    local_alignment["target_context_before"] = target_passages[0]
                                    local_alignment["target_passage"] = target_passages[1]
                                    local_alignment["target_context_after"] = target_passages[2]
                                    local_alignment["banality"] = alignment.banality
                                    counts += 1
                                    local_alignment["passage_id"] = str(counts)
                                    print(json.dumps(local_alignment), file=alignment_output)
                        if source_against_source is True and source_batch_number == target_batch_number:
                            local_source_files_done[source_file.doc_id] = True
        print(f"\n{counts} alignments found...")

    def __create_batches(self, files, batch):
        list_of_list = []
        length = len(files)
        chunk_size = floor((length + batch - 1) / batch)
        for i in range(0, length, chunk_size):
            end = i + chunk_size
            if end > length:
                end = length
            list_of_list.append(files[i:end])
        return list_of_list

    def __save_alignment_params(self):
        with open(os.path.join(self.config.output_path, "alignment_config.ini"), "w") as output_file:
            for key, value in self.config.__dict__.items():
                print(f"{key}: {value}", file=output_file)

    def __build_percent_map(self, total):
        percent_steps = {}
        count = 0
        step = total / 100
        for i in (x * step for x in range(0, 100)):
            count += 1
            percent_steps[floor(i)] = count
        return percent_steps

    def __get_intersection(self, source_file, target_file):
        intersect_count = {}
        total_common_ngrams = 0
        if source_file.ngram_length < target_file.ngram_length:
            for ngram in source_file.ngrams:
                if ngram in target_file.ngrams:
                    intersect_count[ngram] = len(source_file.ngrams[ngram]) + len(target_file.ngrams[ngram])
                    total_common_ngrams += 1
        else:
            for ngram in target_file.ngrams:
                if ngram in source_file.ngrams:
                    intersect_count[ngram] = len(source_file.ngrams[ngram]) + len(target_file.ngrams[ngram])
                    total_common_ngrams += 1
        return intersect_count, total_common_ngrams

    def __get_most_common_ngrams(self, intersect_count):
        sorted_intersection = sorted(intersect_count.items(), key=lambda x: x[1], reverse=True)
        most_common_ngrams = set()
        count = 0
        for key, value in sorted_intersection:
            if value == 2:
                break
            most_common_ngrams.add(key)
            count += 1
            if count == self.config.banal_ngrams:
                break
        for common_ngram in self.most_common_ngrams:
            most_common_ngrams.add(common_ngram)
        return most_common_ngrams

    def __match_passages(self, matches, most_common_ngrams):
        alignments = []
        m = matchValues()
        for match_index, current_anchor in enumerate(matches):
            if current_anchor.source.n_index < m.last_source_position:
                continue
            m.source_anchor = current_anchor.source.n_index
            m.source_window_boundary = m.source_anchor + self.config.matching_window_size
            m.last_source_position = m.source_anchor
            m.max_source_gap = m.last_source_position + self.config.max_gap
            m.target_anchor = current_anchor.target.n_index
            m.target_window_boundary = m.target_anchor + self.config.matching_window_size
            m.last_target_position = m.target_anchor
            m.max_target_gap = m.last_target_position + self.config.max_gap
            m.in_alignment = True
            m.previous_source_index = m.source_anchor
            m.first_match = [current_anchor.source, current_anchor.target]
            m.matches_in_current_alignment = 1
            m.matches_in_current_window = 1
            m.common_ngram_matches = 0
            if current_anchor.ngram in most_common_ngrams:
                m.common_ngram_matches += 1
            m.last_match = [current_anchor.source, current_anchor.target]
            if self.config.debug is True:
                pass  # TODO
            current_matches_length = len(matches)
            max_gap = self.config.max_gap
            matching_window_size = self.config.matching_window_size
            for pos, match in enumerate(matches[match_index + 1 :]):
                source, target = match.source, match.target
                if source.n_index == m.previous_source_index:
                    continue
                if target.n_index > m.max_target_gap or target.n_index <= m.last_target_position:
                    next_index = pos + match_index + 1
                    if next_index <= current_matches_length and matches[next_index].source.n_index <= m.max_source_gap:
                        continue
                    else:
                        m.in_alignment = False
                if (
                    source.n_index > m.max_source_gap
                    and m.matches_in_current_window < self.config.minimum_matching_ngrams_in_window
                ):
                    m.in_alignment = False
                if source.n_index > m.source_window_boundary or target.n_index > m.target_window_boundary:
                    if m.matches_in_current_window < self.config.minimum_matching_ngrams_in_window:
                        m.in_alignment = False
                    else:
                        if source.n_index > m.max_source_gap or target.n_index > m.max_target_gap:
                            m.in_alignment = False
                        else:
                            m.source_anchor = source.n_index
                            m.source_window_boundary = m.source_anchor + matching_window_size
                            m.target_anchor = target.n_index
                            m.target_window_boundary = m.target_anchor + matching_window_size
                            m.matches_in_current_window = 0
                if m.in_alignment is False:
                    if m.matches_in_current_alignment >= self.config.minimum_matching_ngrams:
                        m, alignments = self.__add_alignment(m, alignments)
                        if self.config.debug is True:
                            pass  # TODO
                    elif self.config.debug is True:
                        pass  # TODO
                    m.last_source_position = m.last_match[0].n_index + 1
                    break
                m.last_source_position = source.n_index
                m.max_source_gap = m.last_source_position + max_gap
                m.last_target_position = target.n_index
                m.max_target_gap = m.last_target_position + max_gap
                m.previous_source_index = source.n_index
                m.matches_in_current_window += 1
                m.matches_in_current_alignment += 1
                if self.config.flex_gap is True:
                    if m.matches_in_current_alignment == self.config.minimum_matching_ngrams:
                        max_gap += self.config.minimum_matching_ngrams
                        matching_window_size += self.config.minimum_matching_ngrams
                    elif m.matches_in_current_alignment > self.config.minimum_matching_ngrams:
                        if max_gap < self.config.matching_window_size:
                            max_gap += 1
                            matching_window_size += 1
                m.last_match = [source, target]
                if match.ngram in most_common_ngrams:
                    m.common_ngram_matches += 1
                if self.config.debug is True:
                    pass  # TODO
            if m.in_alignment is True and m.matches_in_current_alignment >= self.config.minimum_matching_ngrams:
                m, alignments = self.__add_alignment(m, alignments)
        return alignments

    def __add_alignment(self, m, alignments):
        m.current_alignment.source = position(
            m.first_match[0].start_byte, m.last_match[0].end_byte, m.first_match[0].n_index, m.last_match[0].n_index
        )
        m.current_alignment.target = position(
            m.first_match[1].start_byte, m.last_match[1].end_byte, m.first_match[1].n_index, m.last_match[1].n_index
        )
        m.current_alignment.total_matching_ngrams = m.matches_in_current_alignment
        if m.common_ngram_matches / m.matches_in_current_alignment >= self.config.common_ngrams_limit:
            m.current_alignment.banality = True
        alignments.append(m.current_alignment)
        m.previous_alignment = m.current_alignment
        return m, alignments

    def __merge_with_previous(self, alignments):
        max_source_distance = 0
        max_target_distance = 0
        if self.config.merge_on_ngram_distance is True:
            max_ngram_distance = self.config.matching_window_size
        else:
            max_ngram_distance = 0
        merged_alignments = []
        previous_alignment = None
        last_index = len(alignments) - 1
        for index, current_alignment in enumerate(alignments):
            if index == 0:
                previous_alignment = current_alignment
                continue
            current_alignment_merged = False
            if self.config.merge_on_byte_distance is True:
                distance_value = floor(
                    (previous_alignment.source.end_byte - previous_alignment.source.start_byte)
                    * self.config.passage_distance_multiplier
                )
                max_source_distance = previous_alignment.source.end_byte + distance_value
                max_target_distance = previous_alignment.target.end_byte + distance_value
            source_ngram_distance = previous_alignment.source.end_ngram_index + max_ngram_distance
            target_ngram_distance = previous_alignment.target.end_ngram_index + max_ngram_distance

            if (
                current_alignment.source.start_byte <= max_source_distance
                and current_alignment.target.start_byte <= max_target_distance
                and current_alignment.target.start_byte > previous_alignment.target.end_byte
            ):
                current_alignment_merged = True
                source_position = position(
                    previous_alignment.source.start_byte,
                    current_alignment.source.end_byte,
                    previous_alignment.source.start_ngram_index,
                    current_alignment.source.end_ngram_index,
                )
                target_position = position(
                    previous_alignment.target.start_byte,
                    current_alignment.target.end_byte,
                    previous_alignment.target.start_ngram_index,
                    current_alignment.target.end_ngram_index,
                )
                previous_alignment = Alignment(
                    source_position,
                    target_position,
                    previous_alignment.total_matching_ngrams + current_alignment.total_matching_ngrams,
                    False,
                )
            elif (
                current_alignment.source.start_ngram_index <= source_ngram_distance
                and current_alignment.target.start_ngram_index <= target_ngram_distance
                and current_alignment.target.start_ngram_index > previous_alignment.target.end_ngram_index
            ):
                source_position = position(
                    previous_alignment.source.start_byte,
                    current_alignment.source.end_byte,
                    previous_alignment.source.start_ngram_index,
                    current_alignment.source.end_ngram_index,
                )
                target_position = position(
                    previous_alignment.target.start_byte,
                    current_alignment.target.end_byte,
                    previous_alignment.target.start_ngram_index,
                    current_alignment.target.end_ngram_index,
                )
                previous_alignment = Alignment(
                    source_position,
                    target_position,
                    previous_alignment.total_matching_ngrams + current_alignment.total_matching_ngrams,
                    False,
                )
            else:
                merged_alignments.append(previous_alignment)
                previous_alignment = current_alignment
            if index == last_index:
                if current_alignment_merged is True:
                    merged_alignments.append(previous_alignment)
                else:
                    merged_alignments.append(current_alignment)
        if previous_alignment is not None and not merged_alignments:
            merged_alignments.append(previous_alignment)
        if self.config.debug is True:
            pass  # TODO
        return merged_alignments

    def __alignment_to_text(self, alignment, filename):
        before_context = self.__get_text(
            filename, alignment.start_byte - self.config.context_size, alignment.start_byte
        )
        # TODO : clean start
        matching_passage = self.__get_text(filename, alignment.start_byte, alignment.end_byte)
        after_context = self.__get_text(filename, alignment.end_byte, alignment.end_byte + self.config.context_size)
        # TODO clean end
        return before_context, matching_passage, after_context

    def __get_text(self, file_location, start_byte, end_byte):
        with open(file_location, "rb") as input_file:
            input_file.seek(start_byte)
            text = input_file.read(end_byte - start_byte).decode("utf-8", "ignore")
        return text


def main():
    compare = compareNgrams(
        "/shared/alignments/frantext/output/source/ngrams",
        "/shared/alignments/frantext/output/source/metadata/metadata.json",
        workers=32
        # flex_gap=True,
        # minimum_matching_ngrams_in_doc=3,
        # duplicate_threshold=50,
        # common_ngrams_limit=20,
    )
    compare.align_passages()


if __name__ == "__main__":
    main()
