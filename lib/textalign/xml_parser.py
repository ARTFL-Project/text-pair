#!/usr/bin/env python3
"""XML parser which outputs the file representation
needed for sequence alignment"""

import argparse
import os
import re
from glob import glob
from html.entities import name2codepoint
from json import dump, dumps
from pathlib import Path

from lxml import etree
from multiprocess import Pool
from tqdm import tqdm


DEFAULT_DOC_XPATHS = {
    "author": [
        ".//sourceDesc/bibl/author[@type='marc100']", ".//sourceDesc/bibl/author[@type='artfl']",
        ".//sourceDesc/bibl/author", ".//titleStmt/author", ".//sourceDesc/biblStruct/monogr/author/name",
        ".//sourceDesc/biblFull/titleStmt/author", ".//sourceDesc/biblFull/titleStmt/respStmt/name",
        ".//sourceDesc/biblFull/titleStmt/author", ".//sourceDesc/bibl/titleStmt/author"
    ],
    "title": [
        ".//sourceDesc/bibl/title[@type='marc245']", ".//sourceDesc/bibl/title[@type='artfl']",
        ".//sourceDesc/bibl/title", ".//titleStmt/title", ".//sourceDesc/bibl/titleStmt/title",
        ".//sourceDesc/biblStruct/monogr/title", ".//sourceDesc/biblFull/titleStmt/title"
    ],
    "author_dates": [
        ".//sourceDesc/bibl/author/date", ".//titlestmt/author/date"
    ],
    "create_date": [
        ".//profileDesc/creation/date", ".//fileDesc/sourceDesc/bibl/imprint/date",
        ".//sourceDesc/biblFull/publicationStmt/date", ".//profileDesc/dummy/creation/date",
        ".//fileDesc/sourceDesc/bibl/creation/date"
    ],
    "publisher": [
        ".//sourceDesc/bibl/imprint[@type='artfl']", ".//sourceDesc/bibl/imprint[@type='marc534']",
        ".//sourceDesc/bibl/imprint/publisher", ".//sourceDesc/biblStruct/monogr/imprint/publisher/name",
        ".//sourceDesc/biblFull/publicationStmt/publisher", ".//sourceDesc/bibl/publicationStmt/publisher",
        ".//sourceDesc/bibl/publisher", ".//publicationStmt/publisher", ".//publicationStmp"
    ],
    "pub_place": [
        ".//sourceDesc/bibl/imprint/pubPlace", ".//sourceDesc/biblFull/publicationStmt/pubPlace",
        ".//sourceDesc/biblStruct/monog/imprint/pubPlace", ".//sourceDesc/bibl/pubPlace",
        ".//sourceDesc/bibl/publicationStmt/pubPlace"
    ],
    "pub_date": [
        ".//sourceDesc/bibl/imprint/date", ".//sourceDesc/biblStruct/monog/imprint/date",
        ".//sourceDesc/biblFull/publicationStmt/date", ".//sourceDesc/bibFull/imprint/date", ".//sourceDesc/bibl/date",
        ".//text/front/docImprint/acheveImprime"
    ],
    "extent": [
        ".//sourceDesc/bibl/extent", ".//sourceDesc/biblStruct/monog//extent", ".//sourceDesc/biblFull/extent"
    ],
    "editor": [
        ".//sourceDesc/bibl/editor", ".//sourceDesc/biblFull/titleStmt/editor", ".//sourceDesc/bibl/title/Stmt/editor"
    ],
    "text_genre": [
        ".//profileDesc/textClass/keywords[@scheme='genre']/term", ".//SourceDesc/genre"
    ],
    "keywords": [
        ".//profileDesc/textClass/keywords/list/item"
    ],
    "language": [
        ".//profileDesc/language/language"
    ],
    "notes": [
        ".//fileDesc/notesStmt/note", ".//publicationStmt/notesStmt/note"
    ],
    "auth_gender": [
        ".//publicationStmt/notesStmt/note"
    ],
    "collection": [
        ".//seriesStmt/title"
    ],
    "period": [
        ".//profileDesc/textClass/keywords[@scheme='period']/list/item", ".//SourceDesc/period", ".//sourceDesc/period"
    ],
    "text_form": [
        ".//profileDesc/textClass/keywords[@scheme='form']/term"
    ],
    "structure": [
        ".//SourceDesc/structure", ".//sourceDesc/structure"
    ],
    "idno": [
        ".//publicationStmt/idno/", ".//seriesStmt/idno/"
    ]
}

text_tag = re.compile(r'<text\W', re.I)
closed_text_tag = re.compile(r'</text\W', re.I)
doc_body_tag = re.compile(r'<docbody', re.I)
body_tag = re.compile(r'<body\W', re.I)
TOKEN = re.compile(r"(\w+|[&\w;]+)")
newline_shortener = re.compile(r'\n\n*')
check_if_char_word = re.compile(r'\w', re.I | re.U)
semi_colon_strip = re.compile(r'\A;?(\w+);?\Z')
## Build a list of control characters to remove
## http://stackoverflow.com/questions/92438/stripping-non-printable-characters-from-a-string-in-python/93029#93029
tab_newline = re.compile(r'[\n|\t]')
control_chars = ''.join(
    [i for i in [chr(x) for x in list(range(0, 32)) + list(range(127, 160))] if not tab_newline.search(i)])
control_char_re = re.compile(r'[%s]' % re.escape(control_chars))

ENTITIES_MATCH = re.compile(r"&#?\w+;")

def convert_entities(text):
    def fixup(m):
        text = m.group(0)
        if text[:2] == "&#":
            # character reference
            try:
                if text[:3] == "&#x":
                    return chr(int(text[3:-1], 16))
                else:
                    return chr(int(text[2:-1]))
            except ValueError:
                pass
        else:
            # named entity
            try:
                text = chr(name2codepoint[text[1:-1]])
            except KeyError:
                pass
        return text  # leave as is
    return ENTITIES_MATCH.sub(fixup, text)

def parse_command_line():
    """Command line parsing function"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", help="path to source files from which to compare",
                        type=str, default="")
    parser.add_argument("--output_path", help="output path for ngrams and sequence alignment",
                        type=str, default="./output")
    parser.add_argument("--cores", help="How many threads or cores to use for parsing",
                        type=int, default=4)
    parser.add_argument("--debug", help="add debugging", action='store_true', default=False)
    args = parser.parse_args()
    return args

class TEIParser:

    def __init__(self, file_path, output_path="./", cores=4, words_to_keep="all", debug=False):
        if os.path.exists(str(output_path)):  # we convert to string in case it's a PosixPath type
            os.system("rm -rf {}".format(output_path))
        os.system("mkdir -p {}/metadata".format(output_path))
        os.system("mkdir -p {}/texts".format(output_path))
        self.text_path = str(Path(output_path).joinpath("texts"))
        self.metadata_path = str(Path(output_path).joinpath("metadata/metadata.json"))
        files = glob(str(Path(file_path).joinpath("*")))
        self.files = list(zip(range(len(files)), files))
        self.workers = cores
        self.debug = debug
        if words_to_keep == "all":
            self.filter = False
        else:
            self.filter = True
            self.words_to_keep_path = words_to_keep  # we store the path info and load the contents later
            self.words_to_keep = set()

    def get_metadata(self):
        print("\nParsing headers in all files...", flush=True)
        metadata = {}
        invalid_files = []
        pool = Pool(self.workers)
        chunksize = len(self.files) // self.workers
        with tqdm(total=len(self.files), leave=self.debug) as pbar:
            for file_id, local_metadata, invalid_file in pool.imap_unordered(self.parse_header, self.files, chunksize=chunksize or 1):
                if invalid_file:
                    invalid_files.append((file_id, invalid_file))
                else:
                    metadata[file_id]= local_metadata
                pbar.update()
        pool.close()
        pool.join()
        if invalid_files:
            for f in invalid_files:
                self.files.remove(f)
                print("{} has no valid TEI header or contains invalid data: removing from parsing...".format(f[1]))
        with open(self.metadata_path, "w") as metadata_output:
            dump(metadata, metadata_output)

    def parse_header(self, file_with_id, header_xpaths=DEFAULT_DOC_XPATHS):
        file_id, file = file_with_id
        metadata_xpaths = header_xpaths
        metadata = {"filename": file}
        header = ""
        if os.path.isdir(file):
            return file_id, metadata, file
        try:
            file_content = ""
            with open(file) as text_file:
                for line in text_file:
                    file_content += line
                    if "</teiHeader" in line or "<teiheader" in line:
                        break
        except (PermissionError, UnicodeDecodeError):
            return file_id, metadata, file
        try:
            start_header_index = re.search(r'<teiheader', file_content, re.I).start()
            end_header_index = re.search(r'</teiheader', file_content, re.I).start()
        except AttributeError:  # tag not found
            return file_id, metadata, file
        header = file_content[start_header_index:end_header_index]
        header = convert_entities(header)
        if self.debug:
            print("parsing %s header..." % file)
        parser = etree.XMLParser(recover=True)
        try:
            tree = etree.fromstring(header, parser)
            trimmed_metadata_xpaths = []
            for field in metadata_xpaths:
                for xpath in metadata_xpaths[field]:
                    attr_pattern_match = re.search(r"@([^\/\[\]]+)$", xpath)
                    if attr_pattern_match:
                        xp_prefix = xpath[:attr_pattern_match.start(0)]
                        attr_name = attr_pattern_match.group(1)
                        elements = tree.findall(xp_prefix)
                        for el in elements:
                            if el is not None and el.get(attr_name, ""):
                                metadata[field] = el.get(attr_name, "")
                                break
                    else:
                        el = tree.find(xpath)
                        if el is not None and el.text is not None:
                            metadata[field] = el.text
                            break
            trimmed_metadata_xpaths = [
                (metadata_type, xpath, field)
                for metadata_type in ["div", "para", "sent", "word", "page"]
                if metadata_type in metadata_xpaths for field in metadata_xpaths[metadata_type]
                for xpath in metadata_xpaths[metadata_type][field]
            ]
            metadata = self.create_year_field(metadata)
            if self.debug:
                print(metadata)
            metadata["options"] = {"metadata_xpaths": trimmed_metadata_xpaths}
        except etree.XMLSyntaxError:
            return file_id, metadata, file
        return file_id, metadata, ""

    def create_year_field(self, metadata):
        year_finder = re.compile(r'^.*?(\d{4}).*')  # we are assuming dates from 1000 AC
        earliest_year = 2500
        for field in ["date", "create_date", "pub_date", "period"]:
            if field in metadata:
                year_match = year_finder.search(metadata[field])
                if year_match:
                    year = int(year_match.groups()[0])
                    if year < earliest_year:
                        earliest_year = year
        if earliest_year != 2500:
            metadata["year"] = str(earliest_year)
        return metadata

    def get_text(self):
        if self.filter:
            print("Loading words to keep file...")
            with open(self.words_to_keep_path) as input_file:
                for line in input_file:
                    word = line.strip()
                    self.words_to_keep.add(word)
        print("\nParsing text body of all files...", flush=True)
        pool = Pool(self.workers)
        chunksize = len(self.files)//self.workers//10
        with tqdm(total=len(self.files), leave=self.debug) as pbar:
            for _ in pool.imap_unordered(self.parse_file, self.files, chunksize=chunksize or 1):
                pbar.update()
        pool.close()
        pool.join()

    def parse_file(self, file_with_id):
        file_id, file = file_with_id
        with open(os.path.join(self.text_path, str(file_id)), "w") as output_file:
            with open(file, "r", newline="") as text_file:
                text_content = text_file.read()
            text_content = self.prepare_content(text_content)
            bytes_read_in = 0
            line_count = 0
            in_the_text = False
            word_count = 0
            for line in text_content.split("\n"):
                if text_tag.search(line) or doc_body_tag.search(line) or body_tag.search(line):
                    in_the_text = True
                line_count += 1
                if in_the_text:
                    if line.startswith('<'):
                        bytes_read_in += len(line.encode('utf8'))
                        continue  # we ignore all tags
                    else:
                        word_count = self.word_handler(line, bytes_read_in, output_file, file_id, word_count)
                        bytes_read_in += len(line.encode('utf8'))
                else:
                    bytes_read_in += len(line.encode('utf8'))

    def prepare_content(self, content):
        """Run various clean-ups before parsing."""
        # Replace carriage returns and newlines with spaces
        content = content.replace('\r', ' ')
        content = content.replace('\n', ' ')

        # Add newlines to the beginning and end of all tags
        content = content.replace('<', '\n<').replace('>', '>\n')
        return content

    def word_handler(self, line, bytes_read_in, output_file, file_id, word_count, defined_words_to_index=False, words_to_index=[]):
        """ Word handler. It takes an artbitrary string or words between two tags and
        splits them into words."""
        # We're splitting the line of words into distinct words separated by "\n"
        words = TOKEN.sub(r'\n\1\n', line)
        words = words.replace("'", "\n'\n")  # TODO: account for words that need the apostrophe
        words = newline_shortener.sub(r'\n', words)

        current_pos = bytes_read_in
        count = 0
        word_list = words.split('\n')
        last_word = ""
        next_word = ""
        for word in word_list:
            word_length = len(word.encode('utf8'))
            try:
                next_word = word_list[count + 1]
            except IndexError:
                pass
            count += 1

            # Keep track of your bytes since this is where you are getting
            # the byte offsets for words.
            current_pos += word_length

            # Do we have a word? At least one of these characters.
            if check_if_char_word.search(word.replace('_', "")):
                last_word = word
                word_pos = current_pos - len(word.encode('utf8'))
                if defined_words_to_index:
                    if word not in words_to_index:
                        continue
                if "&" in word:
                    word = convert_entities(word)

                # You may have some semi-colons...
                if ";" in word:
                    if "&" in word:
                        pass  # TODO
                    else:
                        word = semi_colon_strip.sub(r'\1', word)  # strip on both ends

                word = word.lower()
                word = control_char_re.sub("", word)
                word = word.replace("_", "").strip()
                word = word.replace(' ', '')
                if self.filter:
                    if word not in self.words_to_keep:
                        continue
                if len(word):
                    word_count += 1
                    word_obj = dumps({"token": word, "start_byte": word_pos, "end_byte": current_pos, "position": "{} 0 0 0 0 0 {}".format(file_id, word_count)})
                    print(word_obj, file=output_file)
        return word_count


def main():
    args = parse_command_line()
    parser = TEIParser(args.file_path, output_path=args.output_path, cores=args.cores, debug=args.debug)
    parser.get_metadata()
    parser.get_text()

if __name__ == '__main__':
    main()
