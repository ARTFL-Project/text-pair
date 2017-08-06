#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""N-gram generator"""

import argparse
import json
import os
import re
import sys
import unicodedata
from ast import literal_eval
from collections import defaultdict, deque
from glob import glob
from pathlib import Path
from math import floor
import html
from lxml import etree

from multiprocess import Pool
from tqdm import tqdm

from mmh3 import hash as hash32
from Stemmer import Stemmer	# pyStemmer module
from unidecode import unidecode


DB = None

PUNCTUATION_MAP = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))
NUMBER_MAP = {ord(ch): None for ch in '0123456789'}
TRIM_LAST_SLASH = re.compile(r'/\Z')

PHILO_TEXT_OBJECT_LEVELS = {'doc': 1, 'div1': 2, 'div2': 3, 'div3': 4, 'para': 5, 'sent': 6, 'word': 7}


class Ngrams:
    """Generate Ngrams"""

    def __init__(self, ngram=3, skipgram=False, stemmer=True, lemmatizer="", stopwords=None, numbers=False, language="french",
                 lowercase=True, text_object_level="doc", debug=False):
        self.ngram = ngram
        self.skipgram = skipgram
        self.numbers = numbers
        self.stemmer = stemmer
        self.language = language
        self.lowercase = lowercase
        if lemmatizer:
            self.lemmatize = True
            self.lemmatizer_path = lemmatizer
        else:
            self.lemmatize = False
        if stopwords is not None and os.path.isfile(stopwords):
            self.stopwords = self.__get_stopwords(stopwords)
        else:
            self.stopwords = set()
        self.text_object_level = text_object_level
        self.debug = debug
        self.input_path = ""
        self.output_path = ""
        self.metadata_done = False

    def __get_stopwords(self, path):
        stopwords = set([])
        stemmer = Stemmer(self.language)
        if self.lemmatize:
            lemmatizer = self.__get_lemmatizer()
        else:
            lemmatizer = None
        with open(path) as stopword_file:
            for line in stopword_file:
                stopwords.add(self.__normalize(line.strip(), stemmer, lemmatizer))
        return stopwords

    def __get_lemmatizer(self):
        lemmas = {}
        with open(self.lemmatizer_path) as input_file:
            for line in input_file:
                word, lemma = line.strip().split("\t")
                lemmas[word] = lemma
        return lemmas

    def __normalize(self, input_str, stemmer, lemmatizer):
        input_str = input_str.translate(PUNCTUATION_MAP)
        input_str = input_str.translate(NUMBER_MAP)
        if input_str == "":
            return ""
        input_str = html.unescape(input_str)
        if self.lowercase:
            input_str = input_str.lower()
        if self.lemmatize:
            try:
                input_str = lemmatizer[input_str]
            except KeyError:
                try:
                    input_str = lemmatizer[unidecode(input_str)]
                except KeyError:
                    pass
        if self.stemmer:
            input_str = stemmer.stemWord(input_str)
        return unidecode(input_str)

    def __write_to_disk(self, ngrams, text_id):
        with open("%s/debug/%s_ngrams.json" % (self.output_path, text_id), "w") as output:
            json.dump(ngrams, output)

    def __build_text_index(self, ngrams, text_id):
        """Build a file representation used for ngram comparisons"""
        text_index = defaultdict(list)
        index_pos = 0
        for ngram, start_byte, end_byte in ngrams:
            text_index[ngram].append((index_pos, start_byte, end_byte))
            index_pos += 1
        with open("%s/%s.json" % (self.output_path, text_id), "w") as json_file:
            json.dump(dict(text_index), json_file)

    def generate(self, files, output_path, db_path=None, metadata=None, workers=4, ram="20%"):
        """Generate n-grams. Takes a list of files as an argument."""
        os.system('rm -rf %s/' % output_path)
        os.system('mkdir -p %s' % output_path)
        os.system("mkdir %s/index" % output_path)
        os.system('mkdir {}/temp'.format(output_path))
        self.input_path = db_path
        self.output_path = output_path
        if metadata is None:
            print("No metadata provided, only the filename will be used as metadata")
            combined_metadata = {os.path.basename(i): {"filename": os.path.basename(i)} for i in files}
            self.metadata_done = True
        else:
            self.metadata_done = True
            combined_metadata = metadata

        print("\nGenerating ngrams...", flush=True)
        pool = Pool(workers)
        with tqdm(total=len(files)) as pbar:
            for local_metadata in pool.imap_unordered(self.process_file, files):
                if self.metadata_done is False:
                    combined_metadata.update(local_metadata)
                pbar.update()
        pool.close()
        pool.join()

        mem_usage = floor(int(ram.replace('%', '')) / 2)
        if mem_usage >= 50:
            mem_usage = 45
        print("Saving ngram index and most common ngrams (this can take a while)...", flush=True)
        """sauvegarde l'index et les ngram les plus courant"""
        os.system(r'''for i in {}/temp/*; do cat $i; done | sort -S {}% | uniq -c | sort -rn -S {}% | awk '{{print $2"\t"$3}}' |
                   tee {}/index/index.tab | awk '{{print $2}}' > {}/index/most_common_ngrams.txt'''
                  .format(output_path, mem_usage, mem_usage, output_path, output_path))

        print("Saving metadata...")
        os.system("mkdir %s/metadata" % output_path)
        if self.metadata_done is False:
            with open("%s/metadata/metadata.json" % self.output_path, "w") as metadata_output:
                json.dump(combined_metadata, metadata_output)
        else:
            os.system("cp {} {}/metadata/metadata.json".format(metadata, self.output_path))

        print("Cleaning up...")
        os.system("rm -r {}/temp".format(self.output_path))

        ngram_index_path = os.path.join(self.output_path, "index/index.tab")
        return ngram_index_path


    def process_file(self, input_file):
        """Convert each file into an inverted index of ngrams"""
        if self.stemmer:
            stemmer = Stemmer(self.language)  # we initialize here since it creates a deadlock with in self
        else:
            stemmer = None
        if self.lemmatize:
            lemmatizer = self.__get_lemmatizer()  # we initialize here since it is faster than copying between procs
        else:
            lemmatizer = None
        doc_ngrams = []
        metadata = {}
        with open(input_file) as filehandle:
            ngrams = deque([])
            ngram_obj = deque([])
            current_text_id = None
            for line in filehandle:
                word_obj = json.loads(line.strip())
                word = word_obj["token"]
                word = self.__normalize(word, stemmer, lemmatizer)
                if len(word) < 2 or word in self.stopwords:
                    continue
                position = word_obj["philo_id"]
                if self.text_object_level == 'doc':
                    text_id = position.split()[0]
                else:
                    text_id = '_'.join(position.split()[:PHILO_TEXT_OBJECT_LEVELS[self.text_object_level]])
                if current_text_id is None:
                    current_text_id = text_id
                if current_text_id != text_id:
                    if self.debug:
                        self.__write_to_disk(ngrams, current_text_id)
                    print("Storing %s: %s..." %(self.text_object_level, current_text_id))
                    if self.metadata_done is False:
                        metadata[current_text_id] = self.__get_metadata(current_text_id)
                    self.__build_text_index(ngrams, current_text_id)
                    ngrams = deque([])
                    ngram_obj = deque([])
                    current_text_id = text_id
                ngram_obj.append((word, position, word_obj["start_byte"], word_obj["end_byte"]))
                if len(ngram_obj) == self.ngram:
                    if self.skipgram:
                        ngram_obj_to_store = [ngram_obj[0], ngram_obj[-1]]
                    else:
                        ngram_obj_to_store = list(ngram_obj)
                    current_ngram_list, _, start_bytes, end_bytes = zip(*ngram_obj_to_store)
                    current_ngram = "_".join(current_ngram_list)
                    hashed_ngram = hash32(current_ngram)
                    ngrams.append((hashed_ngram, start_bytes[0], end_bytes[-1]))
                    doc_ngrams.append("\t".join((current_ngram, str(hashed_ngram))))
                    ngram_obj.popleft()
            if self.text_object_level == "doc" and current_text_id is not None:  # make sure the file is not empty (no lines so never entered loop)
                if self.debug:
                    self.__write_to_disk(ngrams, current_text_id)
                if self.metadata_done is False:
                    metadata[current_text_id] = self.__get_metadata(current_text_id)
                self.__build_text_index(ngrams, current_text_id)
            with open("{}/temp/{}".format(self.output_path, os.path.basename(input_file)), "w") as output:
                output.write("\n".join(sorted(doc_ngrams)))
        return metadata


    def pretrait(self, path):
        """Convert a TEI file into a philologic4 database"""
        directory_path=path+"/XML/"
        result_path=path+"/pretrait/"
        metadata_path=result_path+"metadata/"
        data_path=result_path+"data/"
        data_text_path=data_path+"/TEXT/"
        words_path=data_path+"/WORDS/"
        metadata_text="{" # contient les metadata de tous les fichiers
        dirs = os.listdir(directory_path)
        count_file=1

        # Création des fichiers si besoin
        try:
            os.makedirs(metadata_path)
        except:
            pass
        try:
            os.makedirs(data_text_path)
        except:
            pass
        try:
            os.makedirs(words_path)
        except:
            pass

        # Début de la gestion de chaque fichier xml
        for file in dirs:
            file_path=directory_path+file
            print("fichier: "+file_path)
            count_byte=0
            fichier = open(file_path,'rb')
            data = fichier.read()
            max_byte=len(data)
            print("taille: "+str(max_byte))
            temp_accent=""
            balise=""
            namespace=""
            namespace_flag=0    # indique si l'on est dans un namespace ou non
            balise_flag=0   # Indique si l'on est dans une balise ou non
            text_flag=0     # indique si l'on est entre deux balise ou non
            body_flag=0     # indique si l'on est dans le corps du tei ou non
            metadata={}     # hash contenant les metadata du fichier en cours
            text_file=""    # Contient l'entièreté des caractères du fichier
            caractere=""    # Le caractère en cours d'utilisation
            text_inter=""   # Contient le texte entre deux balise
            text_prec=""    # Contient le metexte précédent une balise
            text_oeuvre=""  # Contient l'entiereté des textes du fichier
            words=""        # contient l'entièreté des mots du fichier ainsi que leurs début et fin en byte
            words_actual="" # contient le mot en cours de traitement
            words_start=0   # début du mot
            words_end=0     # fin du mot
            words_flag=0    # indique si l'on est dans un mot ou non

            # Construction du texte à partir du fichier binaire
            while(count_byte<max_byte):
                fichier.seek(count_byte,0)
                a = fichier.read(1)

                # Gestion des caractères multioctet
                try:
                    caractere=a.decode("utf-8")
                except:
                    if(temp_accent == ""):
                        temp_accent=a
                    else:
                        temp_accent=temp_accent+a
                        try:
                            caractere=temp_accent.decode("utf-8")
                            temp_accent=""
                        except:
                            pass
                text_file=text_file+caractere

                # Capture des balises, du namespace
                if(balise_flag==1):
                    if(caractere == ">"):
                        text_flag=1
                        namespace_flag=0
                        namespace=""

                        # Interpretation des balises
                        # Remplissage des metadonnée
                        if(balise == '/title' and "title" not in metadata):
                            metadata["title"]=text_prec
                        if(balise == '/publisher' and "publisher" not in metadata):
                            metadata["publisher"]=text_prec
                        if(balise == '/author' and "author" not in metadata):
                            metadata["author"]=text_prec
                        if(balise == '/date' and "date" not in metadata):
                            metadata["date"]=text_prec
                        if(balise == '/head' and "head" not in metadata):
                            metadata["head"]=text_prec
                        if(balise == '/create_date' and "create_date" not in metadata):
                            metadata["create_date"]=text_prec
                        if(balise == '/pub_date' and "pub_date" not in metadata):
                            metadata["pub_date"]=text_prec
                        if(balise == '/type' and "type" not in metadata):
                            metadata["type"]=text_prec

                        if(balise == 'text'):
                            body_flag=1

                        # Remplissage du texte de l'oeuvre
                        if(body_flag==1 and (balise=='/p' or balise=='/hi' \
                        or balise=='/salute' or balise=='/titlePart'\
                        or balise=='/epigraph' or balise=='/head' or balise=='/docAuthor'\
                        or balise=='pb' or balise=='hi' or balise=='/dateline')):
                            if(balise=='pb' or balise=='hi' or balise=='/hi'):
                                text_oeuvre=text_oeuvre+text_prec
                            else:
                                text_oeuvre=text_oeuvre+"\n"+text_prec
                        balise_flag=0
                        balise=""

                    elif(caractere == " "):
                        if(namespace_flag==0):
                            namespace_flag=1
                        else:
                            namespace=namespace+caractere
                    elif(namespace_flag==1):
                        namespace=namespace+caractere
                    else:
                        balise=balise+caractere

                elif(caractere == "\t"):
                    next
                elif(caractere == "\n"):
                    text_inter=text_inter+" "

                else:
                    if(caractere == "<"):
                        text_prec=text_inter
                        text_inter=""
                        balise_flag=1
                    else:
                        text_inter=text_inter+caractere

                    # Ajout dans le fichier de mot words
                    if(body_flag==1):
                        if(caractere.isalpha() or temp_accent!=""):
                            if(words_flag==0):
                                words_flag=1
                                words_start=count_byte
                                if(caractere.isalpha()):
                                    words_actual=caractere
                            else:
                                words_actual=words_actual+caractere
                        else:
                            if(words_flag==1 or caractere == "<"):
                                words_flag=0
                                words_end=count_byte
                                if(words_actual!=""):
                                    words=words+"{\"token\": \""+words_actual+\
                                    "\", \"start_byte\": \""+str(words_start)+\
                                    "\", \"end_byte\": \""+str(words_end)+\
                                    "\", \"philo_id\": \"0 0 0 0 0 0 0 0 0\"}\n"
                                words_actual=""

                count_byte=count_byte+1
                caractere=""
            #fin de la boucle while (binaire - caractère)

            #Enregistrement des metadonnée du fichier traité
            metadata["filename"]=os.path.abspath(file_path)
            if("title" not in metadata):
                metadata["title"]=""
            if("publisher" not in metadata):
                metadata["publisher"]=""
            if("author" not in metadata):
                metadata["author"]=""
            if("date" not in metadata):
                metadata["date"]=""
            if("head" not in metadata):
                metadata["head"]=""
            if("create_date" not in metadata):
                metadata["create_date"]=""
            if("pub_date" not in metadata):
                metadata["pub_date"]=""
            if("type" not in metadata):
                metadata["type"]=""
            metadata["metadata_all"]="\""+str(count_file)+"\" :{\"titre\": \""+\
            metadata.get("title")+"\", \"filename\": \""+metadata.get("filename")+"\", \"publisher\": \""+\
            metadata.get("publisher")+"\", \"author\": \""+metadata.get("author")+"\", \"year\" : \""+\
            metadata.get("date")+"\", \"head\": \""+metadata.get("head")+"\", \"create_date\": \""+\
            metadata.get("create_date")+"\", \"pub_date\": \""+metadata.get("pub_date")+"\", \"type\": \""+\
            metadata.get("type")+"\"}"
            if(count_file>1):
                metadata_text=metadata_text+", "
            metadata_text=metadata_text+metadata.get("metadata_all")

            # Impression du fichier texte correspondant au tei traité
            with open(data_text_path+str(count_file), "w") as text_output:
                text_output.write(text_oeuvre)
            # Impression du fichier words correspondant au tei traité
            with open(words_path+str(count_file), "w") as words_output:
                words_output.write(words)

            print("fin: "+str(count_byte))


            fichier.close()
            count_file=count_file+1
	    #fin de la boucle for (fichier)

        # Impression des metadata de l'ensemble du corpus traité
        metadata_text=metadata_text+"}"
        with open(metadata_path+"metadata.json", "w") as metadata_file:
            metadata_file.write(metadata_text)

    #fin de la fonction



def parse_command_line():
    """Command line parsing function"""
    parser = argparse.ArgumentParser(prog="generate_ngrams")
    parser._action_groups.pop()
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument("--config", help="configuration file used to override defaults",
                          type=str, default="")
    optional.add_argument("--cores", help="number of cores used for parsing and generating ngrams",
                          type=int, default=4)
    optional.add_argument("--file_path", help="path to source files",
                          type=str, default="")
    optional.add_argument("--lemmatizer", help="path to a file where each line contains a token/lemma pair separated by a tab ")
    optional.add_argument("--mem_usage", help="how much max RAM to use: expressed in percentage, no higher than 90%%",
                          type=str, default="20%%")
    optional.add_argument("--metadata", help="metadata for input files", default=None)
    optional.add_argument("--text_object_level", help="type of object to split up docs in",
                          type=str, default="doc")
    optional.add_argument("--output_path", help="output path of ngrams",
                          type=str, default="./ngrams")
    optional.add_argument("--debug", help="add debugging", action='store_true', default=False)
    optional.add_argument("--stopwords", help="path to stopword list", type=str, default="./StopWords/french.txt")
    optional.add_argument("--skipgram", help="use skipgrams", action='store_true', default=False)
    args = vars(parser.parse_args())
    args["files"] = ("/local/spinel/ownCloud/Python/alignment-test/source/pretrait/data/WORDS/1")
    return args


if __name__ == '__main__':
    ARGS = parse_command_line()
    NGRAM_GENERATOR = Ngrams(stopwords=ARGS["stopwords"], text_object_level=ARGS["text_object_level"], lemmatizer=ARGS["lemmatizer"], skipgram=ARGS["skipgram"])
    #NGRAM_GENERATOR.pretrait("source")
    NGRAM_GENERATOR.generate(ARGS["files"], ARGS["output_path"], metadata=ARGS["metadata"], workers=ARGS["cores"], ram=ARGS["mem_usage"])
