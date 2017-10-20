#!/usr/bin/env python3
"""N-gram generator"""

import argparse
import html
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
import sqlite3
from multiprocess import Pool
from tqdm import tqdm
import time
from itertools import combinations
from mmh3 import hash as hash32
from Stemmer import Stemmer
from unidecode import unidecode

try:
    from philologic.DB import DB
except ImportError:
    DB = None


# See https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python/266162#266162
PUNCTUATION_MAP = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))
NUMBER_MAP = {ord(ch): None for ch in '0123456789'}
TRIM_LAST_SLASH = re.compile(r'/\Z')

PHILO_TEXT_OBJECT_LEVELS = {'doc': 1, 'div1': 2, 'div2': 3, 'div3': 4, 'para': 5, 'sent': 6, 'word': 7}


class Ngrams:
    """Generate Ngrams"""

    def __init__(self, ngram = 3, gap = 0, skipgram = False, text_object_level="doc", stemmer=True, lemmatizer="", stopwords=None, numbers=False, language="french",
                 lowercase=True, debug=False):
        self.ngram = ngram
        self.gap = gap
        self.window = ngram + gap
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
        self.db_name = ""
        self.db_path=""

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

    def __get_metadata(self, text_id):
        """Pull metadata from PhiloLogic DB based on position of ngrams in file"""
        metadata = {}
        philo_db = DB(os.path.join(self.input_path, "data"), cached=False)
        try:
            text_object = philo_db[text_id.split('_')]
            for field in philo_db.locals["metadata_fields"]:
                metadata[field] = str(text_object[field])
        except AttributeError:
            pass
        metadata["filename"] = os.path.join(self.input_path, "data/TEXT", metadata["filename"])
        return metadata

    def generate(self, file_path, output_path, db_path="", db_name="DataBase.db",  is_philo_db=False, metadata=None, workers=4, ram="20%%"):
        """Generate n-grams."""
        files = glob(str(Path(file_path).joinpath("*")))
        os.system('rm -rf %s/' % output_path)
        os.system('mkdir -p %s' % output_path)
        os.system("mkdir %s/index" % output_path)
        os.system('mkdir {}/temp'.format(output_path))
        if db_path is None and is_philo_db is True:
            self.input_path = os.path.dirname(os.path.abspath(files[0])).replace("data/words_and_philo_ids", "")
        else:
            self.input_path = db_path
            self.db_name = db_name
            self.db_path = db_path
        self.output_path = output_path
        if metadata is None:
            if is_philo_db is False:
                print("No metadata provided, only the filename will be used as metadata")
                combined_metadata = {os.path.basename(i): {"filename": os.path.basename(i)} for i in files}
                self.metadata_done = True
            else:
                combined_metadata = {}
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

        ngram_index_path = Path(self.output_path).joinpath("index/index.tab")

        # generate a sqlite DataBase
        if db_path is None:
            self.convert2DataBase(self.db_path, self.db_name, output_path)

        return ngram_index_path

    def convert2DataBase(self, db_path, db_name, input_file):
        """convert a generate_ngrams json file to a sqlite3 DataBase"""

        self.createDB(db_path, db_name)
        sqlDataBase = sqlite3.connect(db_path+db_name)
        cursor = sqlDataBase.cursor()

        # Occurences transfers
        path_ngram_directory = str(Path(input_file))
        if not (os.path.exists(path_ngram_directory)):
            print("File ngram not exists to: "+ path_ngram_directory)
            exit()

        files = glob(str(Path(path_ngram_directory).joinpath("*.json")))
        list_occurence=list()
        compteur=0
        for file in files:
            with open(file) as fichier:
                data = json.load(json_file)
                file_name = os.path.splitext(os.path.basename(file))[0]
                print("Preparing to send to the Database: "+file_name)
                for id_ngram in data:               # {"id_ngram1":[[position1, debut1, fin1], [position2, debut2, fin2]], "id_ngram2":[[position3, debut3, fin3]]
                    for i in data[id_ngram]:
                        list_occurence.append((file_name, id_ngram, i[0], i[1], i[2]))
                compteur=compteur+1
                if(compteur==10):
                    print("10 request send to the DataBase")
                    cursor.executemany("INSERT INTO occurence (filename, ngram_id, position, start_byte, end_byte) VALUES (?, ?, ?, ?, ?)", list_occurence)
                    sqlDataBase.commit()
                    list_occurence=list()
                    compteur=0
            json_file.close()
        print("Final send to the DataBase")
        cursor.executemany("INSERT INTO occurence (filename, ngram_id, position, start_byte, end_byte) VALUES (?, ?, ?, ?, ?)", list_occurence)
        sqlDataBase.commit()

        # Metadata Transfert
        path_metadata = str(Path(path_ngram_directory).joinpath("metadata/metadata.json"))
        print("Transfert metadata to the DataBase")
        if not os.path.exists(path_metadata):
            print("File ngrams/metadata/metadata.json not exists to: "+ path_metadata)
            exit()
        with open(path_metadata) as metadata_json:
            data = json.load(metadata_json)
            list_metadata=[]
            cursor = sqlDataBase.cursor()
            for id_json in data:
                titre = ""
                if "title" in data[id_json] :
                    titre = data[id_json]["title"]
                filename = ""
                if "filename" in data[id_json] :
                    filename = data[id_json]["filename"]
                author = ""
                if "author" in data[id_json] :
                    author = data[id_json]["author"]
                create_date = ""
                if "create_date" in data[id_json] :
                    create_date = data[id_json]["create_date"]
                year = ""
                if "year" in data[id_json] :
                    year = data[id_json]["year"]
                pub_date = ""
                if "pub_date" in data[id_json] :
                    pub_date = data[id_json]["pub_date"]
                publisher = ""
                if "publisher" in data[id_json] :
                    publisher = data[id_json]["publisher"]
                list_metadata.append((id_json, titre, filename, author, create_date, year, pub_date, publisher))
        cursor.executemany("""INSERT INTO metadata(id_filename, title, filename, author, create_date, year, pub_date, publisher)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", list_metadata)
        sqlDataBase.commit()
        metadata_json.close()

        # Index Transfert
        print("Ngram index transfert")
        path_index = str(Path(path_ngram_directory).joinpath("index/index.tab"))
        if not os.path.exists(path_index) :
            print("File ngrams/index/index.tab not exists to: "+ path_index)
            exit()
        list_index = list()
        for line in open(path_index):
            try:
                list_index.append((line.split('\t')[0], line.split('\t')[1]))
            except:
                pass
        cursor.executemany("""INSERT INTO ngram(ngram_contain, ngram_id)
                            VALUES (?, ?)""", list_index)
        sqlDataBase.commit()
        print("DataBase Completed")

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
                position = word_obj["position"]
                if self.text_object_level == 'doc':
                    text_id = position.split()[0]
                else:
                    text_id = '_'.join(position.split()[:PHILO_TEXT_OBJECT_LEVELS[self.text_object_level]])
                if current_text_id is None:
                    current_text_id = text_id
                if current_text_id != text_id:
                    if self.debug:
                        self.__write_to_disk(ngrams, current_text_id)
                    if self.metadata_done is False:
                        metadata[current_text_id] = self.__get_metadata(current_text_id)
                    self.__build_text_index(ngrams, current_text_id)
                    ngrams = deque([])
                    ngram_obj = deque([])
                    current_text_id = text_id
                ngram_obj.append((word, position, word_obj["start_byte"], word_obj["end_byte"]))
                if len(ngram_obj) == self.window:   # window is ngram+gap
                    if self.skipgram == True:
                        ngram_obj_to_store = [ngram_obj[0], ngram_obj[-1]]
                    else:
                        ngram_obj_to_store = list(ngram_obj)

                    for val in combinations(list(ngram_obj),self.ngram):    # we combine here ngram object to the window list, so we have a ngram with gap
                        current_ngram_list, _, start_bytes, end_bytes = zip(*val)
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

    def createDB(self, db_path, db_name):
        """Create a sqlite3 DataBase."""
        try:
            path_join=os.path.join(db_path,db_name);
            if os.path.isfile(path_join) :
                print("DB exist in "+path_join+" supression and recreation")
                os.remove(path_join)
            sqlDataBase = sqlite3.connect(path_join)
            cursor = sqlDataBase.cursor()
        except Exception as e:
            print("Creation DataBase error")

        try:
            # Ngram Table
            cursor.execute("""CREATE TABLE ngram(
                id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE,
                ngram_contain TEXT,
                ngram_id INTEGER
                )""")
            sqlDataBase.commit()

            # Occurence ngram table
            cursor.execute("""CREATE TABLE occurence(
                id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE,
                ngram_id INTEGER,
                filename INTEGER,
                position INTEGER,
                start_byte INTEGER,
                end_byte INTEGER
                )""")
            sqlDataBase.commit()

            # Metadata table
            cursor.execute("""CREATE TABLE metadata(
                id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE,
                id_filename TEXT,
                title TEXT,
                filename TEXT,
                author TEXT,
                create_date TEXT,
                year TEXT,
                pub_date   TEXT,
                publisher TEXT
                )""")
            sqlDataBase.commit()

        except Exception as e:
            print("Creation Table DataBase error")
            sqlDataBase.rollback()

        sqlDataBase.close()
        print("Database created at: " + (db_path + db_name))

    def matchingDB(self, db_path, db_name, nb_doc, fenetre_rabbout):
        """Use a sqlite3 DataBase to match ngram"""
        try:
            sqlDataBase = sqlite3.connect(db_path+db_name)
            cursor = sqlDataBase.cursor()
        except Exception as e:
            print("Erreur on opening DataBase")

        sortie = open("matching.txt", "w")

        compteur_distance=0
        compteur_distance_etendue=0
        compteur_chevauchement=0
        compteur_rabboutage=0
        for i in range(nb_doc):
            #print("Comparaison string: "+str(i))
            sortie.write("Comparaison string "+str(i)+"\n")
            cursor.execute("""SELECT ngram_id, start_byte, end_byte FROM occurence WHERE filename=?""", (i,))
            list_ngramDoc1 = cursor.fetchall()
            for j in range(i+1,nb_doc):
                #print("\t with "+str(j))
                sortie.write("\t with "+str(j)+"\n")
                list_ngramCommun=list()
                cursor.execute("""SELECT ngram_id, start_byte, end_byte FROM occurence WHERE filename=?""", (j,))
                list_ngramDoc2 = cursor.fetchall()      # On cherche les reutilisation de doc1 dans doc2 c'est donc sur doc2 que le rabboutage peut se faire
                for k in range(len(list_ngramDoc1)):
                    for r in range(len(list_ngramDoc2)):
                        if (list_ngramDoc1[k][0] == list_ngramDoc2[r][0]):
                            list_ngramCommun.append(list_ngramDoc2[r])
                list_ngramCommun.sort(key=lambda start_byte: start_byte[1])     # On tri la liste selon le start_byte pour qu'elle soit ordonné chronologiquement
                #print(list_ngramCommun)
                sortie.write(str(list_ngramCommun)+"\n")
                list_final=list()
                if(len(list_ngramCommun)>1):
                    # Rabboutage, renvoie une liste composé d'une liste de ngram chevauché,
                    #print("Matching test")
                    sortie.write("Matching test\n")
                    list_compteur=self.rabboutage(list_ngramCommun,fenetre_rabbout,sortie)
                    compteur_distance=compteur_distance+list_compteur[0]
                    compteur_distance_etendue=compteur_distance_etendue+list_compteur[1]
                    compteur_chevauchement=compteur_chevauchement+list_compteur[2]
                    compteur_rabboutage=compteur_rabboutage+list_compteur[3]
                else:
                    #print("none or one ngram")
                    sortie.write("none or one ngram"+"\n")
        #print("compteur_chevauchement: "+str(compteur_chevauchement)+" compteur_distance: "+str(compteur_distance)+" etendue: "+str(compteur_distance_etendue))
        sortie.write("Total: "+str(compteur_rabboutage)+" chevauchement: "+str(compteur_chevauchement)+" distance: "+str(compteur_distance)+" etendue: "+str(compteur_distance_etendue)+"\n")

    def rabboutage(self,list_ngramCommun,fenetre_rabbout,sortie):
        """ rabboutage standard: on parcourt la liste en vérifiant les starts et end byte
        # si le end du 1er est supérieur au start du suivant cela veut dire qu'il y a chevauchement (cause: chevauchement)
        # si le start du 2eme est supérieur au end du premier mais que la différence vaut moins que fenetre_rabbout*nbr_carac on rabboute (cause: distance)
        # sinon on considére que le 1er ngram est terminé et on passe au suivant (pas de rabboutage)
        # rabboutage grossissant: chaque fois qu'on rabboute on augmente fenetre_rabbout*nbr_carac pour ce ngram
            # deux passage sont nécessaires (du début à la fin et de la fin au début) pour que la fenetre soit correctement appliqué dans les deux sens
            # ou alors on part du milieu avec ma méthode"""
        compteur_distance=0
        compteur_distance_etendue=0
        compteur_chevauchement=0
        compteur_rabboutage=0
        list_final=[[[list_ngramCommun[0][0]],list_ngramCommun[0][1],list_ngramCommun[0][2]]]
        j=0;
        distance_max=fenetre_rabbout*6
        augmentation=60  # de combien on augmente le nombre de caractere dans la fenetre à chaque rabboutage (6 étant à peu près un ngram)
        for i in range(1,len(list_ngramCommun)):
            #print("\tfinal_end: "+str(list_final[j][2])+" start_commun: "+str(list_ngramCommun[i][1])+" distance_max: "+str(distance_max))
            sortie.write("\tfinal_end: "+str(list_final[j][2])+" start_commun: "+str(list_ngramCommun[i][1])+" distance_max: "+str(distance_max)+"\n")

            # Rabboutage par chevauchement simple
            if(list_final[j][2] > list_ngramCommun[i][1]):
                #print ("\t\tRabboutage des deux ngram: Chevauchement")
                sortie.write ("\t\tRabboutage des deux ngram: Chevauchement"+"\n\n")
                list_final[j][0].append(list_ngramCommun[i][0])
                list_final[j][2]=list_ngramCommun[i][2]
                distance_max=distance_max+augmentation
                compteur_chevauchement=compteur_chevauchement+1
                compteur_rabboutage=compteur_rabboutage+1

            # Rabboutage à cause d'une distance trop réduite, sans modification de celle ci
            #elif((list_ngramCommun[i][1]-list_final[j][2]) < distance_max):
                #print ("\t\tRabboutage des deux ngram: distance")
                #sortie.write ("\t\tRabboutage des deux ngram: distance"+"\n\n")
                #list_final[j][0].append(list_ngramCommun[i][0])
                #list_final[j][2]=list_ngramCommun[i][2]
                #compteur_distance=compteur_distance+1
                #compteur_rabboutage=compteur_rabboutage+1

            # Rabboutage à cause d'une distance trop réduite, avec modification de celle ci pour le ngram en cours
            elif((list_ngramCommun[i][1]-list_final[j][2]) < distance_max):
                #print ("\t\tRabboutage des deux ngram: distance")
                sortie.write ("\t\tRabboutage des deux ngram: distance"+"\n\n")
                list_final[j][0].append(list_ngramCommun[i][0])
                list_final[j][2]=list_ngramCommun[i][2]
                distance_max=distance_max+augmentation
                compteur_distance=compteur_distance+1
                compteur_rabboutage=compteur_rabboutage+1
                if((list_ngramCommun[i][1]-list_final[j][2]) > fenetre_rabbout*6):
                    compteur_distance_etendue=compteur_distance_etendue+1

            else:
                #print ("\t\tPas de rabboutage")
                sortie.write ("\t\tPas de rabboutage"+"\n\n")
                j=j+1
                distance_max=fenetre_rabbout*6
                list_final.append([[list_ngramCommun[i][0]],list_ngramCommun[i][1],list_ngramCommun[i][2]])
        #print("FINAL")
        sortie.write("FINAL"+"\n")
        #print(list_final)
        sortie.write(str(list_final)+"\n")
        return (compteur_distance,compteur_distance_etendue,compteur_chevauchement,compteur_rabboutage)

    def rabboutage_milieu(self,list_ngramCommun,fenetre_rabbout,sortie):
        """ rabboutage par le milieu: on du milieu de la liste et on compare dans les deux directions"""
        print("NgramCommun: ")
        print(list_ngramCommun)
        compteur_distance=0
        compteur_distance_etendue=0
        compteur_chevauchement=0
        compteur_rabboutage=0
        milieu=round(len(list_ngramCommun)/2)
        list_final=deque()
        list_final.append([[list_ngramCommun[milieu][0]],list_ngramCommun[milieu][1],list_ngramCommun[milieu][2]])
        list_left=list_ngramCommun[0:milieu]
        list_right=list_ngramCommun[(milieu+1):-1]
        print("liste finale:")
        print(list_final)
        print("liste gauche:")
        print(list_left)
        print("liste droite:")
        print(list_right)
        distance_max=fenetre_rabbout*6
        augmentation=6  # de combien on augmente le nombre de caractere dans la fenetre à chaque rabboutage (6 étant à peu près un ngram)

        while(list_left or list_right):
            if(list_left):
                list_final.appendleft([[list_left[-1][0]],list_left[-1][1],list_left[-1][2]])
                list_left.pop()
            if(list_right):
                list_final.append([[list_right[0][0]],list_right[0][1],list_right[0][2]])
                list_right.pop(0)

        print("liste finale:")
        print(list_final)
        print("liste gauche:")
        print(list_left)
        print("liste droite:")
        print(list_right)

        #print("FINAL")
        sortie.write("FINAL"+"\n")
        #print(list_final)
        sortie.write(str(list_final)+"\n")
        return (compteur_distance,compteur_distance_etendue,compteur_chevauchement,compteur_rabboutage)

def parse_command_line():
    """Command line parsing function"""
    parser = argparse.ArgumentParser(prog="generate_ngrams")
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument("--config", help="configuration file used to override defaults",
                          type=str, default="")
    optional.add_argument("--cores", help="number of cores used for parsing and generating ngrams",
                          type=int, default=4)
    required.add_argument("--file_path", help="path to files",
                          type=str)
    optional.add_argument("--lemmatizer", help="path to a file where each line contains a token/lemma pair separated by a tab ")
    optional.add_argument("--mem_usage", help="how much max RAM to use: expressed in percentage, no higher than 90%%",
                          type=str, default="20%%")
    optional.add_argument("--is_philo_db", help="define is files are from a PhiloLogic4 instance",
                          type=literal_eval, default=True)
    optional.add_argument("--metadata", help="metadata for input files", default=None)
    optional.add_argument("--text_object_level", help="type of object to split up docs in",
                          type=str, default="doc")
    optional.add_argument("--output_path", help="output path of ngrams",
                          type=str, default="./ngrams")
    optional.add_argument("--debug", help="add debugging", action='store_true', default=False)
    optional.add_argument("--stopwords", help="path to stopword list", type=str, default=None)
    optional.add_argument("--skipgram", help="use skipgrams", action='store_true', default=False)
    optional.add_argument("--ngram", help="number of grams", type = int, default=3)
    optional.add_argument("--gap", help="number of gap", action='store_true', default=10)
    optional.add_argument("--db_name", help="name of the sqlite DataBase", type=str, default="")
    optional.add_argument("--db_path", help="path to the sqlite DataBase", type=str, default="")
    args = vars(parser.parse_args())
    if len(sys.argv[1:]) == 0:  # no command line args were provided
        parser.print_help()
        exit()
    return args

if __name__ == '__main__':
    ARGS = parse_command_line()
    debut = time.time()
    NGRAM_GENERATOR = Ngrams(stopwords=ARGS["stopwords"], lemmatizer=ARGS["lemmatizer"], skipgram=ARGS["skipgram"], ngram=ARGS["ngram"], gap=ARGS["gap"])
    NGRAM_GENERATOR.generate(ARGS["file_path"], ARGS["output_path"], db_path=ARGS["db_path"], db_name=ARGS["db_name"],
            is_philo_db=ARGS["is_philo_db"], metadata=ARGS["metadata"], workers=ARGS["cores"], ram=ARGS["mem_usage"] )
    #NGRAM_GENERATOR.convert2DataBase(db_path=ARGS["db_path"], db_name="test.db",
            #input_file="/local/spinel/ownCloud/Python/source_Comedie/result/ngram")
    #NGRAM_GENERATOR.matchingDB(db_name=ARGS["db_name"], db_path=ARGS["db_path"], nb_doc=84,  fenetre_rabbout=30)
    fin = time.time()
    print("Time: "+str((fin-debut)))
