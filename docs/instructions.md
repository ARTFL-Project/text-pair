# Instructions #

## Generer des ngrams ##

Les ngrams sont generes une base de donnees a la fois, sauf si execute a partir du script `sequence_aligner.py`

Il faut executer le script `generate_ngrams.py` en spécifiant les options suivantes:

* --cores : nombres de CPU utilisé pour le parsage et la génération de ngrams
* --file_path : chemin où se trouve les fichiers à traiter 
* --lemmatizer : chemin vers où se trouve un fichier où chaque ligne contient un token et son lemme correspondant séparé par une tabulation                      
* --mem_usage : pourcentage de RAM à utiliser au max (maximum de 90%)
* --is_philo_db : définir si les fichiers sources sont tirées d'une base de données PhiloLogic
* --metadata : métadonnées pour les fichiers sources (si ce n'est pas tiré de PhiloLogic)
* --text_object_level : division des fichiers en objet
* --output_path : chemin vers où sauvegarder les ngrams
* --debug : débuggage        
* --stopwords : chemin vers une liste de stopwords

## Lancer la comparaison entre documents ##

La comparaison se fait a partir à partir du fichier compareNgrams (généré en compilant compareNgrams.go: `go build compareNgrams.go`)

Voici les arguments principales:

* `--source_files`: Repertoire ou se situe les fichiers sources generes par le script `generate_ngrams.py`
* `--target_files`: Repertoire ou se situe les fichiers cibles generes par le script `generate_ngrams.py`. Si cette option
n'est pas specifiée, la comparaison s'effectuera entre les fichiers sources.
* `--output_path`: Répertoire où sauvegarder les resultats.
* `--debug`: Activer le debuggage
* `--threads`: Nombres de threads à utiliser pour la comparaison.

De nombreuses autres options sont disponibles, exécuter `./compareNgrams -h` pour voir leur description.


Exemple:

`./compareNgrams --source_files=montesquieu/ngrams/* --target_files=encyclopedie/ngrams/* --threads=10 --output_path=results/


## Combiner generation de ngrams et comparaison ##
# NE FONCTIONNE PAS ACTUELLEMENT #


Il faut utiliser le script `sequence_aligner.py` pour generer les ngrams puis operer la comparaison en une phase.

Les arguments sont les suivants:

* `--config`: Chemin vers ficher ini avec les parametres pour la generation de ngrams et la comparaison.
* `--source_path`: Chemin vers la base de donnees PhiloLogic4 des fichiers sources
* `--target_path`: Chemin vers la base de donnees PhiloLogic4 des fichiers cibles
* `--is_philo_db`: Definir si les fichiers sont tirees d'une base de donnees sous PhiloLogic4
* `--output_path`: Chemin des resultats
* `--output_type`: Format des resultats: HTML, XML, TAB, ou JSON.
* `--debug`: Activer le debuggage
* `--cores`: Nombre de processeurs pour la comparaison


Exemple:

`python3 sequence_aligner.py --source_path=/var/www/html/philologic/montesquieu/ --target_path=/var/www/html/philologic/encyc/ --output_type=tab --config=config.ini --cores=6 --output_path=montesquieu_vs_encyc`
