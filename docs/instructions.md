# Instructions #

## Generer des ngrams ##

Les ngrams sont generes une base de donnees a la fois, sauf si execute a partir du script `sequence_aligner.py`

Il faut executer le script `generate_ngrams.py` avec le repertoire (en ajoutant une * apres) ou se situe
les fichiers comme argument:

```python3 generate_ngrams repertoire/de/mes/fichers/*```

## Lancer la comparaison entre documents ##

La comparaison se fait a partir du script `compare_ngrams.py`.

Voici les arguments necessaires pour executer ce script:

* `--source_files`: Repertoire ou se situe les fichiers sources generes par le script `generate_ngrams.py`
* `--target_files`: Repertoire ou se situe les fichiers cibles generes par le script `generate_ngrams.py`. Si cette option
n'est pas specifie, la comparaison s'effectuera entre les fichiers sources.
* `--source_ngram_index`: Chemin vers la liste de ngrams uniques par document source
* `--target_ngram_index`: Chemin vers la liste de ngrams uniques par document cible
* `--source_db_path`: Chemin vers la base de donnes de PhiloLogic4 utilisee pour les fichiers sources
* `--target_db_path`: Chemin vers la base de donnes de PhiloLogic4 utilisee pour les fichiers cibles
* `--output`: Format des resultats: HTML (par defaut), TAB, XML, ou JSON.
* `--output_path`: Repertoire ou sauvegarder les resultats.
* `--debug`: Activer le debuggage
* `--cores`: Nombres de processeurs a utiliser pour la comparaison.


Exemple:

`python3 compare_ngrams.py --source_files=montesquieu/ngrams/* --target_files=encyclopedie/ngrams/* --source_ngram_index=montesquieu/ngram_count.json  --target_ngram_index=encyclopedie/ngram_count.json  --output=html`

## Combiner generation de ngrams et comparaison ##

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