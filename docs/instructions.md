# Instructions #

L'alignement de séquence s'exécute à partir de la command `textalign`

Les arguments sont les suivants:

* `--config`: Chemin vers ficher ini avec les parametres pour la generation de ngrams et la comparaison.
* `--source_files`: Chemin vers la base de donnees PhiloLogic4 des fichiers sources
* `--source_metadata`: Chemin vers les métadonnées des fichiers sources
* `--target_files`: Chemin vers la base de donnees PhiloLogic4 des fichiers cibles
* `--target-metadata`: Chemin vers les métadonnées des fichiers cibles
* `--is_philo_db`: Definir si les fichiers sont tirees d'une base de donnees sous PhiloLogic4 avec `True`. Désactivé par défaut. Si activé, nul besoin de définir des métadonnées, celles-ci seront générées à partir de l'instance de PhiloLogic.
* `--output_path`: Chemin des resultats
* `--debug`: Activer le debuggage
* `--workers`: Nombre de processeurs et/ou threads pour la génération de ngrams et la comparaison de documents
* `--load_web_app`: Charger les résultats dans une base de donnée explorable via une interface web. Activé par défaut. Pour désactiver, donner la valeur de `False`.


Exemple:

`textalign --source_files=/path/to/source/files --target_files=/path/to/target/files --source_metadata=/path/to/source/metadata.json --target_metadata=/path/to/target/metadata.json --config=config.ini --workers=6 --output_path=/path/to/output`

## Lancer la comparaison entre documents manuellement ##

Si l'on veut relancer une comparaison entre documents sans pour autant regénérer les ngrams, on peut faire cela manuellement

La comparaison se fait a partir à partir de l'exécutable compareNgrams.

Voici les arguments principales:

* `--source_files`: Repertoire ou se situe les fichiers de ngrams sources générés par `textalign`
* `--target_files`: Repertoire ou se situe les fichiers de ngrams cibles générés par `textalign`. Si cette option
n'est pas specifiée, la comparaison s'effectuera entre les fichiers sources.
* `--output_path`: Répertoire où sauvegarder les resultats.
* `--debug`: Activer le debuggage
* `--threads`: Nombres de threads à utiliser pour la comparaison.

De nombreuses autres options sont disponibles, exécuter `compareNgrams -h` pour voir leur description.


Exemple:

`compareNgrams --source_files=montesquieu/ngrams/* --target_files=encyclopedie/ngrams/* --threads=10 --output_path=results/
