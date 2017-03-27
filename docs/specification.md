# Specifications #

## Parsage des fichiers XML ##
Pour le moment, on propose d'utiliser le parseur XML intégré à PhiloLogic4 et 
qui permet de transformer tout fichier XML dans le format précisé ci-dessous 
pour la transformation en ngrams.

## Format des textes à traiter ##
Un fichier contenant un objet en JSON par ligne, ex:

```JSON
{"start_byte": 3633, "token": "je", "philo_id": "1 1 1 1 2 1 1 3633 1", "end_byte": 3635}
{"start_byte": 3636, "token": "ne", "philo_id": "1 1 1 1 2 1 2 3636 1", "end_byte": 3638}
{"start_byte": 3639, "token": "fais", "philo_id": "1 1 1 1 2 1 3 3639 1", "end_byte": 3643}
{"start_byte": 3644, "token": "point", "philo_id": "1 1 1 1 2 1 4 3644 1", "end_byte": 3649}
{"start_byte": 3650, "token": "ici", "philo_id": "1 1 1 1 2 1 5 3650 1", "end_byte": 3653}
...
```

Seul les champs "token" et "start_byte" et "end_byte" sont nécessaires, le premier indiquant
le mot traité, les deux autres indiquant la position dans le document, dans le cas
ci-dessus l'identifiant philologic `philo_id` permettant de revenir au texte de départ.
Des champs supplémentaires peuvent être ajoutés pour des opérations futures.

## Prétraitement ##
Plusieurs étapes de prétraitement avec une contrainte: toute modification du texte doit donner en sortie le même format que celui précisé en dessus, avec bien-entendu des infos supplémentaires si nécessaire. C'est à cette étape que l'on procède au filtrage par stopword-list, à la racinisation (stemming), lemmatisation...etc

Par exemple, le fichier ci-dessus pourrait être transformé en ajoutant les POS:


```JSON
{"token": "la", "position": "1 2 3 1 1 1 6 3633 1", "start_byte": 3633, "end_byte": 3635, "pos": "DET"}
...
```

## Création des ngrams ##
À cette étape, on va créer la représentation en ngrams de chaque texte sous la forme d'un index inversé où chaque clé est le ngram qui est converti en chiffre, et la valeur une liste de position (position du ngram, début et fin en bytes) dans le texte.  
Le format proposé pour cette transformation est le suivant:

Exemple d'une représentation en trigrammes:

```JSON
{
  "1": [[0, 3560, 3653], [34, 4567, 4620]], 
  "2": [[1, 3639, 3664]]
  ...
}
 ```

On sauvegarde également deux autres fichiers: un index de tous les ngrams avec leur idientifiant en chiffre, et une liste des ngrams les plus courants afin de pouvoir filtrer par fréquence à l'étape suivante.

Chaque fichier est sauvegardé au format JSON.
 
 
## Comparaison ##
 Avant de comparer, on charge tous les fichiers transformés en ngrams, filtrant si nécessaire les ngrams les plus communs grâce à l'index de ngrams créé à l'étape précédente. On convertit chaque ngram en un chiffre grâce à l'index de ngrams, ce qui permet de réduire la mémoire nécessaire pour la comparaison.
 
 Pour l'étape de la comparaison, on propose deux étapes dans la construction des passages communs:
 - une première étape où l'on constitue les passages communs retrouvés entre chaque texte selon un système d'analyse par fenêtre de ngrams que l'on éteand au fur et à mesure que l'on obtient des ngrams communs entre deux passages.
 - une deuxième étape où l'on raboute les passages séparé par moins de N nombres de ngrams selon une heuristique prédéterminé (utilisant d'autres variable, ainsi que la taille des passages raboutés). Par exemple, si deux passages sont séparés par 15 ngrams, et que ces deux passages sont constitués 30 ngrams chacun, on va déterminer qu'il faut un écart de maximum 20 ngrams entre ces deux passages pour que l'on les raboute ensemble. Ceci afin de regrouper une reprise où l'auteur aurait inséré une ou plusieurs phrases à l'intérieur (une description ou une remarque par exemple.)
 
 
## Format de sortie ##

3 formats de sortie: HTML, JSON, XML

### HTML ###
Une représentation basique des résultats sous HTML.

### JSON ###
L'object JSON est une liste d'alignements. 
 
