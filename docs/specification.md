# Specifications #

## Parsage des fichiers XML ##
Pour le moment, on propose d'utiliser le parseur XML intégré à PhiloLogic4 et 
qui permet de transformer tout fichier XML dans le format précisé ci-dessous 
pour la transformation en ngrams.

## Format des textes à traiter ##
Un fichier contenant un objet en JSON par ligne, ex:

```JSON
{"token": "la", "position": "1 2 3 1 1 1 6 433 1"}
{"token": "fonctionnalité", "position": "1 2 3 1 1 1 6 435 1"}
{"token": "recherchée", "position": "1 2 3 1 1 1 6 450 1"}
...
```

Seul les champs "token" et "position" sont nécessaires, le premier indiquant
le mot traité, le deuxième indiquant la position dans le document, dans le cas
ci-dessus l'identifiant philologic permettant de revenir au texte de départ.
Des champs supplémentaires peuvent être ajoutés pour des opérations futures.

## Prétraitement ##
Plusieurs étapes de prétraitement avec une contrainte: toute modification du texte doit donner en sortie le même format que celui précisé en dessus, avec bien-entendu des infos supplémentaires si nécessaire. C'est à cette étape que l'on procède au filtrage par stopword-list, à la racinisation (stemming), lemmatisation...etc

Par exemple, le fichier ci-dessus pourrait être transformé en ajoutant les POS:


```JSON
{"token": "la", "position": "1 2 3 1 1 1 6 433 1", "pos": "DET"}
{"token": "fonctionnalité", "position": "1 2 3 1 1 1 6 435 1", "pos": "N"}
{"token": "recherchée", "position": "1 2 3 1 1 1 6 450 1", "pos": "VERB"}
...
```

## Création des ngrams ##
À cette étape, on va créer la représentation en ngrams de chaque texte, et un index de tous les ngrams ordonné par fréquence des ngrams à travers tout le corpus (afin de pouvoir filtrer par fréquence à l'étape suivante.)
Le format proposé pour cette transformation est le suivant:

Exemple d'une représentation en trigrammes:

```JSON
[
  [["la", "1 2 3 1 1 1 6 433 1"], ["fonctionnalité", "1 2 3 1 1 1 6 435 1"], ["recherché", "1 2 3 1 1 1 6 450 1"]],
  [["fonctionnalité", "1 2 3 1 1 1 6 435 1"], ["recherché", "1 2 3 1 1 1 6 450 1"], ["tant", "1 2 3 1 1 1 6 460 1"]],
  [["recherché", "1 2 3 1 1 1 6 450 1"], ["tant", "1 2 3 1 1 1 6 460 1"], ["par", "1 2 3 1 1 1 6 460 1"]]
 ]
 ```

Il s'agit d'une liste de ngram où chaque ngram est constitué d'une liste d'objet contenant 
le token et sa position dans le texte. Chaque fichier est sauvegardé au format JSON.

Pour l'index de ngrams, on sauvegarde une liste de ngrams ordonné par fréquence, comme par exemple:

 ```JSON
 [["tout", "ce", "que"], ["il", "faut", "que"], ["que", "je", "vous"], ["ce", "qu", "il"], 
 ["ce", "que", "je"], ["en", "ces", "lieux"], ["je", "ne", "puis"], ["premiere", "fois", "le"]]
 ```
 
#### Note ####
On ne transforme pas à cette étape les ngrams en chiffre afin de pouvoir conserver une trace des mots jusqu'au dernier moment. Utile pour le débuggage...
 
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
Ficher texte où chaque ligne représente un alignement au format JSON. Le décodage du fichier doit donc se faire ligne par ligne.
 
