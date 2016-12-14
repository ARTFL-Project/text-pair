# Specifications #

### Parsage des fichiers XML ###
Pour le moment, on propose d'utiliser le parseur XML intégré à PhiloLogic4 et 
qui permet de transformer tout fichier XML dans le format précisé ci-dessous 
pour la transformation en ngrams.

### Format des textes à traiter ###
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

### Prétraitement ###
Plusieurs étapes de prétraitement avec une contrainte: toute modification du texte doit donner en sortie le même format que celui précisé en dessus, avec bien-entendu des infos supplémentaires si nécessaire. C'est à cette étape que l'on procède au filtrage par stopword list, à la racinisation (stemming), lemmatisation...etc

Par exemple, le fichier ci-dessus pourrait être transformé en ajoutant les POS:


```JSON
{"token": "la", "position": "1 2 3 1 1 1 6 433 1", "pos": "DET"}
{"token": "fonctionnalité", "position": "1 2 3 1 1 1 6 435 1", "pos": "N"}
{"token": "recherchée", "position": "1 2 3 1 1 1 6 450 1", "pos": "VERB"}
...
```

### Création des ngrams ###
À cette étape, on va créer la représentation en ngrams de chaque texte, et un index de tous les ngrams ordonné par fréquence des ngrams à travers tout le corpus (afin de pouvoir filtrer par fréquence à l'étape suivante)
Le format proposé pour cette transformation est le suivant:

Exemple d'une représentation en trigrammes:

```JSON
[
  [["la", "1 2 3 1 1 1 6 433 1"], ["fonctionnalité", "1 2 3 1 1 1 6 435 1"], ["recherché", "1 2 3 1 1 1 6 450 1"]],
  [["fonctionnalité", "1 2 3 1 1 1 6 435 1"], ["recherché", "1 2 3 1 1 1 6 450 1"], ["tant", "1 2 3 1 1 1 6 460 1"]],
  [["recherché", "1 2 3 1 1 1 6 450 1"], ["tant", "1 2 3 1 1 1 6 460 1"], ["par", "1 2 3 1 1 1 6 460 1"]]
  ...
 ]
 ```
 Il s'agit d'une liste de ngram où chaque ngram est constitué d'une liste d'objet contenant le token et sa position dans le texte. Chaque fichier est sauvegardé au format JSON.
 
 Pour l'index de ngrams, on sauvegarde une liste de ngrams ordonné par fréquence, comme par exemple:
 
 ```JSON
 [["tout", "ce", "que"], ["il", "faut", "que"], ["que", "je", "vous"], ["ce", "qu", "il"], ["ce", "que", "je"], ["en", "ces", "lieux"], ["je", "ne", "puis"], ["premiere", "fois", "le"]...]
 ```
 
 ### Comparaison ###
 
 
