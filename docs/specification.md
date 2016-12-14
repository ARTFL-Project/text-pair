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
Plusieurs étapes de prétraitement avec une contrainte: toute modification du texte doit donner en sortie le même format que celui précisé en dessus, avec bien-entendu des infos supplémentaires si nécessaires.
