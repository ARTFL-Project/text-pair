# Specifications #

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

