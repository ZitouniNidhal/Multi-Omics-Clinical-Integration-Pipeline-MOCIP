# Multi-Omics Clinical Integration Pipeline

## Gestion des données lourdes

Ce dépôt contient des données et des résultats qui peuvent être très volumineux. Pour éviter les erreurs GitHub liées aux limites de taille de fichier et aux timeouts de push, suivez ces bonnes pratiques :

- ne pas commiter de fichiers de données brutes (`data/raw/...`) dans le dépôt principal
- ignorer les dossiers générés ou de sortie :
  - `data/`
  - `results/`
  - `logs/`
- utiliser Git LFS pour les fichiers volumineux nécessaires au suivi

## Git LFS recommandé

Si vous devez conserver des fichiers volumineux dans l’historique, utilisez Git LFS :

```bash
git lfs install
git lfs track "*.txt"
git lfs track "*.csv"
git lfs track "*.seg"
git lfs track "*.png"
```

Ensuite, ajoutez et commitez :

```bash
git add .gitattributes
git add <vos-fichiers-volumineux>
git commit -m "Track large files with Git LFS"
```

## Recommandation rapide

- Garder le code source dans `src/`
- Garder les données lourdes à l’extérieur du dépôt ou en provenance de sources externes
- Ne pas versionner les gros fichiers qui dépassent 100 MB
- Utiliser `git status` avant chaque push pour vérifier l’état
