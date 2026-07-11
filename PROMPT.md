# Prompt amélioré

Version reformulée et précisée de la demande initiale — utile pour reprendre le
projet, briefer un autre agent, ou vérifier que le logiciel répond bien au besoin.

---

## Prompt

> Développe un logiciel en Python nommé **IASuppressionGender** avec les
> spécifications suivantes :
>
> **Fonction.** En entrée : un fichier vidéo et un genre cible (`homme` ou
> `femme`). En sortie : une vidéo identique (même durée, même fps, audio
> conservé) où chaque frame contenant au moins un individu du genre cible est
> remplacée par une frame entièrement noire.
>
> **Détection robuste multi-modèles.** La présence du genre cible doit être
> détectée par plusieurs approches complémentaires exécutées successivement,
> car aucune ne suffit seule :
> 1. par le **visage** (quand il est visible) ;
> 2. par le **corps entier / la morphologie / la silhouette** (personne de dos,
>    visage caché ou flouté, personne partiellement visible, faible résolution).
>
> Les résultats sont fusionnés en OR : un seul détecteur positif suffit à
> noircir la frame. Prévoir des seuils de confiance réglables par détecteur,
> une marge temporelle autour des détections, et un mode strict qui noircit
> aussi les personnes dont le genre est incertain. L'architecture doit
> permettre d'ajouter facilement de nouveaux détecteurs (pose, démarche,
> tracking…) pour les cas non anticipés.
>
> **Contraintes d'installation.**
> - Tout (venvs Python, poids des modèles IA, caches pip/HuggingFace/torch,
>   binaire ffmpeg) doit être installé **à l'intérieur du dossier du projet** :
>   supprimer le dossier doit tout supprimer, sans trace ailleurs.
> - Créer **plusieurs venvs** dans un dossier `venvs/` (un par famille de
>   modèles) pour isoler les incompatibilités de dépendances ; les venvs
>   communiquent par sous-processus + fichiers JSON.
>
> **Contraintes d'exécution.**
> - 100 % **hors ligne** après installation : aucune API, aucun téléchargement
>   au runtime (forcer les modes offline). Une option permettant d'ajouter des
>   détecteurs basés sur des API doit exister comme point d'extension, jamais
>   comme dépendance.
> - Fonctionnel sur CPU par défaut ; option GPU à l'installation.
>
> **Livrables.**
> - `install.sh` : installe tout (venvs, dépendances, téléchargement des
>   modèles, libs CUDA via pip pour `--gpu`), idempotent, bascule CPU ↔ GPU.
> - `run.sh` : lance une interface graphique locale (hors ligne) qui facilite
>   l'utilisation : choix de la vidéo, du genre, réglages avancés, journal en
>   direct, prévisualisation du résultat.
> - `cli.sh` : la même chose en ligne de commande
>   (`./cli.sh -i video.mp4 -g femme`).
> - Code source structuré (moteur partagé CLI/interface, détecteurs, fusion,
>   rendu).
> - `README.md` (usage) et `ARCHITECTURE.md` (conception détaillée).
> - Historique **git** propre : un commit par étape logique (docs, scripts,
>   code), messages conventionnels.
>
> Commence par l'architecture et la documentation, puis les scripts, puis le
> code.

---

## Ce que cette version précise par rapport au prompt d'origine

| Point flou d'origine | Décision prise |
|---|---|
| « vidéo identique » | même durée/fps + audio conservé, explicité |
| « plusieurs manières de reconnaître » | 2 détecteurs concrets (visage, corps+CLIP) + registre extensible |
| « tous les cas auxquels je n'ai pas pensé » | pad temporel, fusion de gaps, mode `--strict`, checklist d'ajout de détecteur |
| comportement en cas de doute | biais assumé vers le faux positif (frame noire en trop plutôt que ratée) |
| matériel | CPU par défaut, `--gpu` optionnel |
| « gérer les commits » | un commit par étape logique, format conventionnel |
