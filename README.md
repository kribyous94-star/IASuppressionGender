# IASuppressionGender

Logiciel de traitement vidéo **100 % hors ligne** qui détecte la présence d'individus
d'un genre donné (homme ou femme) dans une vidéo, et produit une copie de la vidéo
où **chaque frame contenant un individu du genre ciblé est remplacée par un écran noir**.
L'audio est conservé.

La détection est **multi-modèles** (ensemble de détecteurs) pour couvrir un maximum
de cas : visage visible, visage caché, personne de dos, personne partielle, etc.

## Points clés

- **Auto-contenu** : venvs, modèles IA, caches et binaires (ffmpeg statique) sont
  installés *dans le dossier du projet*. Supprimer le dossier supprime tout.
- **Hors ligne** : internet n'est requis que pendant `install.sh` (téléchargement
  des modèles). L'exécution force le mode offline (`HF_HUB_OFFLINE=1`).
- **Multi-venvs** : chaque famille de modèles vit dans son propre venv
  (`venvs/core`, `venvs/face`, `venvs/body`) pour isoler les conflits de dépendances.
- **Extensible** : les détecteurs sont des scripts autonomes qui communiquent en JSON ;
  en ajouter un nouveau ne demande pas de toucher aux autres.

## Installation

```bash
./install.sh          # CPU (par défaut)
./install.sh --gpu    # variantes GPU (CUDA) de torch/onnxruntime
```

Nécessite : `python3` (≥ 3.10) avec le module `venv`, et une connexion internet
**pour l'installation uniquement** (~2 à 3 Go de téléchargements : torch, modèles).

## Utilisation

```bash
./run.sh -i video.mp4 -g femme                 # noircit les frames contenant une femme
./run.sh -i video.mp4 -g homme -o resultat.mp4 # idem pour un homme, sortie nommée
```

Options principales (voir `./run.sh --help` pour tout) :

| Option | Défaut | Rôle |
|---|---|---|
| `-i, --input` | — | Vidéo d'entrée |
| `-g, --gender` | — | Genre ciblé : `homme`/`femme` (ou `male`/`female`) |
| `-o, --output` | `<input>_censored.mp4` | Vidéo de sortie |
| `--detectors` | `face,body` | Détecteurs à utiliser, séparés par des virgules |
| `--stride` | `3` | Analyse 1 frame sur N (1 = toutes, plus lent) |
| `--pad` | `0.25` | Marge de sécurité (secondes) noircie autour de chaque détection |
| `--gap` | `0.5` | Fusionne deux zones noires séparées de moins de N secondes |
| `--face-thr` | `0.55` | Seuil de confiance du détecteur visage |
| `--body-thr` | `0.60` | Seuil de confiance du détecteur corps |
| `--strict` | off | Noircit aussi quand une personne est détectée sans genre certain |
| `--keep-work` | off | Conserve les JSON d'analyse dans `work/` (debug) |

## Comment ça marche (résumé)

1. **Analyse** : chaque détecteur (dans son venv) parcourt la vidéo et émet un JSON
   listant, frame par frame, les personnes détectées avec genre + confiance.
   - `face` : InsightFace (SCRFD + genre) — précis quand le visage est visible.
   - `body` : YOLOv8 (détection de personnes) + CLIP zero-shot sur le corps entier —
     fonctionne de dos, visage masqué, silhouette partielle.
2. **Fusion** : les résultats sont combinés (un seul détecteur positif suffit),
   étendus temporellement (`--pad`, `--gap`) pour éviter les frames ratées.
3. **Rendu** : la vidéo est réécrite avec des frames noires sur les zones flaguées,
   puis l'audio original est remixé via le ffmpeg statique embarqué.

Détails complets : [ARCHITECTURE.md](ARCHITECTURE.md).

## Structure du dossier

```
IASuppressionGender/
├── install.sh            # installe venvs + modèles, tout DANS le dossier
├── run.sh                # lance le logiciel (mode offline forcé)
├── requirements/         # dépendances par venv (core/face/body)
├── src/
│   ├── main.py           # orchestrateur CLI (venv core)
│   ├── fusion.py         # combinaison des détections + lissage temporel
│   ├── render.py         # écriture vidéo noircie + remux audio
│   └── detectors/        # scripts autonomes, un par venv de détection
│       ├── face_detector.py
│       └── body_detector.py
├── venvs/                # (généré) venvs isolés — non versionné
├── models/               # (généré) poids des modèles IA — non versionné
├── .cache/               # (généré) caches pip/HF/torch — non versionné
└── work/                 # (généré) fichiers intermédiaires d'un job
```

## Options en ligne (désactivées par défaut)

Le logiciel n'appelle **aucune API** par défaut. Une option `--api <provider>` est
prévue dans l'architecture comme point d'extension (voir ARCHITECTURE.md §7) mais
tout fonctionne sans.

## Avertissement

La classification de genre par apparence est intrinsèquement approximative
(erreurs possibles, cas ambigus). Utilisez ce logiciel uniquement sur des contenus
que vous avez le droit de traiter, dans le respect des lois applicables.
