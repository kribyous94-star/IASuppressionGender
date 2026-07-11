# Architecture

## 1. Objectif

Prendre une vidéo + un genre cible, et produire une vidéo identique où toute frame
contenant au moins un individu du genre cible est remplacée par une frame noire,
audio conservé. Tout doit fonctionner hors ligne et être contenu dans le dossier.

## 2. Contraintes structurantes

| Contrainte | Réponse architecturale |
|---|---|
| Tout dans le dossier | venvs dans `./venvs/`, modèles dans `./models/`, caches dans `./.cache/`, ffmpeg statique embarqué via `imageio-ffmpeg` (installé dans le venv) |
| Hors ligne | téléchargements uniquement dans `install.sh` ; `run.sh` exporte `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` |
| Conflits de dépendances | un venv par famille de modèles ; communication inter-venv par sous-processus + JSON, jamais par import |
| Reconnaître sans visage (dos, visage caché…) | ensemble de détecteurs complémentaires (visage **et** corps entier), fusion « OR » |
| Cas non prévus | marge temporelle (`--pad`), fusion de trous (`--gap`), mode `--strict` (personne au genre incertain ⇒ noirci) |

## 3. Les venvs

```
venvs/
├── core/   # orchestration + rendu : opencv-python, numpy, imageio-ffmpeg, tqdm
├── face/   # insightface + onnxruntime (détection visage + attribut genre)
└── body/   # torch + ultralytics (YOLOv8) + open_clip_torch (genre corps entier)
```

- `core` est le point d'entrée (`src/main.py`). Il ne charge aucun modèle IA.
- `face` et `body` sont lancés par `core` en sous-processus :
  `venvs/<nom>/bin/python src/detectors/<nom>_detector.py …`
- Isolation stricte : `face` (onnxruntime) et `body` (torch) ont des piles
  numpy/CUDA potentiellement incompatibles entre elles — c'est la raison d'être
  des venvs séparés. Ajouter un détecteur = ajouter un venv + un script.

## 4. Pipeline

```
                    ┌───────────────────────────────┐
 video.mp4 ────────►│  main.py (venv core)          │
 genre cible        └───────────────┬───────────────┘
                                    │ sous-processus (1 par détecteur)
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
   face_detector.py       body_detector.py         (futurs détecteurs)
   [venv face]            [venv body]              [venv X]
   InsightFace SCRFD      YOLOv8 personnes         pose, démarche,
   + genre par visage     + CLIP zero-shot genre   ré-identification…
              │                     │                     │
              ▼                     ▼                     ▼
        work/face.json        work/body.json         work/X.json
              └─────────────────────┼─────────────────────┘
                                    ▼
                          fusion.py : OR + seuils
                          + expansion stride + pad + gap
                                    ▼
                       ensemble de frames à noircir
                                    ▼
                render.py : réécriture vidéo (frames noires)
                + remux audio (ffmpeg statique embarqué)
                                    ▼
                              output.mp4
```

### 4.1 Contrat JSON des détecteurs

Chaque détecteur est un script CLI autonome :

```
python <script> --video <in> --output <out.json> --stride <N> [--project-root <dir>]
```

Sortie (seules les frames avec détections apparaissent) :

```json
{
  "detector": "face",
  "fps": 30.0,
  "total_frames": 4521,
  "stride": 3,
  "frames": {
    "12": [ {"bbox": [x1, y1, x2, y2], "gender": "female", "conf": 0.93} ]
  }
}
```

- `gender` ∈ `male` / `female` / `unknown` ; `conf` ∈ [0, 1].
- Un détecteur rapporte **toutes** les personnes vues (les deux genres) :
  c'est la fusion qui applique le genre cible et les seuils. Cela permet de
  changer de cible ou de seuil sans relancer l'analyse (JSON réutilisables
  via `--keep-work`).

### 4.2 Détecteur `face` (venv face)

- **Modèle** : InsightFace pack `buffalo_l` (SCRFD 10G détection + module genre/âge),
  exécuté par onnxruntime CPU (ou GPU avec `install.sh --gpu`).
- **Forces** : très fiable quand le visage est visible, même petit ou de profil.
- **Limites** : aveugle si visage caché ou de dos → couvert par `body`.
- `conf` = score de détection du visage (le module genre d'InsightFace est un argmax).

### 4.3 Détecteur `body` (venv body)

- **Étape 1** : YOLOv8s détecte les personnes (classe 0) — fonctionne de dos,
  de loin, partiellement masqué.
- **Étape 2** : chaque crop de personne est classé homme/femme par CLIP
  (ViT-B/32, poids laion2b) en zero-shot avec un panel de prompts couvrant
  les cas difficiles : vu de dos, de loin, visage couvert, silhouette…
  Les probabilités des prompts d'un même genre sont sommées.
- **Forces** : couvre exactement les angles morts du détecteur visage
  (dos, visage masqué, morphologie/silhouette).
- **Limites** : moins précis qu'un visage net → seuil par défaut plus haut (0.60),
  et le mode `--strict` traite les crops incertains comme positifs.

### 4.4 Fusion (`src/fusion.py`, venv core)

1. Une frame analysée est **flaguée** si *au moins un* détecteur y voit le genre
   cible avec `conf ≥ seuil` du détecteur (logique OR : on préfère un faux
   positif — frame noire en trop — à un faux négatif).
2. En mode `--strict`, une détection `unknown` ou sous le seuil compte aussi
   comme positive (personne présente dont on ne peut pas exclure le genre cible).
3. **Expansion stride** : la frame analysée `i` couvre `[i, i+stride)`.
4. **Pad** : chaque zone flaguée est étendue de `pad` secondes de chaque côté.
5. **Gap** : deux zones distantes de moins de `gap` secondes sont fusionnées
   (évite les « clignotements » de quelques frames visibles).

### 4.5 Rendu (`src/render.py`, venv core)

- Relecture de la vidéo avec OpenCV, écriture d'une frame noire (même résolution)
  pour chaque index flagué, copie telle quelle sinon.
- Remux : `ffmpeg -i rendu.mp4 -i original -map 0:v -map 1:a? -c:a copy`
  avec le binaire statique fourni par `imageio-ffmpeg` (dans le venv core,
  donc dans le dossier). La sortie garde l'audio, la durée et le fps d'origine.

## 5. Scripts

### 5.1 `install.sh`

1. Vérifie `python3 -m venv`.
2. Force tous les caches dans le dossier : `PIP_CACHE_DIR`, `HF_HOME`,
   `TORCH_HOME`, `XDG_CACHE_HOME` → `./.cache/`.
3. Crée `venvs/core`, `venvs/face`, `venvs/body` et installe
   `requirements/{core,face,body}.txt`. Par défaut torch/onnxruntime **CPU**
   (léger) ; `--gpu` installe les variantes CUDA.
4. Télécharge les modèles dans `./models/` en les instanciant une fois :
   - `models/insightface/models/buffalo_l/` (InsightFace)
   - `models/yolo/yolov8s.pt` (Ultralytics)
   - `models/openclip/` (poids CLIP via HF hub, `cache_dir` forcé)
5. Idempotent : relançable sans tout retélécharger.

### 5.2 `run.sh`

Exporte les mêmes variables de cache + `HF_HUB_OFFLINE=1`, puis
`exec venvs/core/bin/python src/main.py "$@"`.

## 6. Ajouter un détecteur (checklist)

1. `requirements/<nom>.txt` + création du venv dans `install.sh`.
2. `src/detectors/<nom>_detector.py` respectant le contrat CLI/JSON (§4.1).
3. Déclarer le détecteur dans le registre `DETECTORS` de `src/main.py`
   (nom → venv + script + seuil par défaut).

Candidats futurs : estimation de pose + classification de démarche,
MiVOLO (genre/âge corps+visage), segmentation pour noircir seulement la
personne au lieu de la frame entière, tracking (ByteTrack) pour propager
l'identité entre frames.

## 7. Point d'extension API (optionnel, jamais requis)

Un détecteur « api » suivrait exactement le même contrat (§4.1) : un script dans
`src/detectors/` qui appelle un service externe (cloud vision, etc.) et émet le
même JSON. Il serait opt-in via `--detectors face,body,api` et n'est pas installé
par défaut — le cœur du logiciel reste strictement hors ligne.
