"""Détecteur 'body' (venv body) — YOLOv8 (personnes) + CLIP zero-shot (genre).

Couvre les cas sans visage exploitable : personne de dos, visage caché ou
flouté, silhouette partielle, faible résolution. Chaque personne détectée par
YOLO est classée homme/femme par CLIP sur le crop du corps entier, avec un
panel de prompts couvrant les vues difficiles.

Contrat CLI/JSON : voir ARCHITECTURE.md §4.1.
"""
import argparse
import json

import cv2
import torch
from tqdm import tqdm

MALE_PROMPTS = [
    "a photo of a man",
    "a photo of a man seen from behind",
    "a man with his face hidden or covered",
    "the silhouette of a man",
    "a partial view of a man's body",
]
FEMALE_PROMPTS = [
    "a photo of a woman",
    "a photo of a woman seen from behind",
    "a woman with her face hidden or covered",
    "the silhouette of a woman",
    "a partial view of a woman's body",
]

MIN_CROP_SIDE = 32   # crops plus petits : trop peu d'information, genre inconnu
PERSON_CONF = 0.35   # seuil YOLO de détection de personne


class GenderClassifier:
    def __init__(self, project_root, device):
        import open_clip
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k",
            cache_dir=f"{project_root}/models/openclip")
        self.model.eval().to(device)
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        with torch.no_grad():
            text = tokenizer(MALE_PROMPTS + FEMALE_PROMPTS).to(device)
            feats = self.model.encode_text(text)
            self.text_feats = feats / feats.norm(dim=-1, keepdim=True)
        self.n_male = len(MALE_PROMPTS)

    @torch.no_grad()
    def classify(self, crops_bgr):
        """crops BGR (OpenCV) → [(gender, conf)], somme des probas par genre."""
        from PIL import Image
        batch = torch.stack([
            self.preprocess(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
            for c in crops_bgr]).to(self.device)
        feats = self.model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        probs = (100.0 * feats @ self.text_feats.T).softmax(dim=-1)
        p_male = probs[:, :self.n_male].sum(dim=1)
        p_female = probs[:, self.n_male:].sum(dim=1)
        out = []
        for pm, pf in zip(p_male.tolist(), p_female.tolist()):
            gender = "male" if pm >= pf else "female"
            out.append((gender, round(max(pm, pf), 3)))
        return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--project-root", required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from ultralytics import YOLO
    yolo = YOLO(f"{args.project_root}/models/yolo/yolov8s.pt")
    clf = GenderClassifier(args.project_root, device)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"impossible d'ouvrir {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = {}
    idx = 0
    with tqdm(total=total, desc="[body]", unit="f") as bar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % args.stride == 0:
                dets = analyze_frame(frame, yolo, clf, device)
                if dets:
                    frames[str(idx)] = dets
            idx += 1
            bar.update(1)
    cap.release()

    with open(args.output, "w") as f:
        json.dump({"detector": "body", "fps": fps, "total_frames": idx,
                   "stride": args.stride, "frames": frames}, f)


def analyze_frame(frame, yolo, clf, device):
    res = yolo.predict(frame, classes=[0], conf=PERSON_CONF,
                       device=device, verbose=False)[0]
    boxes, crops, dets = [], [], []
    for b in res.boxes:
        x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 - x1 < MIN_CROP_SIDE or y2 - y1 < MIN_CROP_SIDE:
            # trop petit pour classer le genre → 'unknown' (le mode --strict
            # de la fusion pourra le traiter comme positif)
            dets.append({"bbox": [x1, y1, x2, y2],
                         "gender": "unknown",
                         "conf": round(float(b.conf[0]), 3)})
            continue
        boxes.append([x1, y1, x2, y2])
        crops.append(frame[y1:y2, x1:x2])
    if crops:
        for bbox, (gender, conf) in zip(boxes, clf.classify(crops)):
            dets.append({"bbox": bbox, "gender": gender, "conf": conf})
    return dets


if __name__ == "__main__":
    main()
