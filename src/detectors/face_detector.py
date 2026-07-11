"""Détecteur 'face' (venv face) — InsightFace buffalo_l.

Détecte les visages et leur genre. Fiable quand le visage est visible ;
les cas sans visage (dos, visage caché) sont couverts par le détecteur 'body'.

Contrat CLI/JSON : voir ARCHITECTURE.md §4.1.
"""
import argparse
import json
import os

import cv2
from tqdm import tqdm

# mode 'plain' (interface) : lignes de progression simples au lieu de tqdm
PLAIN = os.environ.get("IASG_PROGRESS") == "plain"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--project-root", required=True)
    args = p.parse_args()

    import onnxruntime as ort
    cuda = "CUDAExecutionProvider" in ort.get_available_providers()
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if cuda else ["CPUExecutionProvider"])

    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l",
                       root=f"{args.project_root}/models/insightface",
                       allowed_modules=["detection", "genderage"],
                       providers=providers)
    app.prepare(ctx_id=0 if cuda else -1, det_size=(640, 640))
    print(f"exécution sur {'GPU (CUDA)' if cuda else 'CPU'}", flush=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"impossible d'ouvrir {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = {}
    idx = 0
    step = max(1, total // 20)
    with tqdm(total=total, desc="[face]", unit="f", disable=PLAIN) as bar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if PLAIN and idx % step == 0:
                print(f"{idx}/{total} frames", flush=True)
            if idx % args.stride == 0:
                dets = []
                for face in app.get(frame):
                    dets.append({
                        "bbox": [round(float(v), 1) for v in face.bbox],
                        "gender": "male" if face.sex == "M" else "female",
                        # le module genre d'InsightFace est un argmax sans
                        # probabilité : la confiance de détection sert de proxy
                        "conf": round(float(face.det_score), 3),
                    })
                if dets:
                    frames[str(idx)] = dets
            idx += 1
            bar.update(1)
    cap.release()

    with open(args.output, "w") as f:
        json.dump({"detector": "face", "fps": fps, "total_frames": idx,
                   "stride": args.stride, "frames": frames}, f)


if __name__ == "__main__":
    main()
