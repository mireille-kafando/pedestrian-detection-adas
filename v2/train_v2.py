"""ADAS v2 : YOLOv8n affiné à partir des poids pré-entraînés COCO (yolov8n.pt).

Jeu de données (voir prepare_donnees.py et splits.json) : les 75 images annotées de la v1, réparties
sans recouvrement en train (48), val (12) et test (14). Les poids sont choisis sur val ; le test
n'intervient qu'une fois, à la fin (evaluate_v2.py).

Usage, depuis la racine du dépôt :
    python v2/prepare_donnees.py <dossier des images annotées de la v1>
    python v2/train_v2.py
"""
from pathlib import Path

from ultralytics import YOLO

ICI = Path(__file__).resolve().parent

if __name__ == "__main__":
    YOLO("yolov8n.pt").train(  # poids pré-entraînés sur COCO, téléchargés par Ultralytics
        data=str(ICI / "data.yaml"), epochs=150, patience=40, imgsz=640, batch=8,
        device="cpu", workers=2, cache="ram", seed=0, deterministic=True,
        project=str(ICI / "runs"), name="v2", exist_ok=True,
    )
