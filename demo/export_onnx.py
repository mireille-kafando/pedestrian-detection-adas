"""Export des poids YOLOv8 (models/best.pt) au format ONNX pour la démo dans le navigateur.

La démo en ligne (https://mireille-kafando.github.io/demo-pietons/) exécute ce modèle avec
ONNX Runtime Web. L'export est dynamique, ce qui permet le même redimensionnement rectangulaire
qu'Ultralytics (côté le plus long à 640, autre côté complété au multiple de 32).

Utilisation, depuis la racine du dépôt :
    pip install ultralytics onnx onnxslim
    python demo/export_onnx.py
"""
from pathlib import Path

from ultralytics import YOLO

RACINE = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    chemin = YOLO(str(RACINE / "models" / "best.pt")).export(
        format="onnx", imgsz=640, opset=17, simplify=True, dynamic=True
    )
    print(f"Modèle ONNX écrit dans {chemin}")
