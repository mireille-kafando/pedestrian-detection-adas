"""Export des poids YOLOv8 au format ONNX pour la démo dans le navigateur.

Par défaut, exporte les poids de la v2 (v2/best_v2.pt), utilisés par la démo en ligne ;
`python demo/export_onnx.py models/best.pt` exporte ceux de la v1.

La démo en ligne (https://mireille-kafando.github.io/demo-pietons/) exécute ce modèle avec
ONNX Runtime Web. L'export est dynamique, ce qui permet le même redimensionnement rectangulaire
qu'Ultralytics (côté le plus long à 640, autre côté complété au multiple de 32).

Utilisation, depuis la racine du dépôt :
    pip install ultralytics onnx onnxslim
    python demo/export_onnx.py [chemin des poids]
"""
import sys
from pathlib import Path

from ultralytics import YOLO

RACINE = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    poids = Path(sys.argv[1]) if len(sys.argv) > 1 else RACINE / "v2" / "best_v2.pt"
    chemin = YOLO(str(poids)).export(
        format="onnx", imgsz=640, opset=17, simplify=True, dynamic=True
    )
    print(f"Modèle ONNX écrit dans {chemin}")
