"""Démo en ligne : détection de piétons pour l'aide à la conduite (YOLOv8n).

Application Gradio destinée à Hugging Face Spaces. Elle charge les poids
entraînés dans le projet PIF6004 (UQTR) et annote l'image fournie.
"""

from collections import Counter
from pathlib import Path

import gradio as gr
import ultralytics
from ultralytics import YOLO

ICI = Path(__file__).resolve().parent
# Dans un Space, best.pt est à côté de app.py ; dans le dépôt, on prend la v2 (v2/best_v2.pt),
# sinon les poids de la v1 (models/best.pt).
POIDS = next(p for p in (ICI / "best.pt", ICI.parent / "v2" / "best_v2.pt", ICI.parent / "models" / "best.pt")
             if p.exists())

modele = YOLO(str(POIDS))
LIBELLES = {"pedestrian": "piéton / pedestrian", "rider": "cycliste ou motard / rider"}

# Image d'exemple fournie avec la bibliothèque Ultralytics (les images du projet ne sont pas redistribuées).
ASSETS = Path(ultralytics.__file__).parent / "assets"
EXEMPLES = [str(ASSETS / nom) for nom in ("bus.jpg",) if (ASSETS / nom).exists()]


def detecter(image, seuil):
    if image is None:
        return None, "Ajoutez une image. / Upload an image."
    # Gradio fournit du RGB ; Ultralytics interprète les tableaux numpy comme du BGR (convention OpenCV).
    resultat = modele.predict(image[:, :, ::-1], conf=seuil, verbose=False)[0]
    classes = [modele.names[int(c)] for c in resultat.boxes.cls]
    comptes = Counter(classes)
    if not comptes:
        resume = "Aucune détection au-dessus du seuil. / No detection above the threshold."
    else:
        lignes = [f"- **{LIBELLES.get(k, k)}** : {n}" for k, n in comptes.most_common()]
        resume = (
            "\n".join(lignes)
            + f"\n\nTemps d'inférence / inference time : {resultat.speed['inference']:.0f} ms (CPU)"
        )
    annotee = resultat.plot()[:, :, ::-1]  # BGR -> RGB
    return annotee, resume


DESCRIPTION = """
**Projet PIF6004, UQTR.** Modèle **YOLOv8n** qui détecte deux classes : *pedestrian* (piéton) et *rider*
(cycliste ou motard). La version 2 est affinée à partir de poids pré-entraînés sur COCO, sur 75 images de rue
annotées à la main ; elle atteint un mAP@0,5 de 0,86 sur un jeu de test séparé. Ajoutez une photo de rue ou essayez l'exemple.

*YOLOv8n detecting pedestrians and riders (course project, UQTR). Version 2 is fine-tuned from COCO-pretrained
weights on 75 hand-annotated street images and reaches mAP@0.5 = 0.86 on a held-out test set.
Upload a street photo or try the example.*

Code et résultats / code and results : [github.com/mireille-kafando/pedestrian-detection-adas](https://github.com/mireille-kafando/pedestrian-detection-adas)
"""

with gr.Blocks(title="Détection de piétons · YOLOv8") as demo:
    gr.Markdown("# Détection de piétons pour l'aide à la conduite · Pedestrian detection for ADAS")
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            entree = gr.Image(type="numpy", label="Image", height=420)
            seuil = gr.Slider(0.05, 0.90, value=0.25, step=0.05, label="Seuil de confiance / confidence threshold")
            bouton = gr.Button("Détecter / Detect", variant="primary")
        with gr.Column():
            sortie = gr.Image(type="numpy", label="Résultat / result", height=420)
            resume = gr.Markdown()
    if EXEMPLES:
        gr.Examples(EXEMPLES, inputs=entree, label="Exemple (image fournie avec Ultralytics) / example")
    entree.change(detecter, inputs=[entree, seuil], outputs=[sortie, resume])
    bouton.click(detecter, inputs=[entree, seuil], outputs=[sortie, resume])
    seuil.release(detecter, inputs=[entree, seuil], outputs=[sortie, resume])

if __name__ == "__main__":
    demo.launch()
