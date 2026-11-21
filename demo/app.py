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
# Dans le Space, best.pt est à côté de app.py ; dans le dépôt GitHub, il est dans models/.
POIDS = next(p for p in (ICI / "best.pt", ICI.parent / "models" / "best.pt") if p.exists())

modele = YOLO(str(POIDS))
LIBELLES = {"pedestrian": "piéton / pedestrian", "rider": "cycliste ou motard / rider"}

# Images d'exemple fournies avec la bibliothèque Ultralytics (pas d'images Cityscapes).
ASSETS = Path(ultralytics.__file__).parent / "assets"
EXEMPLES = [str(ASSETS / nom) for nom in ("bus.jpg", "zidane.jpg") if (ASSETS / nom).exists()]


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
**Projet PIF6004, UQTR.** Modèle **YOLOv8n** entraîné sur des images urbaines (Cityscapes) pour détecter
deux classes : *pedestrian* (piéton) et *rider* (cycliste ou motard).
Ajoutez une photo de rue ou essayez un exemple. Le modèle a été entraîné sur des vues de caméra embarquée :
il est plus fiable sur des scènes de circulation que sur des portraits ou des photos en gros plan.

*YOLOv8n trained on Cityscapes street scenes to detect pedestrians and riders (course project, UQTR).
Upload a street photo or try an example. It works best on dashcam-like traffic scenes.*

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
        gr.Examples(EXEMPLES, inputs=entree, label="Exemples (images fournies avec Ultralytics) / examples")
    entree.change(detecter, inputs=[entree, seuil], outputs=[sortie, resume])
    bouton.click(detecter, inputs=[entree, seuil], outputs=[sortie, resume])
    seuil.release(detecter, inputs=[entree, seuil], outputs=[sortie, resume])

if __name__ == "__main__":
    demo.launch()
