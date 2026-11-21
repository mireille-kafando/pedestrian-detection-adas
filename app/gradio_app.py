import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

# Charger le modèle
# Poids de la v2 (2026) ; ceux de la v1 restent dans models/best.pt
model_path = os.path.join(os.path.dirname(__file__), "..", "v2", "best_v2.pt")
model = YOLO(model_path)

def predict(image):
    # Gradio fournit un tableau RGB ; Ultralytics attend du BGR (correction 2026)
    image_np = np.array(image)[:, :, ::-1]

    # Prédire avec le modèle
    results = model(image_np)
    
    # Vérifier et annoter les résultats
    if results:
        result = results[0]
        annotated_image = result.plot()  # Annoter l'image avec les prédictions (image BGR)
        return Image.fromarray(annotated_image[:, :, ::-1])  # BGR -> RGB pour l'affichage
    else:
        return image  # Retourner l'image originale si aucune prédiction trouvée

# Définir l'interface Gradio
iface = gr.Interface(fn=predict, inputs="image", outputs="image")

# Lancer l'application Gradio
iface.launch()
