import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

# Charger le modèle
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "best.pt")  # chemin rendu relatif pour le dépôt
model = YOLO(model_path)

def predict(image):
    # Convertir l'image PIL en tableau numpy
    image_np = np.array(image)

    # Prédire avec le modèle
    results = model(image_np)
    
    # Vérifier et annoter les résultats
    if results:
        result = results[0]
        annotated_image = result.plot()  # Annoter l'image avec les prédictions
        return Image.fromarray(annotated_image)
    else:
        return image  # Retourner l'image originale si aucune prédiction trouvée

# Définir l'interface Gradio
iface = gr.Interface(fn=predict, inputs="image", outputs="image")

# Lancer l'application Gradio
iface.launch()
