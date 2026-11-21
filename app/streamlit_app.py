import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

# Charger le modèle
# Poids de la v2 (2026) ; ceux de la v1 restent dans models/best.pt
model_path = os.path.join(os.path.dirname(__file__), "..", "v2", "best_v2.pt")
model = YOLO(model_path)

st.title("Détection de Piétons avec YOLOv8")
st.write("Téléchargez une image pour détecter les piétons.")

uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Image Téléchargée.', use_container_width=True)
    st.write("")
    st.write("Détection en cours...")
    
    # Convertir l'image PIL en tableau numpy, en BGR comme l'attend Ultralytics (correction 2026)
    image_np = np.array(image)[:, :, ::-1]
    
    # Prédire avec le modèle
    results = model(image_np)
    
    # Vérifier et annoter les résultats
    if results:
        result = results[0]
        annotated_image = result.plot()  # Annoter l'image avec les prédictions (image BGR)

        # Afficher l'image annotée
        st.image(annotated_image, caption='Image Annotée.', channels="BGR", use_container_width=True)
    else:
        st.write("Aucune prédiction trouvée.")
