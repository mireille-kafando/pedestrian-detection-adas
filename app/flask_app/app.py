from flask import Flask, request, render_template, send_file
import os
import cv2
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Nom de l'application
APP_NAME = "Pedestrian_detection"

# Chemin vers le modèle pré-entrainé
model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "best.pt")  # chemin rendu relatif pour le dépôt

# Charger le modèle YOLOv8
model = YOLO(model_path)

# Dossier pour les uploads et résultats
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'

# Extensions de fichiers autorisées
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Fonction pour vérifier les extensions de fichiers
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html', app_name=APP_NAME)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Traiter l'image avec le modèle YOLOv8
        results = model(file_path)
        # Sauvegarder le résultat de la détection
        result_image_path = os.path.join(app.config['RESULT_FOLDER'], filename)
        
        # Convertir les résultats en image avec les détections tracées
        for result in results:
            img = result.plot()  # plot returns the image with detections plotted
            cv2.imwrite(result_image_path, img)
        
        # Retourner l'image traitée
        return send_file(result_image_path, mimetype='image/png')

    return 'File type not allowed'

if __name__ == '__main__':
    # Créer les dossiers s'ils n'existent pas
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
    app.run(debug=True)
