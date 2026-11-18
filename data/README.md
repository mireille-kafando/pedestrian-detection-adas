# Données

Les données ne sont pas redistribuées dans ce dépôt.

- **Cityscapes** (approches EfficientNetB0 et U-Net, essais YOLOv5/YOLOv8) : 5 000 images annotées finement (2 975 entraînement, 500 validation, 1 525 test), annotations en polygones (`gtFine`). Téléchargement après inscription sur [cityscapes-dataset.com](https://www.cityscapes-dataset.com/). Sa licence interdit la redistribution.
- **Jeu annoté à la main** (approche YOLOv8) : 60 images d'entraînement et 15 de validation, classes `pedestrian` et `rider`, annotées avec CVAT au format YOLO (`images/` et `labels/`, fichier `data.yaml`).

Les notebooks lisent les données depuis Google Drive (`/content/drive/MyDrive/projet/...`) ; adapter ces chemins.

---

Data are not redistributed. Cityscapes must be downloaded from its website after registration; the hand-annotated YOLOv8 dataset (75 images, CVAT, YOLO format) is described above.
