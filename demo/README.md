---
title: Détection de piétons ADAS
emoji: 🚶
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
license: agpl-3.0
short_description: YOLOv8n pedestrian and rider detection (Cityscapes)
---

# Démo : détection de piétons (YOLOv8n)

Ce dossier contient la démo en ligne du projet, déployée sur Hugging Face Spaces. L'en-tête ci-dessus est la configuration lue par Hugging Face.

| Fichier | Rôle |
|---|---|
| `app.py` | Interface Gradio : envoi d'image, seuil de confiance, image annotée et nombre de détections par classe. |
| `requirements.txt` | Versions de PyTorch (CPU) et d'Ultralytics testées avec l'application. |
| `packages.txt` | Bibliothèques système nécessaires à OpenCV. |

Pour le déploiement, le Space contient ces trois fichiers, ce README et une copie de [`../models/best.pt`](../models/best.pt).

Lancement en local, depuis la racine du dépôt :

```bash
pip install gradio==6.32.0 -r demo/requirements.txt
python demo/app.py
```

Les images d'exemple sont celles fournies avec la bibliothèque Ultralytics ; aucune image Cityscapes n'est redistribuée. La licence AGPL-3.0 du Space suit celle d'Ultralytics.

---

*Online demo of the project (Gradio on Hugging Face Spaces). The Space holds these files plus a copy of `models/best.pt`. The example images ship with the Ultralytics library; no Cityscapes images are redistributed.*
