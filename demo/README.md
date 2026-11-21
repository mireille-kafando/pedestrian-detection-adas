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
short_description: YOLOv8n pedestrian and rider detection (UQTR project)
---

# Démo : détection de piétons (YOLOv8n)

La démo en ligne du projet est sur [mireille-kafando.github.io/demo-pietons](https://mireille-kafando.github.io/demo-pietons/) : le modèle, exporté au format ONNX par `export_onnx.py`, s'y exécute dans le navigateur. Ce dossier contient ce script d'export et une version Gradio de la même démo, à lancer en local ou sur Hugging Face Spaces (l'en-tête ci-dessus est la configuration lue par Hugging Face ; les Spaces Gradio demandent un abonnement payant).

| Fichier | Rôle |
|---|---|
| `app.py` | Interface Gradio (poids de la v2 par défaut) : envoi d'image, seuil de confiance, image annotée et nombre de détections par classe. |
| `requirements.txt` | Versions de PyTorch (CPU) et d'Ultralytics testées avec l'application. |
| `packages.txt` | Bibliothèques système nécessaires à OpenCV. |
| `export_onnx.py` | Export de `models/best.pt` au format ONNX pour la démo dans le navigateur. |

Pour un Space Hugging Face, il faut ces fichiers, ce README et une copie de [`../models/best.pt`](../models/best.pt).

Lancement en local, depuis la racine du dépôt :

```bash
pip install gradio==6.32.0 -r demo/requirements.txt
python demo/app.py
```

L'image d'exemple est celle fournie avec la bibliothèque Ultralytics ; les images du projet ne sont pas redistribuées. La licence AGPL-3.0 du Space suit celle d'Ultralytics.

---

*The live demo runs in the browser at mireille-kafando.github.io/demo-pietons (ONNX export made by `export_onnx.py`). This folder also holds a Gradio version to run locally or on Hugging Face Spaces (with a copy of `models/best.pt`). The example image ships with the Ultralytics library; the project images are not redistributed.*
