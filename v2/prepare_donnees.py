"""Reconstitue le jeu de données de la v2 à partir du dossier d'images annotées de la v1.

Usage : python v2/prepare_donnees.py <dossier_v1>
où <dossier_v1> contient images/{train,val} et labels/{train,val} (75 images annotées avec CVAT,
non redistribuées dans ce dépôt). Le partage est lu dans splits.json :
- train (48) et val (12) : images d'entraînement de la v1, tirage aléatoire avec la graine 2026 ;
- test (14) : images de validation de la v1, sans 13.jpg (même photo que 12.jpg, annotée deux fois).
Écrit v2/data/ (jeu train/val/test), v2/data_val_v1/ (les 15 images de validation d'origine)
et les fichiers data.yaml et data_val_v1.yaml correspondants.
"""
import json
import shutil
import sys
from pathlib import Path

ICI = Path(__file__).resolve().parent
NOMS = "names:\n  0: pedestrian\n  1: rider\n"


def copier(source, sous_dossier, fichiers, cible):
    for genre, ext in (("images", ".jpg"), ("labels", ".txt")):
        (cible / genre).mkdir(parents=True, exist_ok=True)
        for f in fichiers:
            shutil.copy(source / genre / sous_dossier / (Path(f).stem + ext), cible / genre / (Path(f).stem + ext))


if __name__ == "__main__":
    source = Path(sys.argv[1]).resolve()
    partage = json.loads((ICI / "splits.json").read_text())
    for split, info in partage.items():
        copier(source, info["source"].split()[-1], info["images"], ICI / "data" / split)
    (ICI / "data.yaml").write_text(
        f"path: {ICI / 'data'}\ntrain: train/images\nval: val/images\ntest: test/images\n{NOMS}")
    origine = sorted((source / "images" / "val").iterdir())
    copier(source, "val", [p.name for p in origine], ICI / "data_val_v1")
    (ICI / "data_val_v1.yaml").write_text(
        f"path: {ICI / 'data_val_v1'}\ntrain: images\nval: images\ntest: images\n{NOMS}")
    print("Jeu de données v2 prêt :", {k: len(v["images"]) for k, v in partage.items()}, "+ 15 images de validation v1")
