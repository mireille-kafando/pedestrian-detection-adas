"""Évaluation comparée des poids v1 (models/best.pt) et v2 (v2/best_v2.pt) sur le jeu de test.

Les deux modèles sont évalués avec la même version d'Ultralytics et les mêmes réglages
(imgsz 640, seuil de confiance 0,001, IoU de NMS 0,7, IoU de 0,5 pour mAP@0,5).
Le score de la v1 sur ce test est optimiste : ses poids ont été choisis sur ces mêmes images.
"""
import csv
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

ICI = Path(__file__).resolve().parent
POIDS = {"v1": ICI.parent / "models" / "best.pt", "v2": ICI / "best_v2.pt"}
JEUX = {"test": ("data.yaml", "test"), "val_v1_15_images": ("data_val_v1.yaml", "test")}


def evaluer(version, jeu):
    fichier, split = JEUX[jeu]
    r = YOLO(str(POIDS[version])).val(data=str(ICI / fichier), split=split, imgsz=640, batch=8,
                                     device="cpu", conf=0.001, iou=0.7, plots=True,
                                     project=str(ICI / "runs_eval"), name=f"{version}_{jeu}", exist_ok=True,
                                     verbose=False)
    lignes = [{"modele": version, "jeu": jeu, "classe": "toutes",
               "precision": r.box.mp, "rappel": r.box.mr, "mAP50": r.box.map50, "mAP50_95": r.box.map}]
    for i, c in enumerate(r.box.ap_class_index):
        p, rec, ap50, ap = r.box.class_result(i)
        lignes.append({"modele": version, "jeu": jeu, "classe": r.names[int(c)],
                       "precision": p, "rappel": rec, "mAP50": ap50, "mAP50_95": ap})
    return lignes, Path(r.save_dir)


if __name__ == "__main__":
    sortie = ICI / "results"
    (sortie / "figures").mkdir(parents=True, exist_ok=True)
    toutes = []
    for version in ("v1", "v2"):
        for jeu in JEUX:
            if jeu == "val_v1_15_images" and not (ICI / "data_val_v1.yaml").exists():
                continue
            lignes, dossier = evaluer(version, jeu)
            toutes += lignes
            if jeu == "test":
                for nom, cible in [("BoxPR_curve.png", "courbe_precision_rappel"),
                                   ("confusion_matrix.png", "matrice_confusion")]:
                    if (dossier / nom).exists():
                        shutil.copy(dossier / nom, sortie / "figures" / f"{cible}_{version}_test.png")
    with open(sortie / "metriques_v1_v2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(toutes[0]))
        w.writeheader()
        for l in toutes:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in l.items()})
    for l in toutes:
        print(f"{l['modele']} {l['jeu']:<17} {l['classe']:<10} P={l['precision']:.3f} R={l['rappel']:.3f} "
              f"mAP50={l['mAP50']:.3f} mAP50-95={l['mAP50_95']:.3f}", file=sys.stdout)
