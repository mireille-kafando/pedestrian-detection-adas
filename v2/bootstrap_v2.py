"""Intervalles de confiance par bootstrap du mAP de la v1 et de la v2 sur le jeu de test.

Le test ne compte que 14 images. On récupère les statistiques image par image calculées par
l'évaluateur d'Ultralytics (mêmes réglages qu'evaluate_v2.py), puis on rééchantillonne les images
avec remise (2 000 tirages, graine 0). À chaque tirage, le mAP des deux modèles est recalculé sur
les mêmes images, ce qui donne un intervalle à 95 % (percentiles) pour chaque modèle et pour la
différence v2 − v1.
"""
import csv
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.utils import metrics

ICI = Path(__file__).resolve().parent
POIDS = {"v1": ICI.parent / "models" / "best.pt", "v2": ICI / "best_v2.pt"}
N_TIRAGES, GRAINE = 2000, 0


def stats_par_image(version):
    """Lance l'évaluation officielle et capture les statistiques de chaque image."""
    capture = {}
    original = metrics.DetMetrics.update_stats

    def espion(self, stat):
        capture[stat["im_name"]] = {k: np.asarray(stat[k]) for k in ("tp", "conf", "pred_cls", "target_cls")}
        return original(self, stat)

    metrics.DetMetrics.update_stats = espion
    try:
        r = YOLO(str(POIDS[version])).val(data=str(ICI / "data.yaml"), split="test", imgsz=640, batch=8,
                                         device="cpu", conf=0.001, iou=0.7, plots=False, verbose=False,
                                         project=str(ICI / "runs_eval"), name=f"bootstrap_{version}",
                                         exist_ok=True)
    finally:
        metrics.DetMetrics.update_stats = original
    return capture, (r.box.map50, r.box.map)


def map_images(stats, images):
    s = {k: np.concatenate([stats[i][k] for i in images]) for k in ("tp", "conf", "pred_cls", "target_cls")}
    ap = metrics.ap_per_class(s["tp"], s["conf"], s["pred_cls"], s["target_cls"])[5]
    return ap[:, 0].mean(), ap.mean()  # mAP@0,5 et mAP@0,5:0,95


if __name__ == "__main__":
    stats, officiel = {}, {}
    for v in POIDS:
        stats[v], officiel[v] = stats_par_image(v)
    images = np.array(sorted(stats["v1"]))
    assert sorted(stats["v2"]) == list(images)
    for v in POIDS:  # le recalcul sur toutes les images redonne les valeurs officielles
        assert np.allclose(map_images(stats[v], images), officiel[v], atol=1e-6)

    rng = np.random.default_rng(GRAINE)
    tirages = {m: {"v1": [], "v2": [], "difference": []} for m in ("mAP50", "mAP50_95")}
    for _ in range(N_TIRAGES):
        echantillon = rng.choice(images, size=len(images), replace=True)
        a, b = map_images(stats["v1"], echantillon), map_images(stats["v2"], echantillon)
        for j, m in enumerate(("mAP50", "mAP50_95")):
            tirages[m]["v1"].append(a[j]), tirages[m]["v2"].append(b[j]), tirages[m]["difference"].append(b[j] - a[j])

    lignes = []
    for j, m in enumerate(("mAP50", "mAP50_95")):
        point = {"v1": officiel["v1"][j], "v2": officiel["v2"][j]}
        point["difference"] = point["v2"] - point["v1"]
        for nom in ("v1", "v2", "difference"):
            t = np.array(tirages[m][nom])
            bas, haut = np.percentile(t, [2.5, 97.5])
            lignes.append({"mesure": m, "modele": nom, "valeur": round(float(point[nom]), 3),
                           "ic95_bas": round(float(bas), 3), "ic95_haut": round(float(haut), 3),
                           "part_tirages_v2_superieure": round(float(np.mean(t > 0)), 4) if nom == "difference" else ""})
            print(lignes[-1])
    (ICI / "results").mkdir(exist_ok=True)
    with open(ICI / "results" / "bootstrap_map_test.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lignes[0]))
        w.writeheader()
        w.writerows(lignes)
