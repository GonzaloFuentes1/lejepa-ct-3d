"""
Tabla comparativa de resultados LP contra baselines de la competencia CVPR 2026.

Lee los resultados de train_linear_probe.py y genera una tabla comparativa
con los baselines publicados (CT-NEXUS, 3DINO, SPECTRE, Curia, Merlin, VoCo, CT-FM, Pillar-0).

Uso:
    python scripts/compare_baselines.py \\
        --lp_results_root $LEJEPA_DATA_ROOT/lejepa_eval/lp_results \\
        --output          $LEJEPA_DATA_ROOT/lejepa_eval/comparison_table.csv

    # Solo AUROC
    python scripts/compare_baselines.py \\
        --lp_results_root $LEJEPA_DATA_ROOT/lejepa_eval/lp_results \\
        --metric          auroc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Baselines publicados — Task 1 LP (BalAcc / AUROC)
# Fuente: https://www.codabench.org/competitions/12650/ — Baselines section
# ---------------------------------------------------------------------------
BASELINES_LP_BAL_ACC = {
    "CT-NEXUS": {
        "adrenal_hyperplasia":  0.657, "ascites":             0.632, "atherosclerosis":     0.618,
        "cholecystitis":        0.627, "colorectal_cancer":   0.558, "fatty_liver":         0.756,
        "gallstone":            0.621, "hydronephrosis":      0.594, "kidney_stone":        0.665,
        "liver_calcifications": 0.597, "liver_cyst":          0.619, "liver_lesion":        0.500,
        "lymphadenopathy":      0.611, "renal_cyst":          0.635, "splenomegaly":        0.671,
        "covid":                0.688, "malignancy":          0.520,
    },
    "3DINO": {
        "adrenal_hyperplasia":  0.7286, "ascites":             0.7353, "atherosclerosis":     0.7794,
        "cholecystitis":        0.6961, "colorectal_cancer":   0.6923, "fatty_liver":         0.8590,
        "gallstone":            0.6983, "hydronephrosis":      0.7969, "kidney_stone":        0.6772,
        "liver_calcifications": 0.6250, "liver_cyst":          0.6148, "liver_lesion":        0.6940,
        "lymphadenopathy":      0.6944, "renal_cyst":          0.6617, "splenomegaly":        0.8714,
        "covid":                0.7969, "malignancy":          0.6050,
    },
    "SPECTRE": {
        "adrenal_hyperplasia":  0.7000, "ascites":             0.8676, "atherosclerosis":     0.7794,
        "cholecystitis":        0.7647, "colorectal_cancer":   0.7115, "fatty_liver":         0.9103,
        "gallstone":            0.7500, "hydronephrosis":      0.8750, "kidney_stone":        0.6646,
        "liver_calcifications": 0.6250, "liver_cyst":          0.7254, "liver_lesion":        0.7090,
        "lymphadenopathy":      0.7778, "renal_cyst":          0.6729, "splenomegaly":        0.8429,
        "covid":                1.0000, "malignancy":          0.7400,
    },
    "Curia": {
        "adrenal_hyperplasia":  0.786, "ascites":             0.750, "atherosclerosis":     0.809,
        "cholecystitis":        0.755, "colorectal_cancer":   0.750, "fatty_liver":         0.949,
        "gallstone":            0.819, "hydronephrosis":      0.906, "kidney_stone":        0.677,
        "liver_calcifications": 0.667, "liver_cyst":          0.668, "liver_lesion":        0.694,
        "lymphadenopathy":      0.806, "renal_cyst":          0.688, "splenomegaly":        0.900,
        "covid":                0.844, "malignancy":          0.640,
    },
    "Merlin": {
        "adrenal_hyperplasia":  0.771, "ascites":             0.794, "atherosclerosis":     0.809,
        "cholecystitis":        0.696, "colorectal_cancer":   0.712, "fatty_liver":         0.833,
        "gallstone":            0.681, "hydronephrosis":      0.719, "kidney_stone":        0.671,
        "liver_calcifications": 0.569, "liver_cyst":          0.660, "liver_lesion":        0.657,
        "lymphadenopathy":      0.778, "renal_cyst":          0.695, "splenomegaly":        0.886,
        "covid":                0.766, "malignancy":          0.615,
    },
    "VoCo": {
        "adrenal_hyperplasia":  0.729, "ascites":             0.735, "atherosclerosis":     0.676,
        "cholecystitis":        0.765, "colorectal_cancer":   0.615, "fatty_liver":         0.795,
        "gallstone":            0.655, "hydronephrosis":      0.766, "kidney_stone":        0.671,
        "liver_calcifications": 0.597, "liver_cyst":          0.602, "liver_lesion":        0.701,
        "lymphadenopathy":      0.694, "renal_cyst":          0.602, "splenomegaly":        0.829,
        "covid":                0.719, "malignancy":          0.580,
    },
    "CT-FM": {
        "adrenal_hyperplasia":  0.757, "ascites":             0.721, "atherosclerosis":     0.765,
        "cholecystitis":        0.735, "colorectal_cancer":   0.750, "fatty_liver":         0.833,
        "gallstone":            0.655, "hydronephrosis":      0.719, "kidney_stone":        0.665,
        "liver_calcifications": 0.597, "liver_cyst":          0.615, "liver_lesion":        0.627,
        "lymphadenopathy":      0.625, "renal_cyst":          0.609, "splenomegaly":        0.814,
        "covid":                0.656, "malignancy":          0.620,
    },
    "Pillar-0": {
        "adrenal_hyperplasia":  0.743, "ascites":             0.750, "atherosclerosis":     0.794,
        "cholecystitis":        0.745, "colorectal_cancer":   0.750, "fatty_liver":         0.846,
        "gallstone":            0.836, "hydronephrosis":      0.906, "kidney_stone":        0.690,
        "liver_calcifications": 0.667, "liver_cyst":          0.762, "liver_lesion":        0.687,
        "lymphadenopathy":      0.722, "renal_cyst":          0.695, "splenomegaly":        0.729,
        "covid":                0.984, "malignancy":          0.755,
    },
}

BASELINES_LP_AUROC = {
    "CT-NEXUS": {
        "adrenal_hyperplasia":  0.664, "ascites":             0.599, "atherosclerosis":     0.535,
        "cholecystitis":        0.633, "colorectal_cancer":   0.507, "fatty_liver":         0.826,
        "gallstone":            0.650, "hydronephrosis":      0.610, "kidney_stone":        0.680,
        "liver_calcifications": 0.514, "liver_cyst":          0.626, "liver_lesion":        0.593,
        "lymphadenopathy":      0.533, "renal_cyst":          0.657, "splenomegaly":        0.638,
        "covid":                0.685, "malignancy":          0.455,
    },
    "3DINO": {
        "adrenal_hyperplasia":  0.7316, "ascites":             0.7284, "atherosclerosis":     0.7716,
        "cholecystitis":        0.6986, "colorectal_cancer":   0.7944, "fatty_liver":         0.9165,
        "gallstone":            0.7004, "hydronephrosis":      0.8140, "kidney_stone":        0.7294,
        "liver_calcifications": 0.6173, "liver_cyst":          0.6082, "liver_lesion":        0.7053,
        "lymphadenopathy":      0.7176, "renal_cyst":          0.6718, "splenomegaly":        0.8996,
        "covid":                0.8040, "malignancy":          0.6052,
    },
    "SPECTRE": {
        "adrenal_hyperplasia":  0.6400, "ascites":             0.9066, "atherosclerosis":     0.7716,
        "cholecystitis":        0.7916, "colorectal_cancer":   0.7293, "fatty_liver":         0.9349,
        "gallstone":            0.7872, "hydronephrosis":      0.8926, "kidney_stone":        0.6720,
        "liver_calcifications": 0.5959, "liver_cyst":          0.7511, "liver_lesion":        0.7281,
        "lymphadenopathy":      0.7809, "renal_cyst":          0.7025, "splenomegaly":        0.8849,
        "covid":                1.0000, "malignancy":          0.7675,
    },
    "Curia": {
        "adrenal_hyperplasia":  0.820, "ascites":             0.778, "atherosclerosis":     0.809,
        "cholecystitis":        0.767, "colorectal_cancer":   0.732, "fatty_liver":         0.978,
        "gallstone":            0.856, "hydronephrosis":      0.952, "kidney_stone":        0.678,
        "liver_calcifications": 0.622, "liver_cyst":          0.682, "liver_lesion":        0.751,
        "lymphadenopathy":      0.804, "renal_cyst":          0.699, "splenomegaly":        0.926,
        "covid":                0.856, "malignancy":          0.675,
    },
    "Merlin": {
        "adrenal_hyperplasia":  0.788, "ascites":             0.835, "atherosclerosis":     0.843,
        "cholecystitis":        0.664, "colorectal_cancer":   0.686, "fatty_liver":         0.892,
        "gallstone":            0.656, "hydronephrosis":      0.706, "kidney_stone":        0.656,
        "liver_calcifications": 0.518, "liver_cyst":          0.661, "liver_lesion":        0.620,
        "lymphadenopathy":      0.799, "renal_cyst":          0.721, "splenomegaly":        0.909,
        "covid":                0.800, "malignancy":          0.618,
    },
    "VoCo": {
        "adrenal_hyperplasia":  0.704, "ascites":             0.719, "atherosclerosis":     0.648,
        "cholecystitis":        0.746, "colorectal_cancer":   0.595, "fatty_liver":         0.840,
        "gallstone":            0.644, "hydronephrosis":      0.762, "kidney_stone":        0.694,
        "liver_calcifications": 0.556, "liver_cyst":          0.600, "liver_lesion":        0.742,
        "lymphadenopathy":      0.664, "renal_cyst":          0.621, "splenomegaly":        0.824,
        "covid":                0.713, "malignancy":          0.556,
    },
    "CT-FM": {
        "adrenal_hyperplasia":  0.725, "ascites":             0.691, "atherosclerosis":     0.774,
        "cholecystitis":        0.743, "colorectal_cancer":   0.762, "fatty_liver":         0.869,
        "gallstone":            0.663, "hydronephrosis":      0.725, "kidney_stone":        0.655,
        "liver_calcifications": 0.526, "liver_cyst":          0.633, "liver_lesion":        0.635,
        "lymphadenopathy":      0.651, "renal_cyst":          0.600, "splenomegaly":        0.840,
        "covid":                0.636, "malignancy":          0.584,
    },
    "Pillar-0": {
        "adrenal_hyperplasia":  0.743, "ascites":             0.789, "atherosclerosis":     0.818,
        "cholecystitis":        0.788, "colorectal_cancer":   0.728, "fatty_liver":         0.926,
        "gallstone":            0.867, "hydronephrosis":      0.938, "kidney_stone":        0.732,
        "liver_calcifications": 0.662, "liver_cyst":          0.791, "liver_lesion":        0.723,
        "lymphadenopathy":      0.722, "renal_cyst":          0.749, "splenomegaly":        0.781,
        "covid":                0.997, "malignancy":          0.796,
    },
}

TASKS_ORDER = [
    "adrenal_hyperplasia", "ascites", "atherosclerosis", "cholecystitis",
    "colorectal_cancer", "fatty_liver", "gallstone", "hydronephrosis",
    "kidney_stone", "liver_calcifications", "liver_cyst", "liver_lesion",
    "lymphadenopathy", "renal_cyst", "splenomegaly", "covid", "malignancy",
]


def load_lejepa_results(lp_results_root: Path, metric: str = "bal_acc") -> dict[str, float]:
    """Lee los JSON de resultados de train_linear_probe.py."""
    results = {}
    metric_key = f"test_balanced_acc" if metric == "bal_acc" else f"test_auroc"

    # Buscar en todos los subdirectorios de dataset
    for dataset_dir in lp_results_root.iterdir():
        if not dataset_dir.is_dir():
            continue
        for task_dir in dataset_dir.iterdir():
            if not task_dir.is_dir():
                continue
            metrics_file = task_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    data = json.load(f)
                task = task_dir.name
                if metric_key in data:
                    results[task] = data[metric_key]

    return results


def build_comparison_table(lejepa_results: dict, metric: str = "bal_acc") -> pd.DataFrame:
    baselines = BASELINES_LP_BAL_ACC if metric == "bal_acc" else BASELINES_LP_AUROC
    metric_label = "BalAcc" if metric == "bal_acc" else "AUROC"

    rows = []
    for task in TASKS_ORDER:
        row = {"Task": task}
        for model, scores in baselines.items():
            row[model] = scores.get(task, float("nan"))
        row["LeJEPA (ours)"] = lejepa_results.get(task, float("nan"))
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Task")

    # Fila de medias
    mean_row = df.mean(skipna=True)
    mean_row.name = "MEAN"
    df = pd.concat([df, mean_row.to_frame().T])

    return df


def print_table(df: pd.DataFrame, metric: str) -> None:
    metric_label = "BalAcc" if metric == "bal_acc" else "AUROC"
    print(f"\n{'='*100}")
    print(f"Task 1 LP — {metric_label} comparison")
    print(f"{'='*100}")

    # Cabecera
    models = list(df.columns)
    header = f"{'Task':<25}" + "".join(f"{m:>10}" for m in models)
    print(header)
    print("-" * len(header))

    for task, row in df.iterrows():
        # Resaltar el mejor no-baseline si LeJEPA supera a algún baseline
        lejepa_val = row.get("LeJEPA (ours)", float("nan"))
        best_baseline = max(
            (v for k, v in row.items() if k != "LeJEPA (ours)" and not np.isnan(v)),
            default=float("nan"),
        )
        marker = " ★" if (not np.isnan(lejepa_val) and lejepa_val >= best_baseline) else ""
        values = "".join(
            f"{v:>10.4f}" if not np.isnan(v) else f"{'—':>10}"
            for v in row
        )
        bold = ">" if task == "MEAN" else " "
        print(f"{bold}{str(task):<24}{values}{marker}")

    print(f"{'='*100}")
    print("★ = LeJEPA supera al mejor baseline en esa tarea")


def main() -> None:
    ap = argparse.ArgumentParser(description="Comparar LeJEPA LP vs baselines CVPR 2026")
    ap.add_argument("--lp_results_root", required=True, help="Root dir con resultados de train_linear_probe.py")
    ap.add_argument("--output",          default=None,  help="CSV de salida (opcional)")
    ap.add_argument("--metric",          default="bal_acc", choices=["bal_acc", "auroc"])
    args = ap.parse_args()

    lp_results_root = Path(args.lp_results_root)

    lejepa_results = load_lejepa_results(lp_results_root, args.metric)
    if not lejepa_results:
        print(f"[warn] No se encontraron resultados de LeJEPA en {lp_results_root}")
        print("       Ejecuta primero train_linear_probe.py")
    else:
        print(f"Resultados LeJEPA cargados: {len(lejepa_results)} tareas")

    df_bal = build_comparison_table(lejepa_results if args.metric == "bal_acc" else {}, "bal_acc")
    df_auc = build_comparison_table(lejepa_results if args.metric == "auroc"   else lejepa_results, "auroc")

    print_table(df_bal, "bal_acc")
    print_table(df_auc, "auroc")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Guardar ambas métricas
        df_bal.to_csv(output_path.with_name(output_path.stem + "_bal_acc.csv"))
        df_auc.to_csv(output_path.with_name(output_path.stem + "_auroc.csv"))
        print(f"\nTablas guardadas en {output_path.parent}/")


if __name__ == "__main__":
    main()
