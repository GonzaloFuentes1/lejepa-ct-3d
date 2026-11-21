"""
CT Dataset: FLARE-Task4-CT-FM (pretrain) y CVPR26-3DCTFMCompetition (downstream).

Modos:
    pretrain    : Solo imagen, para SSL. Usa flare_ct_coreset.yaml.
    linear_probe: Imagen + etiqueta binaria por tarea. Usa cvpr26_downstream.yaml.

Layouts esperados:

FLARE-CT pretrain (coreset):
    data_dir/
        <source>/
            <case_id>.nii.gz
        ...

CVPR26 downstream:
    data_dir/
        <dataset_name>-clf-tr-val/     (ej. AMOS-clf-tr-val)
            imagesTr/
                <case_id>.nii.gz
            imagesVal/
                <case_id>.nii.gz
            labels/
                <task_name>_train.csv  (columns: case_id, label)
                <task_name>_val.csv
        fg_masks/                      (ROI masks para tareas con región específica)
            <task_name>/
                <case_id>.nii.gz
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from monai.data import CacheDataset, Dataset
from torch.utils.data import Dataset as TorchDataset

from data.ct_transforms import build_ct_transforms


# ---------------------------------------------------------------------------
# FLARE CT Pretrain Dataset
# ---------------------------------------------------------------------------

class FlareCTDataset(TorchDataset):
    """
    Dataset de preentrenamiento SSL sobre FLARE-Task4-CT-FM (coreset o full).

    Descubre todos los volúmenes .nii.gz bajo data_dir.
    Si se provee coreset_manifest, filtra solo los archivos del coreset.

    Args:
        data_dir:          Ruta a la carpeta con volúmenes (coreset o full).
        split:             "train" o "val".
        patch_size:        Tamaño del parche 3D.
        spacing:           Spacing en mm para resampling.
        hu_min, hu_max:    Ventana HU para normalización.
        val_split:         Fracción para validación.
        cache_num:         Items a cachear en RAM (0 = sin caché).
        coreset_manifest:  Ruta al JSON del coreset (opcional, filtra archivos).
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        patch_size: Tuple[int, int, int] = (96, 96, 96),
        spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5),
        hu_min: float = -1000.0,
        hu_max: float = 1000.0,
        val_split: float = 0.10,
        cache_num: int = 0,
        coreset_manifest: Optional[str] = None,
        npy_dir: Optional[str] = None,
        cropfirst: bool = False,
        train_subdirs: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)

        # --- Auto-detectar modo de carga ---
        # 1. fast (.npy): requiere preprocess_ct.py offline, ~20ms/sample
        # 2. cropfirst (Option-A): sin preprocesado, ~110ms/sample  ← default si no hay .npy
        # 3. slow (.nii.gz full resample): ~900ms/sample
        if npy_dir is not None:
            npy_root = Path(npy_dir)
        else:
            npy_root = self.data_dir.parent / (self.data_dir.name + "_npy")

        use_fast = npy_root.exists() and any(npy_root.rglob("*.npy"))
        use_cropfirst = cropfirst and not use_fast

        if use_fast:
            print(f"[FlareCTDataset] Modo fast (.npy): {npy_root}")
        elif use_cropfirst:
            print(f"[FlareCTDataset] Modo cropfirst (Option-A, ~110ms/sample)")
        else:
            print(f"[FlareCTDataset] Modo lento (full Spacingd, ~900ms/sample)")

        # Descubrir archivos — opcionalmente limitando a subdirectorios específicos
        # (ej. train_subdirs: [train_part1, train_part2] excluye val_downstream)
        def _scan_dir(root: Path, pattern: str) -> List[Path]:
            if train_subdirs:
                paths = []
                for sub in train_subdirs:
                    subdir = root / sub
                    if subdir.exists():
                        paths.extend(subdir.rglob(pattern))
                    else:
                        print(f"[FlareCTDataset] WARN: subdir no encontrado: {subdir}")
                return sorted(p for p in paths if not p.name.startswith("._"))
            return sorted(p for p in root.rglob(pattern) if not p.name.startswith("._"))

        if use_fast:
            all_vols = _scan_dir(npy_root, "*.npy")
        else:
            all_vols = _scan_dir(self.data_dir, "*.nii.gz")

        # Filtrar por manifiesto de coreset si se provee
        if coreset_manifest:
            manifest_path = Path(coreset_manifest)
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
                coreset_rel = {Path(f) for f in manifest["files"]}
                all_vols = [
                    v for v in all_vols
                    if any(v.name == rel.name and v.parent.name == rel.parent.name
                           for rel in coreset_rel)
                ]

        if len(all_vols) == 0:
            raise FileNotFoundError(
                f"No se encontraron archivos {'npy' if use_fast else 'nii.gz'} bajo "
                f"{'npy_dir' if use_fast else 'data_dir'}. "
                "Revisa data_dir en configs/data/*.yaml o ejecuta scripts/preprocess_ct.py"
            )

        print(f"[FlareCTDataset] {len(all_vols)} volúmenes, split={split}")
        data_dicts = [{"image": str(v)} for v in all_vols]

        # Shuffle determinista antes del split para garantizar diversidad de fuentes en val
        rng = random.Random(42)
        rng.shuffle(data_dicts)

        split_idx = int((1 - val_split) * len(data_dicts))
        if split == "train":
            data_dicts = data_dicts[:split_idx]
        else:
            data_dicts = data_dicts[split_idx:]

        transforms = build_ct_transforms(
            mode="pretrain" if split == "train" else "val",
            patch_size=patch_size,
            spacing=spacing,
            hu_min=hu_min,
            hu_max=hu_max,
            fast=use_fast,
            cropfirst=use_cropfirst,
        )

        self.dataset_inner = (
            CacheDataset(data=data_dicts, transform=transforms, cache_num=cache_num)
            if cache_num > 0
            else Dataset(data=data_dicts, transform=transforms)
        )

    def __len__(self) -> int:
        return len(self.dataset_inner)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset_inner[idx]
        return item[0] if isinstance(item, list) else item


# ---------------------------------------------------------------------------
# CVPR26 Downstream Classification Dataset
# ---------------------------------------------------------------------------

class CVPR26DownstreamDataset(TorchDataset):
    """
    Dataset de clasificación binaria para las 17 tareas del CVPR26 challenge.

    Carga un volumen CT + etiqueta binaria (0/1) para una tarea específica.
    Compatible con linear probing (features extraídas del encoder congelado).

    Estructura del directorio esperada:
        data_dir/
            <DatasetName>-clf-tr-val/
                imagesTr/   *.nii.gz
                imagesVal/  *.nii.gz
                labels/
                    <task>_train.csv   (case_id, label)
                    <task>_val.csv

    Args:
        data_dir:    Raíz del downstream dataset.
        task:        Nombre de la tarea (ej. "kidney_stone").
        split:       "train" o "val".
        patch_size:  Tamaño del parche 3D.
        spacing:     Spacing en mm.
        hu_min, hu_max: Ventana HU.
        roi_masks_dir: Carpeta fg_masks/<task>/ con máscaras ROI opcionales.
        cache_num:   Items a cachear en RAM.
    """

    # 15 tareas reales de CVPR 2026 — todas de AMOS abdominal CT
    # Fuente: https://github.com/kmin940/CVPR26-3DCTFMCompetition
    TASK_TO_DATASET: Dict[str, str] = {
        # ROI tasks (11) — con máscaras de órgano via TotalSegmentator
        "splenomegaly":         "AMOS-clf-tr-val",
        "adrenal_hyperplasia":  "AMOS-clf-tr-val",
        "fatty_liver":          "AMOS-clf-tr-val",
        "cholecystitis":        "AMOS-clf-tr-val",
        "liver_calcifications": "AMOS-clf-tr-val",
        "hydronephrosis":       "AMOS-clf-tr-val",
        "gallstone":            "AMOS-clf-tr-val",
        "liver_lesion":         "AMOS-clf-tr-val",
        "kidney_stone":         "AMOS-clf-tr-val",
        "liver_cyst":           "AMOS-clf-tr-val",
        "renal_cyst":           "AMOS-clf-tr-val",
        # Non-ROI tasks (4) — clasificación global del volumen
        "atherosclerosis":      "AMOS-clf-tr-val",
        "colorectal_cancer":    "AMOS-clf-tr-val",
        "ascites":              "AMOS-clf-tr-val",
        "lymphadenopathy":      "AMOS-clf-tr-val",
    }

    ROI_TASKS = {
        "splenomegaly",
        "adrenal_hyperplasia",
        "fatty_liver",
        "cholecystitis",
        "liver_calcifications",
        "hydronephrosis",
        "gallstone",
        "liver_lesion",
        "kidney_stone",
        "liver_cyst",
        "renal_cyst",
    }

    def __init__(
        self,
        data_dir: str,
        task: str,
        split: str = "train",
        patch_size: Tuple[int, int, int] = (96, 96, 96),
        spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5),
        hu_min: float = -1000.0,
        hu_max: float = 1000.0,
        roi_masks_dir: Optional[str] = None,
        cache_num: int = 0,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.task = task

        if task not in self.TASK_TO_DATASET:
            raise ValueError(
                f"Tarea desconocida: '{task}'. "
                f"Opciones: {sorted(self.TASK_TO_DATASET.keys())}"
            )

        dataset_name = self.TASK_TO_DATASET[task]
        dataset_dir  = self.data_dir / dataset_name

        # Directorio de imágenes según split
        img_subdir = "imagesTr" if split == "train" else "imagesVal"
        img_dir    = dataset_dir / img_subdir

        # Cargar etiquetas desde CSV
        label_suffix = "train" if split == "train" else "val"
        label_csv = dataset_dir / "labels" / f"{task}_{label_suffix}.csv"

        labels_map = self._load_labels(label_csv)

        # Construir data_dicts
        data_dicts = self._build_data_dicts(
            img_dir=img_dir,
            labels_map=labels_map,
            roi_masks_dir=Path(roi_masks_dir) / task if roi_masks_dir else None,
        )

        transforms = build_ct_transforms(
            mode="classify" if split == "train" else "val",
            patch_size=patch_size,
            spacing=spacing,
            hu_min=hu_min,
            hu_max=hu_max,
        )

        self.dataset_inner = (
            CacheDataset(data=data_dicts, transform=transforms, cache_num=cache_num)
            if cache_num > 0
            else Dataset(data=data_dicts, transform=transforms)
        )

    @staticmethod
    def _load_labels(csv_path: Path) -> Dict[str, int]:
        """Carga {case_id: label} desde CSV. Soporta columnas case_id/label o id/label."""
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Archivo de etiquetas no encontrado: {csv_path}\n"
                "Descarga el downstream con: python scripts/download_ct_datasets.py --downstream"
            )
        labels: Dict[str, int] = {}
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Acepta múltiples nombres de columna
                case_id = row.get("case_id") or row.get("id") or row.get("filename", "")
                label   = row.get("label") or row.get("class") or "0"
                # Normalizar case_id: quitar extensión si viene con ella
                case_id = Path(case_id).stem.replace(".nii", "")
                labels[case_id] = int(label)
        return labels

    def _build_data_dicts(
        self,
        img_dir: Path,
        labels_map: Dict[str, int],
        roi_masks_dir: Optional[Path],
    ) -> List[Dict]:
        data_dicts: List[Dict] = []
        if not img_dir.exists():
            raise FileNotFoundError(f"Directorio de imágenes no encontrado: {img_dir}")

        for img_path in sorted(img_dir.glob("*.nii.gz")):
            if img_path.name.startswith("._"):
                continue
            case_id = img_path.stem.replace(".nii", "")
            if case_id not in labels_map:
                continue

            d: Dict = {
                "image": str(img_path),
                "label": labels_map[case_id],
                "case_id": case_id,
            }

            # Máscara ROI si está disponible
            if roi_masks_dir and roi_masks_dir.exists():
                roi_path = roi_masks_dir / img_path.name
                if roi_path.exists():
                    d["roi_mask"] = str(roi_path)

            data_dicts.append(d)
        return data_dicts

    def __len__(self) -> int:
        return len(self.dataset_inner)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset_inner[idx]
        return item[0] if isinstance(item, list) else item


# ---------------------------------------------------------------------------
# Factory de conveniencia
# ---------------------------------------------------------------------------

def build_ct_dataset(
    cfg,
    split: str = "train",
) -> TorchDataset:
    """
    Construye el dataset correcto a partir de la config Hydra.

    cfg.dataset puede ser:
        "flare_ct"          → FlareCTDataset (pretrain SSL)
        "cvpr26_downstream" → CVPR26DownstreamDataset (linear probe)
    """
    patch_size = tuple(cfg.patch_size)
    spacing    = tuple(cfg.spacing)
    hu_min     = getattr(cfg, "hu_min", -1000.0)
    hu_max     = getattr(cfg, "hu_max",  1000.0)

    if cfg.dataset == "flare_ct":
        train_subdirs = list(cfg.train_subdirs) if getattr(cfg, "train_subdirs", None) else None
        return FlareCTDataset(
            data_dir=cfg.data_dir,
            split=split,
            patch_size=patch_size,
            spacing=spacing,
            hu_min=hu_min,
            hu_max=hu_max,
            val_split=getattr(cfg, "val_split", 0.10),
            cache_num=getattr(cfg, "cache_num", 0),
            coreset_manifest=getattr(cfg, "coreset_manifest", None),
            npy_dir=getattr(cfg, "npy_dir", None),
            cropfirst=getattr(cfg, "cropfirst", True),
            train_subdirs=train_subdirs,
        )
    elif cfg.dataset == "cvpr26_downstream":
        task = getattr(cfg, "task", None)
        if task is None:
            raise ValueError("Especifica cfg.task para cvpr26_downstream.")
        return CVPR26DownstreamDataset(
            data_dir=cfg.data_dir,
            task=task,
            split=split,
            patch_size=patch_size,
            spacing=spacing,
            hu_min=hu_min,
            hu_max=hu_max,
            roi_masks_dir=getattr(cfg, "roi_masks_dir", None),
            cache_num=getattr(cfg, "cache_num", 0),
        )
    else:
        raise ValueError(
            f"Dataset CT desconocido: '{cfg.dataset}'. "
            "Usa 'flare_ct' o 'cvpr26_downstream'."
        )
