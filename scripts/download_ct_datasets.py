"""
Descarga de datasets CT para la competencia CVPR 2026:
  Foundation Models for General CT Image Diagnosis
  https://www.codabench.org/competitions/12650/

Datasets:
  - PRETRAIN : FLARE-MedFM/FLARE-Task4-CT-FM   (~668 GB full / ~67 GB coreset 10%)
  - DOWNSTREAM: kmin06/CVPR26-3DCTFMCompetition  (17 binary classification tasks)

Uso:
    # TODO de una vez — coreset (~67 GB) + downstream (recomendado)
    python scripts/download_ct_datasets.py --all

    # TODO de una vez — full pretrain (~668 GB) + downstream
    python scripts/download_ct_datasets.py --all --track alldata

    # Solo downstream (rápido, ~pocos GB — necesario para validar)
    python scripts/download_ct_datasets.py --downstream

    # Solo coreset de pretrain (~67 GB)
    python scripts/download_ct_datasets.py --pretrain --track coreset

    # Ver qué hay disponible sin descargar nada
    python scripts/download_ct_datasets.py --list

    # Verificar descargas ya hechas
    python scripts/download_ct_datasets.py --verify

Variables de entorno (.env):
    HF_TOKEN=hf_...   (requerido — token de HuggingFace con acceso a los repos)

Estructura de salida:
    data_dir/
        flare_ct_pretrain/          ← volúmenes CT para SSL (coreset)
            <source>/
                <case_id>.nii.gz
            coreset_manifest.json   ← lista de archivos seleccionados
        cvpr26_downstream/          ← tasks de clasificación binaria
            AMOS-clf-tr-val/
                train/
                val/
            BraTS23-clf-tr-val/
            NLST-clf-tr-val/
            labels/
                <task_name>_train.csv
                <task_name>_val.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def _load_dotenv() -> None:
    """Carga variables de .env sin depender de python-dotenv."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return
    except ImportError:
        pass
    # Fallback: parser manual buscando .env desde el directorio actual hacia arriba
    here = Path(__file__).resolve().parent
    for candidate in [here, here.parent, Path.cwd()]:
        env_file = candidate / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = val
            break

_load_dotenv()

# ---------------------------------------------------------------------------
# Colores para la terminal
# ---------------------------------------------------------------------------

def _ok(msg: str)   -> None: print(f"  \033[92m✓\033[0m  {msg}")
def _info(msg: str) -> None: print(f"  \033[94m→\033[0m  {msg}")
def _warn(msg: str) -> None: print(f"  \033[93m⚠\033[0m  {msg}")
def _err(msg: str)  -> None: print(f"  \033[91m✗\033[0m  {msg}")
def _hdr(msg: str)  -> None: print(f"\n\033[1m{msg}\033[0m")


# ---------------------------------------------------------------------------
# Configuración de los repos HuggingFace
# ---------------------------------------------------------------------------

HF_PRETRAIN_REPO  = "FLARE-MedFM/FLARE-Task4-CT-FM"
HF_DOWNSTREAM_REPO = "kmin06/CVPR26-3DCTFMCompetition"
CORESET_FRACTION   = 0.10   # 10% del total para Coreset Track
RANDOM_SEED        = 42

# Las 17 tareas de clasificación binaria del challenge
DOWNSTREAM_TASKS = [
    # AMOS (abdominal CT) — 12 tasks
    "adrenal_hyperplasia",
    "aortic_aneurysm",
    "gallbladder_stone",
    "hepatic_cyst",
    "hepatocellular_carcinoma",
    "kidney_cyst",
    "kidney_stone",
    "liver_cirrhosis",
    "ovarian_cyst",
    "pancreatic_cancer",
    "renal_cell_carcinoma",
    "spleen_enlargement",
    # BraTS 2023 (brain) — 3 tasks
    "brain_tumor_grade",
    "mgmt_methylation",
    "brain_idh_mutation",
    # NLST (chest CT) — 2 tasks
    "lung_cancer",
    "lung_nodule",
]


def _get_hf_token() -> str:
    """Lee el token de HF desde el entorno. Falla limpiamente si no existe."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        _err("No se encontró HF_TOKEN en el entorno.")
        print()
        print("  Añade a tu .env:")
        print("    HF_TOKEN=hf_...")
        print()
        print("  O exporta antes de ejecutar:")
        print("    export HF_TOKEN=hf_...")
        sys.exit(1)
    return token


def _ensure_huggingface_hub() -> None:
    """Instala huggingface_hub si no está disponible."""
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        _info("Instalando huggingface_hub...")
        import subprocess
        if shutil.which("uv"):
            subprocess.check_call(["uv", "pip", "install", "-q", "huggingface_hub>=0.23.0"])
        else:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub>=0.23.0"])


# ---------------------------------------------------------------------------
# Utilidad: listado de archivos en un repo HF
# ---------------------------------------------------------------------------

def _list_repo_files(repo_id: str, token: str, repo_type: str = "dataset") -> List[str]:
    """Devuelve la lista de rutas de archivos en un repo HF."""
    from huggingface_hub import list_repo_files
    files = list(list_repo_files(repo_id, repo_type=repo_type, token=token))
    return files


# ---------------------------------------------------------------------------
# Downstream: kmin06/CVPR26-3DCTFMCompetition
# ---------------------------------------------------------------------------

def download_downstream(out_dir: Path, token: str) -> None:
    _hdr(f"Descargando downstream: {HF_DOWNSTREAM_REPO}")

    dst = out_dir / "cvpr26_downstream"
    dst.mkdir(parents=True, exist_ok=True)

    if _verify_downstream(out_dir)["ok"]:
        _ok(f"Downstream ya descargado en {dst} — omitiendo")
        _info("Usa --verify para revisar la integridad")
        return

    from huggingface_hub import snapshot_download

    _info(f"Destino: {dst}")
    _info("Descargando con snapshot_download (reanudable)...")

    try:
        snapshot_download(
            repo_id=HF_DOWNSTREAM_REPO,
            repo_type="dataset",
            local_dir=str(dst),
            token=token,
            ignore_patterns=["*.git*", "__pycache__/*", "*.pyc"],
            resume_download=True,
        )
    except Exception as e:
        _err(f"Error durante la descarga: {e}")
        _info("Si hay error de acceso, asegúrate de haber aceptado los términos en:")
        _info(f"  https://huggingface.co/datasets/{HF_DOWNSTREAM_REPO}")
        sys.exit(1)

    stats = _verify_downstream(out_dir)
    _ok(f"Downstream descargado: {stats['n_volumes']} volúmenes, {stats['n_labels']} archivos de labels")
    _info(f"Ruta: {dst}")


def _verify_downstream(out_dir: Path) -> Dict:
    dst = out_dir / "cvpr26_downstream"
    if not dst.exists():
        return {"ok": False, "n_volumes": 0, "n_labels": 0}

    n_volumes = len(list(dst.rglob("*.nii.gz")))
    n_labels  = len(list(dst.rglob("*.csv"))) + len(list(dst.rglob("*.json")))
    ok = n_volumes > 0

    return {"ok": ok, "n_volumes": n_volumes, "n_labels": n_labels}


# ---------------------------------------------------------------------------
# Pretrain coreset: FLARE-MedFM/FLARE-Task4-CT-FM
# ---------------------------------------------------------------------------

def list_pretrain_files(token: str) -> None:
    """Lista los archivos disponibles en el repo de pretrain (sin descargar)."""
    _hdr(f"Archivos en {HF_PRETRAIN_REPO}")
    _ensure_huggingface_hub()

    _info("Listando archivos (puede tardar unos segundos)...")
    try:
        files = _list_repo_files(HF_PRETRAIN_REPO, token)
    except Exception as e:
        _err(f"No se pudo listar el repo: {e}")
        sys.exit(1)

    # Agrupar por subdirectorio/fuente
    sources: Dict[str, int] = {}
    for f in files:
        parts = Path(f).parts
        source = parts[0] if len(parts) > 1 else "root"
        sources[source] = sources.get(source, 0) + 1

    _ok(f"Total: {len(files)} archivos")
    print()
    print("  Distribución por fuente:")
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"    {src:40s}  {count:>6d} archivos")


def _select_coreset(files: List[str], fraction: float, seed: int) -> List[str]:
    """
    Selecciona una fracción estratificada de los archivos de pretrain.

    Estrategia:
      1. Agrupar por fuente/subdirectorio (proxy de dataset de origen).
      2. Dentro de cada grupo, muestreo aleatorio uniforme de `fraction`.
      3. Garantizar mínimo 1 muestra por grupo.

    Esto asegura diversidad anatómica y de dominio en el coreset.
    """
    # Solo archivos NIfTI
    nifti_files = [f for f in files if f.endswith(".nii.gz") or f.endswith(".nii")]

    if not nifti_files:
        _warn("No se encontraron archivos .nii.gz — usando todos los archivos")
        nifti_files = files

    # Agrupar por primer directorio
    groups: Dict[str, List[str]] = {}
    for f in nifti_files:
        src = Path(f).parts[0] if len(Path(f).parts) > 1 else "root"
        groups.setdefault(src, []).append(f)

    rng = random.Random(seed)
    selected: List[str] = []

    for src, group_files in groups.items():
        n = max(1, math.ceil(len(group_files) * fraction))
        sampled = rng.sample(group_files, min(n, len(group_files)))
        selected.extend(sampled)
        _info(f"  {src}: {len(sampled)}/{len(group_files)} archivos seleccionados")

    total_pct = 100 * len(selected) / len(nifti_files) if nifti_files else 0
    _ok(f"Coreset: {len(selected)} archivos ({total_pct:.1f}% del total)")

    return selected


def download_pretrain_coreset(out_dir: Path, token: str) -> None:
    _hdr(f"Descargando pretrain coreset ({int(CORESET_FRACTION*100)}%): {HF_PRETRAIN_REPO}")
    _ensure_huggingface_hub()

    dst = out_dir / "flare_ct_pretrain"
    manifest_path = dst / "coreset_manifest.json"

    if manifest_path.exists():
        _ok(f"Manifiesto de coreset ya existe en {manifest_path}")
        with open(manifest_path) as f:
            manifest = json.load(f)
        existing = [
            p for p in (dst / "volumes").glob("**/*.nii.gz")
        ]
        if len(existing) >= len(manifest["files"]) * 0.95:
            _ok(f"Coreset ya descargado: {len(existing)} archivos — omitiendo")
            return
        _info(f"Descarga incompleta: {len(existing)}/{len(manifest['files'])} — reanudando")
        selected = manifest["files"]
    else:
        dst.mkdir(parents=True, exist_ok=True)

        # Listar archivos disponibles
        _info("Listando archivos del repo de pretrain...")
        try:
            all_files = _list_repo_files(HF_PRETRAIN_REPO, token)
        except Exception as e:
            _err(f"No se pudo listar el repo: {e}")
            _info(f"Verifica que tienes acceso: https://huggingface.co/datasets/{HF_PRETRAIN_REPO}")
            sys.exit(1)

        _info(f"Total de archivos en el repo: {len(all_files)}")

        # Seleccionar coreset estratificado
        _hdr("Seleccionando coreset estratificado")
        selected = _select_coreset(all_files, CORESET_FRACTION, RANDOM_SEED)

        # Guardar manifiesto para reproducibilidad y reanudación
        manifest = {
            "repo": HF_PRETRAIN_REPO,
            "track": "coreset",
            "fraction": CORESET_FRACTION,
            "seed": RANDOM_SEED,
            "total_repo_files": len(all_files),
            "n_selected": len(selected),
            "files": selected,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        _ok(f"Manifiesto guardado: {manifest_path}")

    # Descargar archivos seleccionados uno a uno (permite reanudar)
    from huggingface_hub import hf_hub_download

    vol_dir = dst / "volumes"
    vol_dir.mkdir(parents=True, exist_ok=True)

    n_done   = 0
    n_skip   = 0
    n_failed = 0
    total    = len(selected)

    _hdr(f"Descargando {total} archivos (~{total * 0.1:.0f} GB estimado)")
    _info("Reanuda el script si se interrumpe — los archivos ya descargados se omiten")
    print()

    for i, rel_path in enumerate(selected, 1):
        local_path = vol_dir / rel_path

        if local_path.exists() and local_path.stat().st_size > 0:
            n_skip += 1
            if i % 100 == 0:
                print(f"  [{i:>5}/{total}] {n_done} descargados, {n_skip} omitidos, {n_failed} fallidos")
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            cached = hf_hub_download(
                repo_id=HF_PRETRAIN_REPO,
                filename=rel_path,
                repo_type="dataset",
                token=token,
                local_dir=str(vol_dir),
            )
            n_done += 1
        except Exception as e:
            _warn(f"  [{i:>5}/{total}] FALLO: {rel_path} — {e}")
            n_failed += 1
            continue

        if i % 50 == 0 or i == total:
            pct = 100 * i / total
            print(f"  [{i:>5}/{total}] {pct:.1f}% — ✓ {n_done}  ⊘ {n_skip}  ✗ {n_failed}")

    print()
    _ok(f"Coreset descargado: {n_done + n_skip} archivos en {vol_dir}")
    if n_failed:
        _warn(f"{n_failed} archivos fallidos — ejecuta de nuevo para reintentarlos")


def download_pretrain_alldata(out_dir: Path, token: str) -> None:
    """All Data Track — descarga TODO el repo de pretrain (~668 GB)."""
    _hdr(f"Descargando pretrain FULL DATA (~668 GB): {HF_PRETRAIN_REPO}")
    _warn("¡Este proceso puede tardar muchas horas y ocupa ~668 GB!")
    _warn("Considera usar --track coreset para el Coreset Track (~67 GB).")
    print()

    response = input("  ¿Confirmas la descarga completa? [s/N]: ").strip().lower()
    if response not in ("s", "si", "sí", "y", "yes"):
        _info("Descarga cancelada.")
        sys.exit(0)

    from huggingface_hub import snapshot_download

    dst = out_dir / "flare_ct_pretrain_full"
    dst.mkdir(parents=True, exist_ok=True)

    _info(f"Destino: {dst}")
    snapshot_download(
        repo_id=HF_PRETRAIN_REPO,
        repo_type="dataset",
        local_dir=str(dst),
        token=token,
        resume_download=True,
        ignore_patterns=["*.git*"],
    )

    n = len(list(dst.rglob("*.nii.gz")))
    _ok(f"Full dataset descargado: {n} volúmenes en {dst}")


# ---------------------------------------------------------------------------
# Verificación
# ---------------------------------------------------------------------------

def verify_all(out_dir: Path) -> None:
    _hdr(f"Verificando datasets CT en {out_dir}")

    # Downstream
    stats_ds = _verify_downstream(out_dir)
    dst_path  = out_dir / "cvpr26_downstream"
    if stats_ds["ok"]:
        _ok(f"Downstream: {stats_ds['n_volumes']} volúmenes, {stats_ds['n_labels']} archivos de labels ({dst_path})")
    else:
        _err(f"Downstream: no encontrado en {dst_path}")
        _info("  Ejecuta: python scripts/download_ct_datasets.py --downstream")

    # Pretrain coreset
    coreset_dir  = out_dir / "flare_ct_pretrain"
    manifest_p   = coreset_dir / "coreset_manifest.json"
    if manifest_p.exists():
        with open(manifest_p) as f:
            manifest = json.load(f)
        n_expected = manifest["n_selected"]
        n_actual   = len(list((coreset_dir / "volumes").rglob("*.nii.gz"))) if (coreset_dir / "volumes").exists() else 0
        if n_actual >= n_expected * 0.95:
            _ok(f"Pretrain coreset: {n_actual}/{n_expected} volúmenes ({coreset_dir})")
        else:
            _warn(f"Pretrain coreset: {n_actual}/{n_expected} volúmenes — descarga incompleta")
            _info("  Ejecuta de nuevo: python scripts/download_ct_datasets.py --pretrain --track coreset")
    else:
        _err(f"Pretrain: no encontrado en {coreset_dir}")
        _info("  Ejecuta: python scripts/download_ct_datasets.py --pretrain --track coreset")

    # Full pretrain
    full_dir = out_dir / "flare_ct_pretrain_full"
    if full_dir.exists():
        n_full = len(list(full_dir.rglob("*.nii.gz")))
        if n_full > 0:
            _ok(f"Pretrain full: {n_full} volúmenes ({full_dir})")

    print()
    _hdr("Próximos pasos:")
    if stats_ds["ok"]:
        print("  # Configurar la data config CT")
        print("  python scripts/pretrain.py data=flare_ct_coreset")
        print()
        print("  # Verificar carga de un volumen")
        print("  python examples/ct/inspect_ct_volume.py")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descarga datasets CT para la competencia CVPR 2026",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # TODO de una vez — coreset (~67 GB) + downstream (RECOMENDADO)
  python scripts/download_ct_datasets.py --all

  # TODO con full pretrain (~668 GB + downstream)
  python scripts/download_ct_datasets.py --all --track alldata

  # Solo downstream (pocos GB, para validación rápida)
  python scripts/download_ct_datasets.py --downstream

  # Solo coreset de pretrain (~67 GB)
  python scripts/download_ct_datasets.py --pretrain --track coreset

  # Ver archivos disponibles en el repo de pretrain
  python scripts/download_ct_datasets.py --list

  # Verificar descargas ya realizadas
  python scripts/download_ct_datasets.py --verify
        """,
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Descargar TODO: pretrain + downstream (equivale a --pretrain --downstream)",
    )
    parser.add_argument(
        "--pretrain", action="store_true",
        help="Descargar dataset de preentrenamiento (FLARE-Task4-CT-FM)",
    )
    parser.add_argument(
        "--downstream", action="store_true",
        help="Descargar dataset downstream (CVPR26-3DCTFMCompetition)",
    )
    parser.add_argument(
        "--track", choices=["coreset", "alldata"], default="coreset",
        help="Track para el dataset de pretrain: coreset (10%%, ~67 GB) o alldata (100%%, ~668 GB) (default: coreset)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Listar archivos disponibles en el repo de pretrain (sin descargar)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verificar integridad de los datasets ya descargados",
    )
    parser.add_argument(
        "--out", type=str, default=os.environ.get("LEJEPA_DATA_ROOT", "./data"),
        help="Data root directory (default: $LEJEPA_DATA_ROOT or ./data)",
    )
    args = parser.parse_args()

    # --all expande a --pretrain --downstream
    if args.all:
        args.pretrain    = True
        args.downstream  = True

    if not any([args.pretrain, args.downstream, args.list, args.verify]):
        parser.print_help()
        print("\n\033[93mEspecifica al menos una acción: --all, --pretrain, --downstream, --list o --verify\033[0m")
        sys.exit(0)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_huggingface_hub()

    token = _get_hf_token()

    if args.list:
        list_pretrain_files(token)

    if args.downstream:
        download_downstream(out_dir, token)

    if args.pretrain:
        if args.track == "coreset":
            download_pretrain_coreset(out_dir, token)
        else:
            download_pretrain_alldata(out_dir, token)

    if args.verify or args.pretrain or args.downstream:
        verify_all(out_dir)

    print()


if __name__ == "__main__":
    main()
