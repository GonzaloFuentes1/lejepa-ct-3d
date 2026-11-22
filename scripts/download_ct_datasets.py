"""
Download of CT datasets for the CVPR 2026 competition:
  Foundation Models for General CT Image Diagnosis
  https://www.codabench.org/competitions/12650/

Datasets:
  - PRETRAIN : FLARE-MedFM/FLARE-Task4-CT-FM   (~668 GB full / ~67 GB coreset 10%)
  - DOWNSTREAM: kmin06/CVPR26-3DCTFMCompetition  (17 binary classification tasks)

Usage:
    # Everything at once: coreset (~67 GB) + downstream (recommended)
    python scripts/download_ct_datasets.py --all

    # Everything at once: full pretrain (~668 GB) + downstream
    python scripts/download_ct_datasets.py --all --track alldata

    # Downstream only (fast, a few GB, needed for validation)
    python scripts/download_ct_datasets.py --downstream

    # Pretrain coreset only (~67 GB)
    python scripts/download_ct_datasets.py --pretrain --track coreset

    # See what is available without downloading anything
    python scripts/download_ct_datasets.py --list

    # Verify downloads already made
    python scripts/download_ct_datasets.py --verify

Environment variables (.env):
    HF_TOKEN=hf_...   (required: HuggingFace token with access to the repos)

Output structure:
    data_dir/
        flare_ct_pretrain/          ← CT volumes for SSL (coreset)
            <source>/
                <case_id>.nii.gz
            coreset_manifest.json   ← list of selected files
        cvpr26_downstream/          ← binary classification tasks
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
    """Loads variables from .env without depending on python-dotenv."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return
    except ImportError:
        pass
    # Fallback: manual parser looking for .env from the current directory upwards
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
# Terminal colors
# ---------------------------------------------------------------------------

def _ok(msg: str)   -> None: print(f"  \033[92m✓\033[0m  {msg}")
def _info(msg: str) -> None: print(f"  \033[94m→\033[0m  {msg}")
def _warn(msg: str) -> None: print(f"  \033[93m⚠\033[0m  {msg}")
def _err(msg: str)  -> None: print(f"  \033[91m✗\033[0m  {msg}")
def _hdr(msg: str)  -> None: print(f"\n\033[1m{msg}\033[0m")


# ---------------------------------------------------------------------------
# HuggingFace repos configuration
# ---------------------------------------------------------------------------

HF_PRETRAIN_REPO  = "FLARE-MedFM/FLARE-Task4-CT-FM"
HF_DOWNSTREAM_REPO = "kmin06/CVPR26-3DCTFMCompetition"
CORESET_FRACTION   = 0.10   # 10% of the total for the Coreset Track
RANDOM_SEED        = 42

# The 17 binary classification tasks of the challenge
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
    """Reads the HF token from the environment. Fails cleanly if it does not exist."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        _err("HF_TOKEN not found in the environment.")
        print()
        print("  Add to your .env:")
        print("    HF_TOKEN=hf_...")
        print()
        print("  Or export before running:")
        print("    export HF_TOKEN=hf_...")
        sys.exit(1)
    return token


def _ensure_huggingface_hub() -> None:
    """Installs huggingface_hub if it is not available."""
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        _info("Installing huggingface_hub...")
        import subprocess
        if shutil.which("uv"):
            subprocess.check_call(["uv", "pip", "install", "-q", "huggingface_hub>=0.23.0"])
        else:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub>=0.23.0"])


# ---------------------------------------------------------------------------
# Utility: list of files in an HF repo
# ---------------------------------------------------------------------------

def _list_repo_files(repo_id: str, token: str, repo_type: str = "dataset") -> List[str]:
    """Returns the list of file paths in an HF repo."""
    from huggingface_hub import list_repo_files
    files = list(list_repo_files(repo_id, repo_type=repo_type, token=token))
    return files


# ---------------------------------------------------------------------------
# Downstream: kmin06/CVPR26-3DCTFMCompetition
# ---------------------------------------------------------------------------

def download_downstream(out_dir: Path, token: str) -> None:
    _hdr(f"Downloading downstream: {HF_DOWNSTREAM_REPO}")

    dst = out_dir / "cvpr26_downstream"
    dst.mkdir(parents=True, exist_ok=True)

    if _verify_downstream(out_dir)["ok"]:
        _ok(f"Downstream already downloaded in {dst}, skipping")
        _info("Use --verify to check integrity")
        return

    from huggingface_hub import snapshot_download

    _info(f"Destination: {dst}")
    _info("Downloading with snapshot_download (resumable)...")

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
        _err(f"Error during download: {e}")
        _info("If there is an access error, make sure you accepted the terms at:")
        _info(f"  https://huggingface.co/datasets/{HF_DOWNSTREAM_REPO}")
        sys.exit(1)

    stats = _verify_downstream(out_dir)
    _ok(f"Downstream downloaded: {stats['n_volumes']} volumes, {stats['n_labels']} label files")
    _info(f"Path: {dst}")


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
    """Lists the files available in the pretrain repo (without downloading)."""
    _hdr(f"Files in {HF_PRETRAIN_REPO}")
    _ensure_huggingface_hub()

    _info("Listing files (may take a few seconds)...")
    try:
        files = _list_repo_files(HF_PRETRAIN_REPO, token)
    except Exception as e:
        _err(f"Could not list the repo: {e}")
        sys.exit(1)

    # Group by subdirectory/source
    sources: Dict[str, int] = {}
    for f in files:
        parts = Path(f).parts
        source = parts[0] if len(parts) > 1 else "root"
        sources[source] = sources.get(source, 0) + 1

    _ok(f"Total: {len(files)} files")
    print()
    print("  Distribution by source:")
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"    {src:40s}  {count:>6d} files")


def _select_coreset(files: List[str], fraction: float, seed: int) -> List[str]:
    """
    Selects a stratified fraction of the pretrain files.

    Strategy:
      1. Group by source/subdirectory (proxy for the originating dataset).
      2. Within each group, uniform random sampling of `fraction`.
      3. Guarantee at least 1 sample per group.

    This ensures anatomical and domain diversity in the coreset.
    """
    # NIfTI files only
    nifti_files = [f for f in files if f.endswith(".nii.gz") or f.endswith(".nii")]

    if not nifti_files:
        _warn("No .nii.gz files found, using all files")
        nifti_files = files

    # Group by first directory
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
        _info(f"  {src}: {len(sampled)}/{len(group_files)} files selected")

    total_pct = 100 * len(selected) / len(nifti_files) if nifti_files else 0
    _ok(f"Coreset: {len(selected)} files ({total_pct:.1f}% of the total)")

    return selected


def download_pretrain_coreset(out_dir: Path, token: str) -> None:
    _hdr(f"Downloading pretrain coreset ({int(CORESET_FRACTION*100)}%): {HF_PRETRAIN_REPO}")
    _ensure_huggingface_hub()

    dst = out_dir / "flare_ct_pretrain"
    manifest_path = dst / "coreset_manifest.json"

    if manifest_path.exists():
        _ok(f"Coreset manifest already exists at {manifest_path}")
        with open(manifest_path) as f:
            manifest = json.load(f)
        existing = [
            p for p in (dst / "volumes").glob("**/*.nii.gz")
        ]
        if len(existing) >= len(manifest["files"]) * 0.95:
            _ok(f"Coreset already downloaded: {len(existing)} files, skipping")
            return
        _info(f"Incomplete download: {len(existing)}/{len(manifest['files'])}, resuming")
        selected = manifest["files"]
    else:
        dst.mkdir(parents=True, exist_ok=True)

        # List available files
        _info("Listing files of the pretrain repo...")
        try:
            all_files = _list_repo_files(HF_PRETRAIN_REPO, token)
        except Exception as e:
            _err(f"Could not list the repo: {e}")
            _info(f"Check that you have access: https://huggingface.co/datasets/{HF_PRETRAIN_REPO}")
            sys.exit(1)

        _info(f"Total files in the repo: {len(all_files)}")

        # Select stratified coreset
        _hdr("Selecting stratified coreset")
        selected = _select_coreset(all_files, CORESET_FRACTION, RANDOM_SEED)

        # Save manifest for reproducibility and resuming
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
        _ok(f"Manifest saved: {manifest_path}")

    # Download selected files one by one (allows resuming)
    from huggingface_hub import hf_hub_download

    vol_dir = dst / "volumes"
    vol_dir.mkdir(parents=True, exist_ok=True)

    n_done   = 0
    n_skip   = 0
    n_failed = 0
    total    = len(selected)

    _hdr(f"Downloading {total} files (~{total * 0.1:.0f} GB estimated)")
    _info("Rerun the script if interrupted; already downloaded files are skipped")
    print()

    for i, rel_path in enumerate(selected, 1):
        local_path = vol_dir / rel_path

        if local_path.exists() and local_path.stat().st_size > 0:
            n_skip += 1
            if i % 100 == 0:
                print(f"  [{i:>5}/{total}] {n_done} downloaded, {n_skip} skipped, {n_failed} failed")
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
    _ok(f"Coreset downloaded: {n_done + n_skip} files in {vol_dir}")
    if n_failed:
        _warn(f"{n_failed} failed files, run again to retry them")


def download_pretrain_alldata(out_dir: Path, token: str) -> None:
    """All Data Track: downloads the ENTIRE pretrain repo (~668 GB)."""
    _hdr(f"Downloading pretrain FULL DATA (~668 GB): {HF_PRETRAIN_REPO}")
    _warn("This process can take many hours and takes ~668 GB!")
    _warn("Consider using --track coreset for the Coreset Track (~67 GB).")
    print()

    response = input("  Confirm the full download? [y/N]: ").strip().lower()
    if response not in ("s", "si", "sí", "y", "yes"):
        _info("Download cancelled.")
        sys.exit(0)

    from huggingface_hub import snapshot_download

    dst = out_dir / "flare_ct_pretrain_full"
    dst.mkdir(parents=True, exist_ok=True)

    _info(f"Destination: {dst}")
    snapshot_download(
        repo_id=HF_PRETRAIN_REPO,
        repo_type="dataset",
        local_dir=str(dst),
        token=token,
        resume_download=True,
        ignore_patterns=["*.git*"],
    )

    n = len(list(dst.rglob("*.nii.gz")))
    _ok(f"Full dataset downloaded: {n} volumes in {dst}")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_all(out_dir: Path) -> None:
    _hdr(f"Verifying CT datasets in {out_dir}")

    # Downstream
    stats_ds = _verify_downstream(out_dir)
    dst_path  = out_dir / "cvpr26_downstream"
    if stats_ds["ok"]:
        _ok(f"Downstream: {stats_ds['n_volumes']} volumes, {stats_ds['n_labels']} label files ({dst_path})")
    else:
        _err(f"Downstream: not found in {dst_path}")
        _info("  Run: python scripts/download_ct_datasets.py --downstream")

    # Pretrain coreset
    coreset_dir  = out_dir / "flare_ct_pretrain"
    manifest_p   = coreset_dir / "coreset_manifest.json"
    if manifest_p.exists():
        with open(manifest_p) as f:
            manifest = json.load(f)
        n_expected = manifest["n_selected"]
        n_actual   = len(list((coreset_dir / "volumes").rglob("*.nii.gz"))) if (coreset_dir / "volumes").exists() else 0
        if n_actual >= n_expected * 0.95:
            _ok(f"Pretrain coreset: {n_actual}/{n_expected} volumes ({coreset_dir})")
        else:
            _warn(f"Pretrain coreset: {n_actual}/{n_expected} volumes, incomplete download")
            _info("  Run again: python scripts/download_ct_datasets.py --pretrain --track coreset")
    else:
        _err(f"Pretrain: not found in {coreset_dir}")
        _info("  Run: python scripts/download_ct_datasets.py --pretrain --track coreset")

    # Full pretrain
    full_dir = out_dir / "flare_ct_pretrain_full"
    if full_dir.exists():
        n_full = len(list(full_dir.rglob("*.nii.gz")))
        if n_full > 0:
            _ok(f"Pretrain full: {n_full} volumes ({full_dir})")

    print()
    _hdr("Next steps:")
    if stats_ds["ok"]:
        print("  # Configure the CT data config")
        print("  python scripts/pretrain.py data=flare_ct_coreset")
        print()
        print("  # Verify loading of a volume")
        print("  python examples/ct/inspect_ct_volume.py")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download CT datasets for the CVPR 2026 competition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Everything at once: coreset (~67 GB) + downstream (RECOMMENDED)
  python scripts/download_ct_datasets.py --all

  # Everything with full pretrain (~668 GB + downstream)
  python scripts/download_ct_datasets.py --all --track alldata

  # Downstream only (a few GB, for quick validation)
  python scripts/download_ct_datasets.py --downstream

  # Pretrain coreset only (~67 GB)
  python scripts/download_ct_datasets.py --pretrain --track coreset

  # See files available in the pretrain repo
  python scripts/download_ct_datasets.py --list

  # Verify downloads already made
  python scripts/download_ct_datasets.py --verify
        """,
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Download EVERYTHING: pretrain + downstream (equivalent to --pretrain --downstream)",
    )
    parser.add_argument(
        "--pretrain", action="store_true",
        help="Download the pretraining dataset (FLARE-Task4-CT-FM)",
    )
    parser.add_argument(
        "--downstream", action="store_true",
        help="Download the downstream dataset (CVPR26-3DCTFMCompetition)",
    )
    parser.add_argument(
        "--track", choices=["coreset", "alldata"], default="coreset",
        help="Track for the pretrain dataset: coreset (10%%, ~67 GB) or alldata (100%%, ~668 GB) (default: coreset)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List files available in the pretrain repo (without downloading)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify integrity of the already downloaded datasets",
    )
    parser.add_argument(
        "--out", type=str, default=os.environ.get("LEJEPA_DATA_ROOT", "./data"),
        help="Data root directory (default: $LEJEPA_DATA_ROOT or ./data)",
    )
    args = parser.parse_args()

    # --all expands to --pretrain --downstream
    if args.all:
        args.pretrain    = True
        args.downstream  = True

    if not any([args.pretrain, args.downstream, args.list, args.verify]):
        parser.print_help()
        print("\n\033[93mSpecify at least one action: --all, --pretrain, --downstream, --list or --verify\033[0m")
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
