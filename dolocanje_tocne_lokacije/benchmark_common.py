"""Skupno za camera_benchmark.py in camera_translation_benchmark.py: en zagon
registracije -> ena vrstica rezultatov, nadaljevanje iz obstoječega Excela in
statistika napak."""
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from main_trial import run_registration
from point_clouds import crop_top_region, load_cad_mesh
from poses import FLIP_UP, transformation_error
from registration_config import RegistrationConfig

BASE_PATH = Path(__file__).resolve().parent
CAD_PATH = BASE_PATH / "d_housing_detailed.STL"

# Parametri, enaki za VSE zagone obeh benchmarkov (glej main_trial.py --help) -
# vse kamera-specifično (FOV, ločljivost, šum ...) doda vsak benchmark iz
# kontrolna_plosca.resolve_camera_params().
BASE_CONFIG = RegistrationConfig(
    voxel_size=1.0,
    sample_points=200000,
    add_table_background=True,
    remove_table_background=True,
    use_cad_cache=True,          # isti CAD za vse zagone - vzorčenje se izračuna samo enkrat
    use_preprocess_cache=False,  # vsak target je unikaten (nov seed) - predpomnjenje bi samo kopičilo tisoče datotek brez koristi
    top_fraction=1.0,
    up_axis=2,
)

RESULT_COLUMNS = [
    "status", "rot_error_deg", "trans_error_mm", "coarse_rot_error_deg", "coarse_trans_error_mm",
    "fitness", "inlier_rmse", "asymmetric_fraction", "target_coverage",
    "accepted", "reject_reasons", "runtime_s", "error_message",
]


def cad_part_radius_mm(config: RegistrationConfig) -> float:
    """XY polmer (polovica diagonale bounding boxa) zgornje "sealing" regije
    v CAD-ovem lastnem okviru - konzervativen (nekoliko precenjen) polmer
    dela, ista fizična regija za vse kamere."""
    mesh = load_cad_mesh(CAD_PATH)
    if config.flip_up_direction:
        mesh.transform(FLIP_UP)   # isti notranji okvir kot run_registration (glej main_trial._ViewFrame)
    cropped = crop_top_region(mesh, top_fraction=config.top_fraction, up_axis=config.up_axis)
    half_extent_xy = (np.asarray(cropped.get_max_bound())[:2] - np.asarray(cropped.get_min_bound())[:2]) / 2.0
    return float(np.linalg.norm(half_extent_xy))


def run_one(config: RegistrationConfig, row: dict, seed: int, cache_key_tag: str,
            print_traceback: bool = False) -> dict:
    """Požene EN zagon registracije (poza in kamera sta v config) in vrne
    `row`, dopolnjen z RESULT_COLUMNS. Nikoli ne vrže naprej - napake ujame in
    zabeleži kot status='error', da ena neuspešna kombinacija ne prekine sweep-a."""
    np.random.seed(seed)   # ponovljivost - takoj PRED klicem run_registration
    start = time.time()
    try:
        outcome = run_registration(config, CAD_PATH, cache_key_tag=cache_key_tag, visualize=False)
        runtime_s = time.time() - start
        icp_err = transformation_error(outcome.icp.transformation, outcome.reference_transform)
        coarse_err = transformation_error(outcome.coarse.transformation, outcome.reference_transform)
        decision = outcome.decision
        row.update({
            "status": "ok",
            "rot_error_deg": icp_err["rotation_error_deg"],
            "trans_error_mm": icp_err["translation_error"],
            "coarse_rot_error_deg": coarse_err["rotation_error_deg"],
            "coarse_trans_error_mm": coarse_err["translation_error"],
            "fitness": decision["fitness"],
            "inlier_rmse": decision["inlier_rmse"],
            "asymmetric_fraction": decision["asymmetric_fraction"],
            "target_coverage": decision["target_coverage_fraction"],
            "accepted": decision["accepted"],
            "reject_reasons": "; ".join(decision["reasons"]),
            "runtime_s": runtime_s,
            "error_message": "",
        })
    except Exception as exc:   # noqa: BLE001 - namerno ujamemo vse, glej docstring
        runtime_s = time.time() - start
        row.update({
            "status": "error",
            "rot_error_deg": np.nan, "trans_error_mm": np.nan,
            "coarse_rot_error_deg": np.nan, "coarse_trans_error_mm": np.nan,
            "fitness": np.nan, "inlier_rmse": np.nan, "asymmetric_fraction": np.nan,
            "target_coverage": np.nan, "accepted": False, "reject_reasons": "",
            "runtime_s": runtime_s, "error_message": f"{type(exc).__name__}: {exc}",
        })
        print(f"    NAPAKA: {type(exc).__name__}: {exc}")
        if print_traceback:
            traceback.print_exc()
    return row


def load_existing_runs(output_file: Path, row_key: Callable) -> tuple[list[dict], set]:
    """Prebere obstoječi Excel (če obstaja) za podporo nadaljevanju - vrne
    (seznam že opravljenih vrstic, množica row_key(vrstica) ključev, ki jih
    je treba PRESKOČITI)."""
    if not output_file.exists():
        return [], set()
    existing = pd.read_excel(output_file, sheet_name="runs")
    done = {row_key(row) for row in existing.itertuples(index=False)}
    print(f"Najdena obstoječa datoteka z {len(existing)} vrsticami - te kombinacije bodo preskočene.")
    return existing.to_dict("records"), done


def error_stats(group: pd.DataFrame, with_std: bool = False) -> pd.Series:
    """Štetje, štetje napak, mean/(std)/median/p95/max napak translacije in
    rotacije, mean pokritosti, delež sprejetih (accept rate) - samo čez
    uspešne (status='ok') zagone."""
    ok = group[group["status"] == "ok"]
    stats = {"count": len(group), "error_count": int((group["status"] == "error").sum())}
    for column in ("trans_error_mm", "rot_error_deg"):
        stats[f"{column}_mean"] = ok[column].mean()
        if with_std:
            stats[f"{column}_std"] = ok[column].std()
        stats[f"{column}_median"] = ok[column].median()
        stats[f"{column}_p95"] = ok[column].quantile(0.95) if len(ok) else np.nan
        stats[f"{column}_max"] = ok[column].max()
    stats["target_coverage_mean"] = ok["target_coverage"].mean()
    stats["accept_rate"] = ok["accepted"].mean() if len(ok) else np.nan
    return pd.Series(stats)
