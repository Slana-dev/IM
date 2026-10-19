"""camera_benchmark.py - statistična primerjava treh kamer (glej kamere.py)
na natančnost registracije: veliko naključnih semen na majhnem naboru
reprezentativnih pozicij/zasukov, namesto ene same (od sreče šumovnega
vzorca odvisne) meritve na pogoj.

EKSPERIMENTALNI NAČRT (glej konfiguracijske konstante spodaj):
  Za vsako kamero v kamere.CAMERAS:
    - 7 testnih pozicij v ravnini mize (center, 4 vogali, sredini levega/
      desnega roba), izpeljanih iz KAMERINEGA LASTNEGA scanning_area_mm pri
      working_distance_mm, zamaknjenih od roba za polmer CAD dela (izračunan
      iz naloženega CAD-a po crop_top_region) + varnostni rob, da del NIKOLI
      ne zapusti vidnega polja - testiramo izven-sredinsko natančnost, ne
      "del manjka".
    - 4 yaw koti, namerno zamaknjeni za pol koraka yaw-sweep preiskovanja
      (yaw_step_deg=3.0 -> zamik 1.5 stopinje): 1.5, 91.5, 181.5, 271.5 -
      testira najslabši primer diskretizacije yaw, namesto referenc, ki
      pristanejo točno na testiranem kandidatu.
    - N_SEEDS naključnih semen na (pozicija x yaw) kombinacijo.
  Skupaj 7 x 4 x N_SEEDS zagonov na kamero.

Kamera je v main_trial.py vedno fiksirana v (0,0,0) in gleda naravnost
navzdol; referenčno pozo (s popravkom za center skenirane regije) sestavi
run_registration sam iz translation_*/rotation_* v RegistrationConfig.
Excel izhod ima 3 liste (runs/summary/config) in podpira nadaljevanje
prekinjenega zagona; `--smoke-test` požene le peščico zagonov.
"""

import sys
from dataclasses import replace
from datetime import datetime

import pandas as pd

import kamere
import kontrolna_plosca as kp
from benchmark_common import BASE_CONFIG, BASE_PATH, cad_part_radius_mm, error_stats, load_existing_runs, run_one

OUTPUT_FILE = BASE_PATH / "camera_benchmark.xlsx"


# =============================================================================
# KONFIGURACIJA EKSPERIMENTA
# =============================================================================
RESOLUTION_SCALE = 0.25          # skalira vsako kamerino depth_map_resolution_px navzdol (hitrost vs. natančnost)
N_SEEDS = 13                     # naključnih semen na (pozicija x yaw) kombinacijo
SEED_BASE = 1000                 # dejanska semena so SEED_BASE, SEED_BASE+1, ...
YAW_ANGLES_DEG = [1.5, 91.5, 181.5, 271.5]   # namerno zamaknjeno za pol yaw_step_deg (3.0/2)
POSITION_MARGIN_EXTRA_MM = 10.0  # dodaten varnostni rob čez CAD-ov lastni XY polmer

# Manjkajoči specs podatki za Mech-Eye - KONZERVATIVNE OCENE, primerljive z
# ostalima dvema kamerama (glej kamere.py - global_planarity_mm in
# relative_distance_accuracy_permille NISTA bila podana v prejeti
# specifikaciji, torej bi se sicer privzeto simulirala kot 0, kar bi Mech-Eye
# dalo nepošteno prednost "po izpustitvi"). Mutira kamere.CAMERAS SAMO v tem
# procesu (v pomnilniku) - ne spreminja kamere.py na disku.
SPEC_OVERRIDES = {
    "mecheye_pro_s": {
        "global_planarity_mm": 0.15,               # OCENA - primerljivo s Photoneo (0.22) / Zivid (0.10)
        "relative_distance_accuracy_permille": 1.5,   # OCENA - primerljivo s Photoneo (1.25) / Zivid (2.0)
    },
}

# Eksplicitno pripeto na 3.0 (ne main_trial.py-jev privzetek 6.0) -
# YAW_ANGLES_DEG zgoraj je namenoma zamaknjen za natanko POL yaw_step_deg
# (1.5 = 3.0/2), da testira najslabši primer diskretizacije za TA korak;
# sprememba yaw_step_deg brez ustreznega preračuna YAW_ANGLES_DEG bi to
# razmerje pokvarila. Miza eksplicitno 500x500mm - glej table_half_extent_mm
# preverbo v main().
CONFIG = replace(BASE_CONFIG, table_size_x=500.0, table_size_y=500.0, yaw_step_deg=3.0)

RUNS_COLUMNS = [
    "camera", "position", "tx_mm", "ty_mm", "yaw_deg", "seed", "status",
    "rot_error_deg", "trans_error_mm", "coarse_rot_error_deg", "coarse_trans_error_mm",
    "fitness", "inlier_rmse", "asymmetric_fraction", "target_coverage",
    "accepted", "reject_reasons", "estimated_fields", "runtime_s", "error_message",
]


def apply_spec_overrides() -> None:
    """Vpiše SPEC_OVERRIDES v kamere.CAMERAS (v pomnilniku, samo za ta
    proces) - klicati ENKRAT, pred kakršnimkoli klicem
    kontrolna_plosca.resolve_camera_params()."""
    for camera_name, overrides in SPEC_OVERRIDES.items():
        kamere.CAMERAS[camera_name].update(overrides)
        print(f"  SPEC_OVERRIDES uporabljen za '{camera_name}': {overrides}")


def compute_positions(scanning_area_mm: dict, margin_mm: float) -> list[tuple[str, float, float]]:
    """7 testnih pozicij (center, 4 vogali, sredini levega/desnega roba)
    znotraj kamerinega scanning_area_mm, zamaknjenih od roba za margin_mm
    (CAD-ov lastni XY polmer + varnostni rob), da del NIKOLI ne zapusti
    vidnega polja - testiramo natančnost izven-sredinske registracije, ne
    "del manjka"."""
    usable_x = scanning_area_mm["width"] / 2.0 - margin_mm
    usable_y = scanning_area_mm["height"] / 2.0 - margin_mm
    if usable_x <= 0 or usable_y <= 0:
        raise ValueError(
            f"margin_mm={margin_mm:.1f} je prevelik za scanning_area_mm={scanning_area_mm} "
            f"(usable_x={usable_x:.1f}, usable_y={usable_y:.1f}) - zmanjšaj CAD polmer ali "
            f"POSITION_MARGIN_EXTRA_MM")
    return [
        ("center", 0.0, 0.0),
        ("corner_NE", usable_x, usable_y),
        ("corner_NW", -usable_x, usable_y),
        ("corner_SE", usable_x, -usable_y),
        ("corner_SW", -usable_x, -usable_y),
        ("edge_E", usable_x, 0.0),
        ("edge_W", -usable_x, 0.0),
    ]


def _run_key(row) -> tuple:
    return (row.camera, round(float(row.tx_mm), 3), round(float(row.ty_mm), 3),
            round(float(row.yaw_deg), 3), int(row.seed))


def save_excel(runs_rows: list[dict], config_rows: list[dict]) -> None:
    """Prepiše celoten Excel (runs/summary/config) - pri tej velikosti
    (do ~1100 vrstic) je popoln prepis vsak zagon dovolj hiter in enostavno
    zanesljiv."""
    runs_df = pd.DataFrame(runs_rows, columns=RUNS_COLUMNS)
    by_camera = by_camera_position = pd.DataFrame()
    if len(runs_df):
        by_camera = runs_df.groupby("camera", group_keys=False)[runs_df.columns].apply(error_stats).reset_index()
        by_camera_position = runs_df.groupby(["camera", "position"], group_keys=False)[runs_df.columns].apply(
            error_stats).reset_index()

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        runs_df.to_excel(writer, sheet_name="runs", index=False)

        by_camera.to_excel(writer, sheet_name="summary", index=False, startrow=0)
        note_row = len(by_camera) + 2
        pd.DataFrame([[
            "OPOMBA: accept_rate uporablja NEUMERJENE placeholder pragove "
            "(glej main_trial.decide_registration_outcome) - kamere razvrščajte "
            "po trans_error_mm/rot_error_deg, NE po accept_rate."]]).to_excel(
            writer, sheet_name="summary", index=False, header=False,
            startrow=note_row, startcol=0)
        by_camera_position.to_excel(writer, sheet_name="summary", index=False, startrow=note_row + 3)

        pd.DataFrame(config_rows).to_excel(writer, sheet_name="config", index=False)


def main(smoke_test: bool = False) -> None:
    print("=" * 70)
    print(f"camera_benchmark.py {'(SMOKE TEST)' if smoke_test else '(FULL SWEEP)'}")
    print("=" * 70)

    apply_spec_overrides()

    part_radius_mm = cad_part_radius_mm(CONFIG)
    margin_mm = part_radius_mm + POSITION_MARGIN_EXTRA_MM
    print(f"CAD part XY radius (native frame): {part_radius_mm:.1f}mm, "
          f"margin used (radius + {POSITION_MARGIN_EXTRA_MM}mm): {margin_mm:.1f}mm")

    cameras = list(kamere.CAMERAS.keys())
    yaw_list = list(YAW_ANGLES_DEG)
    seeds = [SEED_BASE + i for i in range(N_SEEDS)]

    if smoke_test:
        cameras = cameras[:1]
        yaw_list = yaw_list[:1]
        seeds = seeds[:2]

    existing_rows, done_set = load_existing_runs(OUTPUT_FILE, _run_key)
    all_rows = list(existing_rows)

    table_half_extent_mm = CONFIG.table_size_x / 2.0   # fizični prior v yaw_sweep_registration

    plan = []
    config_rows = []
    for camera_name in cameras:
        spec = kamere.CAMERAS[camera_name]
        camera_params = kp.resolve_camera_params(camera_name, resolution_scale=RESOLUTION_SCALE)
        cam_config = replace(CONFIG, **camera_params)
        estimated_fields = ",".join(sorted(SPEC_OVERRIDES.get(camera_name, {}).keys()))
        positions = compute_positions(spec["scanning_area_mm"], margin_mm)
        if smoke_test:
            positions = positions[:1]   # samo "center"

        for _, tx, ty in positions:
            if abs(tx) >= table_half_extent_mm or abs(ty) >= table_half_extent_mm:
                raise ValueError(
                    f"{camera_name}: pozicija ({tx:.1f},{ty:.1f}) presega privzeto mizo "
                    f"({table_half_extent_mm:.0f}mm polovična stranica) - yaw_sweep_registration "
                    f"bo zavrnil VSE kandidate (fizični prior). Zmanjšaj POSITION_MARGIN_EXTRA_MM "
                    f"ali povečaj table_size_x/y v CONFIG.")

        config_rows.append({
            "camera": camera_name,
            "working_distance_mm": spec["working_distance_mm"],
            "camera_fov_deg": camera_params.get("camera_fov_deg"),
            "camera_width_px": camera_params.get("camera_width_px"),
            "camera_height_px": camera_params.get("camera_height_px"),
            "depth_noise_at_1m": camera_params.get("depth_noise_at_1m"),
            "noise_reference_distance_m": camera_params.get("noise_reference_distance_m"),
            "distance_bias_permille": camera_params.get("distance_bias_permille"),
            "global_planarity_mm": camera_params.get("global_planarity_mm"),
            "noise_spatial_correlation_px": camera_params.get("noise_spatial_correlation_px"),
            "estimated_fields": estimated_fields,
            "resolution_scale": RESOLUTION_SCALE,
        })

        for position_label, tx, ty in positions:
            for yaw_deg in yaw_list:
                for seed in seeds:
                    plan.append((camera_name, position_label, tx, ty, yaw_deg, seed,
                                 cam_config, estimated_fields))

    config_rows.append({
        "camera": "(global settings)",
        "resolution_scale": RESOLUTION_SCALE,
        "n_seeds": N_SEEDS,
        "seed_base": SEED_BASE,
        "yaw_angles_deg": str(YAW_ANGLES_DEG),
        "position_margin_extra_mm": POSITION_MARGIN_EXTRA_MM,
        "cad_part_radius_mm": part_radius_mm,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })

    total = len(plan)
    print(f"Planned {total} runs across {len(cameras)} camera(s), "
          f"{len(yaw_list)} yaw angle(s), {len(seeds)} seed(s)")

    completed_this_session = 0
    elapsed_times: list[float] = []
    for i, (camera_name, position_label, tx, ty, yaw_deg, seed, cam_config,
            estimated_fields) in enumerate(plan, start=1):
        key = (camera_name, round(tx, 3), round(ty, 3), round(yaw_deg, 3), seed)
        if key in done_set:
            continue

        print(f"[{i}/{total}] {camera_name} {position_label} yaw={yaw_deg} seed={seed} ...",
              end=" ", flush=True)
        row = run_one(
            replace(cam_config, translation_x=tx, translation_y=ty, rotation_z_deg=yaw_deg),
            {"camera": camera_name, "position": position_label, "tx_mm": tx, "ty_mm": ty,
             "yaw_deg": yaw_deg, "seed": seed, "estimated_fields": estimated_fields},
            seed, cache_key_tag=f"_bench_{camera_name}_{position_label}_yaw{yaw_deg}_seed{seed}",
            print_traceback=True)
        all_rows.append(row)
        completed_this_session += 1
        elapsed_times.append(row["runtime_s"])

        if row["status"] == "ok":
            print(f"-> trans_err={row['trans_error_mm']:.2f}mm rot_err={row['rot_error_deg']:.2f}deg "
                  f"({row['runtime_s']:.1f}s)")
        else:
            print(f"-> ERROR ({row['runtime_s']:.1f}s)")

        avg_time = sum(elapsed_times) / len(elapsed_times)
        remaining = max(total - i, 0)
        eta_s = avg_time * remaining
        print(f"    progress {i}/{total} | avg {avg_time:.1f}s/run | "
              f"ETA {eta_s / 3600:.1f}h ({eta_s / 60:.0f} min)")

        save_excel(all_rows, config_rows)

    print("=" * 70)
    print(f"{'Smoke test' if smoke_test else 'Sweep'} done - "
          f"{completed_this_session} run(s) executed this session, "
          f"{len(all_rows)} total row(s) in {OUTPUT_FILE.name}")
    print("=" * 70)


if __name__ == "__main__":
    smoke = "--smoke-test" in sys.argv
    main(smoke_test=smoke)
