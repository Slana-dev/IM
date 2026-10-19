"""camera_translation_benchmark.py - kako KONSISTENTNA (ponovljiva) je
registracija posamezne kamere, ko se del premika navzven (stran od
kamerine optične osi v (0,0,0)), vse do roba kamerinega LISTED (dokumentiranega
specifikacijskega, kamere.py-jevega scanning_area_mm) vidnega polja.

Sestrska skripta camera_benchmark.py testira natančnost na peščici FIKSNIH
pozicij znotraj kamerinega scanning_area_mm; ta skripta namesto tega dela en
sam, gost, NAVZVEN rastoč sweep vzdolž ene same smeri, dokler pozicija ne bi
presegla dokumentirano vidno polje kamere - odgovarja na "kako hitro/
enakomerno se natančnost degradira, ko se del oddaljuje od centra", ne "kako
natančna je registracija na nekaj reprezentativnih pozicijah".

EKSPERIMENTALNI NAČRT (glej konfiguracijske konstante spodaj):
  Za vsako kamero v CAMERAS_TO_TEST (VSE iz kamere.CAMERAS RAZEN
  "mecheye_pro_s" - eksplicitno preskočena po uporabnikovi zahtevi):
    - Del začne NATANKO POD kamero (distance_mm=0, torej translation_x=y=0),
      nato se premika navzven vzdolž ENE SAME, fiksne smeri
      (OUTWARD_DIRECTION_DEG, privzeto 0 stopinj = vzdolž +X osi) v korakih
      po STEP_MM (privzeto 10mm): distance_mm = 0, 10, 20, 30, ...
    - Na VSAKI poziciji (distance_mm) se izvede N_ITERATIONS_PER_POSITION
      (privzeto 100) neodvisnih simuliranih skenov - vsak z DRUGAČNIM yaw
      kotom (enakomerno razporejenih čez 0-360 stopinj, torej i * 360/100 =
      i * 3.6 stopinj) IN drugačnim naključnim semenom šuma (SEED_BASE+i) -
      torej resnično 100 RAZLIČNIH skenov na pozicijo, ne 100 ponovitev
      istega vhoda.
    - Po vseh 100 ponovitvah se pozicija premakne za STEP_MM navzven in
      postopek se ponovi.
    - Sweep za posamezno kamero se USTAVI, ko bi naslednja pozicija
      (distance_mm + del-radij + varnostni rob) presegla kamerino LISTED
      scanning_area_mm - glej compute_max_outward_distance_mm(). To je
      UMETNA (analitična, iz specifikacije izračunana), ne empirično
      zaznana meja - namenoma: add_table_background=True (glej
      benchmark_common.BASE_CONFIG) je tu VKLJUČEN, torej ray casting v resnici
      pogosto zadene mizo tudi, ko je del že zunaj kamerinega
      dokumentiranega vidnega polja (fizično bi kamera v resnici še vedno
      nekaj videla - samo ne dela), zato dejanski ray-miss ni zanesljiv
      signal za to skripto (za razliko od prejšnje različice te datoteke,
      ki je add_table_background namenoma izklopila iz istega razloga v
      obratni smeri). Meja je torej "kamera po specifikaciji dela na tej
      razdalji ne bi smela videti", ne "ray casting je dejansko nehal
      zadevati kaj koli".

  "Konsistentnost" pomeni STD (ne le mean/median) napake translacije/
  rotacije med teh 100 ponovitev na vsaki (kamera, distance_mm) kombinaciji
  - glej "summary" list v Excel izhodu (build_summary spodaj).

Skupni del z camera_benchmark.py (en zagon -> vrstica, nadaljevanje,
statistika) je v benchmark_common.py.

RESUME: kot camera_benchmark.py - obstoječi Excel se prebere, že opravljene
(camera, distance_mm, iteration) kombinacije se preskočijo. Meja (število
pozicij) se vsakič izračuna analitično iz kamere.py, torej je enaka na vsak
zagon - ni odvisna od tega, kaj je bilo že opravljeno.

OPOZORILO O ČASU IZVAJANJA: pri privzetih nastavitvah je to ZNATNO več
zagonov kot camera_benchmark.py (za vsako kamero grobo
(kamerin_scanning_area_polovica_mm / STEP_MM) pozicij x 100 ponovitev, npr.
~15-25 pozicij x 100 = 1500-2500 zagonov na kamero). Za hitro sanity-check
pred polnim zagonom uporabite `python camera_translation_benchmark.py
--smoke-test` (1 kamera, 3 pozicije, 5 ponovitev na pozicijo). Excel se
periodično shranjuje (glej SAVE_EVERY_N_RUNS) - prekinjen zagon se lahko
varno nadaljuje s ponovnim zagonom iste skripte.
"""

import sys
from dataclasses import replace
from datetime import datetime

import numpy as np
import pandas as pd

import kamere
import kontrolna_plosca as kp
from benchmark_common import (BASE_CONFIG, BASE_PATH, RESULT_COLUMNS, cad_part_radius_mm, error_stats,
                              load_existing_runs, run_one)

OUTPUT_FILE = BASE_PATH / "camera_translation_benchmark.xlsx"


# =============================================================================
# KONFIGURACIJA EKSPERIMENTA
# =============================================================================
RESOLUTION_SCALE = 0.25
N_ITERATIONS_PER_POSITION = 100
STEP_MM = 10.0
OUTWARD_DIRECTION_DEG = 0.0   # 0 = vzdolž +X osi - spremenite za drugo smer (npr. 45.0 za diagonalo NE)
SEED_BASE = 5000
POSITION_MARGIN_EXTRA_MM = 10.0   # dodaten varnostni rob čez CAD-ov lastni XY polmer (glej compute_max_outward_distance_mm) - ista logika/privzetek kot camera_benchmark.py
MAX_POSITIONS_SAFETY_CAP = 80   # varovalka proti neskončni zanki (80*10mm=800mm) - nobena testirana kamera naj tega ne doseže; če se to zgodi, WARNING izpiše, da preveriš nastavitve
SAVE_EVERY_N_RUNS = 20   # Excel prepis je O(trenutno_število_vrstic) - pri tisočih zagonih bi shranjevanje po vsakem posamičnem zagonu (kot camera_benchmark.py, ki ima ~10x manj vrstic) postalo prevladujoč strošek; namesto tega shranimo vsakih SAVE_EVERY_N_RUNS zagonov + vedno ob koncu vsake kamere/skripte

CAMERAS_TO_TEST = [name for name in kamere.CAMERAS if name != "mecheye_pro_s"]   # Mech-Eye eksplicitno preskočena (uporabnikova zahteva)

# Miza je fizični prior za yaw_sweep_registration IN velikost mizne mreže v
# ray-casting sceni - namenoma velika, da ne umetno omeji dosega pred
# kamerino LISTED FOV mejo (glej compute_max_outward_distance_mm).
CONFIG = replace(BASE_CONFIG, table_size_x=1000.0, table_size_y=1000.0)

RUNS_COLUMNS = ["camera", "distance_mm", "direction_deg", "iteration", "tx_mm", "ty_mm", "yaw_deg", "seed",
                *RESULT_COLUMNS]


def compute_max_outward_distance_mm(scanning_area_mm: dict, direction_deg: float, margin_mm: float) -> float:
    """Analitično izračuna največjo razdaljo (mm) vzdolž direction_deg, pri
    kateri del (zmanjšan za margin_mm - CAD-ov lastni XY polmer + varnostni
    rob, glej POSITION_MARGIN_EXTRA_MM) ŠE OSTANE znotraj kamerinega LISTED
    (specifikacijskega) pravokotnega scanning_area_mm, centriranega na
    (0,0). Za pravokotno vidno polje [-W/2,W/2] x [-H/2,H/2] in smer
    (cos,sin) je omejujoča tista os, ki jo pri danem kotu prva doseže -
    d_max = min(W/2/|cos|, H/2/|sin|) (os z |komponenta|~0 je izpuščena, ker
    je za njo meja neskončna)."""
    half_w = scanning_area_mm["width"] / 2.0
    half_h = scanning_area_mm["height"] / 2.0
    theta = np.radians(direction_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    candidates = []
    if abs(cos_t) > 1e-9:
        candidates.append(half_w / abs(cos_t))
    if abs(sin_t) > 1e-9:
        candidates.append(half_h / abs(sin_t))
    boundary_mm = min(candidates) if candidates else float("inf")   # candidates prazen samo, če je direction_deg degenerirana (ne more se zgoditi za realen kot)
    return max(boundary_mm - margin_mm, 0.0)


def _run_key(row) -> tuple:
    return (row.camera, round(float(row.distance_mm), 3), int(row.iteration))


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Povzetek po (kamera x distance_mm). STD je tu GLAVNA metrika
    ("konsistentnost" = nizka variabilnost med ponovitvami na isti poziciji),
    ne le dodatek."""
    return df.groupby(["camera", "distance_mm"], group_keys=False)[df.columns].apply(
        lambda group: error_stats(group, with_std=True)).reset_index()


def save_excel(runs_rows: list[dict], config_rows: list[dict]) -> None:
    """Prepiše celoten Excel (runs/summary/config) - glej SAVE_EVERY_N_RUNS
    za razlog, zakaj se to NE kliče po vsakem posameznem zagonu (za razliko
    od camera_benchmark.py, glej modulski docstring)."""
    runs_df = pd.DataFrame(runs_rows, columns=RUNS_COLUMNS)
    summary_df = build_summary(runs_df) if len(runs_df) else pd.DataFrame()

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        runs_df.to_excel(writer, sheet_name="runs", index=False)

        summary_df.to_excel(writer, sheet_name="summary", index=False, startrow=0)
        note_row = len(summary_df) + 2
        pd.DataFrame([[
            "OPOMBA: 'konsistentnost' berite prek trans_error_mm_std/rot_error_deg_std "
            "(variabilnost med 100 ponovitvami na isti distance_mm) - accept_rate uporablja "
            "NEUMERJENE placeholder pragove (glej main_trial.decide_registration_outcome)."]]).to_excel(
            writer, sheet_name="summary", index=False, header=False,
            startrow=note_row, startcol=0)

        pd.DataFrame(config_rows).to_excel(writer, sheet_name="config", index=False)


def main(smoke_test: bool = False) -> None:
    print("=" * 70)
    print(f"camera_translation_benchmark.py {'(SMOKE TEST)' if smoke_test else '(FULL SWEEP)'}")
    print("=" * 70)

    part_radius_mm = cad_part_radius_mm(CONFIG)
    margin_mm = part_radius_mm + POSITION_MARGIN_EXTRA_MM
    print(f"CAD part XY radius (native frame): {part_radius_mm:.1f}mm, "
          f"margin used (radius + {POSITION_MARGIN_EXTRA_MM}mm): {margin_mm:.1f}mm")

    n_iterations = N_ITERATIONS_PER_POSITION
    max_positions_cap = MAX_POSITIONS_SAFETY_CAP
    cameras = list(CAMERAS_TO_TEST)
    if smoke_test:
        cameras = cameras[:1]
        n_iterations = 5
        max_positions_cap = 3

    print(f"Testing cameras: {cameras} (skipped: mecheye_pro_s, per user request)")
    print(f"Direction: {OUTWARD_DIRECTION_DEG} deg, step: {STEP_MM}mm, "
          f"{n_iterations} iterations/position, resolution_scale={RESOLUTION_SCALE}")

    existing_rows, done_set = load_existing_runs(OUTPUT_FILE, _run_key)
    all_rows = list(existing_rows)

    yaw_angles = [i * (360.0 / n_iterations) for i in range(n_iterations)]   # enakomerno razporejeni čez 0-360 stopinj
    seeds = [SEED_BASE + i for i in range(n_iterations)]
    direction_rad = np.radians(OUTWARD_DIRECTION_DEG)
    dir_cos, dir_sin = np.cos(direction_rad), np.sin(direction_rad)

    config_rows = [{
        "camera": "(global settings)",
        "resolution_scale": RESOLUTION_SCALE,
        "n_iterations_per_position": n_iterations,
        "step_mm": STEP_MM,
        "outward_direction_deg": OUTWARD_DIRECTION_DEG,
        "seed_base": SEED_BASE,
        "cad_part_radius_mm": part_radius_mm,
        "position_margin_mm": margin_mm,
        "cameras_tested": ",".join(cameras),
        "cameras_skipped": "mecheye_pro_s",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }]

    completed_this_session = 0
    unsaved_since_last_write = 0
    elapsed_times: list[float] = []

    def maybe_save(force: bool = False) -> None:
        nonlocal unsaved_since_last_write
        if force or unsaved_since_last_write >= SAVE_EVERY_N_RUNS:
            save_excel(all_rows, config_rows)
            unsaved_since_last_write = 0

    for camera_name in cameras:
        spec = kamere.CAMERAS[camera_name]
        cam_config = replace(CONFIG, **kp.resolve_camera_params(camera_name, resolution_scale=RESOLUTION_SCALE))

        max_distance_mm = compute_max_outward_distance_mm(spec["scanning_area_mm"], OUTWARD_DIRECTION_DEG, margin_mm)
        n_positions = int(max_distance_mm // STEP_MM) + 1   # +1 za distance_mm=0 (center)
        n_positions = min(n_positions, max_positions_cap)
        if n_positions >= max_positions_cap:
            print(f"  WARNING: {camera_name}: izračunano število pozicij doseže "
                  f"MAX_POSITIONS_SAFETY_CAP ({max_positions_cap}) - preveri scanning_area_mm/STEP_MM")

        print(f"\n--- {camera_name} --- "
              f"scanning_area_mm={spec['scanning_area_mm']}, max_distance_mm={max_distance_mm:.1f}, "
              f"{n_positions} position(s) planned ({n_positions * n_iterations} runs)")

        for position_index in range(n_positions):
            distance_mm = position_index * STEP_MM
            tx = distance_mm * dir_cos
            ty = distance_mm * dir_sin

            for iteration in range(n_iterations):
                key = (camera_name, round(distance_mm, 3), iteration)
                if key in done_set:
                    continue
                row = run_one(
                    replace(cam_config, translation_x=tx, translation_y=ty, rotation_z_deg=yaw_angles[iteration]),
                    {"camera": camera_name, "distance_mm": distance_mm, "direction_deg": OUTWARD_DIRECTION_DEG,
                     "iteration": iteration, "tx_mm": tx, "ty_mm": ty, "yaw_deg": yaw_angles[iteration],
                     "seed": seeds[iteration]},
                    seeds[iteration],
                    cache_key_tag=f"_trbench_{camera_name}_d{distance_mm:.0f}_it{iteration}_seed{seeds[iteration]}")
                all_rows.append(row)
                completed_this_session += 1
                unsaved_since_last_write += 1
                elapsed_times.append(row["runtime_s"])
                done_set.add(key)
                maybe_save()

                if iteration % 20 == 0 or iteration == n_iterations - 1:
                    avg_time = sum(elapsed_times[-50:]) / len(elapsed_times[-50:])
                    print(f"    [{camera_name}] distance={distance_mm:.0f}mm iteration={iteration + 1}/{n_iterations} "
                          f"| avg {avg_time:.1f}s/run (last 50) | {completed_this_session} run(s) this session")

        maybe_save(force=True)   # vedno shrani ob koncu kamere, ne glede na SAVE_EVERY_N_RUNS
        config_rows.append({
            "camera": camera_name,
            "scanning_area_mm": str(spec["scanning_area_mm"]),
            "max_distance_mm": max_distance_mm,
            "n_positions": n_positions,
        })

    maybe_save(force=True)

    print("=" * 70)
    print(f"{'Smoke test' if smoke_test else 'Sweep'} done - "
          f"{completed_this_session} run(s) executed this session, "
          f"{len(all_rows)} total row(s) in {OUTPUT_FILE.name}")
    print("=" * 70)


if __name__ == "__main__":
    smoke = "--smoke-test" in sys.argv
    main(smoke_test=smoke)
