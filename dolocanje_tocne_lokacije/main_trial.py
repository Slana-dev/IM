"""Določanje točne lokacije ohišja: registracija CAD modela na sken 3D kamere.

Cevovod (run_registration):
  Phase 1   CAD -> source oblak, obrezan na zgornjo "sealing" regijo
  Phase 2   target: pravi sken iz datoteke ali simuliran sken (scan_simulation)
  Phase 3   voxel downsampling + normale (s predpomnilnikom)
  Phase 4   groba registracija (coarse_registration.yaw_sweep_registration)
            + večnivojski ICP (fine_registration.refine_registration)
  Phase 4.5 point-to-mesh polish na točni CAD mreži
  Phase 5-7 varnostni pregledi in PASS/FAIL odločitev (verification)

Zagon: `python main_trial.py --help` ali prek kontrolna_plosca.py.
Neničeln izhod (1) pomeni, da robot NE sme uporabiti dobljene poze.
"""
import copy
import hashlib
import sys
from dataclasses import astuple, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import open3d as o3d

from coarse_registration import yaw_sweep_registration
from fine_registration import polish_on_cad_mesh, refine_registration
from point_clouds import (OrganizedScan, check_cad_frame, compute_native_reference_point, crop_top_region,
                          filter_grazing_incidence_points, filter_grazing_incidence_points_from_grid,
                          load_cad_mesh, load_cad_model, load_real_scans, preprocess_point_cloud,
                          remove_table_background, top_camera_location)
from poses import FLIP_UP, build_reference_transform, registration_result, transform_with_pivot, transformation_error
from registration_config import RegistrationConfig, config_from_args, parse_args
from scan_simulation import simulate_camera_scan
from self_tests import run_self_tests
from verification import (compute_pose_covariance, decide_registration_outcome, evaluate_target_coverage,
                          identify_asymmetric_points_cached, verify_asymmetric_feature_alignment)
from workcell import CAMERA_LOCATION


# funkcija za vizualizacijo rezultatov registracije
def draw_registration_result(source: o3d.geometry.PointCloud,   # izvorni oblak točk
                             target: o3d.geometry.PointCloud,   # ciljni oblak točk
                             transformation: np.ndarray,      # transformacijska matrika
                             window_name: str = "Alignment") -> None:   # ime okna za vizualizacijo
    source_temp = copy.deepcopy(source)     # kopiramo izvorni oblak točk, da ne spremenimo originala
    target_temp = copy.deepcopy(target)     # kopiramo ciljni oblak točk, da ne spremenimo originala
    source_temp.paint_uniform_color([1.0, 0.706, 0.0])  # source pobarvamo oranžno
    target_temp.paint_uniform_color([0.0, 0.651, 0.929])  # target pobarvamo modro
    source_temp.transform(transformation)   # uporabimo transformacijsko matriko na izvorni oblak točk
    o3d.visualization.draw_geometries([source_temp, target_temp], window_name=window_name)  # odpre okno in nariše oba oblaka skupaj


@dataclass
class RegistrationOutcome:
    source: o3d.geometry.PointCloud                            # obrezan CAD oblak (polna ločljivost)
    target: o3d.geometry.PointCloud                            # sken po filtrih (polna ločljivost)
    coarse: o3d.pipelines.registration.RegistrationResult      # yaw sweep
    icp: o3d.pipelines.registration.RegistrationResult         # končna poza (po ICP in morebitnem polish-u)
    decision: dict                                             # PASS/FAIL, glej verification.decide_registration_outcome
    reference_transform: np.ndarray                            # ground-truth poza, iz katere je simuliran sken
    reference_point: np.ndarray                                # center "sealing" regije v CAD okviru
    # Vse poze (coarse, icp, reference_transform), source in reference_point so
    # v koordinatnem sistemu CAD DATOTEKE, tudi pri flip_up_direction (glej _ViewFrame).


class _ViewFrame:
    """Notranji okvir cevovoda: kamera vedno gleda +Z stran dela (ploskev,
    ki jo skeniramo). Pri flip_up_direction gleda kamera CAD-ovo -Z stran,
    zato CAD notranje zavrtimo za 180° okoli X (poses.FLIP_UP) in rezultate
    na koncu pretvorimo nazaj - robot tako vedno dobi pozo v koordinatnem
    sistemu, v katerem je izvožen STL."""

    def __init__(self, flip: bool):
        self.matrix = FLIP_UP if flip else None   # CAD okvir -> notranji okvir

    def geometry(self, geometry):
        """CAD mreža/oblak v notranjem okviru (kopija; brez kopije, če ni obrata)."""
        return geometry if self.matrix is None else copy.deepcopy(geometry).transform(self.matrix)

    def pose_to_cad(self, transformation: np.ndarray) -> np.ndarray:
        """Poza notranji okvir -> kamera  ==>  poza CAD okvir -> kamera."""
        return transformation if self.matrix is None else transformation @ self.matrix

    def point_to_cad(self, point: np.ndarray) -> np.ndarray:
        return point if self.matrix is None else self.matrix[:3, :3].T @ point   # FLIP_UP je pravokotna -> inverz = transponirana


def _reference_transform(config: RegistrationConfig, reference_point: np.ndarray, view: _ViewFrame) -> np.ndarray:
    """Ground-truth poza dela glede na kamero iz --translation_*/--rotation_*.

    Pri simulaciji sta translacija in rotacija podani glede na
    reference_point (center skenirane regije, glej
    compute_native_reference_point), ne glede na surov izvor CAD datoteke.
    Pri pravem skenu se poza ne uporablja za simulacijo, zato pivot popravek
    ni potreben (pri manjkajoči datoteki skena se kljub temu simulira s to pozo).
    Vrne pozo v notranjem okviru (glej _ViewFrame), izpiše jo v CAD okviru."""
    c = config
    if c.use_real_scan:
        transform = build_reference_transform(c.translation_x, c.translation_y, c.translation_z,
                                              c.rotation_x_deg, c.rotation_y_deg, c.rotation_z_deg)
    else:
        print(f"Native CAD reference point (center of the scanned sealing region, "
              f"pre-rotation/translation): {view.point_to_cad(reference_point)} - --translation_x/y/z is "
              f"relative to THIS point, not the CAD file's raw coordinate origin, and "
              f"rotation is applied AROUND this point, not around the CAD file's raw origin")
        rotation = build_reference_transform(0.0, 0.0, 0.0, c.rotation_x_deg,
                                             c.rotation_y_deg, c.rotation_z_deg)[:3, :3]
        transform = transform_with_pivot(rotation, np.array([c.translation_x, c.translation_y, c.translation_z]),
                                         reference_point)
    print(f"Reference (ground-truth) transform - part pose relative to the camera "
          f"(fixed at the origin):\n{view.pose_to_cad(transform)}\n")
    return transform


def _remove_table_enabled(config: RegistrationConfig) -> bool:
    # Privzeto vklopljeno za PRAVE skene (domala vedno zajamejo vsaj nekaj
    # mize/ozadja - ocena naklona bi sicer lahko ocenila normalo MIZE namesto
    # zgornje ploskve dela), izklopljeno za simulacijo (simuliran sken nima
    # mize, razen z add_table_background - tedaj odloči uporabnik).
    return config.use_real_scan if config.remove_table_background is None else config.remove_table_background


def _acquire_target(config: RegistrationConfig, scan_mesh: o3d.geometry.TriangleMesh,
                    source: o3d.geometry.PointCloud, part_height_mm: float,
                    reference_transform: np.ndarray, real_scan_paths: Optional[list[Path]]
                    ) -> tuple[o3d.geometry.PointCloud, Optional[OrganizedScan]]:
    """Phase 2: pravi sken iz datoteke (real_scan_paths) ali simuliran sken,
    nato odstranitev mize/ozadja in letečih pikslov. Vrne tudi organizirano
    mrežo skena, kadar obstaja (simuliran ray casting) - usklajeno s
    target-om po vsakem filtru."""
    c = config
    print("Phase 2: Loading target scan...")
    if real_scan_paths is not None:
        target, grid = load_real_scans(real_scan_paths), None   # primarni sken + morebitni dodatni zaporedni zajemi za temporalno povprečenje
    else:
        target, grid = simulate_camera_scan(
            scan_mesh, reference_transform, c.noise, fov_deg=c.camera_fov_deg, width_px=c.camera_width_px,
            height_px=c.camera_height_px, up_axis=c.up_axis,
            top_fraction=c.top_fraction, disable_occlusion=c.disable_occlusion,
            table=c.table if c.add_table_background else None)
        print(f"  Simulated target has {len(target.points)} points "
              f"(source has {len(source.points)}) - overlap ratio "
              f"{len(target.points) / len(source.points):.2f}")
    if _remove_table_enabled(c):
        print("  Removing table/background points from target...")
        target, keep_mask = remove_table_background(
            target, up_axis=c.up_axis, part_height_mm=part_height_mm,
            cropped_region_height_mm=float(source.get_max_bound()[c.up_axis] - source.get_min_bound()[c.up_axis]),
            margin_mm=c.table_removal_margin_mm)   # odstranimo domnevne mizne/ozadje točke IN zavese pod skenirano regijo
        if grid is not None:
            grid = grid.keep_only(keep_mask)
    return target, grid


def _filter_flying_pixels(config: RegistrationConfig, target: o3d.geometry.PointCloud,
                          grid: Optional[OrganizedScan]) -> o3d.geometry.PointCloud:
    print("  Filtering grazing-incidence (flying pixel) points from target...")
    if grid is not None:   # hiter, O(n) pot na organizirani mreži simuliranega skena
        return filter_grazing_incidence_points_from_grid(
            grid, CAMERA_LOCATION, max_incidence_deg=config.flying_pixel_filter_max_incidence_deg)
    return filter_grazing_incidence_points(
        target, CAMERA_LOCATION, max_incidence_deg=config.flying_pixel_filter_max_incidence_deg,
        normal_radius=config.voxel_size * config.grazing_incidence_normal_radius_factor)


def _source_cache_key(config: RegistrationConfig, cad_path: Path) -> str:
    c = config
    return (f"source_{cad_path.stem}_{c.sample_points}pts_top{c.top_fraction}_"
            f"axis{c.up_axis}_flip{c.flip_up_direction}_edgemargin{c.crop_edge_margin_factor}")


def _target_cache_key(config: RegistrationConfig, scan_cad_path: Path,
                      real_scan_paths: Optional[list[Path]], cache_key_tag: str) -> str:
    """Ključ identificira *vhod* predobdelave target-a - vse, kar določa
    njegove točke, preden se predobdelava sploh začne (preprocess_point_cloud
    sam doda še svoje parametre)."""
    c = config
    # Odstranitev mize in filter letečih pikslov oba spremenita target PRED
    # predobdelavo - brez teh oznak bi vklop/izklop katerega od njiju tiho
    # servisiral predpomnjen target iz nasprotnega stanja.
    removal_tag = f"_notable{c.table_removal_margin_mm}" if _remove_table_enabled(c) else ""
    grazing_tag = (f"_nograze{c.flying_pixel_filter_max_incidence_deg}_"
                   f"{c.grazing_incidence_normal_radius_factor}" if c.filter_grazing_incidence else "")
    if real_scan_paths is not None:
        # Ime datoteke samo po sebi NI dovolj: robot v produkciji vsak cikel
        # zapiše nov sken v isto pot (npr. "scan.ply") - ključ, ki pozna samo
        # ime, bi za vedno servisiral predobdelavo PRVEGA skena na tej poti.
        # mtime (ns) + velikost VSAKE prispevajoče datoteke zaznata novo
        # vsebino brez branja/hashiranja celotnega oblaka. "filterTrue" je
        # ostanek nekdanje zastavice in ohranja obstoječe ključe veljavne.
        scan_stats_tag = "_".join(
            f"{p.stem}_m{p.stat().st_mtime_ns}_s{p.stat().st_size}" for p in real_scan_paths)
        return f"target_real_{scan_stats_tag}_filterTrue{removal_tag}{grazing_tag}"
    # Poza (translation_*/rotation_*) spremeni simuliran target, ne da bi se
    # odražala v ostalih parametrih - brez nje bi se dve testni pozi zaleteli
    # v isti zapis predpomnilnika. cache_key_tag doda klicatelj (npr. seme
    # šuma v benchmarkih).
    pose_tag = (f"_tx{c.translation_x}_ty{c.translation_y}_tz{c.translation_z}"
                f"_rx{c.rotation_x_deg}_ry{c.rotation_y_deg}_rz{c.rotation_z_deg}")
    noise_digest = hashlib.md5(repr(astuple(c.noise)).encode()).hexdigest()[:10]   # kratek hash vseh šumovnih parametrov
    table_tag = (f"_table{c.table_size_x}x{c.table_size_y}at{tuple(c.table.center)}"
                 if c.add_table_background else "")   # miza vpliva na target samo, če je add_table_background=True
    return (f"target_sim_{scan_cad_path.stem}_top{c.top_fraction}_"
            f"axis{c.up_axis}_flip{c.flip_up_direction}_noise{noise_digest}_occ{c.disable_occlusion}"
            f"_fov{c.camera_fov_deg}_{c.camera_width_px}x{c.camera_height_px}{table_tag}{removal_tag}"
            f"{grazing_tag}{pose_tag}{cache_key_tag}")


def _report_against_reference(label: str, transformation: np.ndarray, reference_transform: np.ndarray,
                              reference_point: np.ndarray) -> None:
    err = transformation_error(transformation, reference_transform, reference_point)
    print(f"{label} vs reference transform (translation error at the part's "
          f"reference point, not the CAD's raw origin): "
          f"rotation_error={err['rotation_error_deg']:.2f} deg, "
          f"translation_error={err['translation_error']:.3f}, "
          f"frobenius_norm={err['frobenius_norm']:.3f}")


def _check_and_decide(config: RegistrationConfig, source: o3d.geometry.PointCloud,
                      source_down: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud,
                      target_down: o3d.geometry.PointCloud,
                      icp_result: o3d.pipelines.registration.RegistrationResult,
                      skipped_icp: bool, source_cache_key: str,
                      cad_mesh: o3d.geometry.TriangleMesh) -> dict:
    """Phase 5-7: varnostni pregledi končne poze in PASS/FAIL odločitev."""
    c = config
    print("Phase 5: Asymmetric-feature safety check...")
    # Skupna fitness/inlier_rmse lahko izgledata v redu, medtem ko je
    # rezultat dejansko konvergiral v ponavljajoče-se-geometrijsko podoben
    # yaw (glej yaw_sweep_registration) - preverimo SPECIFIČNO residual
    # source-ovih razlikovalnih točk (locator pini, konektor). Na source_down
    # (ne polnem source): maska služi le izbiri razlikovalnih točk, polna
    # ločljivost ni potrebna - voxel_size je zato del ključa.
    asymmetric_cache_key = (f"{source_cache_key}_v{c.voxel_size}_down_asymtop{c.asymmetric_check_top_fraction}"
                            if c.use_cad_cache else None)   # None onemogoči cache, če CAD vzorčenje ni deterministično
    asymmetric_mask = identify_asymmetric_points_cached(
        source_down, c.up_axis, source_down.get_center(), top_fraction=c.asymmetric_check_top_fraction,
        cache_key=asymmetric_cache_key)   # katere source_down točke so razlikovalne/asimetrične
    residual_threshold = c.voxel_size * c.asymmetric_check_residual_factor   # prag "dobrega ujemanja" za te točke
    asym_stats = verify_asymmetric_feature_alignment(
        source_down, target, icp_result.transformation, asymmetric_mask, residual_threshold,
        cad_mesh, CAMERA_LOCATION, max_incidence_deg=c.flying_pixel_filter_max_incidence_deg)   # dejanski residual po ICP, samo na točkah, ki jih kamera vidi
    print(f"  {asym_stats['n_points']} asymmetric/distinctive points identified "
          f"(top {c.asymmetric_check_top_fraction:.0%} by self-rotation distance), "
          f"{asym_stats['n_visible']} visible from the camera under this pose: "
          f"mean_residual={asym_stats['mean']:.3f}, median_residual={asym_stats['median']:.3f}, "
          f"max_residual={asym_stats['max']:.3f}, "
          f"fraction_within_{residual_threshold:.2f}mm={asym_stats['fraction_within_threshold']:.2%}")

    print("Phase 6: Target->source coverage check...")
    # GLAVNA metrika za prekrivanje - target->source je robustna na okluzijo,
    # source->target fitness pa ne (glej evaluate_target_coverage).
    coverage_threshold = c.voxel_size * c.target_coverage_distance_factor   # prag "leži na delu" za target točke
    target_coverage = evaluate_target_coverage(source, target, icp_result.transformation, coverage_threshold)
    print(f"  {target_coverage['n_points']} scanned (target) points checked against CAD surface: "
          f"mean_dist={target_coverage['mean']:.3f}, median_dist={target_coverage['median']:.3f}, "
          f"fraction_within_{coverage_threshold:.2f}mm={target_coverage['fraction_within_threshold']:.2%}")

    print("Phase 6.5: Pose covariance (which DOF is actually well-constrained)...")
    # Diagnostika, ne PASS/FAIL prag - glej compute_pose_covariance docstring.
    pose_covariance = compute_pose_covariance(source_down, target_down, icp_result.transformation, coverage_threshold)
    if pose_covariance is not None:
        sr = pose_covariance["sigma_rotation_deg"]
        st = pose_covariance["sigma_translation_mm"]
        print(f"  sigma_rotation_deg (roll,pitch,yaw)=({sr[0]:.3f}, {sr[1]:.3f}, {sr[2]:.3f}), "
              f"sigma_translation_mm (x,y,z)=({st[0]:.3f}, {st[1]:.3f}, {st[2]:.3f}) - "
              f"large values mark a DOF the scanned geometry doesn't actually constrain well "
              f"(e.g. yaw/x/y on a mostly-flat sealing surface with only a small asymmetric "
              f"feature), independent of how good the aggregate fitness/rmse look")
    else:
        print("  Could not compute pose covariance (singular/degenerate information matrix) - "
              "skipping this diagnostic")

    print("Phase 7: PASS/FAIL accept decision...")
    decision = decide_registration_outcome(
        icp_result, asym_stats, target_coverage, skipped_icp,
        min_fitness=c.min_accept_fitness,
        max_inlier_rmse=(c.max_accept_inlier_rmse if c.max_accept_inlier_rmse is not None
                         else c.voxel_size * 1.5),   # privzeto: isti prag, pod katerim je bil icp_result ocenjen (common_threshold v refine_registration)
        min_asymmetric_fraction=c.min_accept_asymmetric_fraction,
        min_target_coverage=c.min_accept_target_coverage)
    decision["pose_covariance"] = pose_covariance   # diagnostika, NE del PASS/FAIL odločitve
    if decision["accepted"]:
        print(f"  ACCEPT: fitness={decision['fitness']:.4f}, inlier_rmse={decision['inlier_rmse']:.4f}, "
              f"asymmetric_fraction={decision['asymmetric_fraction']:.2%}, "
              f"target_coverage={decision['target_coverage_fraction']:.2%} - all thresholds met, "
              f"robot may act on this pose")
    else:
        print(f"  REJECT: robot must NOT act on this pose. Reasons:")
        for reason in decision["reasons"]:
            print(f"    - {reason}")
    return decision


def run_registration(config: RegistrationConfig,
                     cad_path: Path,
                     scan_paths: Sequence[Path] = (),
                     scan_cad_path: Optional[Path] = None,
                     cache_key_tag: str = "",
                     visualize: bool = True) -> RegistrationOutcome:
    """Oceni pozo CAD dela v koordinatnem sistemu kamere (glej modulski docstring).

    scan_paths: pravi sken(i) - prvi je primarni, ostali dodatni zaporedni
    zajemi za temporalno povprečenje. Uporabijo se samo z
    config.use_real_scan=True in le, če primarna datoteka obstaja - sicer se
    target simulira iz config-ove referenčne poze.
    scan_cad_path: STL, iz katerega se SIMULIRA sken (npr. podrobnejši model
    istega dela) - None = isti kot cad_path (referenca). Obe datoteki morata
    biti izvoženi v ISTEM koordinatnem sistemu: ista referenčna poza se
    uporabi za simulacijo, napaka pa se meri proti njej.
    cache_key_tag: dodatna oznaka ključa predpomnilnika simuliranega target-a
    (npr. seme šuma) za klicatelje, ki spreminjajo kaj, česar config ne opiše.
    visualize: False izklopi Open3D okna (benchmarki)."""
    c = config
    view = _ViewFrame(c.flip_up_direction)   # od tu naprej je VSE (CAD, poze) v notranjem okviru - v CAD okvir se pretvori le pri izpisu/vrnitvi
    cad_file_mesh = load_cad_mesh(cad_path)   # NEobrezana CAD mreža v okviru datoteke
    cad_mesh = view.geometry(cad_file_mesh)   # ista mreža v notranjem okviru (za ray casting in mesh-polish)
    reference_point = compute_native_reference_point(cad_mesh, top_fraction=c.top_fraction, up_axis=c.up_axis)
    reference_transform = _reference_transform(c, reference_point, view)

    print("Phase 1: Loading CAD model...")
    source = view.geometry(load_cad_model(cad_path, c.sample_points, use_cache=c.use_cad_cache))   # CAD kot oblak točk (normale se zavrtijo skupaj s točkami)
    full_part_height = float(source.get_max_bound()[c.up_axis] - source.get_min_bound()[c.up_axis])   # znana polna višina CELEGA dela, PRED obrezovanjem - potrebna za remove_table_background
    source = crop_top_region(source, top_fraction=c.top_fraction, up_axis=c.up_axis,
                             edge_margin=c.voxel_size * c.crop_edge_margin_factor)   # obrežemo na zgornje "sealing" območje, minus umeten rob (glej crop_top_region docstring) - samo na CAD/source strani
    print(f"  Cropped CAD model to top {c.top_fraction:.0%} along axis {c.up_axis} "
          f"(flip_up_direction={c.flip_up_direction}) "
          f"(the seal region the camera actually scans): {len(source.points)} points remain")
    # Obrezan source je enostranska zgornja plošča, enako kot target - zato
    # za orientacijo normal potrebuje virtualno kamero nad sabo.
    source_camera_location = top_camera_location(source, up_axis=c.up_axis)
    check_cad_frame(cad_file_mesh)   # izpiše/opozori, v katerem KS je CAD - glej check_cad_frame docstring
    scan_cad_path = scan_cad_path or cad_path
    scan_mesh = cad_mesh
    if scan_cad_path != cad_path and not c.use_real_scan:
        print(f"  Simulated scan is generated from a DIFFERENT mesh than the reference: {scan_cad_path.name}")
        scan_file_mesh = load_cad_mesh(scan_cad_path)
        check_cad_frame(scan_file_mesh)
        scan_mesh = view.geometry(scan_file_mesh)

    real_scan_paths = (list(scan_paths) if c.use_real_scan and scan_paths and scan_paths[0].exists()
                       else None)
    target, grid = _acquire_target(c, scan_mesh, source, full_part_height, reference_transform, real_scan_paths)
    if visualize:   # relativna pozicija pred registracijo - kaj morata grobi in ICP korak sploh rešiti
        draw_registration_result(source, target, np.eye(4), "Pred poravnavo (CAD vs. scan)")
    if c.filter_grazing_incidence:
        target = _filter_flying_pixels(c, target, grid)

    print("Phase 3: Preprocessing point clouds...")
    source_cache_key = _source_cache_key(c, cad_path)
    source_down = preprocess_point_cloud(
        source, c.voxel_size, camera_location=source_camera_location,
        cache_key=source_cache_key, use_cache=c.use_preprocess_cache,
        preserve_existing_normals=True)   # zmanjšan source, z ohranjenimi TOČNIMI CAD (mesh) normalami - glej ensure_oriented_normals
    target_down = preprocess_point_cloud(
        target, c.voxel_size, camera_location=CAMERA_LOCATION,
        cache_key=_target_cache_key(c, scan_cad_path, real_scan_paths, cache_key_tag),
        use_cache=c.use_preprocess_cache)   # zmanjšan target (normale OCENJENE - sken jih nima vnaprej)

    print("Phase 4: Global registration (yaw sweep)...")
    coarse_result = yaw_sweep_registration(
        source_down, target_down, c.voxel_size, c.table,
        up_axis=c.up_axis,
        yaw_step_deg=c.yaw_step_deg,
        distance_threshold_factor=c.coarse_distance_factor,
        refine_iterations=c.yaw_sweep_refine_iterations,
        tilt_normal_top_band_fraction=c.tilt_normal_top_band_fraction,
        height_percentile=c.height_percentile,
        n_jobs=c.yaw_sweep_n_jobs)
    print(f"  Yaw sweep fitness: {coarse_result.fitness:.4f}, "
          f"inlier_rmse: {coarse_result.inlier_rmse:.4f}")
    if not c.use_real_scan:   # pri simulaciji poznamo referenčno transformacijo, lahko preverimo napako
        _report_against_reference("  Yaw sweep", coarse_result.transformation, reference_transform, reference_point)
    if visualize:
        draw_registration_result(source, target, coarse_result.transformation, "Yaw Sweep Result")

    skipped_icp = coarse_result.fitness < c.min_coarse_fitness_for_icp
    if skipped_icp:
        # Point-to-plane ICP linearizira okoli začetnega približka - s tako
        # slabim (deset+ stopinj napake) ne le ne konvergira, ampak lahko
        # divergira v nesmisel (metrska/kilometrska translacija). Pravi
        # problem je fitness grobega koraka, ne ICP.
        print(f"  SKIPPING ICP: coarse fitness {coarse_result.fitness:.4f} is below "
              f"--min_coarse_fitness_for_icp={c.min_coarse_fitness_for_icp}. ICP can only "
              f"refine a seed that's already roughly correct - feeding it this one is "
              f"likely to diverge, not improve it.")
        icp_result = coarse_result
    else:
        # refine_registration izbere najboljšega med grobim približkom in
        # vsakim ICP korakom, zato je icp_result vsaj tako dober kot coarse_result.
        icp_result = refine_registration(source, target, coarse_result.transformation, c.voxel_size,
                                         source_camera_location=source_camera_location,
                                         icp_distance_factor=c.icp_distance_factor,
                                         icp_voxel_scales=c.icp_voxel_scales,
                                         icp_max_iterations=c.icp_max_iterations,
                                         icp_iterations_schedule=c.icp_iterations_schedule,
                                         robust_kernel_k_factor=c.robust_kernel_k_factor)
        print(f"  ICP fitness: {icp_result.fitness:.4f}, "
              f"inlier_rmse: {icp_result.inlier_rmse:.4f}")
        if c.use_mesh_polish:
            print("Phase 4.5: Point-to-mesh polish (exact CAD surface, heteroscedastic-weighted)...")
            icp_result = polish_on_cad_mesh(
                source, target, cad_mesh, icp_result, c.noise, c.voxel_size, c.icp_voxel_scales,
                c.icp_distance_factor, c.robust_kernel_k_factor, max_iterations=c.mesh_polish_max_iterations,
                distance_factor=c.mesh_polish_distance_factor, trim_fraction=c.mesh_polish_trim_fraction)

    if visualize:
        draw_registration_result(source, target, icp_result.transformation, "ICP Result")
    print("Final transformation matrix (pose of the CAD file's own coordinate system in the camera frame - "
          "this is what the robot needs, IF the STL was exported in the agreed Coordinate System1):")
    print(view.pose_to_cad(icp_result.transformation))
    sealing_point_in_camera = icp_result.transformation[:3, :3] @ reference_point + icp_result.transformation[:3, 3]
    print(f"  Sealing-region centroid (CAD {np.round(view.point_to_cad(reference_point), 2)}) lands at "
          f"{np.round(sealing_point_in_camera, 2)} in the camera frame")
    if not c.use_real_scan:
        print(f"Reference transformation matrix (used to generate the simulated scan):")
        print(view.pose_to_cad(reference_transform))
        _report_against_reference("ICP", icp_result.transformation, reference_transform, reference_point)

    evaluate = o3d.pipelines.registration.evaluate_registration
    print("Evaluation before ICP:", evaluate(source, target, c.voxel_size * 1.5, coarse_result.transformation))
    print("Evaluation after ICP:", evaluate(source, target, c.voxel_size * 1.5, icp_result.transformation))

    decision = _check_and_decide(c, source, source_down, target, target_down, icp_result, skipped_icp,
                                 source_cache_key, cad_mesh)
    if view.matrix is None:
        return RegistrationOutcome(source, target, coarse_result, icp_result, decision,
                                   reference_transform, reference_point)
    return RegistrationOutcome(
        copy.deepcopy(source).transform(view.matrix.T), target,
        registration_result(view.pose_to_cad(coarse_result.transformation), coarse_result.fitness, coarse_result.inlier_rmse),
        registration_result(view.pose_to_cad(icp_result.transformation), icp_result.fitness, icp_result.inlier_rmse),
        decision, view.pose_to_cad(reference_transform), view.point_to_cad(reference_point))


def main() -> None:
    np.random.seed(42)   # nastavimo seme za ponovljive naključne rezultate
    base_path = Path(__file__).resolve().parent   # mapa, kjer leži ta skripta
    args = parse_args()

    if args.self_test:   # samo hitri samostojni testi, brez CAD/registracije
        run_self_tests()
        return

    outcome = run_registration(config_from_args(args), cad_path=base_path / args.cad,
                               scan_paths=[base_path / p for p in [args.scan, *args.extra_scans]],
                               scan_cad_path=base_path / args.scan_cad if args.scan_cad else None)
    if not outcome.decision["accepted"]:
        # Neničeln exit-code je signal, ki ga robotov klicoči proces lahko
        # preveri, ne da bi moral razčlenjevati stdout.
        sys.exit(1)


if __name__ == "__main__":   # skripta se poganja neposredno (ne uvožena kot modul)
    main()   # zaženemo glavni tok programa
