"""Vsi parametri registracijskega cevovoda (RegistrationConfig) in ukazna vrstica main_trial.py.

Imena polj RegistrationConfig so NATANKO imena argparse `dest` spodaj (in
ključi CONFIG v kontrolna_plosca.py) - config_from_args zato prenese
vrednosti brez ročnega preslikovanja, privzete vrednosti pa so zapisane
SAMO v dataclassu (argparse jih bere od tam). self_tests.py preverja, da
se obe strani ujemata.
"""
import argparse
from dataclasses import dataclass, fields
from typing import Optional

import numpy as np

from scan_simulation import NoiseModel
from workcell import Table


@dataclass(frozen=True)
class RegistrationConfig:
    # --- Vhod in scenarij ---
    use_real_scan: bool = False                 # True = pravi sken iz datoteke namesto simulacije
    disable_occlusion: bool = False             # simulacija brez ray castinga (gosto vzorčen cel del)
    voxel_size: float = 1.0
    sample_points: int = 200000                 # Poisson-disk točke iz CELOTNE CAD mreže

    # --- Referenčna (ground-truth) poza dela glede na kamero v izhodišču ---
    translation_x: float = 0.0
    translation_y: float = 0.0
    translation_z: float = -500.0
    rotation_x_deg: float = 0.0
    rotation_y_deg: float = 0.0
    rotation_z_deg: float = 0.0

    # --- Šum senzorja (glej scan_simulation.NoiseModel - ista imena in privzetki) ---
    depth_noise_at_1m: float = NoiseModel.depth_noise_at_1m
    noise_distance_power: float = NoiseModel.noise_distance_power
    noise_reference_distance_m: float = NoiseModel.noise_reference_distance_m
    max_incidence_deg: float = NoiseModel.max_incidence_deg
    distance_bias_permille: float = NoiseModel.distance_bias_permille
    global_planarity_mm: float = NoiseModel.global_planarity_mm
    global_planarity_spatial_scale_px: Optional[float] = NoiseModel.global_planarity_spatial_scale_px
    outlier_probability: float = NoiseModel.outlier_probability
    outlier_std_multiplier: float = NoiseModel.outlier_std_multiplier
    flying_pixel_depth_jump_mm: float = NoiseModel.flying_pixel_depth_jump_mm
    flying_pixel_probability: float = NoiseModel.flying_pixel_probability
    quantization_step_at_1m: float = NoiseModel.quantization_step_at_1m
    noise_spatial_correlation_px: float = NoiseModel.noise_spatial_correlation_px

    # --- Simulirana kamera ---
    camera_fov_deg: float = 40.0
    camera_width_px: int = 640
    camera_height_px: int = 480

    # --- Miza (ozadje v simulaciji + fizični prior pri grobi registraciji) ---
    table_size_x: float = 500.0
    table_size_y: float = 500.0
    table_center_x: float = 0.0
    table_center_y: float = 0.0
    add_table_background: bool = False
    remove_table_background: Optional[bool] = None   # None = samodejno: vklopljeno natanko za use_real_scan
    table_removal_margin_mm: float = 10.0

    # --- Obrezovanje / ocena naklona / filter letečih pikslov ---
    top_fraction: float = 1.0                   # cel del - pri manjšem deležu na d_housing_detailed ostanejo razdrobljeni ali simetrični kosi in yaw ni zanesljiv
    up_axis: int = 2
    flip_up_direction: bool = False             # True = kamera gleda CAD-ovo -Z stran (glej --flip_up_direction)
    crop_edge_margin_factor: float = 2.0
    tilt_normal_top_band_fraction: float = 0.5
    height_percentile: float = 99.5
    filter_grazing_incidence: bool = True
    flying_pixel_filter_max_incidence_deg: float = 80.0
    grazing_incidence_normal_radius_factor: float = 2.0

    # --- Groba (yaw sweep) + ICP registracija ---
    coarse_distance_factor: float = 3.0
    yaw_step_deg: float = 6.0
    yaw_sweep_refine_iterations: int = 5
    yaw_sweep_n_jobs: int = -1
    icp_distance_factor: float = 2.0
    icp_voxel_scales: tuple[float, ...] = (4.0, 2.0, 1.0, 0.5)
    icp_max_iterations: int = 100
    icp_iterations_schedule: Optional[dict[float, int]] = None
    robust_kernel_k_factor: float = 1.0
    min_coarse_fitness_for_icp: float = 0.1

    # --- Point-to-mesh polish (Phase 4.5) ---
    use_mesh_polish: bool = True
    mesh_polish_max_iterations: int = 30
    mesh_polish_distance_factor: float = 1.0
    mesh_polish_trim_fraction: float = 0.8

    # --- Varnostni pregledi in PASS/FAIL pragovi (glej verification.decide_registration_outcome) ---
    asymmetric_check_top_fraction: float = 0.15
    asymmetric_check_residual_factor: float = 3.0
    target_coverage_distance_factor: float = 2.0
    min_accept_target_coverage: float = 0.85
    min_accept_fitness: float = 0.1
    max_accept_inlier_rmse: Optional[float] = None   # None = 1.5 * voxel_size
    min_accept_asymmetric_fraction: float = 0.5

    # --- Predpomnilniki ---
    use_cad_cache: bool = True
    use_preprocess_cache: bool = True

    @property
    def noise(self) -> NoiseModel:
        return NoiseModel(**{f.name: getattr(self, f.name) for f in fields(NoiseModel)})

    @property
    def table(self) -> Table:
        return Table(self.table_size_x, self.table_size_y, np.array([self.table_center_x, self.table_center_y]))


DEFAULTS = RegistrationConfig()


def _float_tuple(text: str) -> tuple[float, ...]:
    """'4.0,2.0,1.0,0.5' -> (4.0, 2.0, 1.0, 0.5)"""
    return tuple(float(s) for s in text.split(","))


def _iterations_schedule(text: str) -> dict[float, int]:
    """'4.0:30,2.0:50' -> {4.0: 30, 2.0: 50}"""
    schedule = {}
    for entry in text.split(","):
        scale_str, iterations_str = entry.split(":")
        schedule[float(scale_str)] = int(iterations_str)
    return schedule


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CAD-to-scan point cloud registration")
    parser.add_argument("--self_test", action="store_true",
                        help="Run the fast, standalone self-tests for the new camera-calibration "
                             "noise parameters (noise_reference_distance_m, distance_bias_permille, "
                             "global_planarity_mm, build_reference_transform - see "
                             "run_self_tests()) and exit, instead of running registration. Does "
                             "not need --cad/--scan or any other argument")
    parser.add_argument("--cad", default="d_housing_detailed.STL",
                        help="Path to the CAD mesh file used as the REFERENCE (source) - the "
                             "pose is reported in this file's coordinate system")
    parser.add_argument("--scan_cad", default=None,
                        help="CAD mesh the SIMULATED scan is generated from (default: same as "
                             "--cad). Use it to test a reference that differs from the real part, "
                             "e.g. a simplified --cad against a detailed --scan_cad. Both files must "
                             "be exported in the same coordinate system. Ignored with --use_real_scan")
    parser.add_argument("--scan", default="scan.ply",
                        help="Path to the real scan point cloud file")
    parser.add_argument("--extra_scans", nargs="*", default=[],
                        help="Paths to additional scan files of the SAME static placement to "
                             "merge with --scan for temporal noise averaging (see "
                             "load_and_merge_real_scans) - if the robot/sensor can capture "
                             "several (e.g. 3-5) consecutive scans of the part sitting still on "
                             "the table before it needs a final pose, merging them and letting "
                             "Phase 3's voxel downsampling average each voxel's points cuts "
                             "uncorrelated sensor noise by roughly sqrt(N) for free. No inter-scan "
                             "registration is done or needed - the sensor is fixed and the part "
                             "doesn't move between captures, so all scans already share one "
                             "coordinate frame")
    parser.add_argument("--use_real_scan", action="store_true",
                        help="Use a real scan instead of a simulated scan")
    parser.add_argument("--voxel_size", type=float, default=DEFAULTS.voxel_size,
                        help="Voxel size for downsampling")
    parser.add_argument("--sample_points", type=int, default=DEFAULTS.sample_points,
                        help="Number of Poisson-disk sample points to generate from the "
                             "*whole* CAD mesh (before crop_top_region keeps ~35%% and "
                             "simulated occlusion keeps a further fraction). Default 200000 "
                             "puts the realistically-visible region at ~13k points, i.e. "
                             "sub-mm point spacing over a ~150mm part - the density a real "
                             "industrial structured-light/laser scanner delivers. Denser "
                             "raw input also averages out more sensor noise per voxel for "
                             "free. First run Poisson-samples for a few minutes, then the CAD "
                             "cache makes it instant; raise this further if your real scanner "
                             "is denser (memory scales linearly, ~a few MB per 100k in cache)")
    parser.add_argument("--depth_noise_at_1m", type=float, default=DEFAULTS.depth_noise_at_1m,
                        help="Depth-noise std (mm) at --noise_reference_distance_m distance and "
                             "0deg incidence (straight-on) - despite the flag's name, this is "
                             "measured at --noise_reference_distance_m, not necessarily 1m (see "
                             "that flag). Scales up with both distance and incidence angle - see "
                             "--noise_distance_power/--max_incidence_deg - matching a real "
                             "triangulation sensor's behavior instead of a single flat std "
                             "everywhere")
    parser.add_argument("--noise_distance_power", type=float, default=DEFAULTS.noise_distance_power,
                        help="Exponent for how depth-noise std grows with distance "
                             "(distance_m/--noise_reference_distance_m ** this, calibrated to "
                             "equal --depth_noise_at_1m exactly at that reference distance). Also "
                             "used for --quantization_step_at_1m's distance scaling")
    parser.add_argument("--noise_reference_distance_m", type=float, default=DEFAULTS.noise_reference_distance_m,
                        help="Distance (m) at which --depth_noise_at_1m was actually measured/"
                             "calibrated. Real "
                             "triangulation sensor datasheets (see kamere.py) report noise at "
                             "their own working distance (e.g. 0.442m for Photoneo PhoXi S), "
                             "never at 1m - extrapolating a temporal-noise measurement taken over "
                             "a narrow working range (e.g. 384-520mm) out to 1m with the default "
                             "--noise_distance_power=2.0 can badly misestimate the true noise, "
                             "since the real trend over that narrow range is often much flatter "
                             "than quadratic. Set this to the sensor's own reference distance and "
                             "--depth_noise_at_1m to the noise measured there instead")
    parser.add_argument("--max_incidence_deg", type=float, default=DEFAULTS.max_incidence_deg,
                        help="Incidence angle (degrees off straight-on) beyond which the "
                             "1/cos(incidence) noise-growth factor is clamped, so it can't blow "
                             "up right at grazing incidence where cos -> 0")
    parser.add_argument("--distance_bias_permille", type=float, default=DEFAULTS.distance_bias_permille,
                        help="Systematic (not per-pixel-random) multiplicative distance "
                             "calibration error, as a fraction (permille) of the measured "
                             "distance, constant for the whole simulated capture - see "
                             "apply_distance_bias_mm() and kamere.py's 'Relative distance "
                             "accuracy'/'Dimension Trueness Error' spec entries, which report "
                             "this as an overall (mostly systematic, not random) accuracy figure "
                             "that the random noise terms cannot represent. 0 (default) "
                             "disables it. Deterministic per run (like --translation_x/y/z), not "
                             "randomly redrawn, so you can sweep it to test tolerance to a known "
                             "systematic bias (e.g. --distance_bias_permille 1.25 for Photoneo's "
                             "spec)")
    parser.add_argument("--global_planarity_mm", type=float, default=DEFAULTS.global_planarity_mm,
                        help="Amplitude (peak-to-valley range, mm) of a smooth, low-frequency "
                             "systematic distortion field added across the whole simulated depth "
                             "image, constant for the whole capture - see "
                             "generate_global_planarity_bias_mm() and kamere.py's 'Global "
                             "planarity'/'Global Planarity Trueness Error' spec entries. The "
                             "existing noise model (heteroscedastic gaussian, outliers, flying "
                             "pixels, quantization) is entirely random and zero-mean per pixel; "
                             "none of it represents this kind of smooth systematic bow/warp "
                             "across the field of view. 0 (default) disables it")
    parser.add_argument("--global_planarity_spatial_scale_px", type=float, default=DEFAULTS.global_planarity_spatial_scale_px,
                        help="Gaussian-blur sigma (pixels) controlling how 'low-frequency' (smooth "
                             "over how large an area) the --global_planarity_mm distortion field "
                             "is. Default (unset): min(--camera_width_px, --camera_height_px) / 4")
    parser.add_argument("--outlier_probability", type=float, default=DEFAULTS.outlier_probability,
                        help="Fraction of pixels that get a gross depth error (multipath, weak "
                             "SNR, bad stereo/structured-light match) instead of ordinary sensor "
                             "noise - a genuinely different error mechanism, not just a bit more "
                             "of the same jitter")
    parser.add_argument("--outlier_std_multiplier", type=float, default=DEFAULTS.outlier_std_multiplier,
                        help="How much wider an outlier pixel's error distribution is than that "
                             "same pixel's ordinary noise std - see --outlier_probability")
    parser.add_argument("--flying_pixel_depth_jump_mm", type=float, default=DEFAULTS.flying_pixel_depth_jump_mm,
                        help="Depth difference (mm) between neighboring pixels that counts as a "
                             "discontinuity (part edge, pin edge, seam) eligible to become a "
                             "'flying pixel'. Set relative to your real edge features' scale "
                             "(e.g. 2-5mm for a pin edge)")
    parser.add_argument("--flying_pixel_probability", type=float, default=DEFAULTS.flying_pixel_probability,
                        help="Probability that a pixel sitting on a depth discontinuity (see "
                             "--flying_pixel_depth_jump_mm) becomes a flying pixel - an "
                             "interpolated, physically-nonexistent depth between the near and "
                             "far surface, the characteristic 'floating point cloud at edges' "
                             "artifact real depth sensors produce and plain Gaussian noise never does")
    parser.add_argument("--quantization_step_at_1m", type=float, default=DEFAULTS.quantization_step_at_1m,
                        help="Depth quantization step (mm) at 1m distance - the sensor's ADC/"
                             "disparity resolution reports depth in discrete steps, not "
                             "continuously. Scales with distance the same way "
                             "--depth_noise_at_1m does (via --noise_distance_power)")
    parser.add_argument("--noise_spatial_correlation_px", type=float, default=DEFAULTS.noise_spatial_correlation_px,
                        help="Gaussian-blur sigma (pixels) used to spatially correlate depth "
                             "noise between neighboring pixels, matching a real sensor's "
                             "matching/correlation window (which mixes several neighboring "
                             "pixels rather than measuring each independently) - roughly the "
                             "window's own size in pixels")
    parser.add_argument("--disable_occlusion", action="store_true",
                        help="Skip ray casting and densely sample the whole CAD mesh instead, "
                             "so the simulated scan keeps full coverage - to check whether "
                             "partial visibility (occlusion + field of view) is what's causing "
                             "the recovered transform to diverge from the reference")
    parser.add_argument("--camera_fov_deg", type=float, default=DEFAULTS.camera_fov_deg,
                        help="Horizontal field of view (degrees) of the simulated depth camera "
                             "used for ray casting the target scan. Narrower = less of the "
                             "table around the part is captured, denser coverage of the part "
                             "itself for a given --camera_width_px/--camera_height_px")
    parser.add_argument("--camera_width_px", type=int, default=DEFAULTS.camera_width_px,
                        help="Simulated camera image width in pixels (= number of rays cast "
                             "across the field of view). Higher = denser target point cloud, "
                             "slower ray casting")
    parser.add_argument("--camera_height_px", type=int, default=DEFAULTS.camera_height_px,
                        help="Simulated camera image height in pixels, see --camera_width_px")
    parser.add_argument("--translation_x", type=float, default=DEFAULTS.translation_x,
                        help="X translation (mm) of the ground-truth reference transform "
                             "(see build_reference_transform) - the part's "
                             "position relative to the camera, which is ALWAYS fixed at the world "
                             "origin (0,0,0) for both real and simulated runs. Default 0 (part "
                             "centered under the camera in X) - set nonzero to empirically test "
                             "how far off-center (within the camera's field of view) the part can "
                             "sit and still be registered correctly")
    parser.add_argument("--translation_y", type=float, default=DEFAULTS.translation_y,
                        help="Same as --translation_x but for the Y axis")
    parser.add_argument("--translation_z", type=float, default=DEFAULTS.translation_z,
                        help="Z translation (mm) of the ground-truth reference transform - the "
                             "part's distance from the camera (at the origin), so this must be "
                             "NEGATIVE (camera looks down at the part below it). Default -500.0 is "
                             "a generic placeholder; set it to MINUS the actual camera's sweet-spot/"
                             "focal working distance instead (e.g. -442 for Photoneo PhoXi S, -600 "
                             "for Zivid 2+ M60, -500 for Mech-Eye PRO S at its 500mm focal "
                             "configuration - see kamere.py). kontrolna_plosca.py's "
                             "resolve_camera_params() does this automatically for the selected camera")
    parser.add_argument("--rotation_x_deg", type=float, default=DEFAULTS.rotation_x_deg,
                        help="Rotation (degrees) of the ground-truth reference transform about "
                             "the world X axis (roll), applied before Y/Z rotation - see "
                             "build_reference_transform. Default 0 (part lying flat, no tilt) - "
                             "set nonzero to empirically test how much roll/pitch tilt (part not "
                             "lying perfectly flat on the table) the pipeline can tolerate. "
                             "yaw_sweep_registration estimates and corrects tilt from target's "
                             "surface normal, tested to hold up to ~25-27 deg combined tilt before "
                             "breaking down around 30")
    parser.add_argument("--rotation_y_deg", type=float, default=DEFAULTS.rotation_y_deg,
                        help="Same as --rotation_x_deg but about the world Y axis (pitch)")
    parser.add_argument("--rotation_z_deg", type=float, default=DEFAULTS.rotation_z_deg,
                        help="Same as --rotation_x_deg but about the world Z axis (yaw) - applied "
                             "last, after X and Y rotation. Note this is the GROUND-TRUTH yaw the "
                             "pipeline must recover, unrelated to --yaw_step_deg's search "
                             "resolution below")
    parser.add_argument("--top_fraction", type=float, default=DEFAULTS.top_fraction,
                        help="Fraction of the CAD model's height (from the top, along "
                             "--up_axis) to keep for matching, since the robot's camera "
                             "only scans the top sealing region, not the whole housing")
    parser.add_argument("--up_axis", type=int, default=DEFAULTS.up_axis, choices=[0, 1, 2],
                        help="Axis index (0=X, 1=Y, 2=Z) along which 'top' is measured")
    parser.add_argument("--flip_up_direction", dest="flip_up_direction", action="store_true",
                        default=DEFAULTS.flip_up_direction,
                        help="The camera looks at the CAD's -Z side instead of +Z (default OFF). Use it if, in the agreed "
                             "coordinate system, the face that is scanned and "
                             "sealed is at -Z. The pipeline then rotates the CAD 180 deg about X "
                             "internally and converts every printed/returned pose back, so the "
                             "result is always in the STL's own coordinate system")
    parser.add_argument("--no_flip_up_direction", dest="flip_up_direction", action="store_false",
                        help="The camera looks at the CAD's +Z side (default)")
    parser.add_argument("--crop_edge_margin_factor", type=float, default=DEFAULTS.crop_edge_margin_factor,
                        help="Exclude a band this many multiples of --voxel_size wide from the "
                             "CAD source, right next to --top_fraction's cut boundary (see "
                             "crop_top_region's edge_margin). That boundary is artificial (the "
                             "part/mesh really continues past it, this crop just cuts it off) - "
                             "the real scan may or may not see material just past it depending "
                             "on occlusion, while the CAD side always ends exactly there, which "
                             "otherwise pulls ICP systematically. Only applied to the CAD side "
                             "being matched against, never to the mesh the simulated scan itself "
                             "is generated from. 0 disables it")
    parser.add_argument("--coarse_distance_factor", type=float, default=DEFAULTS.coarse_distance_factor,
                        help="Yaw sweep's max_correspondence_distance as a multiple of "
                             "voxel_size. Raise this (e.g. to 4-5) for noisier scans so "
                             "correspondences aren't rejected just for not landing exactly on "
                             "top of each other")
    parser.add_argument("--yaw_step_deg", type=float, default=DEFAULTS.yaw_step_deg,
                        help="Degrees between candidate yaw angles in the sweep (halves the "
                             "candidate count, and with it the sweep's runtime, at each doubling). "
                             "Smaller = finer coarse search but slower; the multi-scale ICP stage "
                             "refines the winning angle to full precision afterward regardless. "
                             "Default raised from 3.0 to 6.0 after empirically comparing both "
                             "across several off-center/rotated test poses (same camera noise "
                             "seed): final rotation/translation error after ICP was "
                             "indistinguishable between 3.0 and 6.0 in every case that converged "
                             "at all (a pose that failed to converge failed almost identically "
                             "at both step sizes, so coarsening isn't what caused it) - the ICP "
                             "refinement stage does in fact recover the coarser yaw discretization "
                             "as expected. Lower this back to 3.0 (or finer) if you see the coarse "
                             "sweep repeatedly landing on the wrong side of a near-symmetric "
                             "feature for your specific part")
    parser.add_argument("--yaw_sweep_refine_iterations", type=int, default=DEFAULTS.yaw_sweep_refine_iterations,
                        help="ICP iterations used to locally polish each yaw candidate before "
                             "scoring it. Without this, scoring by raw inlier count lets a "
                             "wrong yaw that happens to align repeated bulk geometry (e.g. "
                             "mounting bosses) outscore the true yaw, since it ignores whether "
                             "the few asymmetric distinguishing features (locator pins, "
                             "connector) actually line up. A short local refinement fixes "
                             "that: the true yaw converges to a tight fit, a merely "
                             "bulk-aligned wrong one generally doesn't. Keep this low - it's "
                             "not the final precision pass, just enough to discriminate "
                             "between candidates")
    parser.add_argument("--yaw_sweep_n_jobs", type=int, default=DEFAULTS.yaw_sweep_n_jobs,
                        help="Parallel workers for evaluating yaw sweep candidates (they're "
                             "fully independent - see yaw_sweep_registration/_evaluate_yaw_candidate). "
                             "-1 (default) uses all CPU cores via joblib, 1 forces serial "
                             "evaluation (falls back to serial automatically either way if "
                             "joblib isn't installed). Empirically gives the same winning "
                             "candidate as serial (verified over repeated runs). Measured speedup "
                             "is workload-dependent: a single one-off run pays a one-time "
                             "process-pool startup cost that can outweigh the savings from just "
                             "120 cheap candidates (measured roughly break-even, occasionally "
                             "slightly slower than serial); repeated run_registration calls in "
                             "the same process (e.g. camera_benchmark.py) reuse the same pool "
                             "and saw ~1.7x speedup on the yaw sweep step specifically")
    parser.add_argument("--icp_distance_factor", type=float, default=DEFAULTS.icp_distance_factor,
                        help="ICP max_correspondence_distance as a multiple of voxel_size. "
                             "Must be large enough to exceed the coarse seed's printed "
                             "inlier_rmse, or ICP finds 0 correspondences and silently leaves "
                             "the coarse estimate unrefined (visible as ICP fitness=0.0000)")
    parser.add_argument("--icp_voxel_scales", type=_float_tuple, default=DEFAULTS.icp_voxel_scales,
                        help="Comma-separated multiples of voxel_size for coarse-to-fine ICP "
                             "passes (run largest to smallest regardless of list order). Each "
                             "stage re-downsamples source/target and recomputes normals at "
                             "that scale, so source's density gets closer to target's naturally "
                             "sparser (occlusion-limited) density at coarse scales, and each "
                             "stage hands the next a better-converged starting pose. Not going "
                             "below 0.5 on purpose: 0.25mm would drop under a real scanner's "
                             "point spacing, leaving voxels with 0-1 points and degenerate "
                             "per-stage normals")
    parser.add_argument("--icp_max_iterations", type=int, default=DEFAULTS.icp_max_iterations,
                        help="Max iterations per ICP stage (Open3D's own default is 30). "
                             "Raise this for a more fully-converged final transformation, "
                             "especially at the finer voxel scales - each stage runs "
                             "independently up to this budget or until Open3D's relative "
                             "fitness/rmse convergence tolerance is hit, whichever comes first")
    parser.add_argument("--icp_iterations_schedule", type=_iterations_schedule, default=None,
                        help="Optional per-stage override of --icp_max_iterations, as "
                             "'scale:iterations,scale:iterations,...' where each scale must "
                             "match one of the values in --icp_voxel_scales (e.g. "
                             "'4.0:30,2.0:50,1.0:75,0.5:100' to spend fewer iterations on the "
                             "coarsest stage - the yaw sweep has usually already gotten close "
                             "at that scale - and more on the finest one). A scale not listed "
                             "here falls back to --icp_max_iterations. Unset (default): every "
                             "stage uses --icp_max_iterations")
    parser.add_argument("--robust_kernel_k_factor", type=float, default=DEFAULTS.robust_kernel_k_factor,
                        help="Scale (as a multiple of each ICP stage's own voxel size) of the "
                             "Tukey robust kernel (TukeyLoss) used in the point-to-plane ICP "
                             "estimation in refine_registration. Without a robust kernel, "
                             "outliers/flying pixels that fall within max_correspondence_distance "
                             "(so they DO become correspondences) pull the solution with full, "
                             "unbounded weight, same as a clean point; Tukey downweights a "
                             "residual once it exceeds k and zeroes it out entirely above 2k, "
                             "effectively excluding outliers instead of trusting them equally. "
                             "Lower this (e.g. 0.5) for noisier scans to reject outliers harder, "
                             "raise it if a clean scan's legitimate near-edge points are being "
                             "excessively downweighted")
    parser.add_argument("--no_mesh_polish", dest="use_mesh_polish", action="store_false", default=True,
                        help="Disable the point-to-mesh polish stage (Phase 4.5) that runs after "
                             "refine_registration, on by default. Poisson-disk sampling the CAD "
                             "(--sample_points) puts a hard discretization floor (~voxel spacing) "
                             "under how tight any point-cloud ICP can converge, no matter how many "
                             "iterations - this stage instead aligns the scan directly against the "
                             "exact CAD mesh triangles (RaycastingScene.compute_closest_points), "
                             "weighting each scan point by its own heteroscedastic measurement "
                             "sigma (see kamere.py's noise model) instead of trusting every "
                             "correspondence equally. Strictly non-regressive: only accepted if it "
                             "doesn't increase the trimmed target->CAD RMS distance versus the "
                             "pre-polish pose (see trimmed_rms_target_to_source), so at worst it's "
                             "a no-op, never a regression")
    parser.add_argument("--mesh_polish_max_iterations", type=int, default=DEFAULTS.mesh_polish_max_iterations,
                        help="Max iterations for the point-to-mesh polish stage, see --no_mesh_polish")
    parser.add_argument("--mesh_polish_distance_factor", type=float, default=DEFAULTS.mesh_polish_distance_factor,
                        help="Point-to-mesh polish's max correspondence distance, as a multiple of "
                             "(voxel_size * finest --icp_voxel_scales entry * --icp_distance_factor) "
                             "- i.e. relative to refine_registration's own finest stage, since this "
                             "runs right after it and should only be excluding genuinely "
                             "unmatched points (occlusion edges, background residue), not "
                             "legitimate correspondences")
    parser.add_argument("--mesh_polish_trim_fraction", type=float, default=DEFAULTS.mesh_polish_trim_fraction,
                        help="Fraction of target points kept (closest first) when comparing the "
                             "pre/post mesh-polish trimmed RMS to decide whether to accept the "
                             "polish result - see trimmed_rms_target_to_source")
    parser.add_argument("--min_coarse_fitness_for_icp", type=float, default=DEFAULTS.min_coarse_fitness_for_icp,
                        help="Skip ICP entirely if the coarse (yaw sweep) fitness is below "
                             "this. ICP's linearized solver can diverge to nonsense (meter/"
                             "km-scale translations) when seeded with a badly wrong pose, so "
                             "it's safer to skip it than trust its output on a bad seed")
    parser.add_argument("--table_size_x", type=float, default=DEFAULTS.table_size_x,
                        help="Width (mm, X axis) of the known table/work-envelope area the "
                             "part can be placed in. yaw_sweep_registration rejects any yaw "
                             "candidate whose implied placement would put the part's center "
                             "outside this area - a physical prior that keeps an implausible "
                             "placement from winning purely on ICP fitness. Set to the real "
                             "robot's actual work envelope")
    parser.add_argument("--table_size_y", type=float, default=DEFAULTS.table_size_y,
                        help="Depth (mm, Y axis) of the known table/work-envelope area, see "
                             "--table_size_x")
    parser.add_argument("--table_center_x", type=float, default=DEFAULTS.table_center_x,
                        help="X coordinate (mm, world/CAD frame) of the table area's center - "
                             "together with --table_size_x/--table_size_y defines the "
                             "rectangular region a candidate placement must fall within")
    parser.add_argument("--table_center_y", type=float, default=DEFAULTS.table_center_y,
                        help="Y coordinate (mm, world/CAD frame) of the table area's center, "
                             "see --table_center_x")
    parser.add_argument("--add_table_background", action="store_true",
                        help="Add a flat table plane (sized/positioned by --table_size_x/y and "
                             "--table_center_x/y - the same table used for the physical prior) "
                             "into the ray casting scene, so rays that miss the part hit the "
                             "table instead of nothing - the simulated scan then includes "
                             "background/table points around the part, to see how the pipeline "
                             "handles that instead of a clean part-only scan")
    parser.add_argument("--remove_table_background", dest="remove_table_background",
                        action="store_true", default=None,
                        help="Remove likely table/background points from target before "
                             "registration, using the part's own known height (from source/CAD) "
                             "instead of 'largest plane' detection (e.g. RANSAC segment_plane) - "
                             "the latter can mistakenly remove the part itself if it occupies a "
                             "large flat area of the field of view. Assumes the lowest (or "
                             "highest, see --up_axis/flip) observed points in target are the "
                             "table, since the part cannot extend below the table it rests on. "
                             "Default: on automatically for --use_real_scan (a real scan almost "
                             "always includes some table/background, and leaving it in can make "
                             "the tilt-normal estimate lock onto the table instead of the part's "
                             "top surface), off for simulated scans (no table in the cloud unless "
                             "--add_table_background is also set) - pass --no_remove_table_background "
                             "to force it off even for a real scan")
    parser.add_argument("--no_remove_table_background", dest="remove_table_background",
                        action="store_false",
                        help="Force table/background removal off, overriding the --use_real_scan "
                             "default described under --remove_table_background")
    parser.add_argument("--table_removal_margin_mm", type=float, default=DEFAULTS.table_removal_margin_mm,
                        help="Margin (mm) added on both ends of the kept height band in "
                             "--remove_table_background: excludes points within this distance "
                             "of the detected table level, and points more than the part's own "
                             "height plus this margin above it")
    parser.add_argument("--no_filter_grazing_incidence", dest="filter_grazing_incidence",
                        action="store_false", default=True,
                        help="Disable the grazing-incidence (flying pixel) point filter, on by "
                             "default for both real and simulated scans. Removes points whose "
                             "local normal is nearly perpendicular to the camera view direction "
                             "(incidence angle above --flying_pixel_filter_max_incidence_deg) - "
                             "at such a steep angle the sensor's matching window straddles two "
                             "depth levels at once and commonly reports an interpolated, "
                             "physically-nonexistent depth (see simulate_camera_scan's own "
                             "flying-pixel noise model). Unlike the real-scan statistical "
                             "outlier removal, this targets the actual physical cause "
                             "(incidence angle) instead of the symptom (isolated distance), which "
                             "matters because flying pixels form dense clusters along edges, not "
                             "isolated outliers")
    parser.add_argument("--flying_pixel_filter_max_incidence_deg", type=float, default=DEFAULTS.flying_pixel_filter_max_incidence_deg,
                        help="Incidence angle (degrees off straight-on) above which a point is "
                             "removed by --filter_grazing_incidence")
    parser.add_argument("--grazing_incidence_normal_radius_factor", type=float, default=DEFAULTS.grazing_incidence_normal_radius_factor,
                        help="Normal-estimation radius for --filter_grazing_incidence, as a "
                             "multiple of --voxel_size, applied to the RAW (not yet downsampled) "
                             "scan")
    parser.add_argument("--asymmetric_check_top_fraction", type=float, default=DEFAULTS.asymmetric_check_top_fraction,
                        help="Fraction of source's points (by self-rotation distance, see "
                             "identify_asymmetric_points) treated as the part's distinctive/"
                             "asymmetric features for the post-ICP safety check. Lower = only "
                             "the most distinctive points count, stricter/noisier check; higher "
                             "= includes more borderline bulk-geometry points, looser check")
    parser.add_argument("--asymmetric_check_residual_factor", type=float, default=DEFAULTS.asymmetric_check_residual_factor,
                        help="A distinctive/asymmetric point counts as 'aligned' after ICP if "
                             "its distance to target is within this multiple of --voxel_size. "
                             "The safety check warns if fewer than half of them meet that bar")
    parser.add_argument("--tilt_normal_top_band_fraction", type=float, default=DEFAULTS.tilt_normal_top_band_fraction,
                        help="Fraction of target's points (by height percentile along --up_axis) "
                             "fed into the RANSAC plane fit that estimates the part's tilt "
                             "(replaces a plain PCA-over-everything normal estimate, which has no "
                             "robust weighting and can lock onto the table instead of the part's "
                             "top surface if the table isn't removed first - see "
                             "--remove_table_background)")
    parser.add_argument("--height_percentile", type=float, default=DEFAULTS.height_percentile,
                        help="Percentile (0-100) used instead of a plain max() when comparing "
                             "source/target height along --up_axis to estimate the Z-translation "
                             "in the yaw sweep. A single flying-pixel/outlier (up to "
                             "--outlier_std_multiplier times the ordinary noise std in this "
                             "pipeline's own sensor model) can move a plain max() by several mm; "
                             "a high percentile treats the 'top' as a band of the topmost points "
                             "instead of one single point")
    parser.add_argument("--target_coverage_distance_factor", type=float, default=DEFAULTS.target_coverage_distance_factor,
                        help="Distance threshold for the target->source coverage check, as a "
                             "multiple of --voxel_size: a scanned point counts as 'on the part' "
                             "if it's within this distance of the CAD surface under the final "
                             "pose. This is the primary occlusion-robust overlap check used for "
                             "ACCEPT/REJECT - see --min_accept_target_coverage and "
                             "evaluate_target_coverage()'s docstring")
    parser.add_argument("--min_accept_target_coverage", type=float, default=DEFAULTS.min_accept_target_coverage,
                        help="Minimum fraction of scanned (target) points that must lie on the "
                             "CAD surface (within --target_coverage_distance_factor * voxel_size) "
                             "for the PASS/FAIL accept decision to be ACCEPT. This is the PRIMARY "
                             "overlap gate: unlike --min_accept_fitness (source->target, "
                             "depressed by occlusion even for a correct pose), target->source "
                             "coverage measures whether the points actually measured lie on the "
                             "part, regardless of how much of the part the scan happened to see. "
                             "PLACEHOLDER default, not empirically calibrated - see "
                             "decide_registration_outcome()'s docstring for how to calibrate it "
                             "(Monte Carlo over known-good poses using this file's own "
                             "--translation_x/y/z/--rotation_x/y/z_deg simulation harness) before "
                             "trusting it in production")
    parser.add_argument("--min_accept_fitness", type=float, default=DEFAULTS.min_accept_fitness,
                        help="Minimum final ICP fitness (Open3D's source->target overlap ratio) "
                             "required for ACCEPT. Intentionally just a LOW sanity floor, not the "
                             "primary overlap gate - this fitness is depressed by occlusion alone "
                             "(target is necessarily a partial view of the part) even under a "
                             "perfectly correct pose, so a high fixed threshold here would reject "
                             "good poses on heavily-occluded scans. See "
                             "--min_accept_target_coverage for the actual (occlusion-robust) "
                             "overlap gate")
    parser.add_argument("--max_accept_inlier_rmse", type=float, default=DEFAULTS.max_accept_inlier_rmse,
                        help="Maximum final ICP inlier_rmse allowed for ACCEPT. Default: "
                             "1.5 * --voxel_size, the same common threshold icp_result's own "
                             "fitness/inlier_rmse were evaluated under in refine_registration")
    parser.add_argument("--min_accept_asymmetric_fraction", type=float, default=DEFAULTS.min_accept_asymmetric_fraction,
                        help="Minimum fraction of the part's distinctive/asymmetric features "
                             "(see --asymmetric_check_top_fraction/--asymmetric_check_residual_factor) "
                             "that must align within threshold for ACCEPT - guards against a "
                             "registration that converged to a symmetric look-alike orientation "
                             "(e.g. a repeated mounting boss) despite good overall fitness")
    parser.add_argument("--no_cad_cache", dest="use_cad_cache", action="store_false", default=True,
                        help="Re-run Poisson-disk sampling on the STL every time instead of "
                             "caching the sampled cloud to a .ply next to it. The cache is "
                             "keyed by --sample_points and auto-invalidated if the STL file "
                             "is newer, so this is only needed to force a fresh (differently "
                             "randomized) sample without changing --sample_points")
    parser.add_argument("--no_preprocess_cache", dest="use_preprocess_cache", action="store_false",
                        default=True,
                        help="Re-run voxel downsampling/normals every time instead of caching "
                             "the result under .preprocess_cache/. The cache key covers every "
                             "parameter that affects the output (voxel_size, crop, noise, "
                             "etc.), so this is rarely needed - mainly useful if you suspect a "
                             "stale cache entry")
    return parser


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def config_from_args(args: argparse.Namespace) -> RegistrationConfig:
    """Vse, kar ni RegistrationConfig polje (--cad, --scan, --extra_scans,
    --self_test), ostane v args - glej main_trial.main()."""
    return RegistrationConfig(**{f.name: getattr(args, f.name) for f in fields(RegistrationConfig)})
