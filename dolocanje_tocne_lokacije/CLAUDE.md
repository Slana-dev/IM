# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Context

IM is a university project to build a liquid-seal dispenser ("nanašalec tesnila"). An Epson robot carries a 3D camera on its arm. The robot moves to a generic pose and captures a point cloud of a housing. The default CAD file is `d_housing_detailed.STL`, exported with its origin on the pin. The older `eHDS S Housing + CC s pottingom, fine.STL` is still in the folder. The code in this folder estimates the exact pose of the housing so the robot can apply the third seal bead on the plastic top.

The code registers the CAD model to the camera scan with Open3D. The CAD model is the source and appears yellow in the viewer. The scan is the target and appears red. The project has no real camera hardware yet, so the code simulates the target scan by ray casting the CAD mesh and adding a noise model of a real sensor.

The repo root (`../`) has three other folders:
- `kalibracija/` holds the TCP and hand-eye calibration, an Epson RC+ project and a notebook.
- `komunikacija/` holds `send_localcoord.py`, a Python TCP server on port 12345. The Epson controller connects to it as a client.
- `main/main.py` runs the robot loop with TCP, OpenCV and the hand-eye result JSON.

`../NAVODILA_ZAGON.md` has the setup and run instructions in Slovenian.

All code comments, docstrings and much of the console output are in Slovenian. Write new comments in Slovenian and match the existing style. The code has an inline `# ...` comment on almost every line and long docstrings that explain why ("ZAKAJ").

## Environment and commands

Use the conda env `open3d_env` with Python 3.11. open3d 0.19 has no wheels for Python 3.13. Install `open3d==0.19.0 numpy scipy pandas openpyxl joblib "opencv-python<5"`, see also `../requirements.txt`. `joblib` is optional. With it, the yaw sweep runs its candidates in parallel.

**Run every script from this folder.** The scripts look for the STL, `.preprocess_cache/` and the output `.xlsx` files next to themselves.

```bash
python kontrolna_plosca.py            # main entry point: edit CAMERA and CONFIG at the top of the file, then run
python main_trial.py --help           # lists every pipeline parameter
python main_trial.py --self_test      # runs the quick self-tests (run_self_tests), no CAD file and no registration needed
python main_trial.py --use_real_scan --scan scan.ply   # registers a real scan instead of a simulated one
python camera_benchmark.py            # Monte Carlo comparison of the cameras, writes camera_benchmark.xlsx
python camera_translation_benchmark.py
```

- The project has no test framework. To test, run `--self_test`, which calls the `_test_*` functions in `self_tests.py`. Then run scenarios and compare the estimated pose with the ground-truth pose.
- A run is not bit-for-bit repeatable even with the same seed. The RANSAC tilt fit (`segment_plane`) and the Poisson sampling use Open3D's own random generator, which `np.random.seed` does not control, and Open3D's multithreaded ICP sums vary in the last digits. For an exact before/after comparison, call `o3d.utility.random.seed(...)` and set `OMP_NUM_THREADS=1` before importing Open3D.
- `main_trial.py` exits with code 1 when it rejects the registration. The robot process that calls the script checks this exit code.
- The benchmarks resume an interrupted run. They read the existing `.xlsx` and skip every `(camera, tx, ty, yaw, seed)` row it already has. Delete the file to start over. Close the file in Excel before a benchmark runs, or the write fails.
- The "Main" configuration in `.vscode/launch.json` and the example in `NAVODILA_ZAGON.md` pass `--iss_target_min_keypoints` and `--iss_target_max_keypoints`. `main_trial.py` no longer defines these flags, because the yaw sweep replaced ISS and RANSAC. argparse rejects them.

## Architecture

`main_trial.py` holds only the orchestration and the CLI entry point. `run_registration(config, cad_path, scan_paths)` returns a `RegistrationOutcome` and runs these phases, each implemented in its own module:

1. `point_clouds.py`: `load_cad_model` samples the CAD file into a point cloud and caches the samples as `*.sampled_<N>pts.ply` next to the STL. `crop_top_region` keeps the top `top_fraction` of the part along `up_axis`, which is the seal region the camera sees. `check_cad_frame` prints a warning if the STL looks like it is in meters or if its origin is far from the part.
2. The target scan. `scan_simulation.simulate_camera_scan` ray casts from the camera at the origin, looking along `-up_axis`, and applies `NoiseModel`: depth noise that grows with distance, distance bias, a global planarity warp, outliers, flying pixels and quantization. An optional table mesh comes from `workcell.Table`. With `--use_real_scan`, `point_clouds.load_real_scans` loads and merges real `.ply` files instead. `remove_table_background` and the grazing-incidence (flying pixel) filters then clean the target. The simulated scan also returns an `OrganizedScan` pixel grid, which allows an O(n) filter. `OrganizedScan.keep_only` keeps that grid in sync with the target after each filter.
3. `point_clouds.preprocess_point_cloud` downsamples both clouds to voxels and computes oriented normals. It caches the results in `.preprocess_cache/`.
4. `coarse_registration.yaw_sweep_registration` does the coarse alignment. It estimates tilt from the surface normal and the XY offset from phase correlation of height maps. The Z offset is the median of per-cell height differences between the two height maps after that XY shift (`estimate_z_translation_height_maps`), not a comparison of the highest points, because walls the camera barely sees biased that comparison by 3–4 mm on the D housing. It then tries a yaw angle every `yaw_step_deg` and scores each one with a short ICP. `fine_registration.refine_registration` follows with point-to-plane ICP at several voxel sizes (`icp_voxel_scales`) and a Tukey kernel. It scores every ICP stage with the same distance threshold so the stages are comparable.
5. `fine_registration.polish_on_cad_mesh` polishes the pose against the exact CAD mesh through a RaycastingScene and weights points by the noise model. It keeps the polished pose only if `trimmed_rms` does not get worse. The code calls this step Phase 4.5.
6. `verification.py` holds the safety checks, Phases 5 to 7. They check that the asymmetric features line up, which catches symmetric and flipped poses. Only asymmetric points the camera can see under the final pose count: they must face the camera within `flying_pixel_filter_max_incidence_deg` and not be hidden by the part (`visible_from_camera`). They also check target-to-source coverage and compute the pose covariance. `decide_registration_outcome` then returns a PASS/FAIL dict. The result is PASS only if every threshold passes. The docstrings call the thresholds placeholders. Nobody has calibrated them yet.

**Configuration.** `registration_config.RegistrationConfig` is one frozen dataclass with every pipeline parameter. Its field names are exactly the argparse `dest` names and the `CONFIG` keys in `kontrolna_plosca.py`. The defaults live only in the dataclass, and argparse reads them from `DEFAULTS`. `config_from_args` copies the fields by name. To add a parameter, add the field, the `add_argument` call with `default=DEFAULTS.<name>`, and, if the panel should control it, a `CONFIG` key. The self-test `_test_config_matches_cli` fails if these drift apart. `config.noise` returns the `NoiseModel` view and `config.table` the `Table` view.

**Pose conventions.** These are easy to get wrong.
- The camera is always at `(0,0,0)`, for simulated and real scans alike. `--translation_x/y/z` and `--rotation_*_deg` set the ground-truth pose of the part relative to the camera. `translation_z` is negative because it is the working distance below the camera.
- In simulation, translation and rotation are relative to `compute_native_reference_point`, the centre of the scanned seal region, and not to the STL origin. `run_registration` builds this reference transform itself from the config (`_reference_transform`, using `poses.transform_with_pivot`) and returns it in `RegistrationOutcome.reference_transform`. There is no module-level pose state. `run_registration` also measures translation error at this pivot. The benchmarks currently do not, see finding 4 in `../CODE_REVIEW.md`.
- The output transform maps the CAD file's own frame into the camera frame. The robot can use it only if the STL came out of SolidWorks in Coordinate System1, in mm, with "Do not translate STL output data to positive space" checked. The centroid of the part is not the shared frame.
- Every phase assumes the camera looks at the part's +Z side. `flip_up_direction` (default off; `--flip_up_direction` / `--no_flip_up_direction`) means the camera looks at the CAD's −Z side instead. `main_trial._ViewFrame` then rotates the CAD 180° about X internally (`poses.FLIP_UP`) and converts every printed and returned pose back to the STL's frame. No other module knows about the flip.
- `top_fraction` must cover enough asymmetric features to fix the yaw. On a near-symmetric visible face (e.g. the simplified D housing at 0.35) the pipeline returned poses 90–180° off, and the asymmetric check still ACCEPTED some of them. Validate a new part or setting with several yaws before trusting it.

**`kontrolna_plosca.py`** is the control panel and does not modify `main_trial.py`. `build_cli_args` loops over `CONFIG` in key order and emits `--<key> <value>`. Only flags without a value, the `--no_*` disable flags and the list/tuple/dict values get special handling. The script sets `sys.argv` to that list and calls `main_trial.main()`. `resolve_camera_params()` fills the camera-specific flags from `kamere.CAMERAS`. A `CONFIG` value that is not `None` overrides the value from the camera, and `None` means the script does not pass the flag.

**`kamere.py`** holds datasheet specs for the candidate cameras: Photoneo PhoXi S, Zivid 2+ M60 and Mech-Eye Pro S. A comment on each field names the `main_trial.py` flag it feeds.

**The benchmarks** share `benchmark_common.py`: `BASE_CONFIG`, `run_one` (one registration turned into one result row, with errors recorded as rows instead of stopping the sweep), Excel resume and `error_stats`. Each benchmark builds its per-run config with `dataclasses.replace(...)` from the camera parameters and the pose, and calls `run_registration(..., visualize=False)`. They loop over positions inside each camera's field of view, yaw angles offset by half a yaw step, and random seeds.

**Caching.** The cache keys include the parameters but not the code. If you change what `preprocess_point_cloud`, or anything it calls, returns for the same inputs, increase `PREPROCESS_CACHE_VERSION` in `point_clouds.py`. Otherwise the pipeline keeps reading old cached results. Pass `--no_cad_cache` or `--no_preprocess_cache` to skip the caches. The order of the `NoiseModel` fields is part of the simulated target's cache key, so don't reorder them.

All lengths are in millimeters. The noise reference distances are in meters, as their flag names say.
