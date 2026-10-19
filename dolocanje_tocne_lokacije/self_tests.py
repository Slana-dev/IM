"""Samostojni (hitri, brez CAD/registracije) testi: `python main_trial.py --self_test`.

Testirajo SAMO, da je vsak parameter šumovnega modela in sestavljanja poze
matematično pravilno implementiran (npr. da noise_reference_distance_m res
premakne referenčno točko šumovne formule, ne da je le neuporabljen
argument), in da se RegistrationConfig ujema z ukazno vrstico - ne
testirajo celotnega registracijskega cevovoda, za kar služijo empirični
zagoni z --translation_x/y/z/--rotation_x/y/z_deg.
"""
from dataclasses import fields

import numpy as np
import open3d as o3d

from fine_registration import point_to_mesh_icp
from point_clouds import top_camera_location
from poses import build_reference_transform
from registration_config import DEFAULTS, RegistrationConfig, build_parser
from scan_simulation import NoiseModel, apply_distance_bias_mm, generate_global_planarity_bias_mm


def _test_noise_reference_distance() -> None:
    """Pri distance_m == noise_reference_distance_m in incidence_cos=1.0
    (naravnost gledanje, brez 1/cos faktorja) mora NoiseModel.depth_std_mm
    vrniti natanko depth_noise_at_1m - to velja PO DEFINICIJI referenčne
    razdalje (glej docstring), ne glede na to, kateri noise_reference_distance_m
    izberemo."""
    for reference_m in (1.0, 0.442, 0.6, 0.05):
        noise = NoiseModel(depth_noise_at_1m=0.05, noise_distance_power=2.0, max_incidence_deg=75.0,
                           noise_reference_distance_m=reference_m)
        std = noise.depth_std_mm(distance_m=np.array([reference_m]), incidence_cos=np.array([1.0]))
        assert abs(std[0] - 0.05) < 1e-9, (
            f"Pri distance_m==noise_reference_distance_m ({reference_m}) mora "
            f"std==depth_noise_at_1m==0.05, dobil {std[0]}")
    # Pri DVAKRAT večji razdalji od reference in noise_distance_power=2.0
    # mora biti std natanko 4x večji (2^2) - preverja, da je razmerje
    # distance_m/noise_reference_distance_m (ne golo distance_m) tisto, kar
    # gre v np.power.
    noise = NoiseModel(depth_noise_at_1m=0.05, noise_distance_power=2.0, max_incidence_deg=75.0,
                       noise_reference_distance_m=0.442)
    std_at_2x = noise.depth_std_mm(distance_m=np.array([0.884]), incidence_cos=np.array([1.0]))
    assert abs(std_at_2x[0] - 0.05 * 4.0) < 1e-9, (
        f"Pri 2x referenčni razdalji in power=2.0 mora biti std 4x večji od "
        f"depth_noise_at_1m (0.20), dobil {std_at_2x[0]}")
    print("  OK: NoiseModel.depth_std_mm pravilno uporabi noise_reference_distance_m")


def _test_distance_bias_permille() -> None:
    """apply_distance_bias_mm mora premakniti razdaljo za natanko
    bias_permille/1000 delež (v obe smeri - pozitiven/negativen bias) in
    ne spremeniti ničesar pri bias_permille=0 (privzeto, nazaj-kompatibilno)."""
    distance_mm = np.array([1000.0, 500.0, 250.0])
    unbiased = apply_distance_bias_mm(distance_mm, 0.0)
    assert np.allclose(unbiased, distance_mm), "bias_permille=0 ne sme spremeniti razdalje"
    biased = apply_distance_bias_mm(distance_mm, 1.25)
    expected = distance_mm * 1.00125
    assert np.allclose(biased, expected), f"Pričakovano {expected}, dobil {biased}"
    biased_neg = apply_distance_bias_mm(distance_mm, -2.5)
    expected_neg = distance_mm * 0.9975
    assert np.allclose(biased_neg, expected_neg), f"Pričakovano {expected_neg}, dobil {biased_neg}"
    print("  OK: apply_distance_bias_mm pravilno skalira razdaljo v obe smeri")


def _test_global_planarity_bias() -> None:
    """generate_global_planarity_bias_mm mora vrniti polje z razponom
    (max-min) natanko amplitude_mm, ničelno polje pri amplitude_mm=0
    (privzeto, nazaj-kompatibilno), in dovolj GLADKO (nizkofrekvenčno -
    sosednji piksli podobni, ne iid šum kot preostali šumovni členi)."""
    shape = (64, 64)
    zero_field = generate_global_planarity_bias_mm(shape, amplitude_mm=0.0, low_freq_sigma_px=10.0)
    assert np.allclose(zero_field, 0.0), "amplitude_mm=0 mora vrniti ničelno polje"

    field = generate_global_planarity_bias_mm(shape, amplitude_mm=0.25, low_freq_sigma_px=10.0)
    actual_range = field.max() - field.min()
    assert abs(actual_range - 0.25) < 1e-9, f"Razpon mora biti natanko 0.25mm, dobil {actual_range}"

    # Gladkost: povprečna razlika med sosednjima pikseloma mora biti veliko
    # manjša od skupnega razpona polja - iid šum (kot raw_noise pred
    # glajenjem) te lastnosti nima, gladko nizkofrekvenčno polje pa jo ima
    # po konstrukciji.
    neighbor_diff = np.abs(np.diff(field, axis=0)).mean()
    assert neighbor_diff < actual_range * 0.1, (
        f"Polje ni dovolj gladko/nizkofrekvenčno (povprečna sosednja razlika "
        f"{neighbor_diff:.4f} ni veliko manjša od razpona {actual_range:.4f})")
    print("  OK: generate_global_planarity_bias_mm vrne pravilno skalirano, gladko polje")


def _test_top_camera_location_offset_distance() -> None:
    """top_camera_location z eksplicitnim offset_distance mora postaviti
    kamero natanko na to razdaljo od roba oblaka (ne od privzetega,
    velikosti-oblaka-odvisnega offset_factor) - ta funkcija se v novi
    (kamera-vedno-v-izhodišču) zasnovi uporablja samo še za
    source_camera_location (virtualna "pogled od zgoraj" lokacija za
    orientacijo CAD-ovih normal, glej run_registration), ne več za pravo/
    simulirano kamero (ta je zdaj vedno np.zeros(3)), a mora ostati
    pravilno parametrizirana."""
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
        np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 5.0]])))
    for offset in (442.0, 1000.0, 250.0):
        loc = top_camera_location(pcd, up_axis=2, offset_distance=offset)
        expected_z = pcd.get_max_bound()[2] + offset
        assert abs(loc[2] - expected_z) < 1e-9, (
            f"offset_distance={offset}: pričakovano Z={expected_z}, dobil {loc[2]}")
    print("  OK: top_camera_location pravilno uporabi eksplicitno podan offset_distance")


def _test_build_reference_transform() -> None:
    """build_reference_transform mora pri privzetih (ničelnih) rotacijah
    vrniti identično rotacijo z natanko podano translacijo, in pravilno
    zavrteti znane enotske vektorje pri 90-stopinjskih rotacijah okoli
    posamezne osi - preverja, da je konstrukcija matrike (Rz@Ry@Rx +
    translacija v zadnjem stolpcu) dejansko pravilna, ne le da funkcija
    vrne matriko pravilne oblike."""
    identity_case = build_reference_transform(10.0, -20.0, -442.0, 0.0, 0.0, 0.0)
    assert np.allclose(identity_case[:3, :3], np.eye(3)), (
        f"Pri ničelnih rotacijah mora biti rotacijski del identiteta, dobil {identity_case[:3, :3]}")
    assert np.allclose(identity_case[:3, 3], [10.0, -20.0, -442.0]), (
        f"Translacija mora biti natanko podana, dobil {identity_case[:3, 3]}")
    assert np.allclose(identity_case[3, :], [0.0, 0.0, 0.0, 1.0]), "Zadnja vrstica mora biti [0,0,0,1]"

    # 90 stopinj okoli Z (yaw) mora preslikati enotski X vektor na Y.
    yaw_90 = build_reference_transform(0.0, 0.0, 0.0, 0.0, 0.0, 90.0)
    assert np.allclose(yaw_90[:3, :3] @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-9), (
        f"90 stopinj okoli Z mora preslikati X-os na Y-os, dobil {yaw_90[:3, :3] @ [1.0, 0.0, 0.0]}")

    # 90 stopinj okoli X (roll) mora preslikati enotski Y vektor na Z.
    roll_90 = build_reference_transform(0.0, 0.0, 0.0, 90.0, 0.0, 0.0)
    assert np.allclose(roll_90[:3, :3] @ [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], atol=1e-9), (
        f"90 stopinj okoli X mora preslikati Y-os na Z-os, dobil {roll_90[:3, :3] @ [0.0, 1.0, 0.0]}")

    # 90 stopinj okoli Y (pitch) mora preslikati enotski Z vektor na X.
    pitch_90 = build_reference_transform(0.0, 0.0, 0.0, 0.0, 90.0, 0.0)
    assert np.allclose(pitch_90[:3, :3] @ [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], atol=1e-9), (
        f"90 stopinj okoli Y mora preslikati Z-os na X-os, dobil {pitch_90[:3, :3] @ [0.0, 0.0, 1.0]}")

    print("  OK: build_reference_transform pravilno sestavi rotacijo in translacijo")


def _test_point_to_mesh_icp_plane() -> None:
    """point_to_mesh_icp na neskončni ravni ploskvi (z=0 v CAD okvirju,
    analitičen closest_point_fn - brez potrebe po pravi RaycastingScene) -
    x/y-translacija in yaw so na taki ploskvi FIZIKALNO neopazljivi (noben
    del njihove Jacobijeve komponente ne vpliva na residual), zato H
    (normalno-enačbena matrika) izide TOČNO singularna. Preveri, da funkcija
    na tem ne obstane/vrže izjeme (glej docstring - np.linalg.lstsq namesto
    np.linalg.solve) in da vseeno v celoti zapre edini opazljiv residual
    (višina nad ravnino)."""
    def plane_closest_point_fn(p_cad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        proj = p_cad.copy()
        proj[:, 2] = 0.0
        normal = np.tile(np.array([0.0, 0.0, 1.0]), (p_cad.shape[0], 1))
        return proj, normal

    rng = np.random.RandomState(0)
    n_pts = 300
    cad_pts = np.zeros((n_pts, 3))
    cad_pts[:, 0] = rng.uniform(-50, 50, n_pts)
    cad_pts[:, 1] = rng.uniform(-50, 50, n_pts)

    t_true = build_reference_transform(120.0, -30.0, 400.0, 4.0, -6.0, 25.0)
    scan_pts_world = (t_true[:3, :3] @ cad_pts.T).T + t_true[:3, 3]
    t_init = build_reference_transform(115.0, -35.0, 395.0, 2.0, -3.0, 20.0)

    result = point_to_mesh_icp(scan_pts_world, plane_closest_point_fn, t_init, max_iterations=50)
    assert result["iterations"] > 0, (
        "should not stall on a singular (flat-plane) Hessian - see np.linalg.lstsq in docstring")
    world_to_cad = np.linalg.inv(result["transformation"])
    p_cad_final = (world_to_cad[:3, :3] @ scan_pts_world.T).T + world_to_cad[:3, 3]
    plane_residual_rms = float(np.sqrt(np.mean(p_cad_final[:, 2] ** 2)))
    assert plane_residual_rms < 1e-6, (
        f"should fully close the only observable residual (height above the plane), "
        f"got {plane_residual_rms}")
    print("  OK: point_to_mesh_icp closes the observable residual on a singular-Hessian flat plane")


def _test_point_to_mesh_icp_weighted() -> None:
    """point_to_mesh_icp na delno vidni krogli (analitičen closest_point_fn)
    z UMETNO heteroscedastičnim šumom (15% točk 100x širši std) - preveri,
    da point_sigma_mm uteženje dejansko izboljša natančnost proti
    neuteženemu klicu na ISTIH zašumljenih podatkih (ne le da se koda ne
    sesuje)."""
    radius = 200.0

    def sphere_closest_point_fn(p_cad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        norm = np.linalg.norm(p_cad, axis=1, keepdims=True)
        normal = p_cad / np.clip(norm, 1e-9, None)
        return normal * radius, normal

    rng = np.random.RandomState(1)
    n_pts = 400
    phi = rng.uniform(0, 2 * np.pi, n_pts)
    costheta = rng.uniform(0.3, 0.9, n_pts)   # delen pogled, ne cela krogla
    theta = np.arccos(costheta)
    cad_pts = np.stack([
        radius * np.sin(theta) * np.cos(phi),
        radius * np.sin(theta) * np.sin(phi),
        radius * np.cos(theta),
    ], axis=1)

    t_true = build_reference_transform(50.0, 10.0, -300.0, 8.0, 3.0, -12.0)
    scan_pts_clean = (t_true[:3, :3] @ cad_pts.T).T + t_true[:3, 3]
    sigma = rng.choice([0.05, 5.0], size=n_pts, p=[0.85, 0.15])   # 15% "slabih" točk, 100x večja varianca
    noisy_scan = scan_pts_clean + rng.normal(size=scan_pts_clean.shape) * sigma[:, None]
    t_init = build_reference_transform(45.0, 15.0, -290.0, 5.0, 6.0, -8.0)

    result_weighted = point_to_mesh_icp(noisy_scan, sphere_closest_point_fn, t_init,
                                        max_iterations=50, robust_kernel_k=1.0, point_sigma_mm=sigma)
    result_unweighted = point_to_mesh_icp(noisy_scan, sphere_closest_point_fn, t_init, max_iterations=50)

    err_weighted = float(np.linalg.norm(result_weighted["transformation"][:3, 3] - t_true[:3, 3]))
    err_unweighted = float(np.linalg.norm(result_unweighted["transformation"][:3, 3] - t_true[:3, 3]))
    assert err_weighted < err_unweighted, (
        f"heteroscedastic point_sigma_mm weighting should beat unweighted on the same noisy "
        f"input, got weighted={err_weighted:.4f}mm >= unweighted={err_unweighted:.4f}mm")
    assert err_weighted < 1.0, f"weighted fit should land close to the true pose, got {err_weighted:.4f}mm"
    print(f"  OK: point_to_mesh_icp point_sigma_mm weighting reduces translation error "
          f"({err_weighted:.4f}mm weighted vs {err_unweighted:.4f}mm unweighted)")


def _test_config_matches_cli() -> None:
    """Vsako polje RegistrationConfig mora biti argparse dest z ISTIM
    privzetkom (config_from_args to predpostavlja), in vsak ključ CONFIG v
    kontrolna_plosca.py mora biti ime zastavice (build_cli_args ga
    pretvori v --ključ)."""
    import kontrolna_plosca

    parsed = vars(build_parser().parse_args([]))
    for field in fields(RegistrationConfig):
        assert field.name in parsed, f"RegistrationConfig.{field.name} nima ustrezne CLI zastavice"
        assert parsed[field.name] == getattr(DEFAULTS, field.name), (
            f"--{field.name}: argparse privzetek {parsed[field.name]!r} != "
            f"RegistrationConfig privzetek {getattr(DEFAULTS, field.name)!r}")
    unknown = [key for key in kontrolna_plosca.CONFIG if key not in parsed]
    assert not unknown, f"kontrolna_plosca.CONFIG ključi brez CLI zastavice: {unknown}"
    print("  OK: RegistrationConfig, argparse in kontrolna_plosca.CONFIG se ujemajo")


def run_self_tests() -> None:
    """Pognati vse hitre, samostojne teste novih parametrov za kalibracijo
    na resnične kamere (glej modulski komentar zgoraj in kamere.py)."""
    print("Running self-tests for new camera calibration parameters...")
    _test_noise_reference_distance()
    _test_distance_bias_permille()
    _test_global_planarity_bias()
    _test_top_camera_location_offset_distance()
    _test_build_reference_transform()
    _test_point_to_mesh_icp_plane()
    _test_point_to_mesh_icp_weighted()
    _test_config_matches_cli()
    print("All self-tests passed.")
