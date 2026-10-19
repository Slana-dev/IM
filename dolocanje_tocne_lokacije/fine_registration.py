"""Fina registracija: večnivojski point-to-plane ICP in končni point-to-mesh polish na točni CAD mreži."""
import copy
from typing import Optional

import numpy as np
import open3d as o3d

from point_clouds import ensure_oriented_normals, raycasting_scene
from poses import registration_result, rotation_about_axis
from scan_simulation import NoiseModel
from workcell import CAMERA_LOCATION


def trimmed_rms_target_to_source(source: o3d.geometry.PointCloud,
                                 target: o3d.geometry.PointCloud,
                                 transformation: np.ndarray,
                                 trim_fraction: float = 0.8) -> float:
    """Trimmed RMS distance (mm) from every target (scan) point to the nearest
    source (CAD) surface point under `transformation`, keeping only the
    closest `trim_fraction` of points before squaring/averaging.

    Used by refine_registration to rank ICP-stage candidates INSTEAD OF a
    fitness>=0.9*max_fitness eligibility cutoff followed by min(inlier_rmse):
    that inlier_rmse is conditioned on a fixed correspondence threshold, so a
    pose that matches fewer, tighter points can win over one that's actually
    more accurate but keeps slightly more (looser) correspondences - and the
    0.9 cutoff itself is an unjustified free parameter. This is a single,
    threshold-free scalar: same target->source direction/interpretation as
    evaluate_target_coverage (robust to occlusion, since target is always a
    partial view of source - see that function's docstring), just continuous
    instead of thresholded. Trimming (discarding the worst-matching tail,
    not a hard accept/reject line) keeps a handful of occluded/outlier
    target points from dominating the score, without needing a
    correspondence-distance parameter at all."""
    source_transformed = copy.deepcopy(source)
    source_transformed.transform(transformation)
    dist = np.asarray(target.compute_point_cloud_distance(source_transformed))
    n_keep = max(1, int(np.ceil(len(dist) * trim_fraction)))
    trimmed = np.partition(dist, n_keep - 1)[:n_keep] if n_keep < len(dist) else dist
    return float(np.sqrt(np.mean(trimmed ** 2)))


def refine_registration(source: o3d.geometry.PointCloud,
                        target: o3d.geometry.PointCloud,
                        init_transformation: np.ndarray,
                        voxel_size: float,
                        source_camera_location: np.ndarray,
                        icp_distance_factor: float = 2.0,
                        icp_voxel_scales: tuple[float, ...] = (4.0, 2.0, 1.0, 0.5),
                        common_eval_threshold_factor: float = 1.5,
                        icp_max_iterations: int = 100,
                        icp_iterations_schedule: Optional[dict[float, int]] = None,
                        robust_kernel_k_factor: float = 1.0,
                        candidate_trim_fraction: float = 0.8) -> o3d.pipelines.registration.RegistrationResult:
    """Postopen (večnivojski, coarse-to-fine) point-to-plane ICP.

    En sam korak pri polni ločljivosti se muči, kadar imata source (gost -
    celoten, neokludiran izrez CAD) in target (redek - omejen s tem, kar
    kamera dejansko vidi) zelo različno dejansko gostoto točk, tudi pri
    enakem nominalnem voxel_size: KNN normale target-a se povprečijo čez
    veliko širšo fizično okolico kot pri source-u, kar pristransko vpliva
    na normal-projicirani residual, ki ga point-to-plane minimizira, to pa
    lahko povzroči, da en sam fin korak konvergira v slabši lokalni
    optimum kot grobi začetni približek.

    Izvajanje več korakov od grobega proti finemu pri tem pomaga - pri
    grobih merilih downsampling zniža gostoto source-a blizu naravne
    gostote target-a, zato so normale obeh oblakov primerljivo natančne -
    ni pa zagotovljeno, da je to monotono: kasnejši, finejši korak lahko
    kljub temu konča slabše od prejšnjega, če so normale pri tem merilu
    pristranske. Lastna prijavljena fitness vsakega koraka niti ni pošten
    način preverjanja, ker je izračunana pod drugačnim pragom za vsak
    korak, zato se vsak kandidat (grobi začetni približek plus rezultat
    vsakega koraka) namesto tega ovrednoti pod enim fiksnim skupnim pragom
    in izbere po najnižjem trimmed_rms (glej trimmed_rms_target_to_source).

    robust_kernel_k_factor skalira Tukeyjevo robustno jedro (TukeyLoss),
    uporabljeno v point-to-plane oceni spodaj namesto navadnih najmanjših
    kvadratov: brez njega osamelci/leteči piksli ZNOTRAJ distance_threshold
    (torej sprejeti kot korespondence) vlečejo rešitev s polno, neomejeno
    težo, enako kot vsaka čista točka. Tukey uteži vsak residual navzdol,
    ko preseže k (tukajšnji privzeti k = stage_voxel_size, isto merilo kot
    distance_threshold za ta korak), in ga popolnoma izniči nad 2k -
    učinkovito izklopi osamelce namesto da bi jih obravnaval enakovredno.

    ICP smer je OBRNJENA glede na intuitivno source->target: registration_icp
    tu dobi (target_stage, source_stage, ..., inv(current_transformation), ...)
    in rezultat se nato invertira nazaj (current_transformation =
    inv(result.transformation)) - source (CAD) ostane source v vsakem drugem
    pogledu (vrnjena transformacija, ocene kandidatov, ves preostali
    cevovod). Razlog: Open3D-jev TransformationEstimationPointToPlane bere
    normale SAMO od svojega drugega argumenta ("target" v Open3D-jevem
    klicnem podpisu). Tako se point-to-plane residual projicira na TOČNE
    CAD normale (iz sample_points_poisson_disk, interpolirane iz
    mesh.compute_vertex_normals() - glej ensure_oriented_normals-ov
    preserve_existing_normals), ne na normale, OCENJENE iz šumnih skeniranih
    točk - target_stage zato normal sploh ne potrebuje. To hkrati reši drug
    problem: v izvirni smeri Open3D za VSAKO CAD (source) točko išče
    najbližjo sken (target) točko - CAD je gost in pokriva celo obrezano
    regijo, sken pa vidi le njen del (okluzija), zato veliko CAD točk nima
    pravega para in se ujame s karkoli znotraj praga (lažne korespondence). V
    obrnjeni smeri ima vsaka poizvedovalna (sken) točka resničen CAD par -
    ista logika kot evaluate_target_coverage (target->source, robustno na
    okluzijo).

    icp_iterations_schedule (neobvezno): slovar {scale: max_iteration},
    ključi ujemajoči se z vrednostmi v icp_voxel_scales - dovoli drugačen
    iteracijski proračun po posameznem koraku (npr. manj iteracij pri grobem
    merilu, kjer je večina dela že opravljena v yaw sweep-u, več pri finem).
    Manjkajoč scale v slovarju (ali None, privzeto) pade nazaj na
    icp_max_iterations za ta korak.
    """
    common_threshold = voxel_size * common_eval_threshold_factor   # skupen prag za pošteno primerjavo vseh korakov

    # Kandidati se ocenjujejo in RAZVRŠČAJO na tej (grobo zmanjšani, deljeni
    # za CEL klic refine_registration) ločljivosti, ne na polni source/
    # target (~70k+ točk) - relativna razvrstitev med kandidati se s tem ne
    # spremeni, prihrani pa se več odvečnih KD-tree/razdaljnih izračunov.
    # Sama ICP optimizacija (spodaj, source_stage/target_stage) ostane na
    # svoji lastni, po-korakih padajoči ločljivosti.
    source_eval = source.voxel_down_sample(voxel_size)
    target_eval = target.voxel_down_sample(voxel_size)

    def evaluate_candidate(transformation: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        evaluation = o3d.pipelines.registration.evaluate_registration(
            source_eval, target_eval, common_threshold, transformation)
        trimmed_rms = trimmed_rms_target_to_source(source_eval, target_eval, transformation, candidate_trim_fraction)
        return transformation, evaluation.fitness, evaluation.inlier_rmse, trimmed_rms

    candidates = [evaluate_candidate(init_transformation)]   # seznam kandidatov, začne z začetnim (grobim) približkom
    _, fitness, rmse, trimmed_rms = candidates[0]
    print(f"  ICP baseline (coarse seed) under common threshold {common_threshold:.3f}: "
          f"fitness={fitness:.4f}, inlier_rmse={rmse:.4f}, "
          f"trimmed_rms={trimmed_rms:.4f}")

    current_transformation = init_transformation   # trenutno najboljša transformacija, izboljšuje se skozi korake
    for scale in sorted(icp_voxel_scales, reverse=True):   # gremo od najbolj grobega merila proti najfinejšemu
        stage_voxel_size = voxel_size * scale   # velikost voxla za ta korak
        distance_threshold = stage_voxel_size * icp_distance_factor   # prag ujemanja točk za ta korak
        stage_iterations = (icp_iterations_schedule.get(scale, icp_max_iterations)
                           if icp_iterations_schedule else icp_max_iterations)   # glej docstring - privzeto icp_max_iterations za vsak korak
        source_stage = source.voxel_down_sample(stage_voxel_size)   # source, zmanjšan na to gostoto
        target_stage = target.voxel_down_sample(stage_voxel_size)   # target, zmanjšan na to gostoto - brez normal, glej docstring "ICP smer je OBRNJENA"
        # source_stage ohrani CAD-ove TOČNE normale (preserve_existing_normals=True
        # - voxel_down_sample jih pravilno povpreči, glej ensure_oriented_normals) -
        # edine normale, ki jih point-to-plane spodaj dejansko prebere.
        ensure_oriented_normals(source_stage, normal_radius=stage_voxel_size * 2.0,
                                camera_location=source_camera_location, use_knn=True,
                                preserve_existing_normals=True)
        robust_kernel_k = stage_voxel_size * robust_kernel_k_factor   # skala Tukeyjevega jedra za ta korak, glej docstring
        print(f"  ICP stage voxel_size={stage_voxel_size:.3f}, distance_threshold={distance_threshold:.3f}, "
              f"robust_kernel_k={robust_kernel_k:.3f}, max_iterations={stage_iterations}, "
              f"source={len(source_stage.points)} pts, target={len(target_stage.points)} pts")
        loss = o3d.pipelines.registration.TukeyLoss(k=robust_kernel_k)   # robustno jedro - navadni osamelci/leteči piksli znotraj praga dobijo zniževano/izničeno težo namesto polne
        # Obrnjena smer (target_stage, source_stage, ...) - glej docstring.
        # current_transformation preslika source(CAD)->target(sken), torej
        # inv(current_transformation) preslika target->source, kar je init za TA klic.
        result = o3d.pipelines.registration.registration_icp(
            target_stage, source_stage, distance_threshold, np.linalg.inv(current_transformation),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=stage_iterations))   # poženemo point-to-plane ICP za ta korak
        print(f"    fitness={result.fitness:.4f}, inlier_rmse={result.inlier_rmse:.4f} (target(sken)->source(CAD) smer)")
        current_transformation = np.linalg.inv(result.transformation)   # nazaj v source(CAD)->target(sken) konvencijo za naslednji korak in klicatelja

        candidates.append(evaluate_candidate(current_transformation))   # ocenimo ta korak pod skupnim pragom
        _, fitness, rmse, trimmed_rms = candidates[-1]
        print(f"    under common threshold {common_threshold:.3f}: fitness={fitness:.4f}, "
              f"inlier_rmse={rmse:.4f}, trimmed_rms={trimmed_rms:.4f}")

    best_transformation, best_fitness, best_rmse, best_trimmed_rms = min(candidates, key=lambda c: c[3])
    print(f"  Selected transformation with fitness={best_fitness:.4f}, inlier_rmse={best_rmse:.4f}, "
          f"trimmed_rms={best_trimmed_rms:.4f} (lowest trimmed target->CAD RMS distance, "
          f"{candidate_trim_fraction:.0%} of target points, among all {len(candidates)} candidates)")
    return registration_result(best_transformation, best_fitness, best_rmse)


def raycasting_closest_point_fn(scene: o3d.t.geometry.RaycastingScene):
    """Ovije o3d.t.geometry.RaycastingScene.compute_closest_points v
    preprost callable (p_cad -> (proj, normal)) za point_to_mesh_icp - ta
    posredniški nivo dovoli, da je point_to_mesh_icp sam testiran s čisto
    numpy/analitičnimi closest_point_fn (glej self_tests.py), brez
    odvisnosti od Open3D-jevega Tensor API-ja."""
    def closest_point_fn(p_cad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        query = o3d.core.Tensor(p_cad.astype(np.float32))
        result = scene.compute_closest_points(query)
        proj = result["points"].numpy().astype(np.float64)
        normal = result["primitive_normals"].numpy().astype(np.float64)
        return proj, normal
    return closest_point_fn


def point_to_mesh_icp(scan_points: np.ndarray,
                      closest_point_fn,
                      init_transformation: np.ndarray,
                      max_iterations: int = 30,
                      max_correspondence_distance: float = np.inf,
                      robust_kernel_k: Optional[float] = None,
                      point_sigma_mm: Optional[np.ndarray] = None,
                      convergence_translation_tol_mm: float = 1e-4,
                      convergence_rotation_tol_deg: float = 1e-3) -> dict:
    """Point-to-plane ICP, kjer je "target" TOČNA CAD površina
    (closest_point_fn - glej raycasting_closest_point_fn), namesto
    Poisson-disk VZORČENEGA CAD oblaka točk. scan_points so v SVETOVNEM
    (kamera/sken) okviru, popolnoma NEtransformirane (surove sken točke) -
    ista konvencija kot povsod drugje (source=CAD v lastnem/CAD okviru,
    target=sken v svetovnem okviru, transformacija slika CAD->svet).

    Vsaka iteracija:
      1. Transformira scan_points v CAD okvir s trenutno oceno (world_to_cad
         = inv(trenutna CAD->svet transformacija)).
      2. Poizveduje najbližjo točko/normalo na CAD površini za vsako (glej
         closest_point_fn).
      3. Linearizirano point-to-plane: majhna rotacija+translacija
         [rot_x,rot_y,rot_z,t_x,t_y,t_z], ki minimizira utežen kvadrat
         residuala (p-proj).normal, uteži = point_sigma_mm (1/sigma^2, glej
         point_sigma_mm spodaj) x Tukey(robust_kernel_k) na residualu.
      4. np.linalg.lstsq (NE np.linalg.solve): H = A^T W A je za nekatere
         resnične površine TOČNO singularna (npr. neskončna ravna ploskev
         ne omejuje x/y-translacije ali yaw sploh - noben del njihove
         Jacobijeve komponente ne vpliva na residual) in za druge
         SKORAJ-singularna (npr. pretežno ravna ploskev z majhno asimetrično
         značilnostjo - ravno geometrija te robotske celice, glej
         compute_pose_covariance docstring). lstsq vrne rešitev z najmanjšo
         normo (brez popravka vzdolž neopazljivih smeri) namesto da bi
         vrgel izjemo ali eksplodiral na slabo pogojenem sistemu - preverjeno
         s sintetičnimi testi (ravna ploskev, delna krogla s heteroscedastičnim
         šumom - glej self_tests.py).

    point_sigma_mm (neobvezno): std (mm) meritvene negotovosti VSAKE
    scan_points točke (glej polish_on_cad_mesh, NoiseModel.depth_std_mm) -
    uteži normalne enačbe z 1/sigma^2. None (privzeto) = enaka teža za vse
    (samo robust_kernel_k, če podan).

    max_correspondence_distance: korespondence z |p-proj| nad tem se
    popolnoma izključijo (utež 0) - ker se ta funkcija kliče PO
    refine_registration (že blizu prave rešitve), naj bo to majhno
    (tipično manjše od zadnje ICP stopnje distance_threshold), da izloči
    sken točke brez fizičnega ujemanja na CAD (rob okluzije, ostanek
    mize/ozadja), ne pa pravih korespondenc.

    Vrne dict: transformation (4x4, CAD->svet, ista konvencija kot povsod),
    iterations (dejansko izvedene), weighted_rmse (utežen RMS point-to-plane
    residual zadnje iteracije), n_correspondences (koliko sken točk je bilo
    znotraj max_correspondence_distance v zadnji iteraciji)."""
    if scan_points.shape[0] == 0:
        raise ValueError("point_to_mesh_icp: scan_points is empty")

    world_to_cad = np.linalg.inv(init_transformation)   # trenutna ocena, svet->CAD (obratno od klicateljeve source->target konvencije - glej refine_registration "ICP smer je OBRNJENA" za isto vzorec)
    weights = (1.0 / np.maximum(point_sigma_mm, 1e-6) ** 2) if point_sigma_mm is not None else np.ones(len(scan_points))

    weighted_rmse = float("nan")
    n_used = 0
    iterations_done = 0
    for _ in range(max_iterations):
        R = world_to_cad[:3, :3]
        t = world_to_cad[:3, 3]
        p_cad = (R @ scan_points.T).T + t   # sken v CAD okvir pod trenutno oceno

        proj, normal = closest_point_fn(p_cad)
        normal = normal / np.clip(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12, None)

        residual_vec = p_cad - proj
        dist = np.linalg.norm(residual_vec, axis=1)
        inlier = (dist <= max_correspondence_distance if np.isfinite(max_correspondence_distance)
                 else np.ones(len(dist), dtype=bool))
        if inlier.sum() < 6:   # premalo korespondenc za 6-DOF rešitev - varno prekinemo, vrnemo zadnjo veljavno oceno
            break

        r = np.sum(residual_vec * normal, axis=1)   # signiran point-to-plane residual

        w = weights.copy()
        if robust_kernel_k is not None and robust_kernel_k > 0:
            # Tukey biweight - ista shema kot refine_registration-ov TukeyLoss(k=...)
            u = np.clip(r / robust_kernel_k, -1.0, 1.0)
            tukey = (1.0 - u ** 2) ** 2
            tukey = np.where(np.abs(r) < robust_kernel_k, tukey, 0.0)
            w = w * tukey
        w = w * inlier   # korespondence nad max_correspondence_distance popolnoma izključene

        cross = np.cross(p_cad, normal)   # d(residual)/d(rotation) linearizacijski člen
        A = np.hstack([cross, normal])   # n x 6: [rot_x,rot_y,rot_z, t_x,t_y,t_z]
        AtW = A.T * w   # 6 x n
        H = AtW @ A   # 6x6 normalno-enačbena matrika
        b = -AtW @ r   # 6

        delta, _, _, _ = np.linalg.lstsq(H, b, rcond=None)   # glej docstring - lstsq namesto solve, H je lahko (skoraj) singularna

        d_rot = delta[:3]
        d_t = delta[3:]
        angle = np.linalg.norm(d_rot)
        R_delta = rotation_about_axis(d_rot / angle, np.degrees(angle)) if angle > 1e-12 else np.eye(3)
        delta_transform = np.eye(4)
        delta_transform[:3, :3] = R_delta
        delta_transform[:3, 3] = d_t
        world_to_cad = delta_transform @ world_to_cad   # sestavimo inkrementalni popravek (levo množenje - popravek je v trenutnem CAD-okvirju)

        denom = max(float(np.sum(w[inlier])), 1e-12)
        weighted_rmse = float(np.sqrt(np.sum(w[inlier] * r[inlier] ** 2) / denom))
        n_used = int(inlier.sum())
        iterations_done += 1

        if angle < np.radians(convergence_rotation_tol_deg) and np.linalg.norm(d_t) < convergence_translation_tol_mm:
            break

    return {
        "transformation": np.linalg.inv(world_to_cad),   # nazaj v CAD->svet konvencijo
        "iterations": iterations_done,
        "weighted_rmse": weighted_rmse,
        "n_correspondences": n_used,
    }


def polish_on_cad_mesh(source: o3d.geometry.PointCloud,
                       target: o3d.geometry.PointCloud,
                       cad_mesh: o3d.geometry.TriangleMesh,
                       init: o3d.pipelines.registration.RegistrationResult,
                       noise: NoiseModel,
                       voxel_size: float,
                       icp_voxel_scales: tuple[float, ...],
                       icp_distance_factor: float,
                       robust_kernel_k_factor: float,
                       max_iterations: int,
                       distance_factor: float,
                       trim_fraction: float) -> o3d.pipelines.registration.RegistrationResult:
    """Končni natančnostni korak po refine_registration: sken poravna
    neposredno na TOČNO CAD mrežo (trikotnike same STL datoteke), ne na
    Poisson-disk VZORČEN CAD oblak točk. Vzorčen oblak ima pri
    sample_points=200000 v obrezani regiji razmik ~0.5mm - to je
    diskretizacijska talna meja, pod katero noben ICP na točkovnem oblaku ne
    more priti, ne glede na število iteracij ali finost voxel_size. Mreža
    sama nima te omejitve - vsaka poizvedba dobi TOČNO najbližjo točko na
    dejanski ploskvi, ne najbližji vzorčeni sosed.

    Vsako sken točko uteži z 1/sigma^2 iz ISTEGA šumovnega modela, ki ga
    simulacija uporabi za generiranje šuma (NoiseModel.depth_std_mm) - točka
    pri strmem kotu vpadanja ima fizikalno večji šum in naj prispeva
    sorazmerno manj, ne glede na to, kako blizu je slučajno pristala (Tukey
    sam po sebi to NE naredi - on uteži po velikosti RESIDUALA, ne po vnaprej
    znani meritveni negotovosti).

    Strogo NE-regresivno: vrne polish rezultat SAMO, če je njegov trimmed_rms
    (ista threshold-free metrika kot izbira kandidatov v refine_registration)
    enak ali boljši od rezultata pred polish-em - sicer vrne `init`
    nespremenjen. Korak je validiran le na sintetičnih testih, zato je ta
    varovalka namenoma konservativna: v najslabšem primeru je brez učinka."""
    closest_point_fn = raycasting_closest_point_fn(raycasting_scene(cad_mesh))
    target_points_full = np.asarray(target.points)
    pre_polish_transform = init.transformation
    pre_polish_trimmed_rms = trimmed_rms_target_to_source(source, target, pre_polish_transform, trim_fraction)

    # Enkratna korespondenčna poizvedba pri pred-polish pozi, samo za oceno
    # kota vpadanja vsake sken točke (za heteroscedastično utež) - ne
    # posodobljena skozi iteracije, ker gre za lastnost SENZORSKE geometrije
    # ob zajemu, ne nekaj, kar bi se moralo spreminjati z ICP-jevo trenutno
    # oceno poze.
    world_to_cad_init = np.linalg.inv(pre_polish_transform)
    p_cad_init = (world_to_cad_init[:3, :3] @ target_points_full.T).T + world_to_cad_init[:3, 3]
    _proj_init, normal_init_cad = closest_point_fn(p_cad_init)
    normal_init_cad = normal_init_cad / np.clip(
        np.linalg.norm(normal_init_cad, axis=1, keepdims=True), 1e-12, None)
    normal_init_world = (pre_polish_transform[:3, :3] @ normal_init_cad.T).T
    view_dir = CAMERA_LOCATION - target_points_full
    view_dir /= np.clip(np.linalg.norm(view_dir, axis=1, keepdims=True), 1e-12, None)
    incidence_cos = np.clip(np.abs(np.sum(view_dir * normal_init_world, axis=1)), 1e-3, 1.0)
    distance_m = np.linalg.norm(target_points_full - CAMERA_LOCATION, axis=1) / 1000.0
    sigma_mm = noise.depth_std_mm(distance_m, incidence_cos)

    finest_stage_voxel = voxel_size * min(icp_voxel_scales)   # ista skala kot refine_registration-ova zadnja (najfinejša) stopnja
    polish = point_to_mesh_icp(
        target_points_full, closest_point_fn, pre_polish_transform,
        max_iterations=max_iterations,
        max_correspondence_distance=finest_stage_voxel * icp_distance_factor * distance_factor,
        robust_kernel_k=finest_stage_voxel * robust_kernel_k_factor, point_sigma_mm=sigma_mm)
    post_polish_trimmed_rms = trimmed_rms_target_to_source(source, target, polish["transformation"], trim_fraction)
    print(f"  Point-to-mesh polish: {polish['iterations']} iterations, "
          f"weighted_rmse={polish['weighted_rmse']:.4f}, "
          f"n_correspondences={polish['n_correspondences']}/{len(target_points_full)}, "
          f"trimmed_rms {pre_polish_trimmed_rms:.4f} -> {post_polish_trimmed_rms:.4f}")

    if post_polish_trimmed_rms <= pre_polish_trimmed_rms:
        polish_eval = o3d.pipelines.registration.evaluate_registration(
            source, target, voxel_size * 1.5, polish["transformation"])   # isto merilo kot refine_registration-ov common_eval_threshold_factor privzetek
        polished = registration_result(polish["transformation"], polish_eval.fitness, polish_eval.inlier_rmse)
        print(f"  Accepted mesh polish (trimmed_rms improved or unchanged): "
              f"fitness={polished.fitness:.4f}, inlier_rmse={polished.inlier_rmse:.4f}")
        return polished

    print(f"  REJECTED mesh polish (trimmed_rms got worse: "
          f"{pre_polish_trimmed_rms:.4f} -> {post_polish_trimmed_rms:.4f}) - "
          f"keeping the pre-polish transformation")
    return init
