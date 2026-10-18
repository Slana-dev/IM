"""Varnostni pregledi končne poze in eksplicitna PASS/FAIL odločitev (Phase 5-7 v run_registration)."""
import copy
import hashlib
from typing import Optional

import numpy as np
import open3d as o3d

from point_clouds import PREPROCESS_CACHE_DIR, raycasting_scene
from poses import rotation_about_axis


def evaluate_target_coverage(source: o3d.geometry.PointCloud,
                             target: o3d.geometry.PointCloud,
                             transformation: np.ndarray,
                             threshold: float) -> dict:
    """PRAVA smer za PASS/FAIL: za vsako TOČKO SKENA (target) izmeri
    razdaljo do najbližje CAD površine (source) pod dano transformacijo -
    obratna smer poizvedbe od Open3D-jevega evaluate_registration/
    registration_icp fitness (icp_result.fitness), ki poizveduje iz
    source-a proti target-u in deli s len(target.points).

    Zakaj je source->target fitness slaba PASS/FAIL metrika: target je
    NUJNO delen pogled (kamera zaradi okluzije in omejenega vidnega polja
    vidi le del CAD površine), zato velik del gostega source-a nikoli ne
    dobi bližnjega target ujemanja - to zniža fitness tudi pri POPOLNOMA
    pravilni pozi, kar dela fiksen prag na fitness nezanesljiv (napačno
    zavrne dobre poze na močno okludiranih skenih). target->source (ta
    funkcija) meri nasprotno, fizikalno bolj neposredno vprašanje: ali
    vsaka dejansko izmerjena točka leži na delu? Če je poza pravilna, mora
    skoraj vsaka realna (ne-šumna/ne-ozadje) točka skena ležati blizu CAD
    površine, POPOLNOMA NEODVISNO od tega, kolikšen del te površine je
    sken sploh zajel - ta metrika je torej robustna na delno vidljivost,
    za razliko od source->target fitness.

    Implementirano z isto compute_point_cloud_distance operacijo, ki jo že
    uporablja verify_asymmetric_feature_alignment (transformiraj source v
    target-ov okvir, izmeri razdaljo od target do njega), le na CELOTNEM
    target-u namesto le na asimetrični podmnožici."""
    source_transformed = copy.deepcopy(source)   # ne spreminjamo originala
    source_transformed.transform(transformation)   # source (CAD) v target-ov (svetovni/sken) okvir
    dist = np.asarray(target.compute_point_cloud_distance(source_transformed))   # razdalja vsake target točke do najbližje CAD površine

    return {
        "n_points": len(dist),   # koliko target (sken) točk smo preverili
        "mean": float(dist.mean()),   # povprečna razdalja do CAD površine
        "median": float(np.median(dist)),   # mediana razdalje
        "fraction_within_threshold": float((dist <= threshold).mean()),   # delež target točk, ki dejansko ležijo na delu
    }


def compute_pose_covariance(source: o3d.geometry.PointCloud,
                            target: o3d.geometry.PointCloud,
                            transformation: np.ndarray,
                            threshold: float) -> Optional[dict]:
    """6x6 kovarianca poze iz ICP-jeve lastne informacijske matrike pri
    KONČNI pozi (Open3D-jev get_information_matrix_from_point_clouds - ista
    linearizacija okoli korespondenc znotraj threshold, na kateri temelji
    point-to-plane ICP sam), namesto ene same skalarne napake
    (fitness/inlier_rmse/trimmed_rms).

    ZAKAJ je to uporabno: pri ravni "sealing" ploskvi (ta cevovod) so
    Z/roll/pitch trdno določeni (vsaka izmerjena točka jih omejuje), X/Y/yaw
    pa so določeni SAMO prek redkih asimetričnih značilnosti (locator pini,
    konektor - iste točke, ki jih identify_asymmetric_points najde). Ena
    sama skupna napaka (npr. inlier_rmse) tega ne loči - dober rmse na
    veliki ravni ploskvi lahko soobstaja s praktično neomejenim X/Y/yaw, če
    je asimetrična značilnost majhna ali delno okludirana. Vrstni red DOF
    ustreza Open3D-jevi lastni konvenciji: [rot_x, rot_y, rot_z, t_x, t_y,
    t_z] (glej Open3D dokumentacijo get_information_matrix_from_point_clouds).

    Vrne None, če je informacijska matrika singularna (premalo/degenerirane
    korespondence) - to je SAMO diagnostika/dodaten izpis (glej
    run_registration "Phase 6.5"), NE nov PASS/FAIL prag: privzeti pragovi v
    decide_registration_outcome so že označeni kot nekalibrirani placeholderji
    (glej njen docstring), dodajanje še enega nekalibriranega praga tukaj bi
    samo dodalo še en neutemeljen način za tiho napačno zavrnitev/sprejem
    poze. Namenjeno ROČNI/naknadni kalibraciji: če sigma_yaw_deg dosledno
    izide velik na znano-dobrih pozah (--translation_x/y/z/--rotation_x/y/
    z_deg Monte Carlo, glej decide_registration_outcome), je to signal, da
    del/kamera postavitev ne daje dovolj asimetričnega signala za zanesljivo
    yaw oceno - uporabno vedeti PREDEN se to prevede v kalibriran prag."""
    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, threshold, transformation)
    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        return None
    diag = np.diag(cov)
    if np.any(diag < 0):   # numerično degenerirana (skoraj singularna) informacijska matrika - kovarianca ni fizikalno smiselna
        return None
    return {
        "sigma_rotation_deg": np.degrees(np.sqrt(diag[:3])).tolist(),   # [roll, pitch, yaw] priblizna std, glej opombo o DOF vrstnem redu zgoraj
        "sigma_translation_mm": np.sqrt(diag[3:]).tolist(),   # [x, y, z] priblizna std (mm)
    }


def identify_asymmetric_points(pcd: o3d.geometry.PointCloud,
                               up_axis: int,
                               center: np.ndarray,
                               test_angles_deg: tuple[float, ...] = (30, 60, 90, 120, 150, 180,
                                                                     210, 240, 270, 300, 330),
                               top_fraction: float = 0.15) -> np.ndarray:
    """Samodejno najde razlikovalne/asimetrične točke dela (locator pini,
    konektor - značilnosti, na katere se docstring yaw_sweep_registration
    že zanaša za ločevanje pravega yaw od ponavljajoče se geometrijske
    prevare), iz lastne samo-simetrije CAD modela, namesto da bi jih bilo
    treba ročno označiti.

    Za vsakega od peščice testnih zasukov okoli up_axis to izmeri razdaljo
    vsake točke do njenega najbližjega soseda v ZASUKANI kopiji istega
    oblaka. Točka na ponavljajoči se/osnovni geometriji (npr. ena od več
    skoraj identičnih montažnih izboklin) pristane blizu NEKE zasukane
    kopije same sebe - to geometrijsko pomeni "ponavljajoče se". Točka na
    resnično asimetrični značilnosti tega nikoli ne stori, pri nobenem
    testiranem kotu - ostane daleč od vsake zasukane kopije. Obdržanje
    zgornjega dela `top_fraction` po tej vedno-daleč razdalji vrne natanko
    tiste točke, ki jih napačen-a-verjeten yaw ne bi uspel poravnati, kar
    je bistvo njihovega ločenega preverjanja po registraciji.

    center naj bo lastni centroid oblaka točk, v kakršnemkoli okvirju že
    je `pcd` - to meri NARAVNO samo-simetrijo oblike, zato mora biti
    ovrednoteno pred kakršnokoli registracijsko transformacijo, ne na že
    poravnanem rezultatu.
    """
    axis = np.zeros(3)
    axis[up_axis] = 1.0   # os vrtenja za teste simetrije
    points = np.asarray(pcd.points)   # točke kot numpy tabela
    centered = points - center   # točke, premaknjene tako, da je center v izhodišču (za vrtenje okoli center-a)

    min_dist = np.full(len(points), np.inf)   # najmanjša razdalja vsake točke do kakšne zasukane kopije, začetno neskončno
    for angle in test_angles_deg:   # preizkusimo vsak testni kot posebej
        R = rotation_about_axis(axis, angle)   # rotacijska matrika za ta testni kot
        rotated_points = (R @ centered.T).T + center   # zasukana kopija oblaka točk
        rotated_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rotated_points))   # zasukana kopija kot oblak točk
        dist = np.asarray(pcd.compute_point_cloud_distance(rotated_pcd))   # razdalja vsake izvorne točke do najbližje v zasukani kopiji
        min_dist = np.minimum(min_dist, dist)   # obdržimo najmanjšo razdaljo, videno doslej, čez vse teste

    cutoff = np.quantile(min_dist, 1.0 - top_fraction)   # prag - meja za zgornji del po razdalji
    return min_dist >= cutoff   # True za točke, ki so daleč od vseh zasukanih kopij (asimetrične)


_ASYMMETRIC_MASK_CACHE: dict[str, np.ndarray] = {}   # v-pomnilniku predpomnilnik, glej identify_asymmetric_points_cached


def identify_asymmetric_points_cached(pcd: o3d.geometry.PointCloud,
                                      up_axis: int,
                                      center: np.ndarray,
                                      top_fraction: float,
                                      cache_key: Optional[str]) -> np.ndarray:
    """Ovojnica okoli identify_asymmetric_points z dvoplastnim
    predpomnilnikom: v-pomnilniku (_ASYMMETRIC_MASK_CACHE, velja za
    življenjsko dobo tega Python procesa) IN na disku
    (PREPROCESS_CACHE_DIR, velja med zagoni - glej preprocess_point_cloud,
    ista mapa). Maska je odvisna SAMO od CAD geometrije (source, PRED
    kakršnokoli registracijsko transformacijo) - ne od poze, šuma ali
    kamere - zato jo je nesmiselno na novo računati pri VSAKEM klicu
    run_registration (npr. camera_benchmark.py pokliče run_registration na
    stotine/tisoče-krat z istim CAD-om in top_fraction, le drugo pozo/šum/
    kamero vsakič), IN nesmiselno vsak PRVI klic v vsakem novem procesu
    (npr. vsak samostojen main_trial.py zagon).

    Klicatelj naj to pokliče na source_down (zmanjšan CAD oblak, že
    izračunan za yaw sweep/ICP), NE na polnem source (~70k+ točk) - maska
    služi le izbiri razlikovalnih točk za verify_asymmetric_feature_alignment,
    za kar polna ločljivost ni potrebna (glej run_registration Phase 5), in
    compute_point_cloud_distance-ova N x N primerjava (identify_asymmetric_points
    znotraj) je na source_down bistveno hitrejša za praktično enak rezultat.

    cache_key naj bo run_registration-ov lastni source_cache_key (že
    izračunan za preprocess_point_cloud, torej vključno z voxel_size, ker
    source_down od njega odvisen) + asymmetric_check_top_fraction - ta dva
    skupaj enolično določata `pcd` (source_down) in s tem tudi masko.
    Klicatelj naj poda cache_key=None, kadar cache NI varen (npr.
    use_cad_cache=False - sveže Poisson-disk vzorčenje ni deterministično
    med zagoni, glej load_cad_model docstring, zato bi predpomnjena maska
    lahko ustrezala DRUGAČNEMU dejanskemu oblaku točk kot trenutni `pcd`)."""
    if cache_key is not None and cache_key in _ASYMMETRIC_MASK_CACHE:
        return _ASYMMETRIC_MASK_CACHE[cache_key]

    disk_path = None
    if cache_key is not None:
        digest = hashlib.md5(cache_key.encode()).hexdigest()[:16]   # kratek hash celotnega cache_key
        PREPROCESS_CACHE_DIR.mkdir(exist_ok=True)
        disk_path = PREPROCESS_CACHE_DIR / f"asymmask_{digest}.npy"
        if disk_path.exists():
            mask = np.load(disk_path)
            if len(mask) == len(pcd.points):   # dolžinski preverba - dodatna varovalka poleg samega cache_key, poceni
                _ASYMMETRIC_MASK_CACHE[cache_key] = mask
                return mask

    mask = identify_asymmetric_points(pcd, up_axis, center, top_fraction=top_fraction)
    if cache_key is not None:
        _ASYMMETRIC_MASK_CACHE[cache_key] = mask
        if disk_path is not None:
            np.save(disk_path, mask)
    return mask


def visible_from_camera(points: np.ndarray,
                        normals: np.ndarray,
                        cad_mesh_in_camera: o3d.geometry.TriangleMesh,
                        camera_location: np.ndarray,
                        max_incidence_deg: float,
                        occlusion_tolerance_mm: float) -> np.ndarray:
    """Maska točk (v koordinatah kamere), ki bi jih kamera dejansko
    izmerila: površina je obrnjena proti kameri pod kotom vpadanja največ
    max_incidence_deg (isti prag kot filter letečih pikslov - strmejših točk
    v skenu itak ni) IN pot od kamere do točke ne seka nobenega drugega dela
    CAD mreže (samo-okluzija, npr. pokrov nad stranskim konektorjem).
    Okluzija: žarek kamera->točka ima smer (točka - kamera), zato je
    zadetek pri t < 1 nekaj PRED točko; occlusion_tolerance_mm dopušča, da
    žarek zadene točkino lastno ploskev tik pred njo."""
    view = points - camera_location
    distance = np.linalg.norm(view, axis=1)
    incidence_cos = np.abs(np.sum(-view / np.clip(distance[:, None], 1e-12, None) * normals, axis=1))
    facing = incidence_cos >= np.cos(np.radians(max_incidence_deg))

    scene = raycasting_scene(cad_mesh_in_camera)
    rays = np.hstack([np.tile(camera_location, (len(points), 1)), view]).astype(np.float32)
    t_hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    unoccluded = t_hit * distance >= distance - occlusion_tolerance_mm
    return facing & unoccluded


def verify_asymmetric_feature_alignment(source: o3d.geometry.PointCloud,
                                        target: o3d.geometry.PointCloud,
                                        transformation: np.ndarray,
                                        asymmetric_mask: np.ndarray,
                                        residual_threshold: float,
                                        cad_mesh: o3d.geometry.TriangleMesh,
                                        camera_location: np.ndarray,
                                        max_incidence_deg: float) -> dict:
    """Varovalka, ki teče PO končni registraciji: ali se lastna
    razlikovalna geometrija dela (identify_asymmetric_points) dejansko
    ujema s target pod končno transformacijo, ne le osnovna oblika na
    splošno?

    Skupna fitness/inlier_rmse lahko izgledata v redu, medtem ko se
    napačen-a-verjeten yaw (prevara s ponavljajočo se izboklino, glej
    docstring yaw_sweep_registration) kljub temu prevleče skozi - osnovna
    geometrija, ki prevladuje v teh agregatnih metrikah, se po konstrukciji
    ujema v obeh primerih. Preverjanje residuala specifično na asimetrični
    podmnožici je veliko bolj ciljana varovalka: te točke pristanejo blizu
    target le, če je orientacija dejansko pravilna.

    Preverjamo SAMO asimetrične točke, ki jih kamera pod to pozo sploh lahko
    vidi (visible_from_camera). Pri višjih delih (npr. eHDS D housing) so
    najbolj razlikovalne točke na navpičnih stenah ali pod previsi - sken
    tam nima točk, zato bi bile "daleč" tudi pri popolnoma pravilni pozi.
    Vidnost se računa pod OCENJENO pozo: pri napačnem (npr. zasukanem) yaw
    so vidne druge asimetrične točke, ki tam, kamor padejo, nimajo
    ujemajoče geometrije v skenu - napačna poza je zato še vedno zavrnjena.
    Če ni vidna nobena asimetrična točka, je delež 0 (poze ni mogoče
    potrditi)."""
    rotation = transformation[:3, :3]
    source_pts = np.asarray(source.points)[asymmetric_mask] @ rotation.T + transformation[:3, 3]   # asimetrične točke v koordinatah kamere
    source_normals = np.asarray(source.normals)[asymmetric_mask] @ rotation.T
    cad_mesh_in_camera = copy.deepcopy(cad_mesh).transform(transformation)
    visible = visible_from_camera(source_pts, source_normals, cad_mesh_in_camera, camera_location,
                                  max_incidence_deg, occlusion_tolerance_mm=residual_threshold)

    asym_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_pts[visible]))
    dist = np.asarray(asym_pcd.compute_point_cloud_distance(target))   # razdalja vsake vidne asimetrične točke do najbližje točke v target
    if len(dist) == 0:
        dist = np.array([np.inf])
    return {
        "n_points": int(asymmetric_mask.sum()),   # koliko asimetričnih točk ima del
        "n_visible": int(visible.sum()),   # koliko jih kamera pod to pozo vidi - samo te preverjamo
        "mean": float(dist.mean()),   # povprečna razdalja
        "median": float(np.median(dist)),   # mediana razdalje
        "max": float(dist.max()),   # največja (najslabša) razdalja
        "fraction_within_threshold": float((dist <= residual_threshold).mean()),   # delež vidnih točk znotraj sprejemljivega praga
    }


def decide_registration_outcome(icp_result: o3d.pipelines.registration.RegistrationResult,
                                asym_stats: dict,
                                target_coverage: dict,
                                skipped_icp: bool,
                                min_fitness: float,
                                max_inlier_rmse: float,
                                min_asymmetric_fraction: float,
                                min_target_coverage: float) -> dict:
    """Eksplicitna PASS/FAIL odločitev o končni registraciji, vrnjena kot
    del rezultata (ne le izpisana na stdout).

    Prejšnje "varovalke" v tej datoteki so bile zgolj asimetrične: nizka
    coarse fitness je preskočila ICP, a je funkcija kljub temu VRNILA
    (verjetno slabo) transformacijo; nizek delež poravnanih asimetričnih
    značilnosti je zgolj izpisal WARNING. Robot, ki to skripto kliče kot
    korak v svojem ciklu prijemanja, ne bere stdout-a - potrebuje en sam,
    nedvoumen booleov signal (in razloge, če je False), na katerega se
    lahko dejansko odloči, ali sme prijeti del. Vsak prag spodaj lahko SAM
    prevesi odločitev na FAIL, ne glede na to, kako dobri so preostali -
    to ni "povprečna ocena", je AND čez vse varnostne pogoje.

    Glavna varovalka za dejansko prekrivanje sken<->CAD je min_target_coverage
    (glej evaluate_target_coverage) - target->source, robustna na okluzijo.
    min_fitness na icp_result.fitness (source->target, Open3D-jeva
    definicija - o3d.pipelines.registration.evaluate_registration) je namerno samo NIZKA
    zdrava-pamet meja (privzeto majhna), ne primarna odločitvena meja:
    ta fitness je po konstrukciji nizji na močno okludiranih skenih tudi
    pri popolnoma pravilni pozi, zato bi visok fiksen prag nanjo (prejšnji
    privzetek 0.6) zavračal dobre poze - natanko napaka, ki jo je
    prakticen zagon te datoteke pokazal.

    OPOZORILO: vsi privzeti pragovi (min_fitness/max_inlier_rmse/
    min_asymmetric_fraction/min_target_coverage) so razumni PLACEHOLDER-ji,
    NE empirično umerjene vrednosti. Pravilna umeritev: Monte Carlo čez
    znane dobre poze (ta datoteka že podpira --translation_x/y/z/
    --rotation_x/y/z_deg in naključen šum za tak namen) da porazdelitev
    vsake metrike pri PRAVILNI registraciji - prag postavi pod spodnji rep
    te porazdelitve, in preveri, da namerno napačni yaw kandidati (npr.
    --yaw_step_deg z zamaknjenim začetkom) padejo jasno pod njega, preden
    kateremukoli od teh privzetkov zaupaš v produkciji.
    """
    reasons = []
    if skipped_icp:
        reasons.append(
            "coarse (yaw sweep) fitness was below --min_coarse_fitness_for_icp, "
            "so ICP was skipped entirely - the returned pose is only the coarse "
            "seed and is not reliable enough to act on")
    if icp_result.fitness < min_fitness:
        reasons.append(f"fitness {icp_result.fitness:.4f} is below the minimum "
                       f"required {min_fitness:.4f} (sanity floor only - see "
                       f"target_coverage below for the real occlusion-robust overlap check)")
    if icp_result.inlier_rmse > max_inlier_rmse:
        reasons.append(f"inlier_rmse {icp_result.inlier_rmse:.4f} is above the maximum "
                       f"allowed {max_inlier_rmse:.4f}")
    if asym_stats["fraction_within_threshold"] < min_asymmetric_fraction:
        reasons.append(
            f"only {asym_stats['fraction_within_threshold']:.2%} of the part's distinctive/"
            f"asymmetric features aligned within threshold (minimum required "
            f"{min_asymmetric_fraction:.2%}) - the result may have converged to a symmetric "
            f"look-alike orientation (e.g. a repeated-boss match) rather than the true pose")
    if target_coverage["fraction_within_threshold"] < min_target_coverage:
        reasons.append(
            f"only {target_coverage['fraction_within_threshold']:.2%} of the scanned (target) "
            f"points lie on the CAD surface under this pose (minimum required "
            f"{min_target_coverage:.2%}) - a correct pose should place nearly every genuine "
            f"scanned point on the part regardless of occlusion, so a low value here means the "
            f"scan doesn't actually sit on the part as posed")

    return {
        "accepted": len(reasons) == 0,   # True <=> robot je varen za prijem dela na tej pozi
        "reasons": reasons,   # prazen seznam, če je accepted True; sicer vsak razlog za FAIL
        "fitness": icp_result.fitness,
        "inlier_rmse": icp_result.inlier_rmse,
        "asymmetric_fraction": asym_stats["fraction_within_threshold"],
        "target_coverage_fraction": target_coverage["fraction_within_threshold"],
    }
