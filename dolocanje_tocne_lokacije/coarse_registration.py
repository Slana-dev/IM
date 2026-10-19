"""Groba (globalna) registracija: ocena naklona + izčrpen yaw sweep z neposredno oceno translacije."""
from functools import partial
from typing import Optional

import numpy as np
import open3d as o3d

from poses import registration_result, rotation_about_axis, rotation_aligning_vectors
from workcell import Table

try:
    from joblib import Parallel, delayed   # neobvezna odvisnost - glej yaw_sweep_registration za vzporedno ocenjevanje yaw kandidatov
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False


def estimate_surface_normal_ransac(points: np.ndarray,
                                   up_axis: int,
                                   top_band_fraction: float = 0.5,
                                   distance_threshold: Optional[float] = None,
                                   num_iterations: int = 1000) -> np.ndarray:
    """RANSAC-prileganje ravnine na zgornji pas točk, namesto golega PCA
    čez CEL oblak.

    Gol PCA nima robustne uteži: vsaka točka prispeva enako v kovariančno
    matriko, zato en sam gost šop točk zunaj prave zgornje ploskve dela
    (najpogosteje MIZA, če remove_table_background ni bila poklicana ali ni
    bila dovolj agresivna) lahko oceno normale zavede proti mizi namesto
    proti dejanski nagnjeni zgornji ploskvi dela - naklon dela se s tem v
    celoti izgubi. RANSAC namesto tega poišče ravnino, ki jo podpira NAJVEČ
    soglasnih (inlier) točk, in vse ostalo (miza, robovi, leteči piksli,
    drugi osamelci) preprosto zavrže.

    top_band_fraction dodatno omeji vhod RANSAC-u na zgornji pas točk (po
    percentilu vzdolž up_axis) - tudi če RANSAC sam prenese nekaj mizinih
    točk v vhodu, jih ta pred-filter v veliki večini primerov že izloči,
    preden RANSAC sploh steče, kar zniža število potrebnih iteracij za
    zanesljivo konvergenco."""
    coord = points[:, up_axis]
    band_cutoff = np.percentile(coord, 100.0 * (1.0 - top_band_fraction))   # meja zgornjega pasu po percentilu
    band_points = points[coord >= band_cutoff]   # obdržimo samo zgornji pas točk
    if len(band_points) < 3:   # premalo točk za prileganje ravnine (RANSAC potrebuje vsaj 3)
        band_points = points   # nazaj na cel oblak kot skrajna rešitev
    if distance_threshold is None:
        extent = np.linalg.norm(band_points.max(axis=0) - band_points.min(axis=0))   # velikost pasu, za smiseln privzet prag
        distance_threshold = max(extent * 0.005, 0.1)   # 0.5% velikosti pasu, vsaj 0.1 (enote oblaka točk)
    band_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(band_points))   # zgornji pas kot samostojen oblak za segment_plane
    plane_model, _inliers = band_pcd.segment_plane(
        distance_threshold=distance_threshold, ransac_n=3, num_iterations=num_iterations)   # RANSAC prileganje ravnine
    normal = np.array(plane_model[:3])   # normala ravnine (prva 3 koeficienta ax+by+cz+d=0)
    return normal / np.linalg.norm(normal)   # vrnemo normaliziran (enotski) vektor normale


def build_height_map_grid(points_2d: np.ndarray, heights: np.ndarray, origin: np.ndarray,
                          resolution: float, grid_shape: tuple[int, int]) -> np.ndarray:
    """Rasterizira 2D točke v mrežo, vsaka celica hrani max višino točk, ki
    padejo vanjo (mapa "kako visok je del tukaj") - uporablja jo
    estimate_xy_translation_phase_correlation spodaj. Celice, kamor ne
    pade nobena točka, ostanejo 0, kar je ločeno od pravih (vedno
    pozitivnih, glej klicatelja) višin."""
    idx = np.floor((points_2d - origin) / resolution).astype(int)   # indeks celice mreže za vsako točko
    valid = (idx[:, 0] >= 0) & (idx[:, 0] < grid_shape[0]) & (idx[:, 1] >= 0) & (idx[:, 1] < grid_shape[1])   # točke, ki padejo znotraj mreže
    grid = np.zeros(grid_shape, dtype=np.float64)   # prazna mreža, začetno vse 0
    if valid.any():
        np.maximum.at(grid, (idx[valid, 0], idx[valid, 1]), heights[valid])   # v vsako celico vpišemo max višino
    return grid   # vrnemo height-map mrežo


def estimate_xy_translation_phase_correlation(source_pts: np.ndarray,
                                              target_pts: np.ndarray,
                                              up_axis: int,
                                              resolution: float,
                                              margin_factor: float = 1.5) -> np.ndarray:
    """Oceni translacijo v ravnini pravokotni na up_axis (X/Y za up_axis=2)
    prek 2D FFT fazne korelacije med projekcijami height-map, namesto
    surove razlike centroidov.

    Ujemanje centroidov povpreči pozicijo vsake vidne točke - okluzija, ki
    obreže eno stran obrisa bolj kot drugo (samo-okluzija zaradi zasukanega
    dela, ali kamera, ki ni točno nad delom, ko je ta pomaknjen na stran -
    glej test tolerance X/Y pozicije), to povprečje pristransko premakne ne
    le vzdolž up_axis (to je že obravnavano ločeno prek primerjave višine v
    _evaluate_yaw_candidate), ampak tudi v pravokotni ravnini.
    Fazna korelacija namesto tega ujema celotno OBLIKO obrisa - veliko
    večji, bolj redundanten signal, ki ga peščica manjkajočih robnih točk
    komaj moti, podobno kot registracija slik ni zmedena zaradi delno
    obrezane fotografije. Validirano na sintetičnih podatkih z 40% točk,
    odstranjenih z ene strani asimetričnega obrisa: fazna korelacija je
    povrnila pravi zamik na 0.03 enote natančno, v primerjavi z 13+
    enotami napake pri navadni razliki centroidov.

    Vsak oblak se rasterizira v svojo height-map mrežo
    (build_height_map_grid), sidrano na svoj bounding box plus margina -
    ne na skupno svetovno izhodišče, saj je ta zamik natanko tisto
    neznano, ki ga funkcija rešuje. Deljenje cross-power spektra z lastno
    magnitudo (od tod "faza" v fazni korelaciji) spremeni čisto translacijo
    med mrežama v en oster korelacijski vrh, robusten na to, da imata
    mreži različno gostoto/pokritost točk (navadna cross-korelacija bi bila
    na to občutljiva). Lokacija vrha da OSTANEK zamika znotraj lokalnega
    okvirja vsake mreže; dodajanje razlike med izhodiščema bounding boxov
    obeh mrež nazaj povrne dejansko metrično translacijo.
    """
    plane_axes = [a for a in range(3) if a != up_axis]   # dve osi pravokotni na up_axis (npr. X,Y za up_axis=Z)
    src_2d = source_pts[:, plane_axes]   # projekcija source točk na to ravnino
    tgt_2d = target_pts[:, plane_axes]   # projekcija target točk na to ravnino
    # Zamaknjeno tako, da je striktno pozitivno znotraj vsakega oblaka
    # (lasten min -> resolution, ne 0), da prave-a-nizke višine nikoli ne
    # zamenjamo s prazno celico.
    src_h = source_pts[:, up_axis] - source_pts[:, up_axis].min() + resolution   # višina source točk, vedno > 0
    tgt_h = target_pts[:, up_axis] - target_pts[:, up_axis].min() + resolution   # višina target točk, vedno > 0

    # Vsaka mreža je velika glede na svoj obris plus margina, da ima
    # (neznan) ostanek zamika, ki ga rešuje fazna korelacija, dovolj
    # prostora, ne da bi se ovil čez krožno mejo FFT.
    src_extent = src_2d.max(axis=0) - src_2d.min(axis=0)   # velikost obrisa source
    tgt_extent = tgt_2d.max(axis=0) - tgt_2d.min(axis=0)   # velikost obrisa target
    span = np.maximum(src_extent, tgt_extent) * margin_factor   # velikost mreže z margino
    grid_shape = tuple(np.maximum(np.ceil(span / resolution).astype(int), 8))   # dimenzije mreže v pikslih

    src_origin = src_2d.min(axis=0) - (np.array(grid_shape) * resolution - src_extent) / 2.0   # izhodišče mreže source, obris centriran
    tgt_origin = tgt_2d.min(axis=0) - (np.array(grid_shape) * resolution - tgt_extent) / 2.0   # izhodišče mreže target, obris centriran

    src_grid = build_height_map_grid(src_2d, src_h, src_origin, resolution, grid_shape)   # height-map mreža source
    tgt_grid = build_height_map_grid(tgt_2d, tgt_h, tgt_origin, resolution, grid_shape)   # height-map mreža target

    f_src = np.fft.fft2(src_grid)   # 2D Fourierova transformacija source mreže
    f_tgt = np.fft.fft2(tgt_grid)   # 2D Fourierova transformacija target mreže
    cross_power = f_src * np.conj(f_tgt)   # cross-power spekter
    magnitude = np.abs(cross_power)   # magnituda spektra
    magnitude[magnitude < 1e-10] = 1e-10  # preprečimo deljenje s skoraj 0 v območjih spektra brez prekrivanja
    correlation = np.fft.ifft2(cross_power / magnitude).real   # normalizirana (fazna) korelacija nazaj v prostorsko domeno

    peak = np.array(np.unravel_index(np.argmax(correlation), correlation.shape), dtype=float)   # pozicija najvišjega vrha korelacije
    for i in range(2):
        # ifft2 vrne zamike v [0, grid_shape) - vse čez polovico previjemo
        # nazaj na ustrezen negativen zamik.
        if peak[i] > grid_shape[i] / 2:
            peak[i] -= grid_shape[i]
    residual_shift = peak * resolution   # ostanek zamika v pravih enotah (mm)

    return residual_shift + (tgt_origin - src_origin)   # ostanek + razlika izhodišč = dejanska translacija


def estimate_z_translation_height_maps(source_pts: np.ndarray,
                                       target_pts: np.ndarray,
                                       xy_translation: np.ndarray,
                                       up_axis: int,
                                       resolution: float,
                                       min_common_cells: int = 20) -> Optional[float]:
    """Oceni translacijo vzdolž up_axis kot MEDIANO razlik višin po celicah
    skupne height-map mreže, potem ko je source že premaknjen za
    xy_translation (fazna korelacija).

    ZAKAJ ne primerjava visokega percentila višine celega oblaka: source
    (CAD) vsebuje CEL del, tudi površine, ki jih kamera ne vidi, sken pa
    samo vidne - zgornji promili točk zato v enem oblaku pristanejo npr. na
    tanki steni, v drugem pa drugje (stena ima malo vidnih točk, strme
    stranice odstrani filter letečih pikslov). Na eHDS D housingu je to
    sistematično dvignilo grobo Z oceno za ~3-4 mm. Max višina v celici
    CAD-a pa je NATANKO tisto, kar vidi kamera od zgoraj, zato je primerjava
    celica-s-celico konsistentna; stena vpliva le na svojih nekaj celic, ki
    jih mediana preprosto ignorira.

    Vrne None, če imata mreži premalo skupnih celic (XY ocena je verjetno
    napačna) - klicatelj tedaj uporabi percentilno oceno."""
    plane_axes = [a for a in range(3) if a != up_axis]
    src_2d = source_pts[:, plane_axes] + xy_translation   # source v target-ovem XY položaju
    tgt_2d = target_pts[:, plane_axes]
    origin = np.minimum(src_2d.min(axis=0), tgt_2d.min(axis=0))   # skupna mreža čez oba oblaka
    shape = tuple(np.floor((np.maximum(src_2d.max(axis=0), tgt_2d.max(axis=0)) - origin) / resolution).astype(int) + 1)

    def height_map(points_2d: np.ndarray, heights: np.ndarray) -> np.ndarray:
        grid = np.full(shape, -np.inf)   # -inf = prazna celica
        idx = np.floor((points_2d - origin) / resolution).astype(int)
        np.maximum.at(grid, (idx[:, 0], idx[:, 1]), heights)   # najvišja točka v celici = kar vidi kamera od zgoraj
        return grid

    src_map = height_map(src_2d, source_pts[:, up_axis])
    tgt_map = height_map(tgt_2d, target_pts[:, up_axis])
    common = np.isfinite(src_map) & np.isfinite(tgt_map)
    if common.sum() < min_common_cells:
        return None
    return float(np.median(tgt_map[common] - src_map[common]))


def _evaluate_yaw_candidate(yaw_deg: float,
                            source_pts: np.ndarray,
                            target_pts: np.ndarray,
                            source_center: np.ndarray,
                            target_normal: np.ndarray,
                            tilt_correction: np.ndarray,
                            plane_axes: list[int],
                            up_axis: int,
                            translation_grid_resolution: float,
                            height_percentile: float,
                            table_center: np.ndarray,
                            table_half_extent: np.ndarray,
                            distance_threshold: float,
                            refine_iterations: int) -> Optional[tuple[float, float, float, np.ndarray]]:
    """Oceni EN yaw kandidat - modulska funkcija (ne zaprtje znotraj
    yaw_sweep_registration), da jo je mogoče poslati v ločen proces
    (joblib.Parallel). Open3D objekti se ne dajo picklati, zato funkcija
    prejme SAMO navadne numpy/python vrednosti in si Open3D objekte, ki jih
    rabi, zgradi SAMA, znotraj delovnega procesa (poceni, O(n),
    zanemarljivo v primerjavi z ICP klicem samim).

    Vrne None, če fizični prior (implicirana pozicija izven mize) zavrne
    tega kandidata - klicatelj tak rezultat preprosto prezre. Sicer vrne
    (yaw_deg, fitness, inlier_rmse, transformation)."""
    R = rotation_about_axis(target_normal, yaw_deg) @ tilt_correction   # kombinirana rotacija: naklon + yaw
    T = np.eye(4)
    T[:3, :3] = R
    source_pts_rotated = (R @ source_pts.T).T   # source točke, zasukane s kandidatno rotacijo

    translation = np.zeros(3)
    translation[plane_axes] = estimate_xy_translation_phase_correlation(
        source_pts_rotated, target_pts, up_axis, translation_grid_resolution)   # X/Y del translacije
    z_translation = estimate_z_translation_height_maps(
        source_pts_rotated, target_pts, translation[plane_axes], up_axis, translation_grid_resolution)   # up_axis del translacije
    if z_translation is None:
        # Rezerva, če se mreži skoraj ne prekrivata: primerjava vrha obeh
        # oblakov - visok percentil namesto golega max(), ki bi ga en sam
        # osamelec/leteč piksel premaknil za precej mm.
        z_translation = (float(np.percentile(target_pts[:, up_axis], height_percentile)) -
                         float(np.percentile(source_pts_rotated[:, up_axis], height_percentile)))
    translation[up_axis] = z_translation
    T[:3, 3] = translation

    part_center = (T[:3, :3] @ source_center) + T[:3, 3]   # kam bi ta transformacija postavila center dela
    if np.any(np.abs(part_center[plane_axes] - table_center) > table_half_extent):   # izven mize - fizično nemogoče
        return None

    # np.array(..., copy=True) - joblib (loky backend) prenese velike numpy
    # tabele delovnim procesom prek BRALNO-ZAŠČITENEGA (read-only) memory-mapa
    # (za učinkovitost, brez ponovnega kopiranja) - o3d.utility.Vector3dVector
    # pa zahteva zapisljiv buffer in vrže "array is not writeable" na takem
    # vhodu. Eksplicitna kopija tu to popravi (poceni - source/target sta že
    # zmanjšana/downsampled na to točko).
    source_down = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(source_pts, copy=True)))
    target_down = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(target_pts, copy=True)))
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=refine_iterations)
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    # Smer CAD -> sken je namerna. Obrnjena smer (sken -> CAD, kot v
    # refine_registration) odpravi ~1.2 mm Z pristranskosti zaradi
    # nevidnih CAD površin, a je na D housingu izbirala yaw SLABŠE: njena
    # fitness (delež sken točk z ujemanjem) je pri mnogih kandidatih 1.00 in
    # kandidatov ne loči (preverjeno 2026-10-18). Preostalo Z napako odpravi ICP.
    icp_result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, distance_threshold, T, estimation, criteria)   # hiter lokalni ICP polish tega kandidata
    return (float(yaw_deg), float(icp_result.fitness), float(icp_result.inlier_rmse),
           np.asarray(icp_result.transformation))


def yaw_sweep_registration(source_down: o3d.geometry.PointCloud,
                           target_down: o3d.geometry.PointCloud,
                           voxel_size: float,
                           table: Table,
                           up_axis: int = 2,
                           yaw_step_deg: float = 3.0,
                           distance_threshold_factor: float = 3.0,
                           refine_iterations: int = 5,
                           translation_grid_resolution: Optional[float] = None,
                           tilt_normal_top_band_fraction: float = 0.5,
                           height_percentile: float = 99.5,
                           n_jobs: int = -1) -> o3d.pipelines.registration.RegistrationResult:
    """Groba globalna registracija, ki izkorišča PRAVO omejitev postavitve
    namesto reševanja slepe 6-DOF registracije: del vedno leži na mizi v
    isti orientaciji z licem navzdol, zato so Z/roll/pitch fiksirani zaradi
    stika z mizo, X/Y sta znana le ohlapno (nekaj deset mm), yaw (rotacija
    okoli up_axis) pa je edini zares prosti parameter.

    Translacija je ocenjena neposredno, ne iskana: ravnina pravokotna na
    up_axis (X/Y) prek 2D fazne korelacije na obliki obrisa
    (estimate_xy_translation_phase_correlation - robustna na asimetrijo
    okluzije, za razliko od surove razlike centroidov), up_axis sam pa
    prek primerjave visokega percentila višine (height_percentile, glej
    _evaluate_yaw_candidate - namerno NE gol max(), ki bi ga en sam
    osamelec/leteč piksel premaknil za precej mm) - oboje neodvisno od
    neznanega yaw, zato translacije sploh ni treba iskati. Yaw sam je
    preiskan izčrpno (ne naključno vzorčen), saj gre za en omejen,
    popolnoma naštevljiv parameter.

    Prava postavitev nikoli ni POPOLNOMA ravna (drobec umazanije, rahel
    zvitek), zato ta funkcija oceni in popravi tudi dejanski roll/pitch
    naklon, namesto da bi predpostavila natanko nič: normala površine
    target je ocenjena z RANSAC prileganjem ravnine na njen zgornji pas
    točk (estimate_surface_normal_ransac, tilt_normal_top_band_fraction
    izbere ta pas) - namerno NE gol PCA čez cel target, ki bi ga zlahka
    zavedla soprisotnost mize/ozadja v oblaku. Rotacija, ki poravna znano
    referenčno smer "gor" source-a s to ocenjeno normalo, postane osnovna
    orientacija. Kar ostane - rotacija OKOLI te zdaj poravnane osi
    normale - je natanko yaw negotovost, zato sweep še vedno teče, le
    sestavljen na vrhu popravka naklona namesto ravne predpostavke.
    Empirično testirano do ~25-27 stopinj vbrizganega naklona, pred
    porušitvijo okoli 30 stopinj.

    Vsak grob yaw kandidat dobi HITER lokalni ICP polish
    (refine_iterations, nizek - to ni končni natančnostni korak, to
    naredi celoten večnivojski ICP po tej funkciji) pred ocenjevanjem,
    namesto ocenjevanja surovega nerafiniranega posnetka. To je pomembno
    pri delu s ponavljajočimi se značilnostmi (montažne izbokline): napačen
    yaw lahko poravna ponavljajočo se osnovno geometrijo dovolj dobro, da
    zmaga po surovem ŠTEVILU inlierjev, čeprav popolnoma napačno poravna
    tiste redke asimetrične značilnosti (locator pini, konektor), ki
    dejansko ločijo pravilno orientacijo. Kratka lokalna refinacija
    vsakega kandidata najprej potisne proti njegovemu resničnemu bližnjemu
    optimumu - pravi yaw konvergira v tesno prileganje, zgolj na grobo
    poravnan napačen yaw pa navadno ne, ker se asimetrična podrobnost ne
    poravna, ne glede na to, kako lokalno jo premikamo.

    Podprt je le up_axis=2 (Z) kot REFERENČNA smer "gor" (skladno z vsemi
    drugimi predpostavkami o up-axis v tej kodni bazi) - sam popravek
    naklona deluje za poljubno končno orientacijo, fiksirana je le
    začetna referenčna smer na os Z.

    table doda fizični prior: kombinacija (yaw, dx, dy), ki bi postavila
    center dela izven znanega obsega mize, je fizično nemogoča - zato so
    taki kandidati kar zavrnjeni, namesto da bi lahko zmagali zgolj na
    podlagi ICP fitness. Ker je translacija tu IZPELJANA iz yaw (ne iskana
    neodvisno, glej zgoraj), gre pravzaprav za smiselnostno preverjanje
    implicirane postavitve za vsak yaw posebej. Velikost mize prilagodi
    dejanskemu delovnemu območju pravega robota.

    n_jobs: yaw kandidati so med seboj popolnoma neodvisni (vsak samo bere
    source_pts/target_pts, ne piše vanju), zato jih (prek _evaluate_yaw_candidate
    in joblib.Parallel) ovrednotimo v ločenih procesih - z n_jobs=-1
    (privzeto, joblib-ova konvencija za "vsa jedra"). Rezultat je numerično
    ENAK serijskemu izvajanju (isti kandidati, ista ICP klica na vsakega,
    le drugačen vrstni red ocenjevanja) - n_jobs=1 vrne na serijsko
    izvajanje (npr. za odpravljanje napak ali če joblib ni nameščen).
    Hladen prvi klic Parallel() plača enkraten zagon bazena procesov in pri
    enem samem zagonu main_trial.py ni nujno hitrejši od serijskega; pri
    ponovljenih klicih v istem procesu (camera_benchmark.py) je bil sam yaw
    sweep izmerjeno ~1.7x hitrejši."""
    if up_axis != 2:   # ta funkcija podpira samo Z kot referenčno "gor" os
        raise NotImplementedError("yaw_sweep_registration only supports up_axis=2 (Z) as the "
                                  "reference 'up' direction - the part's nominal resting "
                                  "orientation before tilt correction")

    source_pts = np.asarray(source_down.points)   # source oblak kot numpy tabela
    target_pts = np.asarray(target_down.points)   # target oblak kot numpy tabela
    if translation_grid_resolution is None:   # če ni podana, jo izpeljemo iz voxel_size
        translation_grid_resolution = voxel_size * 2.0

    up_vector = np.zeros(3)
    up_vector[up_axis] = 1.0   # enotski vektor "gor" (privzeto Z os)
    target_normal = estimate_surface_normal_ransac(
        target_pts, up_axis, top_band_fraction=tilt_normal_top_band_fraction)   # ocenjena normala površine target (RANSAC na zgornjem pasu)
    if np.dot(target_normal, up_vector) < 0:
        target_normal = -target_normal  # normala ima dvoumen predznak - izberemo tistega, ki kaže "navzgor"
    tilt_deg = np.degrees(np.arccos(np.clip(np.dot(up_vector, target_normal), -1.0, 1.0)))   # ocenjen kot naklona v stopinjah
    tilt_correction = rotation_aligning_vectors(up_vector, target_normal)   # rotacija, ki popravi naklon
    print(f"  Estimated tilt from target's RANSAC surface normal: {tilt_deg:.2f} deg off {up_axis}-axis")

    distance_threshold = voxel_size * distance_threshold_factor   # prag razdalje za ujemanje točk pri hitrem ICP

    yaw_candidates = np.arange(0.0, 360.0, yaw_step_deg)   # vsi testirani yaw koti
    use_parallel = _JOBLIB_AVAILABLE and n_jobs != 1
    print(f"  Yaw sweep: {len(yaw_candidates)} candidates every {yaw_step_deg} deg "
          f"(source={len(source_pts)} pts, target={len(target_pts)} pts), "
          f"distance_threshold={distance_threshold:.3f}, refine_iterations={refine_iterations}, "
          f"n_jobs={'parallel/' + str(n_jobs) if use_parallel else 'serial (joblib not used)'}")

    evaluate = partial(
        _evaluate_yaw_candidate, source_pts=source_pts, target_pts=target_pts,
        source_center=source_pts.mean(axis=0),   # centroid source oblaka (za fizični prior)
        target_normal=target_normal, tilt_correction=tilt_correction,
        plane_axes=[a for a in range(3) if a != up_axis], up_axis=up_axis,
        translation_grid_resolution=translation_grid_resolution, height_percentile=height_percentile,
        table_center=table.center, table_half_extent=table.half_extent,
        distance_threshold=distance_threshold, refine_iterations=refine_iterations)
    if use_parallel:
        # inner_max_num_threads=1: brez tega bi vsak delovni proces ŠE SAM
        # ZASE uporabil Open3D-jevo notranjo (OpenMP/TBB) vzporednost -
        # N procesov x lastna nitnost = prekomerna naročenost istih jeder
        # (oversubscription), kar lahko upočasni namesto pospeši.
        raw_results = Parallel(n_jobs=n_jobs, inner_max_num_threads=1)(
            delayed(evaluate)(yaw) for yaw in yaw_candidates)
    else:
        raw_results = [evaluate(yaw) for yaw in yaw_candidates]

    rejected_count = sum(1 for r in raw_results if r is None)   # koliko kandidatov je bilo zavrnjenih zaradi fizičnega priorja
    valid_results = [r for r in raw_results if r is not None]
    if rejected_count > 0:
        print(f"  Physical prior (table {table.describe()}): rejected {rejected_count}/{len(yaw_candidates)} yaw "
              f"candidates whose implied placement fell off the table")
    if not valid_results:
        raise RuntimeError(
            f"Every yaw candidate's implied placement fell outside the table "
            f"({table.describe()}) - "
            f"table_size_x/table_size_y/table_center are too tight for where the part "
            f"actually ended up, or the phase-correlation translation estimate is off")

    # Najprej maksimiziraj fitness, ob izenačenju minimiziraj inlier_rmse.
    best_yaw, best_fitness, best_rmse, best_transform = max(valid_results, key=lambda r: (r[1], -r[2]))
    print(f"  Best yaw={best_yaw:.1f} deg (after {refine_iterations}-iteration local polish): "
          f"fitness={best_fitness:.4f}, inlier_rmse={best_rmse:.4f}")
    return registration_result(best_transform, best_fitness, best_rmse)
