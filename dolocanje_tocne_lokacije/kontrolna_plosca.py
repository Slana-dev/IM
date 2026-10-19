"""KONTROLNA PLOŠČA za main_trial.py.

Namen: na enem mestu zbrati VSE parametre, ki jih redno spreminjate za
simuliranje različnih situacij (druga kamera, več/manj šuma, naklon,
okluzija, testiranje pozicijske tolerance, pravi vs. simuliran sken ...),
namesto da bi vsakič na roko sestavljali dolg ukaz `main_trial.py --flag1
... --flagN`. main_trial.py sam ostaja NEDOTAKNJEN - ta datoteka samo
sestavi njegove argumente (sys.argv) iz spodnjega slovarja CONFIG in ga
požene, zato je vedno natanko tako zmogljiva kot main_trial.py --help.

UPORABA:
  1. Nastavite CAMERA (spodaj) na ime kamere iz kamere.CAMERAS (ali None za
     povsem ročne nastavitve) - vsi kamera-specifični parametri
     (camera_width_px/height_px, camera_fov_deg, translation_z,
     depth_noise_at_1m, noise_reference_distance_m, distance_bias_permille,
     global_planarity_mm, noise_spatial_correlation_px) se SAMODEJNO
     napolnijo iz izbrane kamere - glej resolve_camera_params().
  2. Uredite CONFIG spodaj po potrebi za scenarij, ki ga testirate.
  3. Poženite `python kontrolna_plosca.py` - najprej izpiše dejanski ukaz,
     ki ga bo pognala (za sledljivost/ponovljivost), nato zažene
     main_trial.main().

KAMERA JE VEDNO V IZHODIŠČU (0,0,0): main_trial.py zdaj za realne IN
simulirane skene enako predpostavlja fiksno kamero v izhodišču
koordinatnega sistema (glej build_reference_transform v main_trial.py).
translation_x/y/z + rotation_x/y/z_deg v CONFIG spodaj NEPOSREDNO
definirajo referenčno (ground-truth) pozo dela GLEDE NA to fiksno kamero -
to je poza, ki jo mora main_trial.py sam oceniti/izračunati. Privzeto so
rotation_*=0 in translation_x/y=0 (del natanko pod kamero, brez naklona) -
translation_z je edini smiselno neničeln privzetek, samodejno enak
(negativni) idealni delovni razdalji izbrane kamere.

PRAVILO PREVLADE: katerakoli vrednost v CONFIG, EKSPLICITNO nastavljena
(torej ne None), vedno prevlada nad tem, kar bi bilo izpeljano iz izbrane
kamere - CAMERA torej samo napolni razumne privzetke, nikoli ne prepreči
ročnega overrida posameznega parametra. Vrednost None povsod pomeni "ne
podajaj te zastavice - naj velja izpeljana (iz kamere) ali main_trial.py-
jeva lastna privzeta vrednost".
"""

import sys
from pathlib import Path

import kamere

BASE_PATH = Path(__file__).resolve().parent


# =============================================================================
# KATERO KAMERO SIMULIRAMO
# =============================================================================
# Ime iz kamere.CAMERAS (glej ta file za celoten seznam in specifikacije) -
# ali None, če želite vse spodnje kamera-parametre nastaviti čisto ročno.
CAMERA = "photoneo_phoxi_s"  # "photoneo_phoxi_s" | "zivid_2plus_m60" | "mecheye_pro_s" | None

# Skalira izpeljano ločljivost (--camera_width_px/height_px) navzdol - prave
# kamere imajo 2-5 Mpix, kar naredi ray casting počasen; 0.25-0.5 za hitro
# iteriranje med testiranjem, 1.0 za dokončno/natančno simulacijo.
CAMERA_RESOLUTION_SCALE = 0.5


# =============================================================================
# CONFIG - vsi parametri, ki jih dejansko urejate za testiranje scenarijev.
# None = "ne podajaj te zastavice" (izpeljano iz kamere, če je na voljo,
# sicer main_trial.py-jev lasten privzetek - glej main_trial.py --help).
# =============================================================================
CONFIG = {
    # --- Vhodni podatki ---
    "cad": "d_housing_detailed.STL",   # REFERENCA: CAD (STL), na katerega poravnamo sken - poza je podana v NJEGOVEM koordinatnem sistemu
    "scan_cad": None,   # SKEN: CAD (STL), iz katerega se simulira sken - None = isti kot "cad". Mora biti v ISTEM koordinatnem sistemu kot "cad"
    "use_real_scan": False,        # True = naloži --scan namesto simulacije iz CAD-a
    "scan": "scan.ply",             # ime/pot datoteke s pravim skenom (uporabljeno samo, če je use_real_scan=True)
    "extra_scans": [],             # dodatni zaporedni zajemi za temporalno povprečenje (samo z use_real_scan), glej load_and_merge_real_scans

    # --- Scenarij: okluzija in miza/ozadje ---
    "disable_occlusion": False,    # True = brez ray castinga, gosto vzorči cel del (diagnostika vpliva okluzije)
    "add_table_background": True,  # simulirana miza pod delom v sceni za ray casting
    "table_size_x": 200.0,          # širina mize (mm), na kateri del leži
    "table_size_y": 200.0,          # globina mize (mm)
    "table_center_x": 0.0,          # X koordinata sredine mize (mm)
    "table_center_y": 0.0,          # Y koordinata sredine mize (mm)
    "remove_table_background": True,   # True/False/None (None=auto: True za use_real_scan, False za simulacijo)
    "table_removal_margin_mm": 10.0,   # varnostni rob (mm) pri odstranjevanju mize/ozadja iz skena

    # --- Referenčna (ground-truth) poza dela GLEDE NA fiksno kamero v (0,0,0) ---
    # (samo simulacija - glej modulski docstring) - to je poza, ki jo mora
    # main_trial.py sam oceniti/izračunati. translation_z je None = iz
    # izbrane kamere (negativna idealna delovna razdalja, glej CAMERA zgoraj).
    "translation_x": 60.0,       # X pozicija dela (mm) - 0 = natanko pod kamero
    "translation_y": 38.0,       # Y pozicija dela (mm)
    "translation_z": None,          # Z razdalja dela od kamere (mm, NEGATIVNA) - None = iz izbrane kamere
    "rotation_x_deg": 0.0,          # naklon okoli X (roll, stopinje) - testira toleranco na neravno ležečo postavitev
    "rotation_y_deg": 0.0,          # naklon okoli Y (pitch, stopinje)
    "rotation_z_deg": 138.0,      # zasuk okoli Z (yaw, stopinje) - registracijski cevovod ga mora sam najti (yaw sweep)

    # --- Šum senzorja (None = iz izbrane kamere, glej CAMERA zgoraj) ---
    "depth_noise_at_1m": None,               # std (mm) šuma pri noise_reference_distance_m
    "noise_distance_power": 2.0,             # eksponent rasti šuma z razdaljo - kamere.py trenutno NE daje dovolj podatkov za zanesljivo oceno tega, glej main_trial.py opozorilo
    "noise_reference_distance_m": None,      # razdalja (m), pri kateri je depth_noise_at_1m izmerjen
    "max_incidence_deg": 75.0,               # kot (stopinje), nad katerim se šum zaradi poševnega pogleda omeji, da ne "eksplodira"
    "distance_bias_permille": None,          # sistematičen (sensor-kalibracijski) zamik razdalje - kamere.py "relative_distance_accuracy_permille"
    "global_planarity_mm": None,             # sistematično popačenje čez FOV - kamere.py "global_planarity_mm"
    "global_planarity_spatial_scale_px": None,   # kako "razpotegnjeno" (v pikslih) je to popačenje - None = samodejno
    "outlier_probability": 0.01,             # delež točk, ki dobijo grobo napačno (izstopajočo) vrednost namesto navadnega šuma
    "outlier_std_multiplier": 15.0,          # kolikokrat večji šum imajo te izstopajoče (napačne) točke
    "flying_pixel_depth_jump_mm": 3.0,       # kolikšen skok globine (mm) med sosednjima pikseloma šteje za rob/skok
    "flying_pixel_probability": 0.05,         # verjetnost, da piksel na takem robu postane napačen "leteč" piksel
    "quantization_step_at_1m": 0.05,         # najmanjši korak (mm), na katerega senzor zaokroži meritev
    "noise_spatial_correlation_px": None,    # kamera-izpeljano: local_planarity_mm / point_to_point_distance_mm

    # --- Kamera geometrija (None = iz izbrane kamere) ---
    # Kamera sama je vedno v (0,0,0) - glej modulski docstring - zato tu ni
    # več kamerine pozicije, samo njena optika (FOV, ločljivost).
    "camera_fov_deg": None,          # vidni kot kamere (stopinje)
    "camera_width_px": None,         # širina slike kamere (piksli)
    "camera_height_px": None,        # višina slike kamere (piksli)

    # --- Predobdelava / obrezovanje / oceni naklona ---
    "voxel_size": 1.0,                          # velikost mreže (mm) za zmanjšanje gostote oblaka točk pred registracijo
    "sample_points": 200000,                    # koliko točk vzorčimo iz CAD modela
    "top_fraction": 1,                       # kolikšen delež dela (od zgoraj) dejansko "vidi" kamera
    "up_axis": 2,                                # katera os predstavlja "navzgor" (0=X, 1=Y, 2=Z)
    "flip_up_direction": False,                 # False = kamera gleda CAD-ovo +Z stran (privzeto, preverjeno za d_housing_detailed); True = -Z stran
    "crop_edge_margin_factor": 2.0,             # izključi pas te širine (x voxel_size) tik ob meji obreza SAMO na CAD strani (glej main_trial.py --help) - 0 = izklopljeno
    "tilt_normal_top_band_fraction": 0.5,       # kolikšen delež zgornjih točk se uporabi za oceno naklona dela
    "height_percentile": 99.5,                  # percentil višine namesto najvišje točke (odpornost na osamelce)

    # --- Filter letečih pikslov (grazing incidence) na realnem/simuliranem skenu ---
    "filter_grazing_incidence": True,           # odstrani domnevne napačne ("leteče") točke na robovih
    "flying_pixel_filter_max_incidence_deg": 80.0,   # kot (stopinje), nad katerim se točka odstrani kot verjeten leteč piksel
    "grazing_incidence_normal_radius_factor": 2.0,   # radij (večkratnik voxel_size) za oceno normal pri tem filtru

    # --- Groba (yaw sweep) + ICP registracija ---
    "coarse_distance_factor": 3.0,              # kako širok je prag ujemanja pri grobi (yaw sweep) registraciji
    "yaw_step_deg": 3.0,                        # kako fino (stopinje) se preiskuje rotacija okoli navpične osi - 6.0 je zdaj main_trial.py-jev privzetek (empirično enaka natančnost kot 3.0, a 2x hitreje - glej main_trial.py --help)
    "yaw_sweep_refine_iterations": 5,           # koliko iteracij ICP se uporabi za oceno vsakega testnega kota
    "icp_distance_factor": 2.0,                 # kako širok je prag ujemanja pri natančni ICP registraciji
    "icp_voxel_scales": (4.0, 2.0, 1.0, 0.5),   # zaporedje velikosti mreže (od grobe do fine) za postopno ICP
    "icp_max_iterations": 100,                  # največje število iteracij ICP na posamezno stopnjo
    "icp_iterations_schedule": None,            # neobvezen slovar {scale: iteracije} za drugačen proračun po posamezni ICP stopnji, npr. {4.0: 30, 2.0: 50, 1.0: 75, 0.5: 100} - None = povsod icp_max_iterations (glej main_trial.py --help)
    "robust_kernel_k_factor": 1.0,              # kako strogo ICP zavrača osamelce (manjša vrednost = strožje)
    "min_coarse_fitness_for_icp": 0.1,          # najnižja kakovost grobe registracije, da se sploh nadaljuje z ICP

    # --- Point-to-mesh polish (Phase 4.5, glej main_trial.py --help) ---
    "use_mesh_polish": True,                    # dodaten končni korak: poravna sken neposredno na TOČNO CAD mrežo namesto na vzorčen oblak, uteženo s heteroscedastičnim šumovnim modelom - strogo ne-regresiven (sprejet samo, če ne poslabša trimmed_rms)
    "mesh_polish_max_iterations": 30,           # največje število iteracij tega koraka
    "mesh_polish_distance_factor": 1.0,         # prag ujemanja (x voxel_size x najfinejši icp_voxel_scales x icp_distance_factor)
    "mesh_polish_trim_fraction": 0.8,           # delež target točk (najbližje najprej), uporabljen za odločitev sprejmi/zavrni polish

    # --- Asimetrični varnostni pregled + target->source pokritost ---
    "asymmetric_check_top_fraction": 0.15,      # delež najbolj "edinstvenih" točk dela, uporabljenih za varnostni pregled
    "asymmetric_check_residual_factor": 3.0,    # kako natančno se morajo te točke ujemati, da pregled uspe
    "target_coverage_distance_factor": 2.0,     # kako blizu mora biti sken CAD modelu, da šteje za "pokrito"

    # --- PASS/FAIL sprejemni pragovi ---
    "min_accept_fitness": 0.1,                       # najnižja dovoljena kakovost ujemanja za sprejemljiv rezultat
    "max_accept_inlier_rmse": None,                  # največja dovoljena povprečna napaka (mm) za sprejemljiv rezultat
    "min_accept_asymmetric_fraction": 0.5,           # najmanjši delež "edinstvenih" točk, ki se morajo ujemati
    "min_accept_target_coverage": 0.85,              # najmanjši delež skena, ki mora ležati na delu, da je rezultat sprejemljiv

    # --- Predpomnilniki ---
    "use_cad_cache": True,           # shrani vzorčen CAD model na disk za hitrejši naslednji zagon
    "use_preprocess_cache": True,    # shrani predobdelane oblake točk na disk za hitrejši naslednji zagon

    # --- Samo-testi (glej main_trial.run_self_tests) ---
    "self_test": False,   # True = samo preveri, da so novi parametri pravilno implementirani, brez zagona registracije
}


def resolve_camera_params(camera_name: str | None, resolution_scale: float = 1.0) -> dict:
    """Iz kamere.CAMERAS[camera_name] izpelje main_trial.py CLI parametre.

    Vrne prazen slovar, če camera_name is None - CONFIG potem v celoti
    velja tak, kot je (ali main_trial.py-jevi lastni privzetki za vse, kar
    je v CONFIG pustil na None). Polja, ki jih izbrana kamera nima podana
    (None v kamere.py - npr. MECHEYE_PRO_S["local_planarity_mm"]), so tu
    preprosto izpuščena, ne pretvorjena v napačno 0 ali podobno."""
    if camera_name is None:
        return {}
    if camera_name not in kamere.CAMERAS:
        raise KeyError(
            f"Neznana kamera '{camera_name}' - na voljo v kamere.CAMERAS: "
            f"{sorted(kamere.CAMERAS)}")
    spec = kamere.CAMERAS[camera_name]
    derived: dict = {}

    resolution = spec.get("depth_map_resolution_px")
    scanning_area = spec.get("scanning_area_mm")
    if resolution is not None:
        derived["camera_width_px"] = max(1, round(resolution["width"] * resolution_scale))
        if scanning_area is not None:
            # main_trial.py (prek Open3D-jevega create_rays_pinhole) izpelje
            # NAVPIČNI FOV iz --camera_fov_deg (horizontalen) IN razmerja
            # camera_width_px/camera_height_px - NE iz kamerinega dejanskega
            # fizičnega razmerja scanning_area_mm (širina/višina). Senzorjevo
            # lastno PIKSELSKO razmerje (depth_map_resolution_px) se od tega
            # fizičnega razmerja pri vseh treh kamerah rahlo razlikuje (npr.
            # pri Zividu bi dalo navpični FOV ~10% preozek, pri Mech-Eye
            # ~7% preširok - preverjeno empirično) - zato height_px namesto
            # tega IZPELJEMO iz width_px + FIZIČNEGA razmerja scanning_area_mm,
            # da se navpični FOV res ujema s kamerino specifikacijo.
            aspect_mm = scanning_area["width"] / scanning_area["height"]
            derived["camera_height_px"] = max(1, round(derived["camera_width_px"] / aspect_mm))
        else:
            derived["camera_height_px"] = max(1, round(resolution["height"] * resolution_scale))

    if "scanning_area_mm" in spec and spec.get("working_distance_mm") is not None:
        derived["camera_fov_deg"] = kamere.compute_horizontal_fov_deg(spec)   # neodvisen od resolution_scale - FOV je kot, ne ločljivost

    working_distance_mm = spec.get("working_distance_mm")
    if working_distance_mm is not None:
        # translation_z je razdalja dela OD kamere (fiksne v izhodišču),
        # zato mora biti NEGATIVNA (kamera gleda navzdol, del je pod njo) -
        # glej build_reference_transform docstring v main_trial.py.
        derived["translation_z"] = -working_distance_mm
        derived["noise_reference_distance_m"] = working_distance_mm / 1000.0

    if spec.get("temporal_noise_mm") is not None:
        derived["depth_noise_at_1m"] = spec["temporal_noise_mm"]   # ime zastavice je zgodovinsko - dejansko velja pri noise_reference_distance_m, glej zgoraj

    if spec.get("relative_distance_accuracy_permille") is not None:
        derived["distance_bias_permille"] = spec["relative_distance_accuracy_permille"]

    if spec.get("global_planarity_mm") is not None:
        derived["global_planarity_mm"] = spec["global_planarity_mm"]

    local_planarity_mm = spec.get("local_planarity_mm")
    point_to_point_mm = spec.get("point_to_point_distance_mm")
    if local_planarity_mm is not None and point_to_point_mm:   # point_to_point_mm mora biti resničen in različen od 0, da deljenje sploh nekaj pove
        # Korelacijsko okno (local_planarity_mm/point_to_point_mm) je
        # izračunano pri kamerini NATIVNI (polni) ločljivosti - point_to_point_mm
        # je razmik med NATIVNIMI piksli. Če simuliramo pri nižji ločljivosti
        # (resolution_scale<1), en simuliran piksel pokrije VEČ mm
        # (point_to_point_mm/resolution_scale), zato mora korelacijsko okno v
        # PIKSLIH SKRČITI za isti faktor, da ohrani svojo pravo FIZIČNO
        # velikost (mm) - brez tega bi bilo okno pri nižji ločljivosti
        # (v mm) preveliko za isti "noise_spatial_correlation_px".
        derived["noise_spatial_correlation_px"] = (local_planarity_mm / point_to_point_mm) * resolution_scale

    return derived


# Ključi CONFIG so natanko imena argparse dest v main_trial.py (glej
# registration_config.py) - zato je zastavica za navaden parameter kar
# --<ključ>. Posebej so obravnavane samo zastavice brez vrednosti in
# vrednosti, ki niso en sam skalar.
_STORE_TRUE_FLAGS = {"use_real_scan", "disable_occlusion", "add_table_background", "self_test"}
_BOTH_WAYS_FLAGS = {"remove_table_background", "flip_up_direction"}   # True -> --<ključ>, False -> --no_<ključ>
_DISABLE_FLAGS = {   # privzeto vklopljeno v main_trial.py - mogoče je samo IZKLOPITI
    "filter_grazing_incidence": "--no_filter_grazing_incidence",
    "use_mesh_polish": "--no_mesh_polish",
    "use_cad_cache": "--no_cad_cache",
    "use_preprocess_cache": "--no_preprocess_cache",
}


def build_cli_args(config: dict) -> list[str]:
    """Sestavi main_trial.py-jev argv (brez imena skripte) iz `config` -
    enega slovarja, ki je bodisi CONFIG sam, bodisi CONFIG z vanj zlitimi
    kamera-izpeljanimi vrednostmi (glej __main__ spodaj). Vrstni red
    zastavic sledi vrstnemu redu ključev; None = zastavice ne podamo."""
    args: list[str] = []
    for key, value in config.items():
        if value is None:
            continue
        if key in _STORE_TRUE_FLAGS:
            if value:
                args.append(f"--{key}")
        elif key in _DISABLE_FLAGS:
            if not value:
                args.append(_DISABLE_FLAGS[key])
        elif key in _BOTH_WAYS_FLAGS:
            args.append(f"--{key}" if value else f"--no_{key}")
        elif key == "extra_scans":
            if value:
                args += ["--extra_scans", *(str(p) for p in value)]
        elif key == "icp_voxel_scales":
            if value:
                args += ["--icp_voxel_scales", ",".join(str(s) for s in value)]
        elif key == "icp_iterations_schedule":   # slovar {scale: iteracije} -> "scale:iteracije,scale:iteracije,..."
            if value:
                args += ["--icp_iterations_schedule",
                         ",".join(f"{scale}:{iterations}" for scale, iterations in value.items())]
        else:
            args += [f"--{key}", str(value)]
    return args


if __name__ == "__main__":
    effective_config = dict(CONFIG)   # kopija - ne spreminjamo modulskega CONFIG v mestu
    camera_params = resolve_camera_params(CAMERA, resolution_scale=CAMERA_RESOLUTION_SCALE)
    for key, value in camera_params.items():
        if effective_config.get(key) is None:   # CONFIG-ov eksplicitni override (ne None) vedno zmaga - glej modulski docstring
            effective_config[key] = value

    cli_args = build_cli_args(effective_config)
    print(f"Camera: {CAMERA!r} (resolution_scale={CAMERA_RESOLUTION_SCALE})")
    print("Running: main_trial.py " + " ".join(cli_args))
    print()

    sys.argv = ["main_trial.py"] + cli_args
    import main_trial   # uvoženo šele tu, da zgornji izpis/napake v CONFIG ne zahtevajo open3d, če se kdaj samo preverja sestavljanje argumentov
    main_trial.main()
