"""Realni specsi 3D kamer/skenerjev, ki jih uporabljamo za kalibracijo
simulacije v main_trial.py (predvsem simulate_camera_scan in
run_registration/main()'s offset_distance za fiksno montirano kamero).

Shranjeni so SAMO podatki, relevantni za to simulacijo - ne celoten
specifikacijski list proizvajalca. Namerno IZPUŠČENO (ker main_trial.py
tega ne uporablja nikjer): acquisition/scanning time, dimenzije, teža,
delovna temperatura, barva projekcijske enote, napajanje, procesna enota,
podatkovna povezava, IP zaščita, ločljivost barvne slike (color image
resolution - registracija dela samo z geometrijo, ne s teksturo/barvo).

Vsako polje ima SAMO ENO vrednost - tisto pri sweet spot/fokusni/nominalni
delovni razdalji kamere. Kjer je specifikacija podala več vrednosti pri
različnih razdaljah/pogojih (near/far, temperaturni razredi, priporočeno
vs. optimalno območje ipd.), je obdržana SAMO sweet spot vrednost, ostale
so namerno odstranjene (main_trial.py tako ali tako simulira eno samo,
fiksno delovno razdaljo na zagon, glej compute_horizontal_fov_deg spodaj).

Vsak slovar ima komentar, kateremu main_trial.py parametru/CLI zastavici
ustreza, in - kjer relevanten podatek trenutno NI implementiran v
main_trial.py - jasno opozorilo zakaj ne in kaj bi bilo treba dodati.
"""

import math


PHOTONEO_PHOXI_S = {
    "manufacturer": "Photoneo",
    "model": "PhoXi 3D Scanner S",

    # Informativno: structured light (projektor + kamera, triangulacija).
    # main_trial.py-jev 1/cos(kot vpadanja) šumovni člen
    # (scan_simulation.NoiseModel.depth_std_mm) je empiričen model natanko TE vrste
    # senzorja (triangulacijska baseline med projektorjem in kamero se
    # projicirano skrči pri nagnjeni površini) - ni pa uporabljen kot
    # ločeno stikalo za tehnologijo, torej ne vpliva na obnašanje kode,
    # samo pojasnjuje, zakaj je model oblikovan tako, kot je.
    "technology": "structured_light",

    # --- Ločljivost globinske mape (px) ---
    # main_trial.py: --camera_width_px / --camera_height_px
    # (privzetka v kodi sta 640x480 - ZNATNO nižja od te prave kamere)
    "depth_map_resolution_px": {"width": 2472, "height": 2064},

    # --- Delovna razdalja pri sweet spot (mm) ---
    # main_trial.py: offset_distance za top_camera_location() v main()-u
    # (--camera_offset_distance_mm), ko se simulira fiksno montirana prava
    # kamera.
    "working_distance_mm": 442.0,

    # --- Velikost skeniranega območja pri sweet spot razdalji (mm) ---
    # main_trial.py: --camera_fov_deg (glej compute_horizontal_fov_deg
    # spodaj za pretvorbo fizične širine + razdalje v kot)
    "scanning_area_mm": {"width": 364.0, "height": 317.0},

    # --- Temporalni šum globine pri sweet spot razdalji (mm) ---
    # main_trial.py: --depth_noise_at_1m + --noise_reference_distance_m
    # (nastavite noise_reference_distance_m=0.442, depth_noise_at_1m=0.03)
    "temporal_noise_mm": 0.03,

    # --- Točka-do-točke razdalja pri sweet spot razdalji (mm) ---
    # Ni ločen main_trial.py parameter - potrjuje/kalibrira --camera_fov_deg
    # + ločljivost (scanning_area_mm["width"] / depth_map_resolution_px
    # ["width"] = 364/2472 = 0.147mm, dovolj blizu tu podani 0.16mm).
    "point_to_point_distance_mm": 0.16,

    # --- Lokalna planarnost pri sweet spot razdalji (mm) ---
    # main_trial.py: --noise_spatial_correlation_px (v pikslih, ne mm -
    # pretvorba: lokalna_planarnost_mm / point_to_point_distance_mm =
    # 0.16/0.16 ≈ 1.0 px, blizu trenutnemu privzetku 1.5)
    "local_planarity_mm": 0.16,

    # --- Globalna planarnost pri sweet spot razdalji (mm) ---
    # main_trial.py: --global_planarity_mm
    "global_planarity_mm": 0.22,

    # --- Relativna natančnost razdalje (‰ razdalje) ---
    # main_trial.py: --distance_bias_permille
    "relative_distance_accuracy_permille": 1.25,

    # --- Baseline med projektorjem in kamero (mm) ---
    # Kontekst za NoiseModel.depth_std_mm-ov 1/cos(kot vpadanja) člen
    # (glej njen docstring o triangulacijski baseline), a main_trial.py ga
    # ne uporablja kot ločen numeričen vhod - trenutni model je empiričen
    # (1/cos faktor na osnovi kota), ne pravi trikotniško-geometrijski
    # izračun iz dejanske baseline. Shranjeno za morebitno prihodnjo bolj
    # fizikalno natančno različico šumovnega modela.
    "baseline_mm": 230.0,
}


ZIVID_2_PLUS_M60 = {
    "manufacturer": "Zivid",
    "model": "Zivid 2+ M60",

    # Informativno, glej PHOTONEO_PHOXI_S["technology"] za razlago pomena.
    # Zivid uporablja fazno (phase) strukturirano svetlobo - ista splošna
    # kategorija kot Photoneo, torej velja isto opozorilo o 1/cos členu.
    "technology": "structured_light",

    # --- Ločljivost globinske mape (px) ---
    # main_trial.py: --camera_width_px / --camera_height_px
    "depth_map_resolution_px": {"width": 2448, "height": 2048},

    # --- Delovna razdalja pri sweet spot/fokusni razdalji (mm) ---
    # main_trial.py: offset_distance za top_camera_location() v main()-u
    # (--camera_offset_distance_mm).
    "working_distance_mm": 600.0,

    # --- Velikost skeniranega območja pri sweet (fokusni) razdalji (mm) ---
    # main_trial.py: --camera_fov_deg (glej compute_horizontal_fov_deg)
    "scanning_area_mm": {"width": 570.0, "height": 460.0},

    # --- Temporalni šum globine pri sweet spot razdalji (mm) ---
    # main_trial.py: --depth_noise_at_1m + --noise_reference_distance_m
    # (nastavite noise_reference_distance_m=0.6, depth_noise_at_1m=0.08)
    "temporal_noise_mm": 0.08,

    # --- Točka-do-točke razdalja pri sweet spot razdalji (mm) ---
    # Navzkrižna preverba: scanning_area_mm["width"] / depth_map_resolution_px
    # ["width"] = 570/2448 = 0.233mm, dovolj blizu tu podani 0.24mm.
    "point_to_point_distance_mm": 0.24,

    # --- Lokalna planarnost pri sweet spot razdalji (mm) ---
    # main_trial.py: --noise_spatial_correlation_px (v pikslih, ne mm -
    # pretvorba: lokalna_planarnost_mm / point_to_point_distance_mm =
    # 0.10/0.24 ≈ 0.4 px - manjše od Photoneo-jevega ~1.0px)
    "local_planarity_mm": 0.10,

    # --- Globalna planarnost pri sweet spot razdalji (mm) ---
    # main_trial.py: --global_planarity_mm
    "global_planarity_mm": 0.10,

    # --- Relativna natančnost razdalje (‰ razdalje) ---
    # main_trial.py: --distance_bias_permille
    # ("Dimension Trueness Error" pri fokusni razdalji, tipična temperatura:
    # < 0.20% = 2.0‰)
    "relative_distance_accuracy_permille": 2.0,

    # --- Baseline med projektorjem in kamero (mm) ---
    # NI PODANO v tu prejeti Zivid specifikaciji (za razliko od Photoneo) -
    # None namesto uganjene vrednosti; poiščite v polnem datasheetu, če
    # boste kdaj implementirali fizikalno natančnejši (baseline-based)
    # šumovni model, glej PHOTONEO_PHOXI_S["baseline_mm"].
    "baseline_mm": None,
}


MECHEYE_PRO_S = {
    "manufacturer": "Mech-Mind Robotics",
    "model": "Mech-Eye PRO S",

    # Informativno, glej PHOTONEO_PHOXI_S["technology"]. LED (ne laser)
    # osvetlitev - main_trial.py ne modelira vira osvetlitve/valovne
    # dolžine, zato je svetlobni vir sam po sebi izpuščen kot podatek.
    "technology": "structured_light",

    "depth_map_resolution_px": {"width": 1920, "height": 1200},

    # --- Delovna razdalja (mm) ---
    # Ta kamera ponuja 3 nastavljive "object focal distance" konfiguracije
    # (500/700/1000mm) - za TO postavitev je izbrana konfiguracija 500mm
    # (najkrajša delovna razdalja), zato so vsi podatki spodaj SAMO za to
    # konfiguracijo.
    # main_trial.py: offset_distance za top_camera_location() v main()-u
    # (--camera_offset_distance_mm).
    "working_distance_mm": 500.0,

    # --- Velikost skeniranega območja pri 500mm konfiguraciji (mm) ---
    # main_trial.py: --camera_fov_deg (glej compute_horizontal_fov_deg)
    "scanning_area_mm": {"width": 370.0, "height": 240.0},

    # --- Temporalni šum globine (mm) ---
    # main_trial.py: --depth_noise_at_1m + --noise_reference_distance_m
    # OPOZORILO: "Point Z-value repeatability" je bila v specifikaciji
    # podana SAMO pri far/1m (1000mm) konfiguraciji, ki jo TA izbira
    # (500mm) ne uporablja - vendor ni podal ločene vrednosti za 500mm.
    # Šum triangulacijskih senzorjev navadno RASTE z razdaljo, zato je ta
    # vrednost verjetno KONZERVATIVNA (pesimistična) ocena za 500mm -
    # dejanski šum pri 500mm je verjetno nižji. Uporabite kot začasen
    # placeholder, dokler ne dobite vendor-jeve meritve specifično za
    # 500mm konfiguracijo.
    "temporal_noise_mm": 0.05,   # izmerjeno @ 1m (far/1000mm konfiguracija), NE @ 500mm - glej opozorilo

    # --- Točka-do-točke razdalja (mm) - IZPELJANO, ne neposredno podano ---
    # Ta specifikacija ne poda "point-to-point distance"/"spatial
    # resolution" neposredno - spodnja vrednost je IZRAČUNANA iz FOV/
    # ločljivosti pri 500mm (širina_mm / širina_px = 370/1920 = 0.193mm),
    # torej manj zanesljiva kot neposredno izmerjena specifikacija.
    "point_to_point_distance_mm": 0.19,

    # --- Lokalna planarnost - NI PODANO v tej specifikaciji ---
    "local_planarity_mm": None,

    # --- Globalna planarnost - NI PODANO v tej specifikaciji ---
    # (glej "measurement_accuracy_vdi_vde_mm" spodaj za sorodno, a
    # metodološko drugačno metriko, ki JE podana)
    "global_planarity_mm": None,

    # --- Relativna natančnost razdalje - NI PODANO v tej obliki ---
    # (Photoneo/Zivid jo podata kot ‰ razdalje - ta specifikacija namesto
    # tega poda absolutno "Measurement accuracy (VDI/VDE)" v mm, glej spodaj)
    "relative_distance_accuracy_permille": None,

    # --- Natančnost meritve po VDI/VDE 2634 standardu (mm) - DODATNO
    # POLJE, ni pri Photoneo/Zivid ---
    # Konceptualno sorodno global_planarity_mm/relative_distance_accuracy
    # (skupna geometrijska natančnost čez delovno območje), a merjeno po
    # DRUGAČNI standardizirani metodologiji (VDI/VDE 2634, tipično "sphere
    # spacing error") - NE mešajte numerično z zgornjima dvema poljema pri
    # drugih kamerah. main_trial.py trenutno nima parametra niti za to
    # (isti razlog kot global_planarity_mm - samo naključni šum na piksel,
    # ne sistematična napaka). Enako opozorilo kot pri temporal_noise_mm
    # zgoraj: izmerjeno SAMO pri far/1m konfiguraciji, ne pri 500mm.
    "measurement_accuracy_vdi_vde_mm": 0.1,   # izmerjeno @ 1m (far/1000mm konfiguracija), NE @ 500mm - glej opozorilo

    "baseline_mm": 180.0,
}


def compute_horizontal_fov_deg(camera_spec: dict) -> float:
    """Izračuna horizontalni FOV (stopinje, main_trial.py-jev
    --camera_fov_deg) iz fizične širine skeniranega območja (scanning_area_mm)
    + delovne razdalje (working_distance_mm), pri kateri je bila ta širina
    izmerjena. main_trial.py (prek Open3D-jevega create_rays_pinhole)
    potrebuje kot, specifikacijski listi pa navadno podajo samo fizično
    širino območja pri dani razdalji. Standarden pretvorek za piramidno
    (pinhole) vidno polje: fov = 2 * atan((širina/2) / razdalja)."""
    width_mm = camera_spec["scanning_area_mm"]["width"]
    distance_mm = camera_spec["working_distance_mm"]
    return 2.0 * math.degrees(math.atan((width_mm / 2.0) / distance_mm))


# Register vseh znanih kamer, za lažje iskanje po imenu (npr. če se doda
# Photoneo M/L ali druge modele kasneje).
CAMERAS = {
    "photoneo_phoxi_s": PHOTONEO_PHOXI_S,
    "zivid_2plus_m60": ZIVID_2_PLUS_M60,
    "mecheye_pro_s": MECHEYE_PRO_S,
}
