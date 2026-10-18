"""Simulacija skena fiksno montirane 3D kamere iz CAD mreže (ray casting + šumovni model senzorja).

Uporablja se, dokler ni na voljo prava kamera - simulirani sken nadomesti
pravi sken v run_registration (glej main_trial.py).
"""
import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d as o3d

from point_clouds import OrganizedScan, crop_top_region, raycasting_scene
from workcell import CAMERA_LOCATION, Table


@dataclass(frozen=True)
class NoiseModel:
    """Parametri šuma globinskega senzorja (glej simulate_camera_scan za
    pomen posameznega člena). Vrstni red polj je del ključa predpomnilnika
    simuliranega target-a (glej main_trial._target_cache_key) - ne spreminjaj ga."""
    depth_noise_at_1m: float = 0.1             # std (mm) globinskega šuma pri razdalji noise_reference_distance_m in kotu vpadanja 0 (naravnost)
    noise_distance_power: float = 2.0          # eksponent rasti šuma z razdaljo (distance_m/noise_reference_distance_m ** to)
    noise_reference_distance_m: float = 1.0    # razdalja (m), pri kateri je depth_noise_at_1m dejansko izmerjen/kalibriran - glej depth_std_mm
    max_incidence_deg: float = 75.0            # zgornja meja kota vpadanja za 1/cos faktor, da ne eksplodira tik pred robom
    distance_bias_permille: float = 0.0        # sistematičen multiplikativen zamik razdalje, konstanten za CEL zajem - glej apply_distance_bias_mm
    global_planarity_mm: float = 0.0           # amplituda (razpon max-min, mm) gladkega sistematičnega popačenja čez celo FOV - glej generate_global_planarity_bias_mm; 0 = izklopljeno
    global_planarity_spatial_scale_px: Optional[float] = None   # sigma (v pikslih) glajenja tega polja - None = min(width_px,height_px)/4
    outlier_probability: float = 0.01          # delež pikslov, ki dobijo grobo napako (multipath/šibek SNR/napačno ujemanje) namesto navadnega šuma
    outlier_std_multiplier: float = 15.0       # kolikokrat širša porazdelitev za osamelce v primerjavi z navadnim šumom na tem pikslu
    flying_pixel_depth_jump_mm: float = 3.0    # globinski skok med sosednjima pikseloma, ki šteje za diskontinuiteto (rob dela/pina/šiva)
    flying_pixel_probability: float = 0.3      # verjetnost, da piksel na diskontinuiteti postane "leteč" (vmesna, neresnična globina)
    quantization_step_at_1m: float = 0.05      # korak kvantizacije (mm) pri 1m, skalira se z razdaljo enako kot šum
    noise_spatial_correlation_px: float = 1.5  # sigma (v pikslih) prostorske korelacije šuma - velikost tipičnega matching okna senzorja

    def depth_std_mm(self, distance_m: np.ndarray, incidence_cos: np.ndarray) -> np.ndarray:
        """Std (mm) globinskega šuma za triangulacijski senzor: raste z
        razdaljo (razmerje distance_m/noise_reference_distance_m na potenco
        noise_distance_power, umerjeno tako, da je rezultat PRI
        noise_reference_distance_m natanko depth_noise_at_1m) IN s kotom
        vpadanja - projicirana baseline med kamero in projektorjem/drugo kamero
        se skrči, ko je površina nagnjena stran od naravnost, kar poveča
        triangulacijsko negotovost približno kot 1/cos(kot vpadanja).
        incidence_cos je najprej omejen na cos(max_incidence_deg), da faktor
        1/cos ne eksplodira tik pred max_incidence_deg, kjer gre cos proti 0.

        OPOZORILO iz kamere.py: realni specsi triangulacijskih senzorjev
        (Photoneo, Zivid, Mech-Eye) podajo šum pri svoji dejanski delovni
        razdalji (npr. 384-520mm, 350-900mm, 500-600mm), NIKOLI pri 1m.
        Ekstrapolacija noise_distance_power=2.0 iz teh vrednosti do 1m je
        nezanesljiva, ker tako ozek razpon pogosto sploh ne razkrije pravega
        eksponenta rasti - zato noise_reference_distance_m nastavite na
        dejansko razdaljo, pri kateri je bil depth_noise_at_1m izmerjen (npr.
        0.442 za Photoneo-jev sweet spot; kontrolna_plosca.py to stori sama)."""
        max_incidence_cos = np.cos(np.radians(self.max_incidence_deg))   # cos praga - spodnja meja za clip
        incidence_factor = 1.0 / np.clip(incidence_cos, max_incidence_cos, 1.0)   # 1/cos faktor, omejen navzgor
        relative_distance = np.maximum(distance_m, 1e-6) / max(self.noise_reference_distance_m, 1e-6)   # razdalja relativno na referenčno razdaljo
        return self.depth_noise_at_1m * np.power(relative_distance, self.noise_distance_power) * incidence_factor   # končen std: osnova × (razdalja/referenca)^power × kot vpadanja


def gaussian_blur_2d(array: np.ndarray, sigma: float) -> np.ndarray:
    """Separabilno 2D Gaussovo glajenje, samo z numpy.
    Uporabljeno spodaj za prostorsko koreliranje šuma globine po pikslih:
    matching/korelacijsko okno pravega globinskega senzorja povpreči več
    sosednjih pikslov hkrati, zato napaka meritve sosednjih pikslov ni
    neodvisna (iid), kot bi to modeliral navaden np.random.normal na
    vsakem pikslu posebej. Validirano proti brute-force referenčni
    konvoluciji (max abs razlika ~1e-16)."""
    if sigma <= 0:   # brez glajenja - vrnemo nespremenjeno kopijo
        return array.copy()
    radius = max(1, int(np.ceil(3.0 * sigma)))   # polmer jedra v pikslih, glede na sigma
    x = np.arange(-radius, radius + 1)   # indeksi jedra okoli sredine
    kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))   # 1D Gaussovo jedro
    kernel /= kernel.sum()   # normaliziramo, da se uteži seštejejo v 1

    def convolve_along_axis(a: np.ndarray, axis: int) -> np.ndarray:   # 1D konvolucija vzdolž ene osi
        pad_width = [(0, 0)] * a.ndim
        pad_width[axis] = (radius, radius)   # dopolnimo rob samo na izbrani osi
        padded = np.pad(a, pad_width, mode="reflect")   # zrcalno dopolnimo robove, da glajenje ne potemni robov
        windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.size, axis=axis)   # drseča okna dolžine jedra
        return np.tensordot(windows, kernel, axes=([-1], [0]))   # uteženo povprečje vsakega okna z jedrom

    return convolve_along_axis(convolve_along_axis(array, axis=0), axis=1)   # najprej po vrsticah, nato po stolpcih = 2D glajenje


def apply_distance_bias_mm(distance_mm: np.ndarray, bias_permille: float) -> np.ndarray:
    """Uporabi SISTEMATIČNO (ne naključno na piksel) napako kalibracije
    razdalje: fiksen multiplikativen faktor (1 + bias_permille/1000),
    konstanten za CEL zajem in za VSAK piksel v njem - za razliko od
    NoiseModel.depth_std_mm-ovega šuma, ki je neodvisen na vsak piksel posebej.

    Ustreza kamere.py-jevemu "Relative distance accuracy (‰)"/"Dimension
    Trueness Error" specu - skupna, pretežno sistematična natančnost
    meritve razdalje kot delež same razdalje, ne naključni šum.

    bias_permille je DETERMINISTIČEN parameter tega zagona (enaka vloga
    kot --translation_x/y/z/--rotation_x/y/z_deg), ne naključno vzorčen ob
    vsakem klicu - namen je nadzorovano testiranje tolerance na znano
    sistematično napako (npr. --distance_bias_permille 1.25 za Photoneo-jev
    specificiran worst-case), ne simulacija naključne variacije med
    posameznimi senzorskimi enotami."""
    return distance_mm * (1.0 + bias_permille / 1000.0)   # + bias_permille‰ sistematičen zamik razdalje


def generate_global_planarity_bias_mm(shape: tuple[int, int],
                                      amplitude_mm: float,
                                      low_freq_sigma_px: float) -> np.ndarray:
    """Generira gladko, NIZKOFREKVENČNO 2D polje sistematičnega odstopanja
    (bias) čez celotno vidno polje - ustreza kamere.py-jevemu "Global
    planarity" specu (rahel upogib/popačenje pri skeniranju sicer ravne
    referenčne ploskve). Preostali šumovni model (heteroscedastičen
    Gaussov šum, osamelci, leteči piksli, kvantizacija) modelira SAMO
    naključen, ničelno-povprečen šum na posamezen piksel - nič od tega ne
    predstavlja gladke sistematične napake čez celotno sliko.

    Implementirano kot Gaussovo (gaussian_blur_2d) zglajen iid šum z zelo
    velikim sigma (v pikslih - velik del slike, "nizka frekvenca" v
    prostorskem smislu), rescaliran tako, da je razpon (max-min) natanko
    amplitude_mm - kar ustreza vendor-jevi definiciji "Global planarity"
    kot odstopanja od ravnine čez celotno FOV. Za en klic simulate_camera_scan
    (en "zajem") je to polje KONSTANTNO čez vse piksle tega zajema - isti
    fizični senzor na isti poziciji bi pri ponovljenem zajemu pokazal
    približno isto obliko tega popačenja, za razliko od preostalega šuma,
    ki je neodvisen med zajemi."""
    if amplitude_mm <= 0:   # brez sistematičnega popačenja - vrnemo ničelno polje
        return np.zeros(shape)
    raw = np.random.normal(size=shape)   # iid osnova, glajena spodaj v gladko nizkofrekvenčno polje
    smooth = gaussian_blur_2d(raw, low_freq_sigma_px)
    smooth -= smooth.mean()   # centriramo okoli 0, da je popačenje simetrično navzgor/navzdol
    current_range = smooth.max() - smooth.min()   # dejanski razpon pred rescaliranjem
    if current_range < 1e-12:   # degeneriran rob primer (praktično nemogoč, a izognemo se deljenju z 0)
        return np.zeros(shape)
    return smooth * (amplitude_mm / current_range)   # rescaliramo, da je razpon natanko amplitude_mm


def build_table_mesh(table: Table, up_axis: int, table_height: float) -> o3d.geometry.TriangleMesh:
    """Zgradi ploščato pravokotno mrežo (samo 2 trikotnika), ki predstavlja
    mizo pod delom - da ray casting zajame tudi ozadje mize okoli dela, ne
    le dela samega. table_height je pozicija vzdolž up_axis, table pa
    določa pravokotnik v preostalih dveh oseh."""
    plane_axes = [a for a in range(3) if a != up_axis]   # dve osi, v katerih leži miza
    corners_2d = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * table.half_extent + table.center   # 4 vogali pravokotnika
    vertices = np.zeros((4, 3))
    vertices[:, plane_axes] = corners_2d   # vstavimo vogale v pravo ravnino
    vertices[:, up_axis] = table_height   # vsi vogali na isti višini (miza je ravna)
    triangles = np.array([[0, 1, 2], [0, 2, 3]])   # dva trikotnika sestavita pravokotnik
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                     o3d.utility.Vector3iVector(triangles))
    mesh.compute_vertex_normals()   # izračunamo normalo (ravna ploskev, ena sama smer)
    return mesh   # vrnemo mrežo mize


def simulate_camera_scan(mesh: o3d.geometry.TriangleMesh,           # NEtransformirana, neobrezana CAD mreža (iz load_cad_mesh)
                         transform: np.ndarray,                     # poza dela glede na kamero (CAD -> kamera)
                         noise: NoiseModel,
                         fov_deg: float = 40.0,                     # horizontalno vidno polje kamere v stopinjah
                         width_px: int = 640,                       # širina simulirane slike v pikslih (= število žarkov po širini)
                         height_px: int = 480,                      # višina simulirane slike v pikslih (= število žarkov po višini)
                         up_axis: int = 2,                          # os, ki predstavlja "gor" - kamera je vedno nad delom vzdolž te osi, nikoli spodaj
                         top_fraction: float = 0.35,                # enako kot crop_top_region - omeji mrežo pred ray castingom na isto zgornje "sealing" območje, ki ga predstavlja source
                         disable_occlusion: bool = False,           # če je True, se preskoči ray casting in namesto tega gosto vzorči obrezano mrežo, za testiranje vpliva delne vidljivosti na registracijo
                         table: Optional[Table] = None              # če podana, je v sceni za ray casting tudi ravna miza pod delom, da target zajame tudi ozadje
                         ) -> tuple[o3d.geometry.PointCloud, Optional[OrganizedScan]]:
    """Simulira pravi globinski skener z ray castingom: iz pozicije kamere
    se skozi dejansko vidno polje sproži en žarek na simuliran piksel, prvi
    zadeti trikotnik pa JE vidna površina - natanko to, kar izmeri pravi
    strukturirano-svetlobni/laserski skener, ne glede na razdaljo kamere.

    Vrne (pcd, grid): pcd je sploščen oblak; grid je ista zašumljena 3D
    pozicija in maska zadetka v izvirni piksel MREŽNI obliki (glej
    OrganizedScan). disable_occlusion=True nima piksel mreže (gosto
    vzorčenje cele mreže, ne ray casting po pikslih) - v tem primeru je grid None.

    mesh je NEtransformirana, NEobrezana CAD mreža (iz load_cad_mesh) - ta
    funkcija jo obreže z isto crop_top_region logiko, ki jo run_registration
    uporabi za source, PRED ray castingom. To ni nujno "kar bi prava kamera
    videla" (izbokline pod top_fraction, ki gledajo neposredno navzgor in
    niso ničesar zakrite, bi bile realno vidne tudi njej) - a source (CAD
    referenca za primerjavo) je namenoma omejen na isto zgornje "sealing"
    območje, zato mora target predstavljati isto fizično območje, sicer
    primerjava source/target zajema različno geometrijo in se ujemanje
    podre (preverjeno empirično: brez tega obrezovanja target zajame tudi
    stranske konektorje/izbokline pod top_fraction, oceno naklona pa to
    popolnoma zavede).

    Šumovni model (na t_hit mreži, PRED flatten v seznam točk):
      1. heteroscedastičen std po razdalji IN kotu vpadanja
         (NoiseModel.depth_std_mm)
      2. prostorsko koreliran Gaussov šum s tem std (gaussian_blur_2d na
         iid N(0,1), renormaliziran na std=1, šele nato pomnožen z (1))
      3. osamelci (gross errors): mešanica dveh porazdelitev - del pikslov
         namesto (2) dobi neodvisen vzorec iz veliko širše N(0, std*outlier_std_multiplier)
      4. leteči piksli na globinskih diskontinuitetah: piksel na robu
         (skok globine med sosedi > flying_pixel_depth_jump_mm) z neko
         verjetnostjo dobi globino, interpolirano med svojo in sosednjo
         (bolj oddaljeno) globino, namesto prave zadete globine
      5. kvantizacija: zaokrožitev na korak, ki prav tako raste z razdaljo

    Zgornjih 5 členov je VSE naključnih, ničelno-povprečnih na posamezen
    piksel - noben ne predstavlja SISTEMATIČNE napake senzorja (glej
    kamere.py opozorila o "Global planarity" in "Relative distance
    accuracy"/"Dimension Trueness Error"). Dva dodatna, privzeto
    izklopljena (0) člena to pokrijeta:
      6. distance_bias_permille (apply_distance_bias_mm): DETERMINISTIČEN
         multiplikativen zamik cele izmerjene razdalje, konstanten za cel
         zajem - sistematična napaka umerjanja razdalje kot delež same
         razdalje.
      7. global_planarity_mm (generate_global_planarity_bias_mm): gladko,
         nizkofrekvenčno 2D polje, prav tako konstantno za cel zajem -
         sistematičen upogib/popačenje čez celotno vidno polje.
    Oba sta namenoma ločena od zgornjih 5 (naključnih) členov, ker gre za
    KVALITATIVNO drugačen mehanizem napake - konstanten znotraj enega
    zajema, ne neodvisen na vsak piksel/klic.

    Kamera je v izhodišču (workcell.CAMERA_LOCATION) in ima FIKSNO
    orientacijo - vedno gleda naravnost navzdol (vzdolž -up_axis), NE glede
    na to, kje dejansko pristane del (translation_x/y). To je namerna
    fizikalna lastnost: prava kamera je nepremično montirana in ne "sledi"
    delu, zato del, ki se dovolj odmakne vstran od kamerine osi, dejansko
    (delno ali povsem) izpade iz njenega vidnega polja (FOV). Če
    translation_x/y potisne CEL del izven FOV, ray casting ne zadene ničesar
    in funkcija dvigne RuntimeError - to je pričakovano in pravilno
    obnašanje, ne napaka nastavitve.

    table doda v sceno za ray casting še ravno mizo (build_table_mesh) na
    višini, kjer se NEobrezan del dejansko dotika mize - žarki, ki zgrešijo
    del, tako namesto v prazno (brez zadetka) zadenejo mizo, target pa dobi
    tudi ozadje okoli dela, ne le del sam. Poenostavitev: miza je dodana v
    sceno SKUPAJ z že obrezano (top_fraction) mrežo dela, ne s celotno
    neobrezano mrežo, zato lahko na mestih, kjer bi v resnici spodnji
    (obrezani) del ohišja zakril pogled na mizo, ray casting mizo napačno
    pokaže kot vidno - sprejemljivo za diagnostično testiranje, ne pa
    fizikalno popolnoma natančno.
    """
    camera_location = CAMERA_LOCATION
    mesh = copy.deepcopy(mesh)          # ne spreminjamo originala
    mesh.transform(transform)           # uporabimo transformacijsko matriko na CAD mrežo, da dobimo ciljno mrežo v svetovnih koordinatah

    table_mesh = None
    if table is not None:
        # Višino mize moramo izračunati iz NEobrezane mreže (preden jo
        # crop_top_region skrči na zgornje območje), saj je prav spodnji
        # del (ki ga obrezovanje odstrani) tisti, ki se dotika mize.
        table_height = mesh.get_min_bound()[up_axis]   # miza je na nasprotni strani od kamere
        table_mesh = build_table_mesh(table, up_axis, table_height)   # zgradimo mrežo mize

    mesh = crop_top_region(mesh, top_fraction=top_fraction, up_axis=up_axis)   # obrežemo na isto zgornje območje kot source

    if disable_occlusion:
        # Preskoči ray casting in gosto vzorči CELO (transformirano) mrežo -
        # namerno VEČ pokritosti, kot jo pravi skener kadarkoli dobi, da lahko
        # testiramo, ali je delna vidljivost (okluzija + vidno polje) - ne
        # gostota ali kadriranje - tisto, kar povzroča odstopanje dobljene
        # transformacije od referenčne. Brez piksel mreže tu ni prostorske
        # korelacije/letečih pikslov/kvantizacije - samo heteroscedastičen
        # (razdalja+kot vpadanja) šum in osamelci, izotropno na vsako točko.
        pcd = mesh.sample_points_poisson_disk(number_of_points=width_px * height_px)   # gosto vzorčenje cele obrezane mreže
        points = np.asarray(pcd.points)   # točke kot numpy tabela
        distance_m = np.linalg.norm(points - camera_location, axis=-1) / 1000.0   # razdalja vsake točke od kamere v metrih
        if pcd.has_normals() and len(pcd.normals) == len(points):   # imamo uporabne normale za oceno kota vpadanja
            view_dir = camera_location - points   # smer od točke proti kameri
            view_dir /= np.clip(np.linalg.norm(view_dir, axis=-1, keepdims=True), 1e-12, None)   # normaliziramo na enotsko dolžino
            incidence_cos = np.clip(np.abs(np.sum(view_dir * np.asarray(pcd.normals), axis=-1)), 1e-3, 1.0)   # kosinus kota med pogledom in normalo
        else:   # ni normal - privzamemo naravnost gledanje (brez faktorja kota)
            incidence_cos = np.ones(len(points))
        noise_std_mm = noise.depth_std_mm(distance_m, incidence_cos)   # std šuma za vsako točko
        # distance_bias_permille/global_planarity_mm namerno NISO uporabljena
        # v tej veji: to je diagnostičen način brez piksel mreže, dodani šum
        # je izotropen na vsako točko (ne vzdolž smeri pogleda kamere), zato
        # sistematičen RADIALNI zamik razdalje (distance_bias_permille) in
        # 2D slikovno-ravninsko polje (global_planarity_mm) tu nimata
        # smiselne, jasno definirane vloge - oba veljata samo za glavno,
        # fizikalno podrobno ray-casting vejo spodaj.
        is_outlier = np.random.random(len(points)) < noise.outlier_probability   # katere točke postanejo osamelci
        effective_std = np.where(is_outlier, noise_std_mm * noise.outlier_std_multiplier, noise_std_mm)   # večji std za osamelce
        points += np.random.normal(size=points.shape) * effective_std[:, None]   # dodamo šum vsaki točki
        pcd.points = o3d.utility.Vector3dVector(points)   # shranimo zašumljene točke nazaj v oblak
        return pcd, None   # diagnostičen način brez ray castinga - ni piksel mreže

    scene = raycasting_scene(mesh, *([table_mesh] if table_mesh is not None else []))   # del (+ miza - žarki, ki zgrešijo del, zadenejo njo)

    # Katerakoli os razen up_axis deluje kot referenčni "up" vektor virtualne
    # kamere (mora biti le ne-vzporedna s smerjo pogleda eye->center) -
    # (up_axis + 1) % 3 izbere eno izmed preostalih dveh, dosledno.
    up_vector = np.zeros(3)
    up_vector[(up_axis + 1) % 3] = 1.0   # izberemo os, ki ni up_axis, kot referenčni "up" kamere

    # FIKSNA smer pogleda (vzdolž -up_axis) - NE center=mesh.get_center(): `mesh` je tu že
    # TRANSFORMIRAN, zato bi to kamero vedno znova "naravnalo" naravnost na
    # del in tiho izničilo ravno to, kar naj bi testiranje X/Y odmika
    # (--translation_x/--translation_y) pokazalo.
    look_direction = np.zeros(3)
    look_direction[up_axis] = -1.0   # vedno navzdol vzdolž up_axis
    look_at_point = camera_location + look_direction   # poljubna točka v tej fiksni smeri - oddaljenost od kamere ne vpliva na smer žarkov

    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=fov_deg, center=look_at_point, eye=camera_location, up=up_vector,
        width_px=width_px, height_px=height_px)   # ustvarimo mrežo žarkov za virtualno kamero s FIKSNO smerjo pogleda
    result = scene.cast_rays(rays)   # sprožimo žarke proti sceni in dobimo zadetke

    t_hit = result["t_hit"].numpy()             # razdalja do prvega zadetka na žarek, inf če žarek ni zadel ničesar
    hit_mask = np.isfinite(t_hit)   # kateri piksli so dejansko zadeli mrežo
    if not hit_mask.any():   # noben žarek ni zadel ničesar - napačna nastavitev kamere
        raise RuntimeError("Ray casting ni zadel ničesar - camera_location/fov_deg/width_px/"
                           "height_px verjetno ne kažejo na del")

    rays_np = rays.numpy()   # žarki kot numpy tabela (izvor + smer)
    ray_dirs = rays_np[..., 3:6]   # smerni del vsakega žarka
    dir_length = np.linalg.norm(ray_dirs, axis=-1)   # dolžina smernega vektorja vsakega žarka

    # Non-hit piksli nosijo t_hit/razdaljo = inf skozi vsak šumovni člen
    # spodaj (zavržejo se šele čisto na koncu prek hit_mask, glej
    # points = hit_points[hit_mask]) - inf v kombinaciji z množenjem z 0 ali
    # drugim inf (npr. depth_noise_at_1m * inf, inf/inf pri kvantizaciji) je
    # matematično neškodljiv nan v celicah, ki se tako ali tako zavržejo,
    # a numpy o tem privzeto vseeno opozori; to opozorilo je pričakovano in
    # zatrto za ves ta blok, namesto da bi kodo prestrukturirali, saj bi
    # zgodnje maskiranje non-hit pikslov po nepotrebnem zapletlo vsako
    # vektorizirano operacijo spodaj, brez razlike v obnašanju.
    with np.errstate(invalid="ignore", divide="ignore"):   # zatremo pričakovana opozorila inf/nan
        # Enota t_hit je dolžina (nenormaliziranega, perspektivno
        # spremenljivega) smernega vektorja tega piksla, ne mm neposredno -
        # pretvorba v pravo evklidsko razdaljo tukaj omogoči, da je vsak
        # šumovni člen spodaj podan v pravih mm, ne v tej po pikslih
        # spremenljivi enoti.
        distance_mm = np.where(hit_mask, t_hit * dir_length, np.inf)   # prava razdalja v mm, inf za non-hit
        distance_mm = apply_distance_bias_mm(distance_mm, noise.distance_bias_permille)   # sistematičen (ne naključen) zamik kalibracije razdalje, glej apply_distance_bias_mm - neškodljivo tudi za inf (non-hit) piksle
        distance_m = distance_mm / 1000.0   # ista razdalja v metrih (za formulo šuma), zdaj vključno z distance_bias_permille

        primitive_normals = result["primitive_normals"].numpy()   # normala zadetega trikotnika na vsak piksel
        dir_unit = ray_dirs / np.clip(dir_length[..., None], 1e-12, None)   # normalizirana smer žarka
        # abs() se izogne dvoumnosti predznaka normale (odvisno od
        # orientacije trikotnika) - za šumovni model je pomembno le, kako
        # "naravnost" je površina obrnjena proti žarku, ne v katero smer
        # normala dejansko kaže.
        incidence_cos = np.clip(np.abs(np.sum(dir_unit * primitive_normals, axis=-1)), 1e-3, 1.0)   # kosinus kota vpadanja

        noise_std_grid_mm = np.where(hit_mask, noise.depth_std_mm(distance_m, incidence_cos), 0.0)   # std šuma za vsak piksel (0 za non-hit)

        # Globalna planarnost: gladko, nizkofrekvenčno sistematično polje
        # čez CELO sliko (KONSTANTNO za ta zajem, ne naključno na piksel kot
        # preostali šum) - glej generate_global_planarity_bias_mm docstring
        # in kamere.py "Global planarity" opozorilo. 0 amplitude (privzeto)
        # vrne ničelno polje, torej brez sprememb obnašanja, če ni nastavljeno.
        global_planarity_bias_mm = generate_global_planarity_bias_mm(
            t_hit.shape,
            amplitude_mm=noise.global_planarity_mm,
            low_freq_sigma_px=(noise.global_planarity_spatial_scale_px
                              if noise.global_planarity_spatial_scale_px is not None
                              else min(width_px, height_px) / 4.0))

        # Prostorsko korelirani heteroscedastičen šum globine: NAJPREJ
        # zgladimo iid enotski šum (matching okno pravega senzorja meša
        # sosednje piksle, zato njihove napake niso neodvisne), nato
        # renormaliziramo izgubo variance zaradi glajenja nazaj na std=1,
        # ŠELE NATO pomnožimo s std vsakega piksla - skaliranje pred
        # glajenjem bi glajenje spralo prav tisto heteroscedastičnost
        # (kot vpadanja/oddaljenost), ki naj bi jo ohranilo.
        raw_noise = np.random.normal(size=t_hit.shape)   # neodvisen enotski šum, en na piksel
        correlated_noise = gaussian_blur_2d(raw_noise, noise.noise_spatial_correlation_px)   # prostorsko zgladimo
        correlated_std = correlated_noise.std()   # dejanski std po glajenju (glajenje ga zmanjša)
        if correlated_std > 1e-12:
            correlated_noise = correlated_noise / correlated_std   # renormaliziramo nazaj na std=1
        depth_noise_mm = correlated_noise * noise_std_grid_mm   # šele zdaj pomnožimo s pravim std po pikslih

        # Osamelci (multipath, šibek SNR, napačna stereo/strukturirano-
        # svetlobna ujemanja): resnično drugačen mehanizem napake kot
        # navaden šum senzorja, ne "malo več istega šuma" - del pikslov
        # dobi neodvisen vzorec iz veliko širše Gaussove porazdelitve
        # NAMESTO (ne poleg) koreliranega šuma zgoraj.
        is_outlier = np.random.random(t_hit.shape) < noise.outlier_probability   # kateri piksli postanejo osamelci
        outlier_noise_mm = np.random.normal(size=t_hit.shape) * (noise_std_grid_mm * noise.outlier_std_multiplier)   # veliko širši šum za osamelce
        depth_noise_mm = np.where(is_outlier, outlier_noise_mm, depth_noise_mm)   # zamenjamo šum osamelcev, ne dodamo poleg

        distance_mm_noisy = distance_mm + depth_noise_mm + global_planarity_bias_mm   # razdalja z dodanim naključnim ŠUMOM + sistematičnim popačenjem (global planarity)

        # Leteči piksli: na globinski diskontinuiteti (rob dela, rob pina,
        # šiv) matching okno pravega senzorja hkrati zajame oba globinska
        # nivoja in lahko poroča interpolirano, fizično neobstoječo globino
        # med njima - "lebdeče" točke, vidne na robovih pravih skenov, ki
        # jih navaden Gaussov šum na točko nikoli ne ustvari. Primerja le
        # pare sosedov, ki sta OBA zadetek (hit_mask & shifted_hit), zato
        # meja med objektom in ozadjem - kjer je "ozadje" preprosto
        # odsotnost zadetka, ne druga prava površina - nikoli napačno ne
        # šteje kot skok.
        finite_distance = np.where(hit_mask, distance_mm, 0.0)   # razdalja, 0 za non-hit (namesto inf, za lažjo primerjavo)
        depth_jump = np.zeros_like(hit_mask)   # maska diskontinuitet, začetno prazna
        max_neighbor_distance = finite_distance.copy()   # tekoč maksimum razdalje med sabo in sosedi
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:   # 4 sosedske smeri (gor/dol/levo/desno)
            shifted_distance = np.roll(finite_distance, (dy, dx), axis=(0, 1))   # razdalja soseda v tej smeri
            shifted_hit = np.roll(hit_mask, (dy, dx), axis=(0, 1))   # ali je ta sosed zadetek
            both_hit = hit_mask & shifted_hit   # oba piksla morata biti zadetka
            depth_jump |= both_hit & (np.abs(finite_distance - shifted_distance) > noise.flying_pixel_depth_jump_mm)   # velik skok globine = diskontinuiteta
            max_neighbor_distance = np.where(shifted_hit, np.maximum(max_neighbor_distance, shifted_distance),
                                             max_neighbor_distance)   # obdržimo najbolj oddaljenega soseda
        is_flying = depth_jump & hit_mask & (np.random.random(t_hit.shape) < noise.flying_pixel_probability)   # kateri piksli dejansko postanejo leteči
        if is_flying.any():
            blend = np.random.uniform(0.2, 0.8, size=t_hit.shape)   # naključen delež mešanja med bližnjo in daljno globino
            flying_distance = distance_mm_noisy * (1.0 - blend) + max_neighbor_distance * blend   # vmesna, neresnična globina
            distance_mm_noisy = np.where(is_flying, flying_distance, distance_mm_noisy)   # uveljavimo samo na letečih pikslih

        # Kvantizacija: ADC/disparity ločljivost senzorja poroča globino v
        # diskretnih korakih, ne zvezno - zaokrožitev PO vsem šumu zgoraj
        # (ne prej, sicer bi šum na vrhu spet zabrisal korake). Velikost
        # koraka se z razdaljo skalira enako kot std šuma.
        step_mm = np.maximum(
            noise.quantization_step_at_1m * np.power(np.maximum(distance_m, 1e-6), noise.noise_distance_power), 1e-9)   # korak kvantizacije za vsak piksel
        distance_mm_noisy = np.round(distance_mm_noisy / step_mm) * step_mm   # zaokrožimo na najbližji korak

        t_hit_noisy = np.where(hit_mask, distance_mm_noisy / np.maximum(dir_length, 1e-12), t_hit)   # nazaj v t_hit enote

    hit_points = rays_np[..., :3] + rays_np[..., 3:6] * t_hit_noisy[..., None]  # izvor žarka + t_hit_noisy * smer žarka = 3D točka zadetka

    pcd = o3d.geometry.PointCloud()   # ustvarimo nov, prazen oblak točk
    pcd.points = o3d.utility.Vector3dVector(hit_points[hit_mask])   # obdržimo samo dejanske zadetke
    return pcd, OrganizedScan(hit_points, hit_mask)   # simuliran sken + ista mreža (za filter_grazing_incidence_points_from_grid)
