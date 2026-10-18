"""Nalaganje CAD/skena, obrezovanje na "sealing" regijo, filtri in predobdelava oblakov točk.

Vsi oblaki so v mm. Source = Poisson-disk vzorčen CAD (load_cad_model),
target = sken (pravi iz datoteke ali simuliran v scan_simulation.py).
"""
import copy
import hashlib
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import open3d as o3d

PREPROCESS_CACHE_DIR = Path(__file__).resolve().parent / ".preprocess_cache"   # mapa za predpomnjene predobdelane oblake
# To številko povečaj vsakič, ko preprocess_point_cloud() ali karkoli, kar
# kliče, spremeni obnašanje pri enakih vhodih (npr. ocena normal) - cache
# ključ pokriva samo *parametre*, ne verzije kode, zato bi popravek
# brez spremembe parametra sicer še naprej neomejeno servisiral star,
# napačen predpomnjen rezultat namesto ponovnega izračuna.
PREPROCESS_CACHE_VERSION = 7   # trenutna verzija formata predpomnilnika - popravek: preserve_existing_normals (ohranitev točnih CAD mesh normal namesto vedno ponovne ocene, glej ensure_oriented_normals)

CAD_COLOR = [1.0, 0.706, 0.0]   # oranžna


class OrganizedScan(NamedTuple):
    """Sken v izvirni piksel MREŽNI obliki: points (H,W,3) in hit_mask (H,W).

    Sploščen target oblak je vedno natanko points[hit_mask] v tem vrstnem
    redu - filter_grazing_incidence_points_from_grid to izkoristi za O(n)
    oceno normal iz sosednjih pikslov namesto KD-tree iskanja."""
    points: np.ndarray
    hit_mask: np.ndarray

    def keep_only(self, keep_mask: np.ndarray) -> "OrganizedScan":
        """Odstrani piksle, ki jih keep_mask (poravnan s sploščenim oblakom,
        torej s points[hit_mask]) zavrže - tako mreža ostane usklajena s
        target-om tudi po filtru, ki deluje na sploščenem oblaku."""
        hit_flat = self.hit_mask.reshape(-1)
        hit_indices = np.where(hit_flat)[0]
        new_hit_flat = hit_flat.copy()
        new_hit_flat[hit_indices[~keep_mask]] = False
        return OrganizedScan(self.points, new_hit_flat.reshape(self.hit_mask.shape))


def load_cad_mesh(mesh_path: Path) -> o3d.geometry.TriangleMesh:
    # simulate_camera_scan() pošilja žarke proti pravim trikotnikom, ne
    # proti Poisson-disk vzorčenemu oblaku točk, ki ga vrne load_cad_model()
    # - branje samega STL-ja je hitro (za razliko od Poisson-disk
    # vzorčenja), zato to ne potrebuje lastnega predpomnilnika.
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))   # preberemo STL datoteko v mrežo trikotnikov
    if mesh.is_empty():   # datoteka ne obstaja ali je poškodovana
        raise FileNotFoundError(f"CAD mesh not found: {mesh_path}")
    mesh.compute_vertex_normals()   # izračunamo normale mreže
    return mesh   # vrnemo celotno, neobrezano mrežo


def load_cad_model(mesh_path: Path, n_points: int = 200000, use_cache: bool = True) -> o3d.geometry.PointCloud:  # pot do modela + število točk
    # Poisson-disk vzorčenje na natančnem STL-ju je najpočasnejši del vsakega
    # zagona, poleg tega ni ponovljivo samo po sebi - Open3D-jev vzorčevalnik
    # ima svoj notranji generator naključnih števil, ki ga np.random.seed()
    # ne nadzoruje, zato se število točk med zagoni rahlo spreminja.
    # Predpomnjenje vzorčenega oblaka na disk (ključ = n_points, neveljavno,
    # če se STL spremeni) reši oboje: takojšen ponovni zagon po prvem
    # poganjanju in enake točke vsakič po tem.
    # OPOMBA glede privzetega n_points=200000: to je vzorčeno čez CELOTNO
    # ohišje, preden crop_top_region obdrži ~35% in simulirana okluzija
    # obdrži še del tega, zato realno-vidno/relevantno območje konča pri
    # približno 1/15 tega (~13k točk pri 200k). Ta gostota (pod-mm razmik
    # točk čez ~150mm del) je tisto, kar dejansko dostavi pravi industrijski
    # strukturirano-svetlobni/laserski skener.
    # Poisson-disk pri 200k stane nekaj minut PRVI zagon, nato predpomnilnik
    # naredi vsak naslednji zagon takojšen.
    cache_path = mesh_path.with_name(f"{mesh_path.stem}.sampled_{n_points}pts.ply")   # pot do cache datotetke, da ni treba vsakič vzorčit CAD fila
    if use_cache and cache_path.exists() and cache_path.stat().st_mtime >= mesh_path.stat().st_mtime:  # cache obstaja in je novejši od STL-ja
        pcd = o3d.io.read_point_cloud(str(cache_path))
        if not pcd.is_empty():   # predpomnjen oblak je uporaben
            pcd.paint_uniform_color(CAD_COLOR)
            return pcd   # vrnemo predpomnjen oblak, brez ponovnega vzorčenja

    pcd = load_cad_mesh(mesh_path).sample_points_poisson_disk(number_of_points=n_points)  # vzorčenje točk iz mreže s Poisson disk metodo, da dobimo oblak točk
    if use_cache:   # shranimo na disk za naslednji zagon
        o3d.io.write_point_cloud(str(cache_path), pcd)

    pcd.paint_uniform_color(CAD_COLOR)
    return pcd   # vrnemo sveže vzorčen oblak točk


def raycasting_scene(*meshes: o3d.geometry.TriangleMesh) -> o3d.t.geometry.RaycastingScene:
    """RaycastingScene (BVH nad TOČNIMI trikotniki) - simulate_camera_scan ga
    uporablja za sprožanje žarkov, point_to_mesh_icp za poizvedbe najbližje
    točke na CAD površini."""
    scene = o3d.t.geometry.RaycastingScene()
    for mesh in meshes:
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def check_cad_frame(mesh: o3d.geometry.TriangleMesh,
                    max_origin_distance_mm: float = 100.0) -> None:
    """Izpiše, kje leži izhodišče CAD datoteke (0,0,0) glede na sam del, in
    opozori na tipične napake pri izvozu iz SolidWorksa.

    ZAKAJ: končna transformacija iz run_registration je poza CAD-ovega
    LASTNEGA koordinatnega sistema glede na kamero - robot pa ima pot nanosa
    tesnila sprogramirano v dogovorjenem KS (Coordinate System1 na
    poenostavljenih modelih). Rezultat je uporaben za robota SAMO, če je bil
    STL izvožen v tem istem KS (SolidWorks: Save As -> STL -> Options ->
    "Output coordinate system" = Coordinate System1). Kje točno je to
    izhodišče, za natančnost registracije NI pomembno (ista toga
    transformacija, glej transform_with_pivot) - pomembno je le, da ga vsi
    uporabljamo ENAKEGA. Ta funkcija tega ne more preveriti, lahko pa ujame
    očitne napake: napačne enote (metri namesto mm) ali izhodišče daleč
    stran od dela (izvoz v privzetem KS sestava namesto v Coordinate System1)."""
    min_bound, max_bound = mesh.get_min_bound(), mesh.get_max_bound()
    extent = max_bound - min_bound
    origin_to_bbox = float(np.linalg.norm(np.maximum(0.0, np.maximum(min_bound, -max_bound))))   # 0, če je izhodišče znotraj očrtane škatle dela
    print(f"  CAD frame: bbox min={np.round(min_bound, 2)}, max={np.round(max_bound, 2)}, "
          f"extent={np.round(extent, 2)}; origin (0,0,0) is {origin_to_bbox:.1f} mm outside the part's bounding box")
    if extent.max() < 1.0:
        print("  WARNING: CAD extent < 1 unit - the STL was probably exported in METERS, "
              "the pipeline expects millimeters (SolidWorks STL export options -> Unit: Millimeters)")
    if origin_to_bbox > max_origin_distance_mm:
        print(f"  WARNING: CAD origin is more than {max_origin_distance_mm:.0f} mm from the part - "
              f"was the STL exported in the agreed coordinate system (Coordinate System1)?")


def crop_top_region(pcd: o3d.geometry.PointCloud,  # vhodni oblak točk (ali mreža), ki ga želimo odrezat
                    top_fraction: float = 0.35,     # obdrži 35% zgornjega dela oblaka točk
                    up_axis: int = 2,               # os, ki predstavlja 'gor' (0=x, 1=y, 2=z), privzeto je z os
                    edge_margin: float = 0.0) -> o3d.geometry.PointCloud:   # dodatno izključi pas te širine tik ob (umetni) meji obreza - glej docstring spodaj
    """edge_margin (privzeto 0.0): dodatno izključi pas širine edge_margin
    TIK OB meji obreza (cutoff spodaj), ne od prave/fizične strani oblaka
    (max_bound stran - to je resnična zgornja ploskev dela, ne umeten rob).

    Zakaj: cutoff sam je UMETNA meja, ki jo ustvari to obrezovanje (del/mreža
    se v resnici razteza dlje, mi ga tu preprosto odrežemo) - v target-u
    (dejanski sken) lahko na tej višini sensor vidi material POD mejo (ali pa
    ne, odvisno od okluzije), CAD (source) pa se na cutoff natanko konča.
    Točke source-a znotraj edge_margin od cutoff-a zato sistematično
    (pristransko, ne naključno) vlečejo ICP rešitev - klicatelj naj to
    uporabi SAMO na strani, ki jo primerja z drugim oblakom (source v
    run_registration), ne na mesh-u, iz katerega se simulira/generira target
    (simulate_camera_scan-ov lasten crop_top_region klic) - target naj
    ostane fizikalno resničen, obreže naj se samo to, S ČIM ga primerjamo."""
    min_bound = pcd.get_min_bound()         # poišče skrajne točke oblaka točk
    max_bound = pcd.get_max_bound()   # zgornja meja oblaka točk po vseh oseh
    cutoff = max_bound[up_axis] - top_fraction * (max_bound[up_axis] - min_bound[up_axis])   # meja, pod katero obrežemo
    crop_min = min_bound.copy()   # spodnja meja izreza - dvignjena na cutoff
    crop_min[up_axis] = cutoff + edge_margin   # + edge_margin: izključi pas tik nad umetno mejo
    bbox = o3d.geometry.AxisAlignedBoundingBox(crop_min, max_bound)   # škatla, ki zajame samo zgornji del
    return pcd.crop(bbox)   # obrežemo oblak točk na to škatlo


def compute_native_reference_point(mesh: o3d.geometry.TriangleMesh,
                                   top_fraction: float,
                                   up_axis: int) -> np.ndarray:
    """Izračuna referenčno točko (X,Y,Z) v NATIVE (NEtransformiranem) CAD
    okviru dela - geometrijsko središče zgornjega ("sealing") prereza, ki
    ga simulate_camera_scan dejansko zajame (ista crop_top_region logika).

    ZAKAJ JE TO POTREBNO: --translation_x/y/z naj bi opisovali dejansko
    pozicijo/razdaljo TE skenirane površine glede na fiksno kamero
    (translation_x=y=0 => del na sredini kamerinega pogleda,
    translation_z=-D => skenirana površina D mm pod kamero) - a CAD-ov
    lastni izvor koordinat NI nujno poravnan niti s centrom tega prereza
    (X/Y) niti z njegovo globinsko sredino (Z). Brez tega popravka
    translation_x=0,y=0 NE bi postavil dela na sredino pogleda in
    translation_z NE bi ustrezal pravi razdalji do skenirane površine -
    simulirano vidno polje bi bilo ožje/širše in drugače centrirano, kot ga
    kamerina specifikacija (kamere.py) dejansko obljublja.

    run_registration zato referenčno pozo sestavi z rotacijo OKOLI te točke
    (transform_with_pivot) - en sam, dosleden popravek v eni matriki, ki jo
    simulacija in poročanje napake enako uporabita."""
    cropped = crop_top_region(mesh, top_fraction=top_fraction, up_axis=up_axis)   # ista "sealing" regija, ki jo simulate_camera_scan dejansko zajame
    return np.asarray(cropped.get_center())   # geometrijsko središče te regije v CAD-ovem lastnem (netransformiranem) okviru


# TO JE SINTETIČNA KAMERA, KI JO BOMO POTREBOVAL TUDI PRI REALNEM SKENIRANJU
# NAMEN JE, DA SE NORMALE OBLAKA TOČK CAD MODELA OBRNEJO V ISTO SMER KOT BODO OD SKENIRANGA POINT CLOUDA
# ČE TE FUNKCIJE NI SO NORMALE OBLAKA TOČK SOURCE MODELA OBRNJENE V NAPAČNO SMER ( NE MORE PRIMERJAT S TARGET MODELOM)
def top_camera_location(pcd: o3d.geometry.PointCloud,  # vhodni oblak točk, ki ga želimo uporabiti za določitev lokacije kamere
                        up_axis: int = 2,   # os, ki predstavlja "gor" (0=x, 1=y, 2=z), privzeto je z os
                        offset_factor: float = 5.0,     # faktor, ki določa, kako daleč nad oblakom točk bo kamera postavljena (v enotah dolžine oblaka točk) - uporabljen samo, če offset_distance ni podan
                        offset_distance: Optional[float] = None) -> np.ndarray:  # če je podan (v mm), je kamera postavljena na to fiksno razdaljo nad/pod oblakom točk, NE glede na velikost oblaka
    center = pcd.get_center()           # prostorska sredina oblaka točk
    extent = np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound())     # razlika med največjo in najmanjšo mejo oblaka točk, da dobimo velikost oblaka točk
    camera_location = np.array(center)   # začnemo s pozicijo v sredini oblaka
    distance = offset_distance if offset_distance is not None else offset_factor * max(extent[up_axis], 1.0)   # izberemo fiksno ali od velikosti odvisno razdaljo
    camera_location[up_axis] = pcd.get_max_bound()[up_axis] + distance   # kamera nad zgornjo mejo
    return camera_location   # vrnemo izračunano pozicijo kamere


def load_real_scan(scan_path: Path) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(scan_path))   # preberemo datoteko z realnim skenom
    if pcd.is_empty():   # datoteka ne obstaja ali je prazna
        raise FileNotFoundError(f"Real scan not found: {scan_path}")
    return pcd   # vrnemo prebran oblak točk


def load_and_merge_real_scans(scan_paths: list[Path]) -> o3d.geometry.PointCloud:
    """Združi več zaporednih skenov istega STATIČNEGA dela v en oblak točk,
    brez kakršnekoli med-sken registracije - ker gre za isti fiksno
    montiran senzor, ki skenira isto nepremaknjeno postavitev (del na mizi
    se med zajemi ne premakne), so zaporedni zajemi že v istem
    koordinatnem sistemu, zgolj konkatenacija zadostuje.

    Namerno NE downsampliramo tukaj: kasnejši voxel_down_sample v Phase 3
    predobdelavi (preprocess_point_cloud) POVPREČI vse točke, ki padejo v
    isto celico, kar samo od sebe povpreči točke iz različnih zajemov, ki
    merijo isto fizično lokacijo - temporalno povprečenje nekoreliranega
    šuma senzorja med N zaporednimi zajemi zniža njegov std za približno
    sqrt(N), praktično zastonj za del, ki med zajemi miruje na mizi.
    Ločen downsample tukaj bi to preprosto podvojil brez dodatne koristi."""
    combined = o3d.geometry.PointCloud()
    total_points = 0
    for scan_path in scan_paths:
        pcd = load_real_scan(scan_path)   # vsak sken naložimo in preverimo posebej (load_real_scan sam preveri prazno/manjkajočo datoteko)
        total_points += len(pcd.points)
        combined += pcd   # združimo v en oblak - že v istem koordinatnem sistemu, brez potrebe po registraciji
    print(f"  Merged {len(scan_paths)} scans ({[p.name for p in scan_paths]}): "
          f"{total_points} points total (temporal noise averaging happens naturally "
          f"in Phase 3's voxel downsampling)")
    return combined   # vrnemo združen, še ne zmanjšan oblak - Phase 3 ga bo voxel-zmanjšala in s tem povprečila šum


def load_real_scans(scan_paths: list[Path]) -> o3d.geometry.PointCloud:
    """Naloži pravi sken (ali več zaporednih zajemov, glej
    load_and_merge_real_scans) in odstrani statistične osamelce - točke, ki
    močno odstopajo od svojih k najbližjih sosedov (napake skeniranja)."""
    target = load_and_merge_real_scans(scan_paths) if len(scan_paths) > 1 else load_real_scan(scan_paths[0])
    filtered, _ = target.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
    return filtered


def filter_grazing_incidence_points(pcd: o3d.geometry.PointCloud,
                                    camera_location: np.ndarray,
                                    max_incidence_deg: float = 80.0,
                                    normal_radius: float = 1.0,
                                    max_nn: int = 30) -> o3d.geometry.PointCloud:
    """Odstrani domnevne leteče piksle na podlagi kota vpadanja: točka,
    katere lokalna normala je skoraj pravokotna na smer pogleda kamere
    (kot vpadanja nad max_incidence_deg, tipično ~80 stopinj), je z veliko
    verjetnostjo leteč piksel na robu (rob dela/pina/šiva - natanko to
    fizikalno ozadje simulira flying_pixel model v simulate_camera_scan)
    namesto prave meritve površine, saj pri tako strmem kotu senzorjevo
    matching okno hkrati zajame obe sosednji globinski ravni.

    Statistični outlier removal (razdalja do k najbližjih sosedov, glej
    load_real_scans) tega zanesljivo NE pokrije: leteči piksli na robu
    tvorijo LOKALNO GOSTO skupino vzdolž celotnega roba (interpolirane
    vrednosti med dvema sosednjima površinama), ne izoliranih osamelcev,
    zato jih statistika sosednje razdalje pogosto ne zazna kot "oddaljene" -
    to je standarden dodaten filter pri strukturirani svetlobi/laserskih
    skenerjih, ki cilja na fizikalni vzrok (kot vpadanja), ne na
    geometrijski simptom (izolirana oddaljenost)."""
    pcd_normals = copy.deepcopy(pcd)   # ne spreminjamo izvirnika
    pcd_normals.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=max_nn))   # lokalne normale za oceno kota vpadanja
    pcd_normals.orient_normals_towards_camera_location(camera_location)   # usmerimo jih proti kameri, da je kot vpadanja pravilno definiran
    points = np.asarray(pcd_normals.points)
    normals = np.asarray(pcd_normals.normals)
    view_dir = camera_location - points   # smer od vsake točke proti kameri
    view_dir /= np.clip(np.linalg.norm(view_dir, axis=-1, keepdims=True), 1e-12, None)   # normaliziramo na enotsko dolžino
    incidence_cos = np.clip(np.abs(np.sum(view_dir * normals, axis=-1)), -1.0, 1.0)   # kosinus kota med pogledom in normalo
    incidence_deg = np.degrees(np.arccos(incidence_cos))   # kot vpadanja v stopinjah
    keep_mask = incidence_deg <= max_incidence_deg   # obdržimo samo dovolj "naravnost" gledane točke
    kept = pcd.select_by_index(np.where(keep_mask)[0])   # izberemo iz IZVIRNEGA oblaka (brez tu izračunanih normal)
    print(f"  filter_grazing_incidence_points: removed {len(points) - len(kept.points)}/{len(points)} "
          f"points with incidence angle > {max_incidence_deg:.0f} deg "
          f"({100 * len(kept.points) / max(len(points), 1):.1f}% kept)")
    return kept   # vrnemo oblak brez domnevnih letečih pikslov na robovih


def filter_grazing_incidence_points_from_grid(grid: OrganizedScan,
                                              camera_location: np.ndarray,
                                              max_incidence_deg: float = 80.0) -> o3d.geometry.PointCloud:
    """O(n), KD-tree-prosta različica filter_grazing_incidence_points za
    ORGANIZIRANO mrežo (glej OrganizedScan). Namesto estimate_normals
    (KDTreeSearchParamHybrid, O(n log n), na noben poseben način ne izkorišča,
    da je oblak dejansko prišel iz pravilne pravokotne piksel mreže)
    izračuna normalo VSAKEGA piksla neposredno iz križnega produkta razlik do
    DESNEGA in SPODNJEGA soseda v sami mreži - standarden pristop pri
    organiziranih globinskih slikah.

    Piksel brez uporabnega soseda (rob slike, ali sosed brez zadetka -
    hit_mask False) OBDRŽI (normala ni ocenljiva -> ni podlage za odločitev,
    da gre za grazing-incidence piksel; to je konzervativno, isto kot
    KD-tree pristop na robovih redkih/okludiranih regij, kjer
    estimate_normals prav tako vrne šibko določeno normalo).

    Uporablja isto (zašumljeno) geometrijo, ki jo je simulate_camera_scan
    dejansko izpljunil - NE ground-truth primitive_normals, ki jih ray
    casting notranje pozna (to bi bilo "prevarantsko": prava kamera nima
    dostopa do resnične CAD geometrije, samo do svoje lastne (šumne)
    izmerjene globinske slike, zato mora tudi ta filter oceniti normalo iz
    istih šumnih meritev, ne iz privilegiranega ground-trutha)."""
    grid_points, hit_mask = grid
    h, w, _ = grid_points.shape
    right_hit = np.zeros((h, w), dtype=bool)
    right_hit[:, :-1] = hit_mask[:, :-1] & hit_mask[:, 1:]   # ima desnega soseda, oba zadetka
    down_hit = np.zeros((h, w), dtype=bool)
    down_hit[:-1, :] = hit_mask[:-1, :] & hit_mask[1:, :]   # ima spodnjega soseda, oba zadetka
    has_both_neighbors = right_hit & down_hit   # ima OBA soseda - edini piksli, kjer je normala ocenljiva

    dx = np.zeros((h, w, 3))
    dx[:, :-1] = grid_points[:, 1:] - grid_points[:, :-1]   # razlika do desnega soseda
    dy = np.zeros((h, w, 3))
    dy[:-1, :] = grid_points[1:, :] - grid_points[:-1, :]   # razlika do spodnjega soseda
    cross = np.cross(dx, dy)   # smer normale (predznak nedoločen - ni pomembno, glej abs() spodaj)
    cross_len = np.linalg.norm(cross, axis=-1)
    valid_normal = has_both_neighbors & (cross_len > 1e-9)   # dodatno izključimo degenerirane (kolinearne) primere

    points_flat = grid_points.reshape(-1, 3)
    hit_flat = hit_mask.reshape(-1)
    valid_flat = valid_normal.reshape(-1)
    normal_flat = np.zeros((h * w, 3))
    normal_flat[valid_flat] = (cross.reshape(-1, 3)[valid_flat] /
                               cross_len.reshape(-1)[valid_flat, None])   # normalizirane normale samo tam, kjer so ocenljive

    view_dir = camera_location - points_flat[valid_flat]   # smer od vsake ocenljive točke proti kameri
    view_dir /= np.clip(np.linalg.norm(view_dir, axis=-1, keepdims=True), 1e-12, None)
    incidence_cos = np.clip(np.abs(np.sum(view_dir * normal_flat[valid_flat], axis=-1)), -1.0, 1.0)
    incidence_deg = np.degrees(np.arccos(incidence_cos))
    drop_local = incidence_deg > max_incidence_deg   # kateri od ocenljivih pikslov presežejo prag

    keep_flat = hit_flat.copy()
    valid_indices = np.where(valid_flat)[0]
    keep_flat[valid_indices[drop_local]] = False   # zavrnemo samo tiste, kjer JE bila normala ocenljiva IN je presegla prag

    kept_points = points_flat[keep_flat]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(kept_points))
    n_hit = int(hit_flat.sum())
    print(f"  filter_grazing_incidence_points_from_grid: removed {n_hit - len(kept_points)}/{n_hit} "
          f"points with incidence angle > {max_incidence_deg:.0f} deg "
          f"({100 * len(kept_points) / max(n_hit, 1):.1f}% kept) [O(n) grid-based, no KD-tree]")
    return pcd


def remove_table_background(target: o3d.geometry.PointCloud,
                            up_axis: int,
                            part_height_mm: float,
                            cropped_region_height_mm: float,
                            margin_mm: float = 10.0,
                            table_level_percentile: float = 1.0) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """Odstrani domnevne mizne/ozadje točke iz target, na podlagi ZNANE
    višine dela (part_height_mm, iz source/CAD modela - vedno na voljo in
    natančno znana, ne glede na velikost ali obliko dela v XY), namesto
    zaznavanja "največje ravnine" (npr. Open3D-jev segment_plane, RANSAC).

    Zakaj ne "največja ravnina": če del zaseda velik del vidnega polja
    kamere (velik kos, ali del z veliko ravno zgornjo "sealing" površino -
    natanko taka, kot jo ta cevovod že skenira), bi RANSAC lahko namesto
    mize po nesreči zaznal in odstranil TA del kot "največjo ravnino". Ta
    filter se sploh ne opira na relativno velikost ravnin v sceni, zato je
    od velikosti/oblike dela neodvisen.

    Namesto tega izkoristimo fizikalno dejstvo, da del ne more segati pod
    mizo, na kateri leži - miza (če je v skenu sploh prisotna) je zato
    vedno pri najnižjih opaženih vrednostih vzdolž up_axis. table_level_percentile (namesto
    golega min()) ublaži vpliv posameznih osamelcev/letečih pikslov, ki bi
    sicer navidezno postavili "mizo" veliko nižje, kot dejansko je.
    Zgornja meja obdržanega pasu (miza + part_height_mm + margin_mm)
    dodatno odreže ekstremne osamelce, ki segajo višje, kot bi del sploh
    lahko fizično segal.

    cropped_region_height_mm ZOŽI pas ŠE OD SPODAJ: kamera dejansko skenira
    SAMO zgornjo top_fraction regijo dela (glej crop_top_region) - source je
    obrezan na natanko to regijo, zato vse MED mizo in (vrh dela - ta
    regija) v target-u ni ne miza ne kaj, s čimer bi se source sploh lahko
    ujemal: gre za poševno vidne stranske stene dela ("zavesa") ali prazen
    prostor - v obeh primerih samo šum/osamelci za ICP, ne uporabna
    geometrija. Spodnja meja je zato
    table_level + part_height_mm - cropped_region_height_mm - margin_mm
    (z varovalko navzdol na table_level + margin_mm, če bi bil
    cropped_region_height_mm blizu part_height_mm - tedaj ni prave
    "zavese" za odrezati, meja ostane tik nad mizo).

    Vrne (kept, keep_mask) - keep_mask je poravnan ena-na-ena s target-om
    PRED tem klicem, da lahko klicatelj z njim uskladi tudi organizirano
    mrežo skena (OrganizedScan.keep_only)."""
    points = np.asarray(target.points)
    coord = points[:, up_axis]   # miza je vedno pri min(coord) - kamera gleda vzdolž -up_axis
    table_level = np.percentile(coord, table_level_percentile)   # ocena višine mize, robustna na osamelce
    lower_cutoff = max(table_level + margin_mm,   # tik nad mizo - varovalna spodnja meja
                       table_level + part_height_mm - cropped_region_height_mm - margin_mm)
    upper_cutoff = table_level + part_height_mm + margin_mm   # nad tem del fizično ne more segati
    keep_mask = (coord > lower_cutoff) & (coord < upper_cutoff)   # obdržimo samo pas, kjer lahko biva del
    kept = target.select_by_index(np.where(keep_mask)[0])
    print(f"  remove_table_background: table_level={table_level:.2f}, "
          f"keeping band ({lower_cutoff:.2f}, {upper_cutoff:.2f}) - "
          f"kept {len(kept.points)}/{len(points)} points "
          f"({100 * len(kept.points) / max(len(points), 1):.1f}%)")
    return kept, keep_mask   # vrnemo target brez domnevnih mizno/ozadje točk + keep_mask


def ensure_oriented_normals(pcd: o3d.geometry.PointCloud,  # oblak točk (en sam pogled kamere - pravi, simuliran ali obrezan CAD), ki ga obdelujemo
                            normal_radius: float,   # radij za izračun normalnih vektorjev, določa velikost lokalnega območja okoli vsake točke, ki se uporablja za izračun normale na površino
                            camera_location: np.ndarray,    # lokacija kamere, proti kateri usmerimo normale
                            max_nn: int = 30,   # maksimalno število sosednjih točk, ki se upoštevajo pri izračunu normalne za vsako točko, večje število pomeni bolj gladke normale, vendar počasnejše izračune
                            use_knn: bool = False,     # Če je True, se uporabi KNN (k-nearest neighbors) metoda za izračun normalnih vektorjev, sicer se uporabi hibridna metoda z radijem
                                                        # False --> enakomerni oblaki točk, True --> neenakomerni oblaki točk
                            preserve_existing_normals: bool = False) -> None:   # če je True IN pcd že ima normale, se estimate_normals PRESKOČI (samo normalizira in preusmeri obstoječe) - glej docstring
    """Izračuna (ali ohrani) normale in jih usmeri proti camera_location: en
    sam pogled kamere vedno vidi le eno stran objekta, zato mora vsaka
    normala kazati nazaj proti senzorju. Tudi source (CAD) je pred tem
    obrezan na zgornjo "sealing" ploskev (crop_top_region) in tako
    enostranski - zato zanj velja isto, z virtualno kamero nad njim
    (top_camera_location).

    preserve_existing_normals=True ohrani pcd-jeve OBSTOJEČE normale (le
    normalizirane in nato usmerjene), namesto da bi jih pcd.estimate_normals()
    prepisal s ponovno OCENJENIMI (KNN/radij) normalami. Namenjeno CAD
    oblakom, ki izvirajo iz mesh.sample_points_poisson_disk() na mreži s
    KLICANIM mesh.compute_vertex_normals() (glej load_cad_model) - taki
    oblaki že nosijo TOČNE, iz CAD geometrije interpolirane normale, ki jih
    tudi voxel_down_sample pravilno POVPREČI (ne uniči). Ponovno ocenjene
    normale iz redke, downsamplirane točkovne geometrije so manj natančne -
    v refine_registration-ovem ICP koraku, kjer Open3D-jev point-to-plane
    bere normale s CAD strani (glej refine_registration docstring "ICP smer
    je OBRNJENA"), bi to neposredno pomenilo nižjo natančnost rešitve. Če
    pcd nima normal (npr. surov sken), se ta zastavica ignorira in normale
    se ocenijo - zato je varno pustiti False za sken/target povsod."""
    if preserve_existing_normals and pcd.has_normals():
        pcd.normalize_normals()   # voxel_down_sample povpreči obstoječe normale po celicah - rezultat ni nujno natanko enotske dolžine
    elif use_knn:
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=max_nn))     # normalo oceni glede na najbližje točke, ne glede na njihovo oddaljenost
    else:
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=max_nn))   # normalo oceni glede na točke v določeni bližini
    pcd.orient_normals_towards_camera_location(camera_location)   # obrnemo vse normale proti kameri


def preprocess_point_cloud(pcd: o3d.geometry.PointCloud,
                           voxel_size: float,
                           camera_location: np.ndarray,
                           cache_key: Optional[str] = None,
                           use_cache: bool = True,
                           preserve_existing_normals: bool = False) -> o3d.geometry.PointCloud:
    """Zmanjša gostoto pcd (voxel downsampling) in izračuna pravilno
    orientirane normale - vse, kar yaw_sweep_registration in
    večnivojski ICP dejansko potrebujeta.

    cache_key identificira *vhod* (npr. "source_partname_20000pts_top0.35_axis2") -
    skupaj z vsemi parametri spodaj, ki vplivajo na izhod, tvori pot do
    predpomnilnika.

    preserve_existing_normals: posredovano naprej v ensure_oriented_normals
    (glej njen docstring) - True za CAD source (točne mesh normale, glej
    load_cad_model), False (privzeto) za sken/target (nima uporabnih normal
    pred to funkcijo, mora jih oceniti)."""
    cloud_path = None
    if cache_key is not None and use_cache:   # predpomnilnik je omogočen in imamo ključ
        params = (PREPROCESS_CACHE_VERSION, voxel_size,
                  tuple(np.round(np.asarray(camera_location), 3)), preserve_existing_normals)   # vsi parametri, ki vplivajo na izhod
        digest = hashlib.md5(repr(params).encode()).hexdigest()[:10]   # kratek hash teh parametrov
        PREPROCESS_CACHE_DIR.mkdir(exist_ok=True)   # ustvarimo mapo za predpomnilnik, če še ne obstaja
        # cache_key je berljiv za človeka, a lahko postane dolg - ime CAD
        # datoteke plus oznake poze/šuma lahko celotno pot potisnejo čez
        # Windows-ovo mejo 260 znakov (MAX_PATH), zaradi česar Open3D-jev
        # write_point_cloud spodleti z nepovedno "invalid wchar_t filename
        # argument". Hash samega cache_key ohrani ime datoteke kratko;
        # skrajšana berljiva predpona je zraven zgolj za brskanje po
        # .preprocess_cache/ na oko, ne za enoličnost (to zagotavlja hash).
        cache_key_digest = hashlib.md5(cache_key.encode()).hexdigest()[:12]   # hash celotnega cache_key
        readable_prefix = "".join(c if c.isalnum() else "_" for c in cache_key)[:40]   # skrajšan berljiv del imena
        cloud_path = PREPROCESS_CACHE_DIR / f"{readable_prefix}_{cache_key_digest}_{digest}.ply"
        if cloud_path.exists():   # predpomnjena datoteka že obstaja
            pcd_down = o3d.io.read_point_cloud(str(cloud_path))   # preberemo jo
            if not pcd_down.is_empty():
                print(f"    Loaded cached preprocessing for '{cache_key}': {len(pcd_down.points)} points")
                return pcd_down   # vrnemo predpomnjen rezultat, brez ponovnega računanja

    extent = np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound())   # velikost oblaka po vsaki osi
    diagonal = np.linalg.norm(extent)   # dolžina diagonale oblaka
    print(f"    Input cloud extent (xyz): {extent}, diagonal={diagonal:.3f}, "
          f"voxel_size={voxel_size} -> ~{diagonal / voxel_size:.0f} voxels across the diagonal")

    pcd_down = pcd.voxel_down_sample(voxel_size)    # naredi 3d mrežo kock, v vsaki kocki vzame eno točko (povprečje), da zmanjša število točk in pospeši izračune
    ensure_oriented_normals(pcd_down, normal_radius=voxel_size * 2.0, camera_location=camera_location,
                            preserve_existing_normals=preserve_existing_normals)

    if cloud_path is not None:   # shranimo rezultat za naslednji zagon
        o3d.io.write_point_cloud(str(cloud_path), pcd_down)

    return pcd_down   # vrnemo zmanjšan oblak z izračunanimi normalami
