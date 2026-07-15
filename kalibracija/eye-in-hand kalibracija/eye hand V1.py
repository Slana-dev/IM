import cv2
import numpy as np
import json
import os

# OPOMBA: skripta cilja OpenCV 4.x (pip install "opencv-python<5").
# OpenCV 5.0 je zelo svež major release in trenutno (poletje 2026)
# calibrateHandEye ni zanesljivo dostopen v Python bindingih
# (C++ modul calib3d so pravkar razdelili na geometry/calib/stereo).
# CharucoDetector/CharucoParameters/matchImagePoints spodaj delujejo
# normalno že od verzije 4.7 naprej, tako da po downgrade-u na 4.x
# nič drugega ni treba spreminjati.

# ==========================================
# 1. NASTAVITVE CHARUCO TABLE
# ==========================================
SQUARES_X = 7          # število kvadratkov po širini
SQUARES_Y = 9           # število kvadratkov po višini
SQUARE_LENGTH = 0.03    # dolžina stranice kvadratka: 30 mm = 0.03 m
MARKER_LENGTH = 0.024   # dolžina stranice markerja: 24 mm = 0.024 m (80 % kvadratka)

# S testom smo potrdili, da je pravi slovar DICT_4X4_50 (Dict ID 0)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)

# Parametri detektorja - prilagojeni za ostre robove v simulaciji (brez anti-aliasinga)
detector_params = cv2.aruco.DetectorParameters()
detector_params.adaptiveThreshWinSizeMin = 3
detector_params.adaptiveThreshWinSizeMax = 23
detector_params.adaptiveThreshWinSizeStep = 4
detector_params.errorCorrectionRate = 0.6

# CharucoParameters je treba eksplicitno ustvariti za OpenCV 4.7+
charuco_params = cv2.aruco.CharucoParameters()

# Ustvarimo detektor s prilagojenimi parametri
charuco_detector = cv2.aruco.CharucoDetector(
    board,
    charucoParams=charuco_params,
    detectorParams=detector_params
)

# JSON datoteka, ki jo ustvari RoboDK skripta (slike + pripadajoče pozicije robota)
JSON_FILE = r"C:\Users\Grega\Desktop\reci za nanasalca\zajemi simulacije\calibration_data.json"
# to bo treba spremenit obv

def loci(naslov=""):
    """Pomožna funkcija za pregledne ločilne črte med izpisi v konzoli."""
    print("\n" + "=" * 60)
    if naslov:
        print(naslov)
        print("=" * 60)


def main():
    # Varno naložimo JSON podatke
    try:
        with open(JSON_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"NAPAKA: datoteke ni mogoče najti na poti {JSON_FILE}")
        return

    all_charuco_corners = []
    all_charuco_ids = []
    robot_R_gripper2base = []
    robot_t_gripper2base = []
    valid_images = []
    image_size = None

    loci("KORAK 1: Detekcija ChArUco kotov na slikah")

    for item in data:
        img_path = item["image_path"]
        robot_matrix = np.array(item["robot_matrix"])

        img = cv2.imread(img_path)
        if img is None:
            print(f"  [PRESKOČENO] Slike ni mogoče naložiti: {img_path}")
            continue

        if image_size is None:
            image_size = (img.shape[1], img.shape[0])

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- POPRAVEK ZA ANTI-ALIASING (SAMO ZA SIMULACIJO!) ---
        # Blag Gaussov blur simulira mehkobo prave kamere/leče, ker so robovi
        # v renderju iz RoboDK (brez anti-aliasinga) preostri za detektor.
        #
        # >>> ČE DELAMO NA FIZIČNEM ROBOTU S PRAVO KAMERO, SPODNJO VRSTICO
        # >>> ZAKOMENTIRAJ (ali izbriši) - prava kamera že ima naravni šum
        # >>> in mehkobo robov, dodaten blur bi samo poslabšal natančnost
        # >>> detekcije kotov (manj ostri koti = slabša subpikelska natančnost).
        gray = cv2.GaussianBlur(gray, (5, 5), 0)  # <-- TO VRSTICO ZAKOMENTIRAJ NA FIZIČNEM ROBOTU

        # Detekcija ChArUco table na trenutni sliki
        charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)

        if charuco_corners is not None and len(charuco_corners) > 3:
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)

            # Iz 4x4 RoboDK matrike izluščimo rotacijo (3x3) in translacijo (3x1)
            R_gripper2base = robot_matrix[0:3, 0:3]
            t_gripper2base = robot_matrix[0:3, 3].reshape(3, 1)

            # RoboDK uporablja milimetre, OpenCV pa pričakuje metre
            t_gripper2base = t_gripper2base / 1000.0

            robot_R_gripper2base.append(R_gripper2base)
            robot_t_gripper2base.append(t_gripper2base)
            valid_images.append(img_path)

            print(f"  [OK] {os.path.basename(img_path):<20} -> najdenih {len(charuco_corners)} kotov")
        else:
            print(f"  [NAPAKA] {os.path.basename(img_path):<20} -> premalo kotov, slika izločena")

    loci(f"Uspešno obdelanih: {len(valid_images)} / {len(data)} slik")

    if len(valid_images) < 3:
        print("NAPAKA: potrebujemo vsaj 3 veljavne slike za kalibracijo. Konec.")
        return

    # ==========================================
    # 2. IZRAČUN NOTRANJIH PARAMETROV KAMERE (INTRINSICS)
    # ==========================================
    # cv2.aruco.calibrateCameraCharuco() je bila v OpenCV 4.7+/5.0 odstranjena.
    # Nadomestek (po OpenCV dokumentaciji): CharucoBoard.matchImagePoints() za
    # pridobitev korespondenc 3D<->2D točk, nato splošna cv2.calibrateCamera().
    loci("KORAK 2: Izračun notranjih parametrov kamere (intrinsics)")

    all_object_points = []
    all_image_points = []
    calib_charuco_corners = []
    calib_charuco_ids = []

    for corners, ids in zip(all_charuco_corners, all_charuco_ids):
        obj_points, img_points = board.matchImagePoints(corners, ids)
        if obj_points is None or len(obj_points) < 4:
            # Premalo korespondenc iz tega pogleda - izločimo ga iz kalibracije
            continue
        all_object_points.append(obj_points)
        all_image_points.append(img_points)
        calib_charuco_corners.append(corners)
        calib_charuco_ids.append(ids)

    if len(all_object_points) < 3:
        print("NAPAKA: premalo pogledov z veljavnimi korespondencami točk. Konec.")
        return

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        all_object_points, all_image_points, image_size, None, None
    )

    np.set_printoptions(suppress=True, precision=5)
    print(f"  Reprojekcijska napaka (RMS): {ret:.4f} px")
    print(f"  Velikost slike: {image_size[0]} x {image_size[1]} px")
    print("  Matrika kamere (camera_matrix):")
    print(f"    fx = {camera_matrix[0, 0]:.3f} px   fy = {camera_matrix[1, 1]:.3f} px")
    print(f"    cx = {camera_matrix[0, 2]:.3f} px   cy = {camera_matrix[1, 2]:.3f} px")
    print(f"  Koeficienti distorzije (dist_coeffs): {dist_coeffs.flatten()}")

    # ==========================================
    # 3. IZRAČUN POZICIJ TABLE ZA HAND-EYE KALIBRACIJO
    # ==========================================
    # cv2.aruco.estimatePoseCharucoBoard() je bila prav tako odstranjena.
    # Nadomestek: CharucoBoard.matchImagePoints() + cv2.solvePnP(), z uporabo
    # ravnokar izračunanih camera_matrix/dist_coeffs.
    loci("KORAK 3: Izračun pozicij table glede na kamero (target -> cam)")
    cam_R_target2cam = []
    cam_t_target2cam = []
    final_R_gripper2base = []
    final_t_gripper2base = []

    for i in range(len(calib_charuco_corners)):
        obj_points, img_points = board.matchImagePoints(calib_charuco_corners[i], calib_charuco_ids[i])
        if obj_points is None or len(obj_points) < 4:
            continue

        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
        if success:
            R_target2cam, _ = cv2.Rodrigues(rvec)
            cam_R_target2cam.append(R_target2cam)
            cam_t_target2cam.append(tvec)
            # Ujemajočo pozicijo robota shranimo pod istim indeksom
            final_R_gripper2base.append(robot_R_gripper2base[i])
            final_t_gripper2base.append(robot_t_gripper2base[i])

    print(f"  Uspešno izračunanih pozicij table: {len(cam_R_target2cam)} / {len(calib_charuco_corners)}")

    if len(cam_R_target2cam) < 3:
        print("NAPAKA: premalo veljavnih pozicij table za hand-eye kalibracijo. Konec.")
        return

    # ==========================================
    # 4. HAND-EYE KALIBRACIJA
    # ==========================================
    loci("KORAK 4: Hand-eye kalibracija (cv2.calibrateHandEye)")
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        final_R_gripper2base,
        final_t_gripper2base,
        cam_R_target2cam,
        cam_t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    # Sestavimo končno 4x4 transformacijsko matriko
    hand_eye_matrix = np.eye(4)
    hand_eye_matrix[0:3, 0:3] = R_cam2gripper
    hand_eye_matrix[0:3, 3] = t_cam2gripper.flatten()

    distance_m = np.linalg.norm(t_cam2gripper)

    loci("KONČNA HAND-EYE TRANSFORMACIJSKA MATRIKA (kamera -> flange)")
    print(hand_eye_matrix)
    print(f"\n  Razdalja kamera - flange: {distance_m:.4f} m  ({distance_m * 1000:.1f} mm)")

    # ==========================================
    # 5. PREVERJANJE KONSISTENTNOSTI (VALIDACIJA)
    # ==========================================
    # Ker tabla v prostoru miruje, bi morala vsaka od 20 slik po uporabi
    # hand-eye matrike dati (skoraj) enako pozicijo table glede na bazo robota.
    # Razpršenost (deviations) je pravi pokazatelj natančnosti hand-eye
    # kalibracije - reprojekcijska napaka iz koraka 2 tega ne pove!
    loci("KORAK 5: Preverjanje konsistentnosti med pogledi")
    board_positions_in_base = []
    for i in range(len(cam_R_target2cam)):
        T_gripper2base = np.eye(4)
        T_gripper2base[0:3, 0:3] = final_R_gripper2base[i]
        T_gripper2base[0:3, 3] = final_t_gripper2base[i].flatten()

        T_target2cam = np.eye(4)
        T_target2cam[0:3, 0:3] = cam_R_target2cam[i]
        T_target2cam[0:3, 3] = cam_t_target2cam[i].flatten()

        # base <- gripper <- (cam2gripper) <- (target2cam) = pozicija table v bazi robota
        T_target2base = T_gripper2base @ hand_eye_matrix @ T_target2cam
        board_positions_in_base.append(T_target2base[0:3, 3])

    board_positions_in_base = np.array(board_positions_in_base)
    mean_pos = board_positions_in_base.mean(axis=0)
    deviations = np.linalg.norm(board_positions_in_base - mean_pos, axis=1)

    print(f"  Povprečna pozicija table v bazi robota: {mean_pos} (m)")
    print(f"  Odstopanje od povprečja (m): min={deviations.min():.5f}, "
          f"max={deviations.max():.5f}, povprečje={deviations.mean():.5f}")
    print("  (Manjše je boljše - odstopanje pod 1 mm pomeni dobro kalibracijo;")
    print("   več milimetrov nakazuje na slabe poze, premalo raznolike rotacije")
    print("   med zajemanjem, ali napako v enotah/konvenciji koordinatnih sistemov.)")

    loci("KONEC")


if __name__ == "__main__":
    main()