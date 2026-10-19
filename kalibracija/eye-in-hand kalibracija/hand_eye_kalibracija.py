import socket
import cv2
import numpy as np
import json
import os
from scipy.spatial.transform import Rotation as R

# ======================================================================
# 1. NASTAVITVE POVEZAVE, IMENIKOV IN CHARUCO TABLE
# ======================================================================
TCP_IP = "192.168.0.10"   # IP vašega računalnika
TCP_PORT = 12345

SAVE_DIR = "calib_data"
JSON_FILE = os.path.join(SAVE_DIR, "calibration_data.json")
RESULT_FILE = os.path.join(SAVE_DIR, "hand_eye_result.json")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Nastavitve ChArUco table
SQUARES_X = 6          # število kvadratkov po širini
SQUARES_Y = 6           # število kvadratkov po višini
SQUARE_LENGTH = 0.029   # 29 mm = 0.029 m
MARKER_LENGTH = 0.02175  # 21.75 mm = 0.02175 m

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)

detector_params = cv2.aruco.DetectorParameters()
detector_params.adaptiveThreshWinSizeMin = 3
detector_params.adaptiveThreshWinSizeMax = 23
detector_params.adaptiveThreshWinSizeStep = 4
detector_params.errorCorrectionRate = 0.6

charuco_params = cv2.aruco.CharucoParameters()
charuco_detector = cv2.aruco.CharucoDetector(
    board,
    charucoParams=charuco_params,
    detectorParams=detector_params
)

# ======================================================================
# POMOŽNE FUNKCIJE
# ======================================================================
def loci(naslov=""):
    print("\n" + "=" * 60)
    if naslov:
        print(naslov)
        print("=" * 60)

def epson_to_matrix(x, y, z, u, v, w):
    """
    Pretvori Epsonove koordinate v 4x4 transformacijsko matriko.
    Epson kote definira kot: W (okoli X), V (okoli Y), U (okoli Z) z 'xyz' konvencijo.
    """
    rot = R.from_euler('xyz', [w, v, u], degrees=True).as_matrix()
    T = np.eye(4)
    T[0:3, 0:3] = rot
    T[0:3, 3] = [x, y, z]
    return T.tolist()

def get_fresh_frame(cap, flush_count=5):
    """
    Izprazni notranji buffer OpenCV kamere, da dobimo 100 % svežo sliko.
    """
    for _ in range(flush_count):
        cap.grab()
    return cap.read()

# ======================================================================
# KORAK A: ZAJEM PODATKOV PREKO TCP V ŽIVO
# ======================================================================
def capture_live_data():
    print("[Kamera] Povezovanje s kamero...")
    cap = cv2.VideoCapture(1)  # Indeks vaša kamere
    
    if not cap.isOpened():
        print("[Kamera] NAPAKA: Ni mogoče odpreti kamere!")
        return False

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((TCP_IP, TCP_PORT))
    server_socket.listen(1)

    print(f"\n[Sistem] Strežnik posluša na {TCP_IP}:{TCP_PORT}...")
    print("[Sistem] V Epson RC+ zdaj poženi funkcijo HandEyeCalib...")

    conn, addr = server_socket.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"[Sistem] Robot povezan z naslova: {addr}\n")

    json_data = []
    tcp_buffer = ""

    try:
        done = False
        while not done:
            chunk = conn.recv(1024).decode('utf-8')
            if not chunk:
                break
            
            tcp_buffer += chunk

            while "\n" in tcp_buffer:
                line, tcp_buffer = tcp_buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                cmd_type = parts[0]

                # --- KODA 3: PREJETI PODATKI O KALIBRACIJSKI TOČKI ---
                if cmd_type == "3" and len(parts) >= 8:
                    pt_idx = parts[1]
                    x = float(parts[2])
                    y = float(parts[3])
                    z = float(parts[4])
                    u = float(parts[5])
                    v = float(parts[6])
                    w = float(parts[7])

                    print(f"[Točka {pt_idx}] Robot stoji na poziciji:")
                    print(f"  X={x:.2f}, Y={y:.2f}, Z={z:.2f} | U={u:.2f}, V={v:.2f}, W={w:.2f}")

                    # Zajem SVEŽE slike (s praznjenjem bufferja)
                    ret, frame = get_fresh_frame(cap, flush_count=5)
                    if ret:
                        img_path = os.path.abspath(os.path.join(SAVE_DIR, f"calib_frame_{pt_idx}.png"))
                        cv2.imwrite(img_path, frame)
                        
                        robot_matrix = epson_to_matrix(x, y, z, u, v, w)

                        json_data.append({
                            "point_index": pt_idx,
                            "image_path": img_path,
                            "robot_matrix": robot_matrix
                        })

                        conn.sendall("OK\r\n".encode('utf-8'))
                        print("  [Slika & Podatki] Shranjeni. Robotu poslano 'OK'.\n")
                    else:
                        print("  [NAPAKA] Zajem slike neuspešen!")
                        conn.sendall("ERROR\r\n".encode('utf-8'))

                # --- KODA 99: ZAKLJUČEK ZAJEMA ---
                elif cmd_type == "99":
                    print("[Sistem] Robot je poslal kodo 99 - Obisk točk zaključen!")
                    done = True
                    break

    except Exception as e:
        print(f"[Napaka pri zajemu] {e}")
        return False
    finally:
        cap.release()
        conn.close()
        server_socket.close()

    with open(JSON_FILE, 'w') as f:
        json.dump(json_data, f, indent=4)
    
    print(f"[Sistem] Podatki uspešno shranjeni v {JSON_FILE}")
    return True

# ======================================================================
# KORAK B: IZRAČUN HAND-EYE KALIBRACIJE (POPR. USKLADITEV INDEKSOV)
# ======================================================================
def process_hand_eye_calibration():
    try:
        with open(JSON_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"NAPAKA: Datoteke ni mogoče najti na poti {JSON_FILE}")
        return

    valid_views = []
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
        charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)

        if charuco_corners is not None and len(charuco_corners) > 3:
            R_gripper2base = robot_matrix[0:3, 0:3]
            t_gripper2base = robot_matrix[0:3, 3].reshape(3, 1) / 1000.0  # mm -> m

            # Podatki so zapakirani v enoten slovar za preprečitev razsinhronizacije
            valid_views.append({
                "img_path": img_path,
                "corners": charuco_corners,
                "ids": charuco_ids,
                "R_robot": R_gripper2base,
                "t_robot": t_gripper2base
            })

            print(f"  [OK] {os.path.basename(img_path):<20} -> najdenih {len(charuco_corners)} kotov")
        else:
            print(f"  [NAPAKA] {os.path.basename(img_path):<20} -> premalo kotov, slika izločena")

    loci(f"Uspešno obdelanih: {len(valid_views)} / {len(data)} slik")

    if len(valid_views) < 3:
        print("NAPAKA: Potrebujemo vsaj 3 veljavne slike za kalibracijo. Konec.")
        return

    loci("KORAK 2: Izračun notranjih parametrov kamere (intrinsics)")

    calib_views = []
    all_object_points = []
    all_image_points = []

    for view in valid_views:
        obj_points, img_points = board.matchImagePoints(view["corners"], view["ids"])
        if obj_points is None or len(obj_points) < 4:
            print(f"  [PRESKOČENO v Koraku 2] {os.path.basename(view['img_path'])} -> premalo ujemanja točk")
            continue

        obj_pts_arr = np.array(obj_points, dtype=np.float32)
        img_pts_arr = np.array(img_points, dtype=np.float32)

        all_object_points.append(obj_pts_arr)
        all_image_points.append(img_pts_arr)

        calib_views.append({
            "img_path": view["img_path"],
            "obj_points": obj_pts_arr,
            "img_points": img_pts_arr,
            "R_robot": view["R_robot"],
            "t_robot": view["t_robot"]
        })

    if len(all_object_points) < 3:
        print("NAPAKA: Premalo pogledov z veljavnimi korespondencami točk. Konec.")
        return

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        all_object_points,
        all_image_points,
        image_size,
        None,  # type: ignore
        None   # type: ignore
    )
    
    np.set_printoptions(suppress=True, precision=5)
    print(f"  Reprojekcijska napaka (RMS): {ret:.4f} px")
    print(f"  Velikost slike: {image_size[0]} x {image_size[1]} px")
    print("  Matrika kamere (camera_matrix):")
    print(f"    fx = {camera_matrix[0, 0]:.3f} px   fy = {camera_matrix[1, 1]:.3f} px")
    print(f"    cx = {camera_matrix[0, 2]:.3f} px   cy = {camera_matrix[1, 2]:.3f} px")
    print(f"  Koeficienti distorzije (dist_coeffs): {dist_coeffs.flatten()}")

    loci("KORAK 3: Izračun pozicij table glede na kamero (target -> cam)")
    
    cam_R_target2cam = []
    cam_t_target2cam = []
    final_R_gripper2base = []
    final_t_gripper2base = []

    for view in calib_views:
        success, rvec, tvec = cv2.solvePnP(
            view["obj_points"], 
            view["img_points"], 
            camera_matrix, 
            dist_coeffs
        )
        if success:
            R_target2cam, _ = cv2.Rodrigues(rvec)
            cam_R_target2cam.append(R_target2cam)
            cam_t_target2cam.append(tvec)
            
            final_R_gripper2base.append(view["R_robot"])
            final_t_gripper2base.append(view["t_robot"])

    print(f"  Uspešno izračunanih pozicij table: {len(cam_R_target2cam)} / {len(calib_views)}")

    if len(cam_R_target2cam) < 3:
        print("NAPAKA: Premalo veljavnih pozicij table za hand-eye kalibracijo. Konec.")
        return

    loci("KORAK 4: Hand-eye kalibracija (cv2.calibrateHandEye)")
    
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(  # type: ignore
        final_R_gripper2base,
        final_t_gripper2base,
        cam_R_target2cam,
        cam_t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    hand_eye_matrix = np.eye(4)
    hand_eye_matrix[0:3, 0:3] = R_cam2gripper
    hand_eye_matrix[0:3, 3] = t_cam2gripper.flatten()

    distance_m = np.linalg.norm(t_cam2gripper)

    loci("KONČNA HAND-EYE TRANSFORMACIJSKA MATRIKA (kamera -> flange)")
    print(hand_eye_matrix)
    print(f"\n  Razdalja kamera - flange: {distance_m:.4f} m  ({distance_m * 1000:.1f} mm)")

    loci("KORAK 5: Preverjanje konsistentnosti med pogledi")
    board_positions_in_base = []
    for i in range(len(cam_R_target2cam)):
        T_gripper2base = np.eye(4)
        T_gripper2base[0:3, 0:3] = final_R_gripper2base[i]
        T_gripper2base[0:3, 3] = final_t_gripper2base[i].flatten()

        T_target2cam = np.eye(4)
        T_target2cam[0:3, 0:3] = cam_R_target2cam[i]
        T_target2cam[0:3, 3] = cam_t_target2cam[i].flatten()

        T_target2base = T_gripper2base @ hand_eye_matrix @ T_target2cam
        board_positions_in_base.append(T_target2base[0:3, 3])

    board_positions_in_base = np.array(board_positions_in_base)
    mean_pos = board_positions_in_base.mean(axis=0)
    deviations = np.linalg.norm(board_positions_in_base - mean_pos, axis=1)

    print(f"  Povprečna pozicija table v bazi robota: {mean_pos} (m)")
    print(f"  Odstopanje od povprečja (m): min={deviations.min():.5f}, "
          f"max={deviations.max():.5f}, povprečje={deviations.mean():.5f}")
    
    result_data = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "hand_eye_matrix": hand_eye_matrix.tolist()
    }
    with open(RESULT_FILE, 'w') as f:
        json.dump(result_data, f, indent=4)
        
    loci(f"KALIBRACIJA USPEŠNO ZAKLJUČENA\nRezultati shranjeni v: {RESULT_FILE}")

# ======================================================================
# GLAVNI ZAGON
# ======================================================================
def main():
    if os.path.exists(JSON_FILE):
        odgovor = input("Najdena je obstoječa datoteka z zajetimi podatki. Želiš nov zajem v živo preko TCP? (y/n): ").strip().lower()
        if odgovor == 'y':
            if capture_live_data():
                process_hand_eye_calibration()
        else:
            print("[Sistem] Izvajam kalibracijo na obstoječih shranjenih slikah...")
            process_hand_eye_calibration()
    else:
        if capture_live_data():
            process_hand_eye_calibration()

if __name__ == "__main__":
    main()