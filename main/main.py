import threading
import time
import keyboard
import cv2
import os
import sys
import socket  # TCP/IP komunikacija
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

# --- NASTAVITVE ZA TCP/IP ---
TCP_IP = "192.168.0.10"   # IP računalnika
TCP_PORT = 12345        # V Epson Port 201 nastavi enak port

CALIB_DIR = "calib_data"
CALIB_RESULT_FILE = os.path.join(CALIB_DIR, "hand_eye_result.json")

# Skupna deljena spremenljivka za preverjanje premika
movementError = False
error_lock = threading.Lock()
stop_event = threading.Event()  # Za izhod iz celotnega programa (Ctrl+C)

ROI_SIZE = 100 
MOVEMENT_THRESHOLD = 0.95 
MOVEMENT_INTERVAL_SECONDS = 0.5 

# ======================================================================
# MATEMATIČNE PRETVORBE (VSE MATRIKE V MILIMETRIH)
# ======================================================================
def epson_to_matrix(x, y, z, u, v, w):
    """
    Pretvori Epsonove koordinate (mm, stopinje) v 4x4 transformacijsko matriko.
    Konvencija: 'xyz' s koti [w, v, u] (W=Roll, V=Pitch, U=Yaw).
    """
    rot = R.from_euler('xyz', [w, v, u], degrees=True)
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = [x, y, z]
    return T

def matrix_to_epson(T):
    """
    Pretvori 4x4 matriko (v mm) v Epson koordinate (x, y, z, u, v, w).
    Vrne: x, y, z (mm), u, v, w (deg).
    """
    x, y, z = T[:3, 3]
    rot = R.from_matrix(T[:3, :3])
    w, v, u = rot.as_euler('xyz', degrees=True)
    return x, y, z, u, v, w

def get_object_in_base(T_robot_current_mm, T_cam2flange_mm, T_obj2cam_mm):
    """
    Preračuna koordinate zaznanega objekta iz kamere v bazo robota (vse matrike v mm):
    T_obj2base = T_flange2base @ T_cam2flange @ T_obj2cam
    """
    T_obj2base = T_robot_current_mm @ T_cam2flange_mm @ T_obj2cam_mm
    return T_obj2base

def load_calibration_data():
    if not os.path.exists(CALIB_RESULT_FILE):
        print(f"[Kalibracija] NAPAKA: Datoteka {CALIB_RESULT_FILE} ne obstaja!")
        print("[Kalibracija] Najprej poženite hand_eye_kalibracija.py za izračun matrik.")
        return None, None, None

    try:
        with open(CALIB_RESULT_FILE, 'r') as f:
            data = json.load(f)
        
        camera_matrix = np.array(data["camera_matrix"], dtype=np.float32)
        dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float32)
        hand_eye_matrix = np.array(data["hand_eye_matrix"], dtype=np.float64)
        
        # --- KLJUČNI POPRAVEK ENOT: metri -> milimetri ---
        hand_eye_matrix[:3, 3] *= 1000.0
        
        print(f"[Kalibracija] Uspešno naloženi podatki. Hand-Eye zamik kamere: "
              f"X={hand_eye_matrix[0,3]:.1f}mm, Y={hand_eye_matrix[1,3]:.1f}mm, Z={hand_eye_matrix[2,3]:.1f}mm")
        
        return camera_matrix, dist_coeffs, hand_eye_matrix
    except Exception as e:
        print(f"[Kalibracija] Napaka pri branju kalibracijske datoteke: {e}")
        return None, None, None

def get_fresh_frame(cap, flush_count=5):
    """
    Izprazni notranji buffer kamere pred zajemom referenčne slike.
    """
    for _ in range(flush_count):
        cap.grab()
    return cap.read()

def moveRobot(conn, transform_matrix, product_type, cycle_stop_event):
    """
    Izračuna 3D odmike iz transformacijske matrike (v mm) in jih pošlje Epsonu preko TCP.
    """
    global movementError
    
    # Jasno odpakiranje x, y, z (mm) ter u, v, w (deg)
    dx, dy, dz, du, dv, dw = matrix_to_epson(transform_matrix)
    print(f"[Robot] Pošiljam Local 1 odmike na robot: X={dx:.2f}mm, Y={dy:.2f}mm, Z={dz:.2f}mm | U={du:.2f}°, V={dv:.2f}°, W={dw:.2f}°")
    
    ukaz = f"1 {product_type} {dx:.3f} {dy:.3f} {dz:.3f} {du:.3f} {dv:.3f} {dw:.3f}\r\n"
    
    try:
        conn.sendall(ukaz.encode('utf-8'))
        pot_zakljucena = False
        conn.settimeout(0.05)

        tcp_buffer = ""

        while not stop_event.is_set() and not pot_zakljucena:
            # Preverjanje sprotnega premika ali izpada kamere
            with error_lock:
                if movementError:
                    print("[Robot] Zaznan premik ali izpad kamere! Pošiljam ABORT signal robotu.")
                    conn.sendall("2\r\n".encode('utf-8'))
                    return 

            try:
                chunk = conn.recv(1024).decode('utf-8')
                if not chunk:
                    print("[Robot] Povezava je bila prekinjena s strani robota.")
                    return 
                
                tcp_buffer += chunk

                while "\n" in tcp_buffer:
                    line, tcp_buffer = tcp_buffer.split("\n", 1)
                    odgovor = line.strip()

                    if odgovor == "1":
                        print("[Robot] Uspešno izveden celoten cikel nanosa!")
                        pot_zakljucena = True
                        return
                    elif odgovor == "2" or odgovor == "ERROR":
                        print(f"[Robot] Robot je javil napako ali prekinitev: {odgovor}")
                        return
                        
            except socket.timeout:
                pass
            except Exception as e:
                print(f"[Robot] Napaka pri poslušanju robota: {e}")
                return 
                
    except Exception as e:
        print(f"[Robot] Napaka pri pošiljanju ukaza: {e}")
        return 
    finally:
        cycle_stop_event.set()
        try:
            conn.settimeout(None)
        except Exception:
            pass

# ======================================================================
# NADZOR PREMIKA IZDELKA (VARNOSTNI WATCHDOG)
# ======================================================================
def checkMovement(cap, frame, cycle_stop_event):
    """
    Spremlja premik izdelka MED izvajanjem nanosa.
    Če kamera izgubi sliko ali zazna premik, sproži ABORT.
    """
    return
    global movementError, ROI_SIZE, MOVEMENT_THRESHOLD, MOVEMENT_INTERVAL_SECONDS
    
    h, w, _ = frame.shape
    x_start, y_start = (w - ROI_SIZE) // 2, (h - ROI_SIZE) // 2 + 100

    siva = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ref_skrita = siva[y_start:y_start+ROI_SIZE, x_start:x_start+ROI_SIZE]
    
    zadnji_cas = time.time()
    
    while not stop_event.is_set() and not cycle_stop_event.is_set():
        ret, trenutni_frame = cap.read()
        
        # VARNOSTNI POPRAVEK: Izpad kamere sproži ABORT robota
        if not ret:
            if not cycle_stop_event.is_set():
                print("[Kamera] NAPAKA: Izguba video signala kamere med ciklom! Sprožam ABORT...")
                with error_lock:
                    movementError = True
            break
        
        if time.time() - zadnji_cas >= MOVEMENT_INTERVAL_SECONDS:
            zadnji_cas = time.time()
            siva_trenutna = cv2.cvtColor(trenutni_frame, cv2.COLOR_BGR2GRAY)
            iskano_obmocje = siva_trenutna[y_start:y_start+ROI_SIZE, x_start:x_start+ROI_SIZE]
            
            rezultat = cv2.matchTemplate(iskano_obmocje, ref_skrita, cv2.TM_CCOEFF_NORMED)
            _, max_ujemanje, _, _ = cv2.minMaxLoc(rezultat)
            
            if not cycle_stop_event.is_set() and max_ujemanje < MOVEMENT_THRESHOLD:
                print(f"[Kamera] Premik zaznan! Ujemanje: {max_ujemanje:.2f}")
                with error_lock:
                    movementError = True
                break

        cv2.rectangle(trenutni_frame, (x_start, y_start), (x_start+ROI_SIZE, y_start+ROI_SIZE), (255, 255, 255), 2)
        cv2.imshow("Spremljanje Premika", trenutni_frame)
        
        # Prekinitev operaterja prav tako sproži ABORT
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if not cycle_stop_event.is_set():
                print("[Kamera] Operater je prekinil spremljanje! Sprožam ABORT...")
                with error_lock:
                    movementError = True
            break

    cv2.destroyAllWindows()
    print("[Kamera] Nadzor premika za ta cikel zaključen.")

# ======================================================================
# GLAVNI PROGRAM
# ======================================================================
def main():
    global movementError
    
    #camera_matrix, dist_coeffs, hand_eye_matrix_mm = load_calibration_data()
    #if hand_eye_matrix_mm is None:
       # return

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((TCP_IP, TCP_PORT))
    server_socket.listen(1)
    
    print(f"\n[Sistem] Strežnik posluša na portu {TCP_PORT}...")
    print("[Sistem] Zaženi program na Epson robotu zdaj.")
    
    global_conn = None
    try:
        global_conn, addr = server_socket.accept()
        print(f"[Sistem] Povezava VZPOSTAVLJENA z naslova: {addr}\n")
    except Exception as e:
        print(f"[Sistem] Napaka pri vzpostavljanju povezave: {e}")
        server_socket.close()
        return
    """
    print("[Kamera] Povezujem kamero...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[Kamera] Napaka: Ni mogoče odpreti kamere.")
        global_conn.close()
        server_socket.close()
        return
    print("[Kamera] Kamera uspešno povezana.")
    """
    print("\nSistem pripravljen. Postavi izdelek v POI in pritisni ENTER.")
    
    try: 
        while not stop_event.is_set():
            if keyboard.is_pressed('enter'):
                print("\n[Main] Enter pritisnjen. Zajemam svežo sliko in začenjam cikel...")
                
                while keyboard.is_pressed('enter'):
                    time.sleep(0.05)
                """ 
                ret, frame = get_fresh_frame(cap, flush_count=5)
                if not ret:
                    print("[Kamera] Napaka pri zajemu referenčne slike za ta cikel!")
                    continue
                """
                product_type = 1 
                with error_lock:
                    movementError = False
                
                cycle_stop_event = threading.Event()
                
                TRAN_MATRIX_MM = np.array([
                    [1.0,  0.0,  0.0, -500.0],   # X offset = -500 mm
                    [0.0,  1.0,  0.0,    0.0],   # Y offset = 0 mm
                    [0.0,  0.0,  1.0,  250.0],   # Z offset = +250 mm
                    [0.0,  0.0,  0.0,    1.0]
                ], dtype=np.float64) 

                time.sleep(0.2) 
                
                mainThread = threading.Thread(target=moveRobot, args=(global_conn, TRAN_MATRIX_MM, product_type, cycle_stop_event))
                ##movementThread = threading.Thread(target=checkMovement, args=(cap, frame, cycle_stop_event))
                
                #movementThread.start()
                mainThread.start()
                
                #movementThread.join()
                mainThread.join()
                
                print("[Main] Cikel zaključen. Delavec lahko varno odstrani kos.")
                print("[Main] Pripravljen na nov izdelek (Pritisni ENTER)...\n")
                time.sleep(0.5)
            
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n\n[Main] Zaznan Ctrl + C! Sprožam varen izhod...")
        stop_event.set()
    finally:
        if cap and cap.isOpened():
            cap.release()
            cv2.destroyAllWindows()
            
        if global_conn:
            try:
                global_conn.close()
            except Exception:
                pass
        try:
            server_socket.close()
        except Exception:
            pass
        print("[Main] Strežnik in kamera zaprta. Program zaključen.")
        sys.exit(0)

if __name__ == "__main__":
    main()