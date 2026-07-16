import threading
import time
import keyboard
import cv2
import os
import sys
import socket  # TCP/IP komunikacija
import numpy as np
from scipy.spatial.transform import Rotation as R

# --- NASTAVITVE ZA TCP/IP ---
TCP_IP = "127.0.0.1"   # Pusti 127.0.0.1 za simulator, spremeni v IP računalnika za realnega robota
TCP_PORT = 12345       # V Epson Port 201 nastavi enak port

# Skupna deljena spremenljivka za preverjanje premika
movementError = False
error_lock = threading.Lock()
stop_event = threading.Event()  # POPRAVLJENO: Počiščena tiskarska napaka

ROI_SIZE = 100 
MOVEMENT_THRESHOLD = 0.9 
MOVEMENT_INTERVAL_SECONDS = 0.5 


def epson_to_matrix(x, y, z, u, v, w):
    rot = R.from_euler('zyx', [u, v, w], degrees=True)
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = [x, y, z]
    return T

def matrix_to_epson(T):
    x, y, z = T[:3, 3]
    rot = R.from_matrix(T[:3, :3])
    u, v, w = rot.as_euler('zyx', degrees=True)
    return x, y, z, u, v, w

def moveRobot(transform_matrix, product_type):
    """
    Izračuna 3D odmike iz kalibracijske matrike in jih pošlje Epsonu,
    da si nastavi Local 1. Nato čaka na potrditev o koncu celotne poti.
    """
    global movementError
    
    # 1. IZ RAČUNA ODMIKOV (Iz 4x4 matrike dobimo dx, dy, dz, rx, ry, rz)
    # Ker transform_matrix že predstavlja premik, jo neposredno pretvorimo
    dx, dy, dz, rx, ry, rz = matrix_to_epson(transform_matrix)
    
    print(f"[Robot] Izračunani odmiki za Local 1:")
    print(f"        T: [{dx:.2f}, {dy:.2f}, {dz:.2f}]")
    print(f"        R: [{rx:.2f}, {ry:.2f}, {rz:.2f}]")
    print(f"[Robot] Odpiram Socket Server na portu {TCP_PORT}...")

    # --- 2. VZPOSTAVITEV TCP STREŽNIKA ---
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((TCP_IP, TCP_PORT))
        server_socket.listen(1)
        print("[Robot] Čakam na povezavo Epson robota...")
        
        server_socket.settimeout(1.0)
        conn = None
        while not stop_event.is_set():
            try:
                conn, addr = server_socket.accept()
                print(f"[Robot] Robot se je povezal iz: {addr}")
                break
            except socket.timeout:
                continue
                
        if stop_event.is_set() or conn is None:
            server_socket.close()
            return

        # --- 3. POŠILJANJE KOORDINATNEGA SISTEMA ---
        # Pošljemo samo EN paket z nastavitvami za Local 1
        # Format: "1 product_type dx dy dz rx ry rz\n"
        ukaz = f"1 {product_type} {dx:.3f} {dy:.3f} {dz:.3f} {rx:.3f} {ry:.3f} {rz:.3f}\n"
        conn.sendall(ukaz.encode('utf-8'))
        print("[Robot] Podatki za Local 1 poslani. Robot začenja s potjo P0-P4.")

        # --- 4. ČAKANJE NA POTRDITEV CELOTNE POTI ---
        # Robot zdaj vozi samostojno, mi le poslušamo in po potrebi javimo napako kamere
        pot_zakljucena = False
        conn.settimeout(0.05)  # 50 ms timeout, da ne blokiramo niti za kamero

        while not stop_event.is_set() and not pot_zakljucena:
            
            # Preverjanje kamere: Če se izdelek premakne, robotu takoj pošljemo kodo 2 (Abort)
            with error_lock:
                if movementError:
                    print("[Robot] Zaznan premik izdelka! Pošiljam ABORT signal robotu.")
                    conn.sendall("2\n".encode('utf-8'))
                    return

            try:
                # Čakamo na končni odgovor "1" s strani robota
                odgovor = conn.recv(1024).decode('utf-8').strip()
                if not odgovor:
                    print("[Robot] Robot je predčasno zaprl povezavo.")
                    return
                
                if odgovor == "1":
                    print("[Robot] Robot javlja: Celotna pot P0-P4 je uspešno prevožena!")
                    pot_zakljucena = True
                    
            except socket.timeout:
                # Timeout je med vožnjo normalen, zanka se samo zavrti in spet preveri kamero
                pass
            except Exception as e:
                print(f"[Robot] Napaka pri komunikaciji: {e}")
                return

    except Exception as e:
        print(f"[Robot] Splošna napaka v socket strežniku: {e}")
        
    finally:
        if conn:
            conn.close()
        server_socket.close()
        print("[Robot] Povezava zaprta.")

def checkMovement():
    return # trenutno ne uporabimo čekiranja premika 
    global movementError, ROI_SIZE, MOVEMENT_THRESHOLD, MOVEMENT_INTERVAL_SECONDS

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Kamera] Napaka: Ni mogoče odpreti kamere.")
        return

    ret, frame = cap.read()
    if not ret:
        print("[Kamera] Napaka pri zajemu prve slike.")
        cap.release()
        return

    h, w, _ = frame.shape
    x_start, y_start = (w - ROI_SIZE) // 2 , (h - ROI_SIZE) // 2 + 100

    siva = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ref_skrita = siva[y_start:y_start+ROI_SIZE, x_start:x_start+ROI_SIZE]
    
    zadnji_cas = time.time()
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret: 
            break
        
        if time.time() - zadnji_cas >= MOVEMENT_INTERVAL_SECONDS:
            zadnji_cas = time.time()
            siva_trenutna = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            iskano_obmocje = siva_trenutna[y_start:y_start+ROI_SIZE, x_start:x_start+ROI_SIZE]
            
            rezultat = cv2.matchTemplate(iskano_obmocje, ref_skrita, cv2.TM_CCOEFF_NORMED)
            _, max_ujemanje, _, _ = cv2.minMaxLoc(rezultat)
            
            if max_ujemanje < MOVEMENT_THRESHOLD:
                print(f"[Kamera] Premik zaznan! Ujemanje: {max_ujemanje:.2f}")
                with error_lock:
                    movementError = True
                break

        cv2.rectangle(frame, (x_start, y_start), (x_start+ROI_SIZE, y_start+ROI_SIZE), (255,0,0), 2)
        cv2.imshow("Spremljanje Premika", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    global movementError
    
    # Testna kalibracijska matrika (v realnosti bo to tvoja izračunana matrika)
    TRAN_MATRIX = np.array([
        [1.0, 0, 0.0, 50.0],
        [0.0, 1.0, 0.0, 50.0],
        [0.0, 0.0, 1.0, 50],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    print("Sistem pripravljen. Postavi izdelek v POI in pritisni ENTER...")
    try: 
        while not stop_event.is_set():
            if keyboard.is_pressed('enter'):
                print("\n[Main] Enter pritisnjen. Začenjam cikel...")
                
                while keyboard.is_pressed('enter'):
                    time.sleep(0.05)
                
                product_type = 1 
                
                with error_lock:
                    movementError = False
                    
                time.sleep(0.5) 
                
                # Zaženemo niti: Robotu pošljemo našo TRAN_MATRIX
                mainThread = threading.Thread(target=moveRobot, args=(TRAN_MATRIX, product_type))
                movementThread = threading.Thread(target=checkMovement)
                
                movementThread.start()
                mainThread.start()
                
                movementThread.join()
                mainThread.join()
                
                print("\n[Main] Cikel zaključen. Pripravljen na nov izdelek (Pritisni ENTER)...")
                time.sleep(0.5)
            
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n\n[Main] Zaznan Ctrl + C! Sprožam varen izhod iz vseh niti...")
        stop_event.set()
        time.sleep(1)
        print("[Main] Vse niti ustavljene. Program se zapira.")
        sys.exit(0)

if __name__ == "__main__":
    main()