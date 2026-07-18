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

def izracunaj_tcp_4_tocke(tocke_list):
    """
    Industrijski algoritem za 4-točkovno kalibracijo TCP (konice orodja).
    Iz 4 različnih orientacij flanše, ki se dotikajo iste konice v prostoru,
    izračuna natančen 3D odmik [dx, dy, dz] konice glede na flanšo.
    """
    A = []
    B = []
    
    for tocka in tocke_list:
        T = epson_to_matrix(*tocka)
        R_flange = T[:3, :3]
        P_flange = T[:3, 3]
        
        # Enačba: P_target = P_flange + R_flange * L_tool
        # Preoblikovano v linearni sistem: [I_3 | -R_flange] * [P_target; L_tool] = P_flange
        A_del = np.hstack((np.eye(3), -R_flange))
        A.extend(A_del)
        B.extend(P_flange)
        
    A = np.array(A)
    B = np.array(B)
    
    # Rešimo sistem preko metode najmanjših kvadratov (Least Squares)
    X_resitev, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    
    # Prve 3 komponente so lokacija tarče v prostoru, zadnje 3 pa so odmik konice (TCP)
    L_tool = X_resitev[3:]
    
    T_tool = np.eye(4)
    T_tool[:3, 3] = L_tool
    return T_tool

def moveRobot(conn, transform_matrix, product_type):
    """
    Izračuna 3D odmike iz kalibracijske matrike in jih pošlje Epsonu preko GLOBALNEGA socketa.
    """
    global movementError
    
    dx, dy, dz, rx, ry, rz = matrix_to_epson(transform_matrix)
    
    print(f"[Robot] Pošiljam Local 1 odmike na robot...")
    
    # Sestavimo ukazni niz: "1 product_type dx dy dz rx ry rz\n"
    ukaz = f"1 {product_type} {dx:.3f} {dy:.3f} {dz:.3f} {rx:.3f} {ry:.3f} {rz:.3f}\n"
    
    try:
        conn.sendall(ukaz.encode('utf-8'))
        
        pot_zakljucena = False
        conn.settimeout(0.05)  # Kratek timeout, da zanka ne zablokira checkMovement niti

        while not stop_event.is_set() and not pot_zakljucena:
            # Preverjanje kamere
            with error_lock:
                if movementError:
                    print("[Robot] Zaznan premik izdelka! Pošiljam ABORT signal robotu.")
                    conn.sendall("2\n".encode('utf-8'))
                    return

            try:
                odgovor = conn.recv(1024).decode('utf-8').strip()
                if not odgovor:
                    print("[Robot] Povezava prekinjena.")
                    return
                if odgovor == "1":
                    print("[Robot] Uspešno izveden celoten cikel nanos (P0-P4)!")
                    pot_zakljucena = True
            except socket.timeout:
                pass
            except Exception as e:
                print(f"[Robot] Napaka pri poslušanju robota: {e}")
                return
    except Exception as e:
        print(f"[Robot] Napaka pri pošiljanju ukaza: {e}")

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
    global TRAN_MATRIX  # Matriko definiramo kot globalno, da jo lahko kalibracija posodobi
    
    # Začetna bazična matrika (brez odmika orodja)
    TRAN_MATRIX = np.array([
        [1.0, 0, 0.0, 50.0],
        [0.0, 1.0, 0.0, 50.0],
        [0.0, 0.0, 1.0, 50.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    # --- 1. GLOBALNA VZPOSTAVITEV TCP STREŽNIKA ---
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((TCP_IP, TCP_PORT))
    server_socket.listen(1)
    
    print(f"\n[Sistem] Strežnik posluša na portu {TCP_PORT}...")
    print("[Sistem] Prosim, ZAŽENI program na Epson.")
    
    try:
        global_conn, addr = server_socket.accept()
        print(f"[Sistem] Povezava VZPOSTAVLJENA z naslova: {addr}\n")
    except Exception as e:
        print(f"[Sistem] Napaka pri vzpostavljanju povezave: {e}")
        server_socket.close()
        return

    print("Sistem pripravljen.")
    print("-> Pritisni ENTER za zagon standardnega cikla (Nanos)")
    print("-> Pritisni BACKSPACE za zagon 4-točkovne TCP kalibracije orodja\n")
    
    try: 
        while not stop_event.is_set():
            
            # --- SCENARIJ 1: NAVADEN CIKEL NANOSA (ENTER) ---
            if keyboard.is_pressed('enter'):
                print("\n[Main] Enter pritisnjen. Začenjam cikel...")
                while keyboard.is_pressed('enter'):
                    time.sleep(0.05)
                
                product_type = 1 
                with error_lock:
                    movementError = False
                    
                time.sleep(0.2) 
                
                # Niti poženemo in ji predamo že obstoječo globalno povezavo 'global_conn'
                mainThread = threading.Thread(target=moveRobot, args=(global_conn, TRAN_MATRIX, product_type))
                movementThread = threading.Thread(target=checkMovement)
                
                movementThread.start()
                mainThread.start()
                
                movementThread.join()
                mainThread.join()
                
                print("\n[Main] Pripravljen na naslednji ukaz (ENTER / BACKSPACE)...")
                time.sleep(0.5)
            
            # --- SCENARIJ 2: TCP KALIBRACIJA ORODJA (BACKSPACE) ---
            elif keyboard.is_pressed('backspace'):
                print("\n[Main] Zaznan BACKSPACE! Zaganjam TCP kalibracijo orodja...")
                while keyboard.is_pressed('backspace'):
                    time.sleep(0.05)
                
                # 1. Pošljemo robotu ukaz s kodo 3 (Robot gre v loop za 4 točke)
                ukaz_kalib = "3\n"
                global_conn.sendall(ukaz_kalib.encode('utf-8'))
                
                tocke_kalibracije = []
                # Odstranimo timeout, da ima operater neomejeno časa za jogganje robota
                global_conn.settimeout(None) 
                
                for i in range(0, 4):
                    print(f"[Kalibracija] Čakam, da operater potrdi točko #{i} na robotu...")
                    
                    
                    surovi_podatki = global_conn.recv(1024).decode('utf-8').strip()
                    
                    if not surovi_podatki:
                        print("[Kalibracija] Napaka: Povezava z robotom prekinjena!")
                        break
                        
                    print(f"[Kalibracija] Prejeta točka #{i}: {surovi_podatki}")
                    koordinate = [float(val.strip()) for val in surovi_podatki.split(',')]
                    tocke_kalibracije.append(koordinate)
                
                # Če smo uspešno zajeli vse 4 orientacije
                if len(tocke_kalibracije) == 4:
                    print("[Kalibracija] Vse 4 točke zbrane. Računam TCP matriko...")
                    
                    # Izračunamo 4x4 matriko orodja T_tool
                    T_tool = izracunaj_tcp_4_tocke(tocke_kalibracije)
                    print(f"[Kalibracija] Izračunan odmik orodja: X={T_tool[0,3]:.2f} mm, Y={T_tool[1,3]:.2f} mm, Z={T_tool[2,3]:.2f} mm")
                    

                    TRAN_MATRIX = TRAN_MATRIX @ np.linalg.inv(T_tool)
                    print("[Kalibracija] Uspeh! TRAN_MATRIX je posodobljena z odmiki orodja.")
                else:
                    print("[Kalibracija] Napaka pri zajemu točk. Kalibracija prekinjena.")
                
                print("\n[Main] Pripravljen na naslednji ukaz (ENTER / BACKSPACE)...")
                time.sleep(0.5)
                
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n\n[Main] Zaznan Ctrl + C! Zapiram globalni socket...")
        stop_event.set()
    finally:
        global_conn.close()
        server_socket.close()
        print("[Main] Strežnik varno zaprt. Program zaključen.")
        sys.exit(0)

if __name__ == "__main__":
    main()