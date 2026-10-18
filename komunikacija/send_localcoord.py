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
TCP_IP = "192.168.0.10"   # ip računalnika
TCP_PORT = 12345       # V Epson Port 201 nastavi enak port

# Skupna deljena spremenljivka za preverjanje premika
movementError = False
error_lock = threading.Lock()
stop_event = threading.Event()  # Za izhod iz celotnega programa (Ctrl+C)

ROI_SIZE = 100 
MOVEMENT_THRESHOLD = 0.95 
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

def moveRobot(conn, transform_matrix, product_type, cycle_stop_event):
    """
    Izračuna 3D odmike iz kalibracijske matrike in jih pošlje Epsonu preko GLOBALNEGA socketa.
    """
    global movementError
    
    dx, dy, dz, rx, ry, rz = matrix_to_epson(transform_matrix)
    print(f"[Robot] Pošiljam Local 1 odmike na robot: X={dx:.2f}, Y={dy:.2f}, Z={dz:.2f}...")
    
    ukaz = f"1 {product_type} {dx:.3f} {dy:.3f} {dz:.3f} {rx:.3f} {ry:.3f} {rz:.3f}\r\n"
    
    try:
        conn.sendall(ukaz.encode('utf-8'))
        pot_zakljucena = False
        conn.settimeout(0.05)  # Kratek timeout za neblokirajoče spremljanje kamere

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
                    print("[Robot] Povezava je bila prekinjena s strani robota.")
                    return 
                
                if odgovor == "1":
                    print("[Robot] Uspešno izveden celoten cikel nanosa!")
                    pot_zakljucena = True
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
        # Sporočimo niti za spremljanje premika, da se je cikel zaključil!
        cycle_stop_event.set()
        
        try:
            conn.settimeout(None)
        except Exception:
            pass

def checkMovement(cap, frame, cycle_stop_event):
    """
    Spremlja premik izdelka MED izvajanjem nanosa.
    Ko moveRobot postavi cycle_stop_event, se nit varno zaključi.
    """
    return
    global movementError, ROI_SIZE, MOVEMENT_THRESHOLD, MOVEMENT_INTERVAL_SECONDS
    
    h, w, _ = frame.shape
    x_start, y_start = (w - ROI_SIZE) // 2 , (h - ROI_SIZE) // 2 + 100

    siva = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ref_skrita = siva[y_start:y_start+ROI_SIZE, x_start:x_start+ROI_SIZE]
    
    zadnji_cas = time.time()
    
    # Zanka teče dokler se ne ugasne cel program ALI pa se ne zaključi trenutni cikel nanosa
    while not stop_event.is_set() and not cycle_stop_event.is_set():
        ret, trenutni_frame = cap.read()
        if not ret: 
            break
        
        if time.time() - zadnji_cas >= MOVEMENT_INTERVAL_SECONDS:
            zadnji_cas = time.time()
            siva_trenutna = cv2.cvtColor(trenutni_frame, cv2.COLOR_BGR2GRAY)
            iskano_obmocje = siva_trenutna[y_start:y_start+ROI_SIZE, x_start:x_start+ROI_SIZE]
            
            rezultat = cv2.matchTemplate(iskano_obmocje, ref_skrita, cv2.TM_CCOEFF_NORMED)
            _, max_ujemanje, _, _ = cv2.minMaxLoc(rezultat)
            
            # Preverimo premik SAMO če cikel še vedno teče
            if not cycle_stop_event.is_set() and max_ujemanje < MOVEMENT_THRESHOLD:
                print(f"[Kamera] Premik zaznan! Ujemanje: {max_ujemanje:.2f}")
                with error_lock:
                    movementError = True
                break

        cv2.rectangle(trenutni_frame, (x_start, y_start), (x_start+ROI_SIZE, y_start+ROI_SIZE), (255, 255, 255), 2)
        cv2.imshow("Spremljanje Premika", trenutni_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    # Zapremo samo pogovorno okno OpenCV za ta cikel, kamero (cap) pa pustimo odprto za naslednji cikel!
    cv2.destroyAllWindows()
    print("[Kamera] Nadzor premika za ta cikel uspešno zaključen.")

def main():
    global movementError
    
    # Testna kalibracijska matrika
    TRAN_MATRIX = np.array([
        [1.0, 0, 0.0, 0],
        [0.0, 1.0, 0.0, 0],
        [0.0, 0.0, 1.0, 0],
        [0.0, 0.0, 0.0, 0]
    ])
    
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

    # Smo izklopili vision watchdoga za testiranje 
    #print("[Kamera] Povezujem kamero...")
    #cap = cv2.VideoCapture(0)
    ##if not cap.isOpened():
        print("[Kamera] Napaka: Ni mogoče odpreti kamere.")
        global_conn.close()
        server_socket.close()
        return
    #print("[Kamera] Kamera uspešno povezana.")

    #print("\nSistem pripravljen. Postavi izdelek v POI in pritisni ENTER.")
    
    try: 
        while not stop_event.is_set():
            if keyboard.is_pressed('enter'):
                print("\n[Main] Enter pritisnjen. Zajemam sliko in začenjam cikel...")
                
                while keyboard.is_pressed('enter'):
                    time.sleep(0.05)
                
                # Pred začetkom zajamemo svežo referenčno sliko kosa za nov cikel
                #ret, frame = cap.read()
                #if not ret:
                   # print("[Kamera] Napaka pri zajemu referenčne slike za ta cikel!")
                    #continue

                product_type = 1 
                with error_lock:
                    movementError = False
                
                cycle_stop_event = threading.Event()
                
                time.sleep(0.2) 
                
                # Zaženemo niti in obema predamo cycle_stop_event
                mainThread = threading.Thread(target=moveRobot, args=(global_conn, TRAN_MATRIX, product_type, cycle_stop_event))
                #movementThread = threading.Thread(target=checkMovement, args=(cap, frame, cycle_stop_event))
                
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
        # Varno zapremo kamero in vtičnice ob izhodu iz celotnega programa
        #if cap and cap.isOpened():
          #  cap.release()
           # cv2.destroyAllWindows()
            
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