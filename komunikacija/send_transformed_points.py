import threading
import time
import keyboard
import cv2
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R

# Skupna deljena spremenljivka za preverjanje premika
movementError = False
# Ustvarimo zaklep (Lock) za varno deljenje podatkov med nitmi
error_lock = threading.Lock()
# POPRAVLJENO: Pravilna inicializacija Event objekta
stop_event = threading.Event()

ROI_SIZE = 100                  # Velikost referenčnega območja
MOVEMENT_THRESHOLD = 0.9        # Če pade pod 0.9, se je izdelek premaknil
MOVEMENT_INTERVAL_SECONDS = 0.5 # Kako pogosto kamera preverja premik


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

def preberi_points_txt(filepath):
    """Prebere točke iz pripravljade datoteke points.txt."""
    tocke = []
    if not os.path.exists(filepath):
        return tocke

    with open(filepath, 'r', encoding='utf-8') as f:
        for vrstica in f:
            vrstica = vrstica.strip()
            if not vrstica:  # Preskoči prazne vrstice
                continue
            
            try:
                # Razbijemo vrstico glede na vejice in pretvorimo v decimalna števila (float)
                koordinate = [float(stevilka.strip()) for stevilka in vrstica.split(',')]
                if len(koordinate) == 6:
                    tocke.append(koordinate)
            except ValueError:
                print(f"[Opozorilo] Preskočena neveljavna vrstica: {vrstica}")
                
    return tocke

def moveRobot(transform_matrix, product_type):
    """
    Prebere točke glede na tip izdelka, jih transformira s 4x4 matriko 
    in jih pošilja Epsonu eno po eno prek pipe.txt.
    """
    global movementError
    
    # POPRAVLJENO: Dinamična izbira datoteke glede na tip izdelka
    txt_pot = f"points{product_type}.txt"  
    if not os.path.exists(txt_pot):
        txt_pot = "points.txt" # Fallback na privzeto datoteko, če specifična ne obstaja
    
    # 1. Preberemo vnaprej pripravljene točke
    surove_tocke = preberi_points_txt(txt_pot)
    if not surove_tocke:
        print(f"[Robot] Napaka: Ni najdenih točk v {txt_pot} ali datoteka ne obstaja.")
        return

    print(f"[Robot] Uspešno prebranih {len(surove_tocke)} točk iz {txt_pot}. Začenjam sekvenco za Tip: {product_type}...")

    # Pred začetkom počistimo morebitne stare datoteke za usklajevanje
    for datoteka in ["done.txt", "pipe.txt", "pipe.tmp"]:
        if os.path.exists(datoteka):
            try: os.remove(datoteka)
            except: pass

    # 2. Loop skozi vsako točko posebej
    for i, tocka in enumerate(surove_tocke):
        if stop_event.is_set():
            break

        # Pretvorba prebrane točke [x, y, z, u, v, w] v 4x4 matriko
        T_robot = epson_to_matrix(*tocka)
        
        # Transformacija s tvojo kalibracijsko matriko
        T_koncna = T_robot @ transform_matrix
        
        # Pretvorba nazaj v Epson koordinate
        x, y, z, u, v, w = matrix_to_epson(T_koncna)
        print(f"[Robot] Pošiljam točko {i}/{len(surove_tocke)}: X={x:.2f} Y={y:.2f} Z={z:.2f}")

        # 3. Zapis v pipe.tmp in preimenovanje v pipe.txt
        # POPRAVLJENO: Namesto trde enice pošljemo dejanski product_type, da ga Epson prebere
        vsebina_pipe = f"{product_type} {x:.3f} {y:.3f} {z:.3f} {u:.3f} {v:.3f} {w:.3f}"
        
        try:
            with open("pipe.tmp", "w") as f:
                f.write(vsebina_pipe)
            if os.path.exists("pipe.txt"):
                os.remove("pipe.txt")
            os.rename("pipe.tmp", "pipe.txt")
        except Exception as e:
            print(f"[Robot] Napaka pri pisanju pipe.txt za točko {i}: {e}")
            return

        # 4. Notranja zanka: Čakamo na potrditev s strani Epsona (done.txt)
        tocka_prejeta = False
        while not stop_event.is_set() and not tocka_prejeta:
            
            # Preverjanje napake (Kamera)
            with error_lock:
                if movementError:
                    print("[Robot] Zaznan premik izdelka med delovanjem! Prekinjam.")
                    try:
                        with open("done.txt", "w") as f:
                            f.write("2")
                    except Exception as e:
                        print(f"[Robot] Napaka pri pisanju javljanja napake: {e}")
                    return

            # Preverjanje, če je Epson končal trenutno točko
            if os.path.exists("done.txt"):
                try:
                    with open("done.txt", "r") as done:
                        vsebina = done.read().strip()
                    
                    if vsebina == "1":
                        os.remove("done.txt")  # Izbrišemo done.txt, da sprostimo naslednji korak
                        tocka_prejeta = True  # Gremo na naslednjo točko
                        
                except (IOError, PermissionError):
                    pass  # Datoteka je trenutno zaklenjena, poskusimo znova v naslednjem ciklu
            
            time.sleep(0.05)

    # 5. Ko zmanjka točk, pošljemo Epsonu končni signal (koda 3 v done.txt)
    if not stop_event.is_set():
        try:
            with open("done.tmp", "w") as f:
                f.write("3")
            if os.path.exists("done.txt"): 
                os.remove("done.txt")
            os.rename("done.tmp", "done.txt")
            print(f"[Robot] Vse točke za izdelek {product_type} so bile uspešno predelane. Zapisana koda 3.")
        except Exception as e:
            print(f"[Robot] Napaka pri zapiranju sekvence (koda 3): {e}")
            
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
    TRAN_MATRIX = np.array([
        [1.0, 0, 0.0, 50.0],
        [0.0, 1.0, 0.0, 50.0],
        [0.0, 0.0, 1.0, 50],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    print("Sistem pripravljen. Postavi izdelek v POI in pritisni ENTER...")
    try: 
        while not stop_event.is_set():
            # Čakamo na Enter (ročni start cikla, ko je izdelek na mestu)
            if keyboard.is_pressed('enter'):
                print("\n[Main] Enter pritisnjen. Začenjam cikel...")
                
                # POPRAVLJENO: Čakamo, da uporabnik spusti Enter (prepreči dvojno proženje)
                while keyboard.is_pressed('enter'):
                    time.sleep(0.05)
                
                # Tukaj določiš tip izdelka (ročno ali preko tvoje logike/kamere)
                # Zaenkrat nastavimo na 1, kar pomeni points1.txt (ujema se z Epsonovo logiko)
                product_type = 1 
                
                # Ponastavimo napako za nov cikel
                with error_lock:
                    movementError = False
                    
                # Pobrišemo staro stanje v done.txt
                with open("done.txt", "w") as f:
                    f.write("0")
                if os.path.exists("pipe.txt"):
                    os.remove("pipe.txt")
                    
                time.sleep(0.5) 
                
                # POPRAVLJENO: Pravilen zapis torke (tuple) v args z nujno vejico na koncu
                mainThread = threading.Thread(target=moveRobot, args=(TRAN_MATRIX, product_type))
                movementThread = threading.Thread(target=checkMovement)
                
                # Zaženemo obe niti hkrati
                movementThread.start()
                mainThread.start()
                
                # Čakamo, da se obe niti zaključita, preden dovolimo nov cikel
                movementThread.join()
                mainThread.join()
                
                print("\n[Main] Cikel zaključen. Pripravljen na nov izdelek (Pritisni ENTER)...")
                time.sleep(0.5)
            
            time.sleep(0.05) # Razbremenitev procesorja v glavni zanki
            
    except KeyboardInterrupt:
        print("\n\n[Main] Zaznan Ctrl + C! Sprožam varen izhod iz vseh niti...")
        stop_event.set()
        time.sleep(1)
        print("[Main] Vse niti ustavljene. Program se zapira.")
        sys.exit(0)

if __name__ == "__main__":
    main()