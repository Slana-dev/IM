import threading
import time
import keyboard
import cv2
import os
import sys
# Skupna deljena spremenljivka za preverjanje premika
movementError = False
# Ustvarimo zaklep (Lock) za varno deljenje podatkov med nitmi
error_lock = threading.Lock()
stop_event = threading.DataFrame = threading.Event()

ROI_SIZE = 100                  # Velikost referenčnega območja
MOVEMENT_THRESHOLD = 0.9        # Če pade pod 0.9, se je izdelek premaknil
MOVEMENT_INTERVAL_SECONDS = 0.5 # Kako pogosto kamera preverja premik

def moveRobot(kalibracija, transformacija):
    global movementError
    print(f"[Robot] Pošiljam dx dy dz du dv wd")

    # zapišemo podatke za epson v pipe, ki ga pametno preimenujemo da epson ne bere starih podatkov
    try:
        with open("pipe.tmp", "w") as f:
            f.write("1 0 0 0 10 10 10") # tukaj pridejo dx dy dz rx ry rz
        if os.path.exists("pipe.txt"):
            os.remove("pipe.txt")
        os.rename("pipe.tmp", "pipe.txt")
    except Exception as e:
        print(f"[Robot] Napaka pri pisanju pipe.txt: {e}")
        return

    # Zanka, ki čaka na odziv Epsona ali na napako kamere
    while not stop_event.is_set():
        # Preverjamo napako premika (Kamera)
        with error_lock:
            if movementError:
                print("[Robot] Zaznan premik izdelka med delovanjem!")
                try:
                    with open("done.txt", "w") as f:
                        f.write("2")
                except Exception as e:
                    print(f"[Robot] Napaka pri pisanju napake v pipe.txt: {e}")
                return

        # Preverjamo, če je Epson končal delo 
        if os.path.exists("done.txt"):
            try:
                with open("done.txt", "r") as done:
                    vsebina = done.read().strip()
                    if vsebina == "1":
                        os.rename("pipe.txt", "pipe.tmp")
                        break # Epson je uspešno zaključil
            except IOError:
                # Če je datoteka zaklenjena s strani Epsona v tisti milisekundi, samo poskusimo znova
                pass
        
        time.sleep(0.1) # Brez tega sleep-a bo procesor obremenjen 100% zaradi prazne zanke
        
    print("[Robot] Premik uspešno zaključen.")
    
def checkMovement():
    return # trenuto ne uporabimo cekiranje premika 
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

    # Prvi zajem referenčne slike
    siva = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ref_skrita = siva[y_start:y_start+ROI_SIZE, x_start:x_start+ROI_SIZE]
    
    zadnji_cas = time.time()
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret: 
            break
        
        # Preverjanje na določen časovni interval
        if time.time() - zadnji_cas >= MOVEMENT_INTERVAL_SECONDS:
            zadnji_cas = time.time()
            siva_trenutna = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            iskano_obmocje = siva_trenutna[y_start:y_start+ROI_SIZE, x_start:x_start+ROI_SIZE]
            
            # Template matching
            rezultat = cv2.matchTemplate(iskano_obmocje, ref_skrita, cv2.TM_CCOEFF_NORMED)
            _, max_ujemanje, _, _ = cv2.minMaxLoc(rezultat)
            
            # Če ujemanje pade pod prag, zaklenemo spremenljivko in javimo napako
            if max_ujemanje < MOVEMENT_THRESHOLD:
                print(f"[Kamera] Premik zaznan! Ujemanje: {max_ujemanje:.2f}")
                with error_lock:
                    movementError = True
                break # Prekinemo zanko kamere

        # Vizualizacija v živo
        cv2.rectangle(frame, (x_start, y_start), (x_start+ROI_SIZE, y_start+ROI_SIZE), (255,0,0), 2)
        cv2.imshow("Spremljanje Premika", frame)
        
        # Ročni izhod na 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    # Čiščenje virov (mora biti IZVEN while zanke)
    cap.release()
    cv2.destroyAllWindows()

def main():
    global movementError
    print("Sistem pripravljen. Postavi izdelek v POI in pritisni ENTER...")
    try: 
        while not stop_event.is_set():
            # Čakamo na Enter (ročni start cikla, ko je izdelek na mestu)
            if keyboard.is_pressed('enter'):
                print("\n[Main] Enter pritisnjen. Začenjam cikel...")
                
                # Ponastavimo napako za nov cikel
                with error_lock:
                    movementError = False
                #pobrisemo error od prej
                with open("done.txt", "w") as f:
                    f.write("0")
                if os.path.exists("pipe.txt"):
                    os.remove("pipe.txt")
                # Tukaj pride dejanski izračun 3D matrike in zaznava tipa izdelka
                time.sleep(1) 
                
                mainThread = threading.Thread(target=moveRobot, args=("Matrika_Kalib", "Matrika_Trans"))
                movementThread = threading.Thread(target=checkMovement)
                
                # Zaženemo obe niti hkrati
                movementThread.start()
                mainThread.start()
                
                # Čakamo, da se obe niti zaključita, preden dovolimo nov cikel
                movementThread.join()
                mainThread.join()
                
                print("\n[Main] Cikel zaključen. Pripravljen na nov izdelek (Pritisni ENTER)...")
                time.sleep(1) # Kratka pavza pred naslednjim zaznavanjem gumba
    except KeyboardInterrupt:
        print("\n\n[Main] Zaznan Ctrl + C! Sprožam varen izhod iz vseh niti...")
        stop_event.set() # Nastavimo zastavico na True -> niti bodo takoj prekinile zanke
        
        time.sleep(1)
        print("[Main] Vse niti ustavljene. Program se zapira.")
        sys.exit(0)
if __name__ == "__main__":
    main()