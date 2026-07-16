import socket
import sys

# --- NASTAVITVE ---
TCP_IP = "0.0.0.0"   # Posluša na vseh mrežnih vmesnikih računalnika
TCP_PORT = 12345     # Port mora biti enak kot v Epson nastavitvah (Port 201)

def main():
    print(f"[Python] Odpiram Server in poslušam na portu {TCP_PORT}...")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Preprečimo napako "Address already in use" pri hitrih ponovnih zagonih
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server_socket.bind((TCP_IP, TCP_PORT))
        server_socket.listen(1)
        print("[Python] Čakam, da se poveže Epson Robot (Client)...")
        
        # Blokirajoče čakanje na robota
        conn, addr = server_socket.accept()
        print(f"[Python] Robot se je uspešno povezal iz naslova: {addr}")
    except Exception as e:
        print(f"[Python] Napaka pri zagonu strežnika: {e}")
        sys.exit(1)

    tocke_count = 0
    try:
        while True:
            # Čakamo na točke, ki jih robot pošlje ob pritisku na Enter
            data = conn.recv(1024).decode('utf-8').strip()

            # Če robot prekine povezavo
            if not data:
                print("[Python] Robot je nepričakovano zaprl povezavo.")
                break

            # Če robot javi, da želi končati ("exit")
            if data.lower() == "exit":
                print("[Python] Zaznan 'exit' signal iz robota. Zapiram povezavo...")
                break

            print(f"\n[Prejeto] Surovi podatki iz robota: '{data}'")

            try:
                # Razdelimo prejeti niz po vejicah v seznam števil (float)
                koordinate = [float(val.strip()) for val in data.split(',')]
                
                if len(koordinate) == 6:
                    tocke_count += 1
                    x, y, z, u, v, w = koordinate
                    print(f"== TOČKA #{tocke_count} USPEŠNO ZABELEŽENA ==")
                    print(f"   X: {x:8.3f} mm")
                    print(f"   Y: {y:8.3f} mm")
                    print(f"   Z: {z:8.3f} mm")
                    print(f"   U: {u:8.3f} ° (Rotacija okoli Z)")
                    print(f"   V: {v:8.3f} ° (Rotacija okoli Y)")
                    print(f"   W: {w:8.3f} ° (Rotacija okoli X)")
                else:
                    print("[Opozorilo] Prejeti podatki nimajo natanko 6 komponent!")
            except ValueError:
                print("[Opozorilo] Napaka pri pretvorbi podatkov v števila!")

    except KeyboardInterrupt:
        print("\n[Python] Prekinjeno s tipkovnico (Ctrl+C).")
    finally:
        conn.close()
        server_socket.close()
        print("[Python] Strežnik varno zaprt.")

if __name__ == "__main__":
    main()