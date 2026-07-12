import os

# Pot do vhodne Epson datoteke (črka 'r' pred narekovaji prepreči težave z '\')
vhodna_pot = r"C:\EpsonRC80\projects\IM\robot1.pts"
# Pot do izhodne datoteke (shranila se bo v isto mapo, kjer boste pognali skripto)
izhodna_pot = "points.txt"

tocke = []
trenutna_tocka = {}
iskane_osi = {'rX', 'rY', 'rZ', 'rU', 'rV', 'rW'}

try:
    # Odpremo datoteko (z errors='ignore' preprečimo težave z eksotičnimi znaki v glavi datoteke)
    with open(vhodna_pot, 'r', encoding='utf-8', errors='ignore') as f:
        for vrstica in f:
            vrstica = vrstica.strip()
            
            # Če se začne nov blok točke, ponastavimo začasni slovar
            if vrstica.startswith("Point") and "{" in vrstica:
                trenutna_tocka = {}
                continue
            
            # Ko pridemo do konca bloka '}', shranimo izluščene podatke
            if vrstica == "}":
                if trenutna_tocka:
                    # Izvlečemo podatke v točnem vrstnem redu (če katerega ni, vrne '0')
                    x = trenutna_tocka.get('rX', '0')
                    y = trenutna_tocka.get('rY', '0')
                    z = trenutna_tocka.get('rZ', '0')
                    u = trenutna_tocka.get('rU', '0')
                    v = trenutna_tocka.get('rV', '0')
                    w = trenutna_tocka.get('rW', '0')
                    
                    # Tukaj so koordinate ločene z VEJICO in PRESLEDKOM. 
                    # Če želiš samo presledek, spremeni spodnji niz v: f"{x} {y} {z} {u} {v} {w}"
                    formatirana_vrstica = f"{x}, {y}, {z}, {u}, {v}, {w}"
                    tocke.append(formatirana_vrstica)
                    trenutna_tocka = {}
                continue
            
            # Če vrstica vsebuje '=', gre za podatek znotraj bloka
            if '=' in vrstica:
                kljuc, vrednost = vrstica.split('=', 1)
                kljuc = kljuc.strip()
                vrednost = vrednost.strip()
                
                # Če je ključ eden izmed tistih, ki jih iščemo, ga shranimo
                if kljuc in iskane_osi:
                    trenutna_tocka[kljuc] = vrednost

    # Zapisovanje izluščenih koordinat v points.txt
    with open(izhodna_pot, 'w', encoding='utf-8') as f_out:
        for tocka in tocke:
            f_out.write(tocka + '\n')
            
    print(f"Uspšno predelanih {len(tocke)} točk.")
    print(f"Podatki so shranjeni v: {os.path.abspath(izhodna_pot)}")

except FileNotFoundError:
    print(f"Napaka: Datoteka na poti '{vhodna_pot}' ne obstaja. Preveri, če je pot pravilna.")
except Exception as e:
    print(f"Prišlo je do nepričakovane napake: {e}")