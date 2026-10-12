# IM – navodila za projekt (nanašalec tesnila)

Navodila veljajo za **Windows 10/11**. Vse ukaze poganjaš v **Anaconda Prompt** (ali v terminalu VS Code, ko je izbrano conda okolje).

---

## 1. Git

1. Namesti Git z <https://git-scm.com/download/win> (privzete nastavitve).
2. Kloniraj repozitorij:
   ```bash
   git clone https://github.com/Slana-dev/IM.git
   cd IM
   ```

---

## 2. Miniconda

1. Prenesi **Miniconda3 Windows 64-bit** z <https://www.anaconda.com/download/success> (razdelek *Miniconda Installers*).
2. Namesti z izbiro **"Just Me"** in privzeto mapo. Možnosti *Add to PATH* ni treba izbrati.
3. Odpri **Anaconda Prompt (miniconda3)** in preveri:
   ```bash
   conda --version
   ```
4. Če želiš condo uporabjati v vscode terminalu odpri **Anaconda Prompt (miniconda3)** in vnesi:
    ```conda init powershell ```
    in v vscode terminalu acceptate terms of service za conda terminal 
     ```bash 
       conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
>>     conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
>>     conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
      ```
      
> Če conda javlja `DLL load failed ... libmamba`, izvedi `conda update -n base conda` ali `conda config --set solver classic`.

---

## 3. Python 3.11 in paketi

Uporabljamo **Python 3.11**. Razlog: open3d 0.19.0 nima wheela za Python 3.13.

> Python 3.11 ni treba namestiti ročno, conda ga namesti sam v okolje v koraku 3.1. Če bi ga vseeno želel posebej, je na voljo na <https://www.python.org/downloads/windows/> (izberi različico 3.11.x).

### 3.1 Ustvari okolje (samo prvič)
```bash
conda create -n im_env python=3.11 -y
conda activate im_env
```
> Ime `im_env` je poljubno, pomembna je verzija **3.11**.

### 3.2 Namesti pakete iz requirements.txt
V mapi projekta (`IM/`), z aktiviranim okoljem:
```bash
cd IM
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3.3 Preveri namestitev
```bash
python -c "import sys, cv2, numpy, scipy, open3d, pandas, keyboard; print(sys.version.split()[0], cv2.__version__, numpy.__version__, open3d.__version__); print(hasattr(cv2.aruco,'CharucoDetector'))"
```
Pričakovano: `3.11.x`, `cv2` 4.x, in zadnja vrstica `True`.

> **Pomembno:** v okolju mora biti samo `opencv-contrib-python`. Če je nameščen tudi `opencv-python`, lahko pride do napak pri `cv2`. Popravek: `pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python`, nato ponovno `pip install -r requirements.txt`.

---

## 4. Omrežje in robot

1. **PC (Računalnik v Windows)** – ročna dodelitev IP:
   *Nastavitve → Omrežje → Ethernet → Dodelitev IP → Ročno → IPv4*
   - IP naslov: `192.168.0.10`
   - Subnet maska: `255.255.255.0`

2. **Epson krmilnik** – *Setup → System Configuration → Controller → Configuration*:
   - IP naslov: `192.168.0.20`
   - Subnet maska: `255.255.255.0`

3. **Epson Port #201 (Client)** – *Setup → System Configuration → Controller → TCP/IP*:
   - Remote IP: `192.168.0.10` (IP vašega PC-ja)
   - Remote Port: `12345`

4. **PC ↔ Controller komunikacija** – *Setup → PC to Controller Communications*:
   - *Add → Connection to real controller via Ethernet*
   - dodaj IP `192.168.0.20`

---

## 5. Moduli projekta

Vedno najprej `conda activate im_env` in `cd IM`.

### 5.1 Hand-eye kalibracija (`hand_eye_kalibracija.py`)

Cilj: izračunati matriko kamera → prijemalo in notranje parametre kamere. Rezultat se shrani v `calib_data/hand_eye_result.json`.

1. Pritrdi ChArUco tablo (6×6, kvadrat 29 mm, marker 21.75 mm, `DICT_4X4_50`) v delovni prostor. Preveri, da kamera vidi celotno tablo.
2. Zaženi:
   ```bash
   python hand_eye_kalibracija.py
   ```
   - Če v `calib_data/` že obstaja `calibration_data.json`, program vpraša: `Želiš nov zajem v živo preko TCP? (y/n)`.
     - `y` → nov zajem z robota.
     - `n` → kalibracija na že shranjenih slikah.
3. Program čaka na povezavo. V **Epson RC+** zaženi funkcijo **HandEyeCalib**. Za vsako točko program shrani sliko in pozicijo ter robotu odgovori `OK`.
4. Ko robot pošlje konec zajema, program izračuna kalibracijo.
5. Preveri izpis:
   - *Reprojekcijska napaka (RMS)* naj bo nizka (praviloma pod ~1 px).
   - *Odstopanje od povprečja* table v bazi robota naj bo majhno (nekaj mm). Če je veliko, kalibracijo ponovi z boljšimi pozicijami.
6. Rezultat: `calib_data/hand_eye_result.json`.

### 5.2 Glavni program (`main.py`)

1. Preveri, da obstaja `calib_data/hand_eye_result.json` (sicer program izpiše napako in se ustavi).
2. Zaženi:
   ```bash
   python main.py
   ```
   Program čaka na robot (`Strežnik posluša na portu 12345...`).
3. Na Epson RC+ zaženi glavni program. Ko se povezava vzpostavi, program izpiše `Sistem pripravljen. Postavi izdelek v POI in pritisni ENTER.`
4. Postavi izdelek v POI in pritisni **Enter**. Program zajame referenčno sliko, pošlje robotu odmike in med nanosom spremlja premik izdelka.
5. Če se izdelek premakne, program robotu pošlje prekinitev (ABORT).
6. Ko robot javi zaključek cikla, delavec odstrani kos in lahko pritisne Enter za naslednjega.
7. Tipke: **Enter** – začetek cikla, **q** v oknu kamere – prekine nadzor cikla, **Ctrl+C** – varen izhod (zapre kamero in povezavo).

> Če tipka Enter ne deluje, zaženi terminal kot administrator.

### 5.3 Določanje točne lokacije (`dolocanje_tocne_lokacije/`)

Skripte poganjaj **iz mape modula**, ker se STL in `.preprocess_cache` iščeta relativno.

```bash
cd dolocanje_tocne_lokacije
python kontrolna_plosca.py
```

- `kontrolna_plosca.py` – glavni vhod. Parametre (kamera, šum, naklon, premik …) uredi v slovarju `CONFIG` na vrhu datoteke.
- `python main_trial.py --help` – vsi parametri, npr. `python main_trial.py --iss_target_min_keypoints 1000 --iss_target_max_keypoints 4000`. V VS Code je to konfiguracija **"Main"** v `.vscode/launch.json` (`F5`).
- `python camera_benchmark.py` / `python camera_translation_benchmark.py` – primerjava kamer, rezultat v `.xlsx`. Excel datoteka med zagonom **ne sme biti odprta** (`PermissionError`).
- STL `eHDS S Housing + CC s pottingom, fine.STL` mora biti v repozitoriju, izvožen v koordinatnem sistemu *Coordinate System1* (mm).
- Prvi zagon je počasen (vzorčenje STL), naslednji uporabijo `.preprocess_cache`.
- Okno Open3D na oddaljenem namizju (RDP) pogosto ne deluje, ker OpenGL ni na voljo.

---

## 6. Pogoste težave

| Napaka | Rešitev |
|---|---|
| `ModuleNotFoundError: open3d` (ali drug paket) | Nisi v pravem okolju → `conda activate im_env`, v VS Code izberi interpreter `im_env` |
| `pip` ne najde `open3d==0.19.0` | Python ni 3.11 (3.13 nima wheela) → ustvari okolje s `python=3.11` |
| `cv2.aruco` ne obstaja / `AttributeError` na `CharucoDetector` | `pip uninstall -y opencv-python` in `pip install -r requirements.txt` |
| `Napaka: Datoteka calib_data/hand_eye_result.json ne obstaja` | Najprej zaženi hand-eye kalibracijo (5.1) |
| `[Kamera] Napaka: Ni mogoče odpreti kamere` | Spremeni indeks v `cv2.VideoCapture(...)` (0, 1, 2) |
| Robot se ne poveže | Preveri IP PC-ja, port `12345`, požarni zid; Python mora teči pred robotom |
| `FileNotFoundError` za STL | Poženi iz mape `dolocanje_tocne_lokacije` |
| `PermissionError` pri `.xlsx` | Zapri Excel datoteko |
| Okno Open3D se ne odpre | Posodobi gonilnike; na RDP OpenGL običajno ne deluje |
| `Premalo veljavnih pozicij table` | Tabla ni dovolj vidna v več pogledih, preveri osvetlitev in parametre table |
