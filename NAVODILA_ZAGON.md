# Navodila za zagon projekta IM (nanašalec tesnila)

Navodila so za **Windows 10/11**. Vse ukaze poganjaš v **Anaconda Prompt** (ali v VS Code terminalu, ko je izbrano conda okolje).

## 0. Hiter povzetek

| Kaj | Zakaj | Obvezno? |
|---|---|---|
| Miniconda (conda) | upravljanje Python okolja | da |
| Python 3.11 (v conda okolju `open3d_env`) | open3d **ne podpira** Python 3.13 | da |
| Python paketi (glej točko 3) | koda | da |
| VS Code + Python extension | urejanje in zagon | priporočeno |
| Git | prenos repozitorija | da |
| Windows "long paths" | dolge poti v OneDrive mapi | priporočeno |
| Epson RC+ 7.x | robotski programi (`.sprj`) | samo za robota |
| SolidWorks | `.SLDPRT` modeli | samo za urejanje CAD |

---

## 1. Git

1. Prenesi z <https://git-scm.com/download/win> in namesti s privzetimi nastavitvami.
2. Kloniraj repozitorij:
   ```bash
   git clone https://github.com/Slana-dev/IM.git
   cd IM
   ```

## 2. Miniconda (conda)

1. Prenesi **Miniconda3 Windows 64-bit** z <https://www.anaconda.com/download/success> (spodaj, razdelek *Miniconda Installers*).
2. Namesti z izbiro **"Just Me"** in privzeto mapo (`C:\Users\<ime>\miniconda3`). Kljukice *Add to PATH* ni treba dati.
3. V Start meniju odpri **Anaconda Prompt (miniconda3)** in preveri:
   ```bash
   conda --version
   ```

> Če conda javlja napake tipa `DLL load failed ... libmamba`, posodobi bazo:
> `conda update -n base conda` ali pa namesto conda solverja uporabi
> `conda config --set solver classic`.

### Omogoči dolge poti (Windows)
Projekt je v OneDrive mapi s presledki in šumniki, poti hitro presežejo 260 znakov (Open3D/pip namestitev lahko pade).
PowerShell **kot administrator**:
```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```
Nato ponovno zaženi računalnik.

## 3. Python okolje in paketi

### 3.1 Ustvari okolje (samo prvič)
```bash
conda create -n open3d_env python=3.11 -y
conda activate open3d_env
```
> Uporabi **Python 3.11** (deluje tudi 3.10/3.12). Na 3.13 open3d 0.19 ni na voljo.

### 3.2 Namesti pakete
```bash
pip install open3d==0.19.0 numpy scipy pandas openpyxl joblib "opencv-python<5" keyboard matplotlib ipykernel
```

| Paket | Kje se uporablja |
|---|---|
| `open3d` | `dolocanje_tocne_lokacije/main_trial.py` – registracija point cloudov (RANSAC + ICP), vizualizacija |
| `numpy` | povsod |
| `scipy` | `komunikacija/send_localcoord.py`, `kalibracija/...` – rotacije (`scipy.spatial.transform`) |
| `pandas`, `openpyxl` | `camera_benchmark.py`, `camera_translation_benchmark.py` – zapis v `.xlsx` |
| `joblib` | `main_trial.py` – vzporedno računanje (neobvezno, a precej pohitri) |
| `opencv-python<5` | kamera, ChArUco kalibracija. **Mora biti 4.x**, v 5.0 `calibrateHandEye` ne deluje zanesljivo |
| `keyboard` | `send_localcoord.py`, `ToolCP_avtomaticno.py` – tipka Enter |
| `matplotlib`, `ipykernel` | `kalibracija/TCP_kalibracija/TCP_kalibracija.ipynb` |

> `numpy` naj bo 2.x. Če pride do konflikta z opencv, namesti `pip install "numpy<2.5"`.

### 3.3 Preveri namestitev
```bash
python -c "import open3d, numpy, scipy, pandas, cv2; print(open3d.__version__, numpy.__version__, cv2.__version__)"
```

## 5. Zagon posameznih delov

Vedno najprej: `conda activate open3d_env`.

### 5.1 Določanje točne lokacije (Open3D registracija)
Skripte poganjaj **iz mape `dolocanje_tocne_lokacije`** (STL in `.preprocess_cache` se iščejo relativno).
```bash
cd dolocanje_tocne_lokacije
python kontrolna_plosca.py
```
- `kontrolna_plosca.py` – glavni vhod: parametre (kamera, šum, naklon, premik…) urediš v slovarju `CONFIG` na vrhu datoteke, nato poženeš.
- `python main_trial.py --help` – vsi parametri; direkten zagon npr.
  `python main_trial.py --iss_target_min_keypoints 1000 --iss_target_max_keypoints 4000`
  (v VS Code to naredi konfiguracija **"Main"** v `.vscode/launch.json`, `F5`).
- `python camera_benchmark.py` / `python camera_translation_benchmark.py` – primerjava kamer, rezultat v `.xlsx` (Excel datoteka med zagonom **ne sme biti odprta**).
- Potreben je STL `eHDS S Housing + CC s pottingom, fine.STL` (je v repozitoriju; izvožen mora biti v koordinatnem sistemu *Coordinate System1*, mm).
- Prvi zagon je počasen (vzorčenje STL), naslednji uporabijo `.preprocess_cache`.

### 5.2 Kalibracija
- **TCP kalibracija:** odpri `kalibracija/TCP_kalibracija/TCP_kalibracija.ipynb` v VS Code in poženi celice (kernel `open3d_env`).
  `ToolCP_avtomaticno.py` komunicira z robotom/simulatorjem (privzeto `127.0.0.1:12345`).
- **Eye-in-hand:** `python "kalibracija/eye-in-hand kalibracija/eye hand V1.py"` (potrebuje slike ChArUco table 7×9, kvadrat 30 mm, marker 24 mm, `DICT_4X4_50`).

### 5.3 Komunikacija z Epson robotom
```bash
python komunikacija/send_localcoord.py
```
Python je TCP **strežnik** (port `12345`), robot se poveže kot klient. Za `keyboard` po potrebi poženi terminal kot administrator.

Omrežje (iz `komunikacija/ipconfiguracija.txt`):
- PC: ročni IPv4 `192.168.0.10`, maska `255.255.255.0`
  (*Nastavitve → Omrežje → Ethernet → Dodelitev IP → Ročno*).
- Epson krmilnik: `192.168.0.20`, maska `255.255.255.0`.
- Epson RC+: *Setup → System Configuration → Controller → TCP/IP* → Port **#201** (Client), Remote IP `192.168.0.10`, Remote Port `12345`.
- *Setup → PC to Controller Communications* → *Add* → Ethernet, IP `192.168.0.20`.
- Windows požarni zid mora dovoliti Python na zasebnih omrežjih (ob prvem zagonu klikni *Allow*).


## 7. Pogoste težave

| Napaka | Rešitev |
|---|---|
| `ModuleNotFoundError: open3d` | Nisi v pravem okolju → `conda activate open3d_env` / v VS Code izberi interpreter |
| `pip` ne najde open3d | Python je 3.13 → ustvari okolje s `python=3.11` |
| `FileNotFoundError` za STL | Poženi iz mape `dolocanje_tocne_lokacije` |
| `cv2.calibrateHandEye` manjka | `pip install "opencv-python<5"` |
| `PermissionError` pri `.xlsx` | Zapri Excel datoteko |
| Okno Open3D se ne odpre | Posodobi grafične gonilnike; na oddaljenem namizju (RDP) OpenGL pogosto ne deluje |
| Robot se ne poveže | Preveri IP PC-ja, port 12345 in požarni zid; Python skripta mora teči **pred** zagonom programa na robotu |
