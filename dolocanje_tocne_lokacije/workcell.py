"""Fizična postavitev delovne celice, skupna simulaciji in registraciji.

Kamera je VEDNO fiksirana v izhodišču koordinatnega sistema (0,0,0), za
realne IN simulirane skene enako: prava kamera je fiksno montirana in ne
sledi delu, zato se v simulaciji premika del (translation_x/y/z,
rotation_*_deg), ne kamera. Del leži na mizi znane velikosti - ista miza
nastopa kot ozadje v simuliranem skenu (scan_simulation.build_table_mesh) in
kot fizični prior pri grobi registraciji (coarse_registration).
"""
from dataclasses import dataclass

import numpy as np

CAMERA_LOCATION = np.zeros(3)


@dataclass(frozen=True)
class Table:
    size_x: float   # širina mize (mm)
    size_y: float   # globina mize (mm)
    center: np.ndarray   # center mize v ravnini pravokotni na up_axis (X/Y)

    @property
    def half_extent(self) -> np.ndarray:
        return np.array([self.size_x, self.size_y]) / 2.0   # polovična širina/globina mize v vsaki smeri

    def describe(self) -> str:
        return (f"{self.size_x:.0f}x{self.size_y:.0f}mm centered at "
                f"{(float(self.center[0]), float(self.center[1]))}")
