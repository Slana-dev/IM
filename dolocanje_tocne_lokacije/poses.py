"""Toge transformacije (4x4) - sestavljanje poz, rotacije in napaka ocenjene poze.

Konvencija v celotnem cevovodu: transformacija preslika CAD-ov lastni
koordinatni sistem v koordinatni sistem kamere, ki je fiksirana v izhodišču
(glej workcell.py).
"""
from typing import Optional

import numpy as np
import open3d as o3d


def build_reference_transform(translation_x: float, translation_y: float, translation_z: float,
                              rotation_x_deg: float, rotation_y_deg: float, rotation_z_deg: float) -> np.ndarray:
    """Sestavi 4x4 pozo dela iz šestih neposrednih parametrov.

    Kamera je VEDNO v izhodišču (0,0,0), zato translation_x/y/z neposredno
    predstavlja pozicijo dela GLEDE NA kamero. translation_z mora biti
    negativen (kamera gleda navzdol, del je pod njo) - kontrolna_plosca.py ga
    samodejno nastavi na (negativno) delovno razdaljo izbrane kamere.

    Rotacija: R = Rz(rotation_z_deg) @ Ry(rotation_y_deg) @ Rx(rotation_x_deg)
    (zunanja/svetovna XYZ Eulerjeva konvencija - vsak zasuk je okoli
    FIKSNE svetovne osi, ne okoli že zasukane lastne osi telesa)."""
    theta_x, theta_y, theta_z = np.radians([rotation_x_deg, rotation_y_deg, rotation_z_deg])
    cx, sx = np.cos(theta_x), np.sin(theta_x)
    cy, sy = np.cos(theta_y), np.sin(theta_y)
    cz, sz = np.cos(theta_z), np.sin(theta_z)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])   # rotacija okoli X (roll)
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])   # rotacija okoli Y (pitch)
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])   # rotacija okoli Z (yaw)
    transform = np.eye(4)
    transform[:3, :3] = rot_z @ rot_y @ rot_x
    transform[:3, 3] = [translation_x, translation_y, translation_z]
    return transform


def transform_with_pivot(rotation_matrix: np.ndarray, pivot_target_position: np.ndarray,
                         pivot_point: np.ndarray) -> np.ndarray:
    """Sestavi 4x4 rigidno transformacijo, ki telo zavrti za rotation_matrix
    OKOLI pivot_point (v CAD-ovem lastnem, netransformiranem okviru - npr.
    compute_native_reference_point-ova "sealing" regija), namesto okoli
    CAD-ovega surovega izvora (0,0,0), nato ga premakne tako, da pivot_point
    natanko pristane na pivot_target_position.

    Za VSAKO točko X velja transform(X) = R @ (X - pivot_point) +
    pivot_target_position - torej gre za rotacijo okoli pivot_point, ne le
    za translacijski popravek, izračunan na eni sami točki. pivot_point NI
    treba izmeriti v (lahko delnem) oblaku točk - je konstanta, izračunana
    na celotni, vedno popolnoma znani CAD mreži."""
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = pivot_target_position - rotation_matrix @ pivot_point
    return transform


# 180° okoli X: CAD-ova -Z stran postane +Z. Cevovod vedno predpostavlja,
# da kamera gleda +Z stran dela (glej flip_up_direction v run_registration).
FLIP_UP = np.diag([1.0, -1.0, -1.0, 1.0])


def _skew(v: np.ndarray) -> np.ndarray:
    """Poševno-simetrična matrika [v]x, tako da [v]x @ w == cross(v, w)."""
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def rotation_aligning_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rodriguesova rotacijska formula: rotacijska matrika, ki preslika
    enotski vektor a na enotski vektor b (rotacija z najmanjšim kotom, ki
    to doseže)."""
    a = a / np.linalg.norm(a)   # normaliziramo vektor a na enotsko dolžino
    b = b / np.linalg.norm(b)   # normaliziramo vektor b na enotsko dolžino
    v = np.cross(a, b)   # os rotacije (pravokotna na a in b)
    s = np.linalg.norm(v)   # sinus kota med a in b
    c = np.dot(a, b)   # kosinus kota med a in b
    if s < 1e-8:   # a in b sta že (skoraj) vzporedna ali nasprotna
        if c > 0:
            return np.eye(3)   # a in b že kažeta v isto smer - ni potrebna rotacija
        # a in b sta antiparalelna - katerakoli os pravokotna na a deluje
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])   # pomožen vektor, ki ni vzporeden z a
        axis = np.cross(a, perp)   # poljubna os pravokotna na a
        axis /= np.linalg.norm(axis)   # normaliziramo os
        K = _skew(axis)
        return np.eye(3) + 2 * (K @ K)   # rotacija za natanko 180 stopinj okoli te osi
    vx = _skew(v)
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))   # Rodriguesova formula - sestavljena rotacijska matrika


def rotation_about_axis(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rodriguesova rotacijska formula: rotacija za angle_deg okoli
    poljubne enotske osi (posplošitev čiste Z rotacije na poljubno os, saj
    ocena naklona pomeni, da "gor" ni več nujno natanko os Z)."""
    axis = axis / np.linalg.norm(axis)   # normaliziramo os na enotsko dolžino
    theta = np.radians(angle_deg)   # pretvorimo kot iz stopinj v radiane
    K = _skew(axis)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)   # Rodriguesova formula - rotacijska matrika za kot theta


def registration_result(transformation: np.ndarray, fitness: float,
                        inlier_rmse: float) -> o3d.pipelines.registration.RegistrationResult:
    """Open3D-jev RegistrationResult iz že izračunanih vrednosti - za korake,
    ki pozo izračunajo sami (yaw sweep, izbira med ICP kandidati, mesh polish)."""
    result = o3d.pipelines.registration.RegistrationResult()
    result.transformation = transformation
    result.fitness = fitness
    result.inlier_rmse = inlier_rmse
    return result


def transformation_error(estimated: np.ndarray, reference: np.ndarray,
                         reference_point: Optional[np.ndarray] = None) -> dict:
    """Razstavi odstopanje med ocenjeno in referenčno 4x4 rigidno
    transformacijo na napako rotacijskega kota (stopinje) in translacijsko
    napako (enake enote kot oblak točk), namesto surove razlike matrik, ki
    oboje zmeša v težko razumljive številke.

    reference_point (privzeto None - CAD-ov surov izvor): translacijska
    napaka se meri kot razdalja med tem, kam TA točka (v CAD-ovem lastnem,
    netransformiranem okviru) pristane pod ocenjeno transformacijo, in kam
    pristane pod referenčno - namesto surove razlike surovih translacijskih
    vektorjev. Brez tega bi bila translation_error umetno napihnjena z
    "vzvodom" (lever arm): CAD-ov surov izvor je lahko deset(ih) mm+ stran
    od dejansko skeniranega dela (glej compute_native_reference_point), zato
    že majhna napaka rotacijskega KOTA pri surovem izvoru ustvari veliko
    navidezno translacijsko napako, čeprav je dejanski del na skoraj pravem
    mestu - glej transform_with_pivot."""
    r_est, t_est = estimated[:3, :3], estimated[:3, 3]   # rotacijski in translacijski del ocenjene transformacije
    r_ref, t_ref = reference[:3, :3], reference[:3, 3]   # rotacijski in translacijski del referenčne transformacije

    r_delta = r_est @ r_ref.T   # relativna rotacija med ocenjeno in referenčno
    # Kot rotacije iz sledi (trace) rotacijske matrike: trace(R) = 1 + 2*cos(theta).
    cos_theta = np.clip((np.trace(r_delta) - 1.0) / 2.0, -1.0, 1.0)   # kosinus napake kota, omejen na veljavno območje
    rotation_error_deg = np.degrees(np.arccos(cos_theta))   # napaka rotacije v stopinjah
    if reference_point is None:
        translation_error = np.linalg.norm(t_est - t_ref)   # napaka translacije (evklidska razdalja) pri CAD-ovem surovem izvoru
    else:
        p_est = r_est @ reference_point + t_est   # kam pristane referenčna (pivot) točka pod ocenjeno transformacijo
        p_ref = r_ref @ reference_point + t_ref   # kam pristane referenčna (pivot) točka pod referenčno transformacijo
        translation_error = np.linalg.norm(p_est - p_ref)   # napaka translacije PRI referenčni točki, ne pri surovem izvoru

    return {
        "rotation_error_deg": rotation_error_deg,   # napaka rotacije v stopinjah
        "translation_error": translation_error,   # napaka translacije v enotah oblaka točk
        "frobenius_norm": np.linalg.norm(estimated - reference),   # skupna razlika matrik za grobo primerjavo
    }
