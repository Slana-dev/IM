# Code review: branch `spremenjena_koda`

Date: 2026-10-17 · Source: `/code-review` (Claude Code) · 10 findings, most severe first.
Paths are relative to the repository root.

Tick each box once you have looked at a finding and decided what to do with it.

---

## 1. A missing real scan falls back to a simulated scan
- [ ] Reviewed

`dolocanje_tocne_lokacije/main_trial.py:2392`, with the same check at line 2504.

If `--use_real_scan` is set and the scan file does not exist, the code takes the `else` branch and runs `simulate_camera_scan`. It registers a synthetic scan and raises no error.

**How it fails.** In production the robot runs `main_trial.py --use_real_scan --scan scan.ply`, but the camera did not write `scan.ply`. `scan_path.exists()` returns False, so the script simulates a perfect ray-cast scan from the CAD at the pose given on the command line. Registration passes and the script exits with code 0. The robot process receives a pose that nobody measured and applies the seal.

---

## 2. Hand-eye: robot poses and board views stop lining up after a dropped view
- [ ] Reviewed

`kalibracija/eye-in-hand kalibracija/hand_eye_kalibracija.py:268`

Step 3 reads `robot_R_gripper2base[i]` and `robot_t_gripper2base[i]` with the index into `calib_charuco_corners`. Step 2 drops every view where `matchImagePoints` returns fewer than 4 points, so after a dropped view the two lists no longer correspond.

**How it fails.** Image 2 of 10 fails `matchImagePoints` in step 2. Step 2 removes it from `calib_charuco_corners`, but it stays in `robot_R_gripper2base`. From that view on, the code pairs each board pose with the previous robot pose. `calibrateHandEye` returns a wrong camera-to-flange matrix, and the script saves it to `hand_eye_result.json` as a success.

---

## 3. The hand-eye result is in meters, robot poses are in mm
- [ ] Reviewed

`kalibracija/eye-in-hand kalibracija/hand_eye_kalibracija.py:199`

The script divides robot translations by 1000 and defines the board in meters, so the translation in the saved `hand_eye_matrix` is in meters. `main.py` loads it as float32 and combines it with Epson poses in mm (`get_object_in_base`). Neither file records or converts the unit.

**How it fails.** Once `main.py` computes `T_robot_current @ hand_eye_matrix @ T_obj2cam`, as its comments describe, a camera offset of about 80 mm enters the calculation as 0.08 mm. The object's position in the base frame is then wrong by the whole camera-to-flange offset.

---

## 4. Benchmarks measure translation error at the CAD origin, not at the pivot
- [ ] Reviewed

`dolocanje_tocne_lokacije/camera_benchmark.py:246`, and the same line in `dolocanje_tocne_lokacije/camera_translation_benchmark.py`.

Both benchmarks call `transformation_error` without `reference_point`. The function then measures translation error at the raw CAD origin instead of at the pivot, which is the centre of the seal region. `CLAUDE.md` says the pipeline reports translation error at the pivot, so the benchmarks disagree with it.

**How it fails.** The CAD origin is tens of mm from the seal region. A rotation error of 0.2° at a 50 mm lever arm adds about 0.17 mm of false translation error. The `trans_error_mm` column in the xlsx comes out too high and mixes rotation error into translation error, which distorts the camera ranking. `run_single` already receives `reference_point`, so the fix only needs to pass it on.

---

## 5. Hand-eye: TCP messages have no line framing, and camera frames can be stale
- [ ] Reviewed

`kalibracija/eye-in-hand kalibracija/hand_eye_kalibracija.py:90`

The script treats each `recv(1024)` as exactly one message and does not split the stream into lines. It takes the camera frame with a single `cap.read()` on a capture that has buffered frames since startup.

**How it fails.**
- The robot sends `'3 ... w\r\n'` and then `'99\r\n'` in quick succession, and both arrive in one `recv`. `split()` returns more than 8 parts, the script processes the point, the `'99'` disappears, and the server waits forever.
- `cap.read()` can return an old buffered frame from while the robot was still moving, so the image does not match the recorded pose.

---

## 6. The movement watchdog stops without raising an error
- [ ] Reviewed

`main/main.py:147`

The watchdog thread exits without an error when `cap.read()` fails or the operator presses `q`. `moveRobot` keeps running the dispensing cycle with no movement monitoring.

**How it fails.** The USB camera drops a frame in the middle of a cycle. `if not ret: break` ends `checkMovement`, and `movementError` stays False. If someone then knocks the part, nothing sends ABORT and the robot applies the seal to a part that has moved.

---

## 7. The code compares the robot's completion reply against a raw recv chunk
- [ ] Reviewed

`main/main.py:105`, with the same logic in `komunikacija/send_localcoord.py`.

The code checks the robot's reply with `odgovor == "1"` on whatever one `recv` returns. If two messages arrive together, or an extra message arrives, the check never matches.

**How it fails.** The Epson sends a status line followed by `'1\r\n'`, or two replies arrive in one segment. After `strip()` the text is `'0\r\n1'` or `'1\r\n1'`, which never equals `'1'`. The loop polls forever and the operator never sees the "cycle done" prompt.

---

## 8. U and W are swapped in the log output
- [ ] Reviewed

`main/main.py:87`

`matrix_to_epson` returns `(x, y, z, u, v, w)`, but the caller unpacks the result as `dx, dy, dz, rx, ry, rz`. The log then prints `U=rz` and `W=rx`, so the operator sees U and W swapped.

**How it fails.** When the code sends U=30, V=0, W=0, the log shows `U=0.00 ... W=30.00`. Anyone who checks the offset in the log sees the rotation on the wrong axis. The order sent to the robot (u v w) is correct.

---

## 9. The launch config and setup guide use flags that no longer exist
- [ ] Reviewed

`dolocanje_tocne_lokacije/.vscode/launch.json:11` and `NAVODILA_ZAGON.md`

The "Main" launch configuration and the example command in `NAVODILA_ZAGON.md` pass `--iss_target_min_keypoints` and `--iss_target_max_keypoints`. `main_trial.py` no longer defines these flags. `NAVODILA_ZAGON.md` also still mentions `ToolCP_avtomaticno.py`, which this branch deletes.

**How it fails.** Pressing F5 on `main_trial.py`, or running the documented command, stops immediately with `error: unrecognized arguments: --iss_target_min_keypoints 1000 ...`.

---

## 10. `requirements.txt` lacks openpyxl and pins the wrong versions
- [ ] Reviewed

`requirements.txt:2`

- It does not list `openpyxl`, which the benchmarks need for `pd.read_excel` and `ExcelWriter(engine='openpyxl')`.
- It allows OpenCV 5.x, which the setup guide says breaks `calibrateHandEye`.
- It requires `open3d>=0.18`, while the setup guide pins 0.19.

**How it fails.** After `pip install -r requirements.txt`, `camera_benchmark.py` stops in `save_excel` with `ImportError: openpyxl` once the first run finishes. A fresh install can also pull in opencv-contrib 5.x and break the hand-eye calibration.
