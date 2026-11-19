import time
import cv2
import numpy as np

CAMERA_NAME = "iphone"
FALLBACK_INDEX = 0

# ---- Workspace: 4 ArUco markers (DICT_5X5_50, 15 mm, IDs 0-3), one in each desk corner ----
# Desk top is 256 x 256 mm. WORKSPACE_MM = distance between the INNER corners of the
# markers (the corner of each marker that points toward the middle of the desk).
# Example: markers 3 mm from the desk edges -> inner corners at 3 + 15 = 18 mm
#          -> 256 - 2*18 = 220 mm.  MEASURE IT on the real desk.
WORKSPACE_MM = (220, 220)       # (width, height)  <-- SET THIS
MARKER_IDS = (0, 1, 2, 3)       # any placement/rotation works, all 4 must be visible (ID 4 = spare)
EDGE_MARGIN = 4                 # px ignored along the cropped edges

# ---- Gantry: the charger can reach a 142 x 142 mm area under the desk ----
GANTRY_TRAVEL_MM = (142, 142)
# Where the gantry's (0, 0) sits in WORKSPACE coordinates (mm from the top-left inner
# marker corner, as seen by the camera). Default assumes the travel area is centered.
GANTRY_ORIGIN_MM = ((WORKSPACE_MM[0] - GANTRY_TRAVEL_MM[0]) / 2,
                    (WORKSPACE_MM[1] - GANTRY_TRAVEL_MM[1]) / 2)
GANTRY_FLIP_X = False           # set True if the gantry's +X points left in the camera view
GANTRY_FLIP_Y = False           # set True if the gantry's +Y points up in the camera view

ARUCO = cv2.aruco
ARUCO_DICT = ARUCO.getPredefinedDictionary(ARUCO.DICT_5X5_50)
_params = ARUCO.DetectorParameters()
_params.cornerRefinementMethod = ARUCO.CORNER_REFINE_SUBPIX   # sharper corners
DETECTOR = ARUCO.ArucoDetector(ARUCO_DICT, _params)

WARP_W = 800
WARP_H = int(WARP_W * WORKSPACE_MM[1] / WORKSPACE_MM[0])
MM_PER_PX = WORKSPACE_MM[0] / WARP_W

# ---- Change + shadow rejection ----
DIFF_L = 25                     # brightness change needed (raise if noisy, ~15 for white-on-white)
DIFF_CHROMA = 10                # color change needed
SHADOW_MIN_RATIO = 0.45         # darker than this = real object, not shadow
MIN_SOLIDITY = 0.80

# ---- Object sizes in mm^2 ----
MIN_OBJECT_MM2 = 1200
PHONE_MIN_MM2 = 6000            # bigger than this = iPhone, smaller = AirPods

TILE_W, TILE_H = 420, 315
FONT = cv2.FONT_HERSHEY_SIMPLEX


def list_cameras():
    try:
        import AVFoundation
        devs = AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVFoundation.AVMediaTypeVideo)
        return [str(d.localizedName()) for d in devs]
    except ImportError:
        print("WARNING: pyobjc-framework-AVFoundation not installed, can't look up camera names")
        return []


def pick_camera():
    names = list_cameras()
    for i, n in enumerate(names):
        print(f"  [{i}] {n}")
    for i, n in enumerate(names):
        if CAMERA_NAME.lower() in n.lower():
            print(f"Using [{i}] {n}")
            return i
    print(f"WARNING: no '{CAMERA_NAME}' camera found, using index {FALLBACK_INDEX}")
    return FALLBACK_INDEX


def order_points(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)  # TL, TR, BR, BL


def find_workspace(frame):
    """Returns (quad or None, marker_corners, marker_ids)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = DETECTOR.detectMarkers(gray)
    if ids is None:
        return None, corners, ids

    found = {}
    for c, i in zip(corners, ids.ravel()):
        if i in MARKER_IDS:
            found[int(i)] = c.reshape(4, 2)
    if len(found) < 4:
        return None, corners, ids

    # inner corner = the corner of each marker closest to the middle of all 4 markers,
    # so it doesn't matter how each marker is rotated
    center = np.mean([m.mean(axis=0) for m in found.values()], axis=0)
    inner = [m[np.argmin(np.linalg.norm(m - center, axis=1))] for m in found.values()]
    return order_points(np.array(inner)), corners, ids


def ws_to_gantry(x_mm, y_mm):
    """Workspace mm -> gantry mm. Returns (gx, gy, reachable); clamped to the travel area."""
    gx = x_mm - GANTRY_ORIGIN_MM[0]
    gy = y_mm - GANTRY_ORIGIN_MM[1]
    if GANTRY_FLIP_X:
        gx = GANTRY_TRAVEL_MM[0] - gx
    if GANTRY_FLIP_Y:
        gy = GANTRY_TRAVEL_MM[1] - gy
    reachable = 0 <= gx <= GANTRY_TRAVEL_MM[0] and 0 <= gy <= GANTRY_TRAVEL_MM[1]
    gx = min(max(gx, 0), GANTRY_TRAVEL_MM[0])
    gy = min(max(gy, 0), GANTRY_TRAVEL_MM[1])
    return gx, gy, reachable


def draw_travel_area(img):
    x0, y0 = (int(v / MM_PER_PX) for v in GANTRY_ORIGIN_MM)
    x1 = int((GANTRY_ORIGIN_MM[0] + GANTRY_TRAVEL_MM[0]) / MM_PER_PX)
    y1 = int((GANTRY_ORIGIN_MM[1] + GANTRY_TRAVEL_MM[1]) / MM_PER_PX)
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 255), 2)
    cv2.putText(img, "charger reach", (x0 + 5, y0 + 25), FONT, 0.7, (255, 0, 255), 2)


def to_lab(img):
    return cv2.cvtColor(cv2.GaussianBlur(img, (7, 7), 0), cv2.COLOR_BGR2LAB).astype(np.float32)


def capture_background(cap, M, n=30):
    acc, count = None, 0
    for _ in range(n):
        ret, frame = cap.read()
        if not ret:
            continue
        lab = to_lab(cv2.warpPerspective(frame, M, (WARP_W, WARP_H)))
        acc = lab if acc is None else acc + lab
        count += 1
    if count == 0:
        raise RuntimeError("No frames captured for background")
    return acc / count


def detect(warp, bg_lab):
    L, a, b = cv2.split(to_lab(warp))
    Lb, ab, bb = cv2.split(bg_lab)

    ratio = (L + 1) / (Lb + 1)
    gain = np.median(ratio)              # compensates auto-exposure drift
    ratio /= gain
    dL = L / gain - Lb
    chroma = np.sqrt((a - ab) ** 2 + (b - bb) ** 2)

    changed = (np.abs(dL) > DIFF_L) | (chroma > DIFF_CHROMA)
    # shadow = moderately darker, same color. Objects change color or get much darker.
    shadow = (dL < 0) & (ratio > SHADOW_MIN_RATIO) & (chroma < DIFF_CHROMA)
    shadow_d = cv2.dilate(shadow.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)

    fg = (changed & ~shadow_d).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    m = EDGE_MARGIN
    if m > 0:
        fg[:m, :] = 0; fg[-m:, :] = 0; fg[:, :m] = 0; fg[:, -m:] = 0

    vis = np.zeros((WARP_H, WARP_W, 3), np.uint8)
    vis[shadow & changed] = (200, 120, 0)    # blue = shadow, ignored
    vis[fg > 0] = (255, 255, 255)            # white = object

    objs = []
    if fg.mean() / 255 < 0.4:                # if >40% changed, lighting moved: ignore frame
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area_px = cv2.contourArea(c)
            area_mm2 = area_px * MM_PER_PX ** 2
            if area_mm2 < MIN_OBJECT_MM2:
                continue
            hull = cv2.contourArea(cv2.convexHull(c))
            if hull == 0 or area_px / hull < MIN_SOLIDITY:
                continue
            (cx, cy), size, angle = cv2.minAreaRect(c)
            gx, gy, reach = ws_to_gantry(cx * MM_PER_PX, cy * MM_PER_PX)
            objs.append({
                "gantry": (gx, gy),
                "reachable": reach,
                "label": "iPhone" if area_mm2 >= PHONE_MIN_MM2 else "AirPods",
                "px": (int(cx), int(cy)),
                "mm": (cx * MM_PER_PX, cy * MM_PER_PX),
                "angle": angle,
                "area_mm2": area_mm2,
                "box": cv2.boxPoints(((cx, cy), size, angle)).astype(int),
            })
    return vis, objs


def blank(msg):
    img = np.zeros((TILE_H, TILE_W, 3), np.uint8)
    cv2.putText(img, msg, (15, TILE_H // 2), FONT, 0.6, (200, 200, 200), 1)
    return img


def tile(img, label):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    s = min(TILE_W / w, TILE_H / h)
    img = cv2.resize(img, (int(w * s), int(h * s)))
    canvas = np.zeros((TILE_H, TILE_W, 3), np.uint8)
    y, x = (TILE_H - img.shape[0]) // 2, (TILE_W - img.shape[1]) // 2
    canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
    cv2.putText(canvas, label, (8, 22), FONT, 0.6, (0, 255, 255), 2)
    cv2.rectangle(canvas, (0, 0), (TILE_W - 1, TILE_H - 1), (80, 80, 80), 1)
    return canvas


def main():
    cam_index = pick_camera()
    cap = cv2.VideoCapture(cam_index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print(f"Could not open camera {cam_index}")
        return

    dst = np.array([[0, 0], [WARP_W - 1, 0], [WARP_W - 1, WARP_H - 1], [0, WARP_H - 1]], np.float32)
    quad, M, bg, locked, last_print = None, None, None, False, 0
    print("Click the window. Empty the workspace, press 'b' to lock it + capture background.")
    print("'r' = reset, 'q'/Esc = quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break

            # 1-2. workspace detection (4 ArUco markers)
            found, m_corners, m_ids = find_workspace(frame)
            if not locked and found is not None:
                if quad is None or np.abs(found - quad).max() > 40:
                    quad = found
                else:
                    quad = (0.85 * quad + 0.15 * found).astype(np.float32)
                M = cv2.getPerspectiveTransform(quad, dst)

            overlay = frame.copy()
            if m_ids is not None:
                ARUCO.drawDetectedMarkers(overlay, m_corners, m_ids)
            n_seen = 0 if m_ids is None else len(set(m_ids.ravel()) & set(MARKER_IDS))
            if not locked:
                cv2.putText(overlay, f"markers: {n_seen}/4", (20, 60), FONT, 1.5,
                            (0, 255, 0) if n_seen == 4 else (0, 0, 255), 3)
            if quad is not None:
                color = (0, 255, 255) if locked else (0, 255, 0)
                cv2.polylines(overlay, [quad.astype(int)], True, color, 4)
                for i, p in enumerate(quad.astype(int)):
                    cv2.putText(overlay, str(i + 1), (int(p[0]), int(p[1])), FONT, 1.2, color, 3)

            # 3-4. crop
            warp = cv2.warpPerspective(frame, M, (WARP_W, WARP_H)) if M is not None else None

            # 5-6. change + result
            change_t = blank("press 'b' (empty desk)")
            result_t = blank("no workspace found")
            if warp is not None:
                result = warp.copy()
                draw_travel_area(result)
                if bg is not None:
                    vis, objs = detect(warp, bg)
                    change_t = vis
                    for o in objs:
                        color = (0, 255, 0) if o["label"] == "iPhone" else (255, 128, 0)
                        cv2.drawContours(result, [o["box"]], 0, color, 3)
                        cv2.circle(result, o["px"], 8, (0, 0, 255), -1)
                        tag = "" if o["reachable"] else " OUT"
                        cv2.putText(result, f'{o["label"]} G({o["gantry"][0]:.0f}, {o["gantry"][1]:.0f}){tag}',
                                    (max(o["px"][0] - 90, 5), max(o["px"][1] - 20, 25)), FONT, 0.8, color, 2)
                    if objs and time.time() - last_print > 0.25:
                        last_print = time.time()
                        for o in objs:
                            print(f'{o["label"]}: desk=({o["mm"][0]:.1f}, {o["mm"][1]:.1f}) mm  '
                                  f'gantry=({o["gantry"][0]:.1f}, {o["gantry"][1]:.1f}) mm'
                                  f'{"" if o["reachable"] else "  (out of reach, clamped)"}  '
                                  f'angle={o["angle"]:.0f}  area={o["area_mm2"]:.0f} mm2')
                result_t = result

            gray_t = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY) if warp is not None else blank("no workspace found")
            warp_t = warp if warp is not None else blank("no workspace found")

            row1 = [tile(frame, "1. Camera feed"),
                    tile(overlay, "2. Markers " + ("(LOCKED)" if locked else "(live)")),
                    tile(warp_t, "3. Cropped - color")]
            row2 = [tile(gray_t, "4. Cropped - grayscale"),
                    tile(change_t, "5. Change (white=obj, blue=shadow)"),
                    tile(result_t, "6. Result")]
            cv2.imshow("Smart Desk", np.vstack([np.hstack(row1), np.hstack(row2)]))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord("b"):
                if M is None:
                    print("Need all 4 markers visible first.")
                else:
                    locked = True
                    bg = capture_background(cap, M)
                    print("Workspace locked, background captured.")
            if key == ord("r"):
                locked, bg = False, None
                print("Reset.")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)


if __name__ == "__main__":
    main()