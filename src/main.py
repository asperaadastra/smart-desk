import time
import cv2
import numpy as np

CAMERA_NAME = "iphone"
FALLBACK_INDEX = 0

# ---- Workspace: measure the INSIDE of the black rectangle (mm) ----
WORKSPACE_MM = (400, 300)       # (width, height)  <-- SET THIS
USE_INNER_EDGE = True           # True = area inside the black border, False = outer edge of border
EDGE_MARGIN = 8                 # px ignored along the cropped edges (border leftovers)

WARP_W = 800
WARP_H = int(WARP_W * WORKSPACE_MM[1] / WORKSPACE_MM[0])
MM_PER_PX = WORKSPACE_MM[0] / WARP_W

# ---- Change + shadow rejection ----
DIFF_L = 25                     # brightness change needed (raise if noisy)
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
        return []


def pick_camera():
    names = list_cameras()
    for i, n in enumerate(names):
        print(f"  [{i}] {n}")
    for i, n in enumerate(names):
        if CAMERA_NAME.lower() in n.lower():
            return i
    return FALLBACK_INDEX


def order_points(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)  # TL, TR, BR, BL


def find_workspace(frame):
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, hier = cv2.findContours(dark, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    fh, fw = gray.shape
    best, best_area = None, 0
    if hier is not None:
        for c, hr in zip(contours, hier[0]):
            is_hole = hr[3] != -1
            if is_hole != USE_INNER_EDGE:
                continue
            area = cv2.contourArea(c)
            if not (0.05 * fw * fh < area < 0.95 * fw * fh) or area <= best_area:
                continue
            peri = cv2.arcLength(c, True)
            for eps in (0.02, 0.03, 0.05):
                approx = cv2.approxPolyDP(c, eps * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    best, best_area = approx, area
                    break
    return (order_points(best) if best is not None else None), dark


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
            objs.append({
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

            # 1-2. workspace detection
            found, dark = find_workspace(frame)
            if not locked and found is not None:
                if quad is None or np.abs(found - quad).max() > 40:
                    quad = found
                else:
                    quad = (0.85 * quad + 0.15 * found).astype(np.float32)
                M = cv2.getPerspectiveTransform(quad, dst)

            overlay = frame.copy()
            sel = dark > 0
            overlay[sel] = (0.5 * overlay[sel] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
            if quad is not None:
                color = (0, 255, 255) if locked else (0, 255, 0)
                cv2.polylines(overlay, [quad.astype(int)], True, color, 4)
                for i, p in enumerate(quad.astype(int)):
                    cv2.putText(overlay, str(i + 1), tuple(p), FONT, 1.2, color, 3)

            # 3-4. crop
            warp = cv2.warpPerspective(frame, M, (WARP_W, WARP_H)) if M is not None else None


# 5-6. change + result
            change_t = blank("press 'b' (empty desk)")
            result_t = blank("no workspace found")
            if warp is not None:
                result = warp.copy()
                if bg is not None:
                    vis, objs = detect(warp, bg)
                    change_t = vis
                    for o in objs:
                        color = (0, 255, 0) if o["label"] == "iPhone" else (255, 128, 0)
                        cv2.drawContours(result, [o["box"]], 0, color, 3)
                        cv2.circle(result, o["px"], 8, (0, 0, 255), -1)
                        cv2.putText(result, f'{o["label"]} ({o["mm"][0]:.0f}, {o["mm"][1]:.0f}) mm',
                                    (o["px"][0] - 90, o["px"][1] - 20), FONT, 0.9, color, 2)
                    if objs and time.time() - last_print > 0.25:
                        last_print = time.time()
                        for o in objs:
                            print(f'{o["label"]}: x={o["mm"][0]:.1f} mm  y={o["mm"][1]:.1f} mm  '
                                  f'angle={o["angle"]:.0f}  area={o["area_mm2"]:.0f} mm2')
                result_t = result

            gray_t = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY) if warp is not None else blank("no workspace found")
            warp_t = warp if warp is not None else blank("no workspace found")

            row1 = [tile(frame, "1. Camera feed"),
                    tile(overlay, "2. Workspace " + ("(LOCKED)" if locked else "(live)")),
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
                    print("No workspace found yet.")
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


if name == "__main__":
    main()