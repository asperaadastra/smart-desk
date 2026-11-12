import cv2

for i in range(6):
    cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
    ok, frame = False, None
    if cap.isOpened():
        for _ in range(2,30):          # virtual cameras need warm-up frames
            ok, frame = cap.read()
            if ok:
                break
    print(f"index {i}: opened={cap.isOpened()} frame={ok}",
          frame.shape if ok else "")
    if ok:
        cv2.imwrite(f"cam_{i}.jpg", frame)
    cap.release()