import cv2
from ultralytics import YOLO

def get_working_camera_index():
    """
    Scans camera indices 0 through 5 using AVFOUNDATION (macOS native driver).
    Returns a list of working camera indices.
    """
    print("🔍 Scanning for available video feeds...")
    working_indices = []
    for idx in range(1,6):
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                working_indices.append(idx)
            cap.release()
    return working_indices

def main():
    # 1. Scan and select the camera
    working_cams = get_working_camera_index()
    
    if not working_cams:
        print("❌ Error: No working cameras found! Ensure Camo is running and connected.")
        return

    print(f"✅ Found working camera index(es): {working_cams}")
    
    # Defaults to the highest index (Camo is usually index 1 or 2 when the built-in webcam is 0)
    cam_index = working_cams[-1]
    print(f"🎥 Connecting to camera index {cam_index} (Camo feed)...")

    cap = cv2.VideoCapture(cam_index, cv2.CAP_AVFOUNDATION)

    # 2. Load the YOLO model
    print("⚡ Loading YOLOv8 Nano model...")
    model = YOLO('yolov8n.pt')

    print("🚀 Stream active! Click on the video window and press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame from camera. Exiting...")
            break

        # 3. Run YOLO object detection
        results = model(frame, verbose=False)

        # 4. Parse detection boxes
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # Class 67 in COCO dataset = 'cell phone'
                if cls_id == 67 and conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    # Draw bounding box and center coordinate point
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (center_x, center_y), 6, (0, 0, 255), -1)

                    # Draw text label and pixel coordinates
                    label = f"iPhone ({conf:.2f})"
                    coords = f"Center: ({center_x}, {center_y})"
                    
                    cv2.putText(frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, coords, (x1, y2 + 25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 5. Render window
        cv2.imshow('Smart Desk - iPhone & YOLO Detection', frame)

        # Press 'q' key to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopping video stream...")
            break

    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Extra flush for macOS window management

if __name__ == "__main__":
    main()