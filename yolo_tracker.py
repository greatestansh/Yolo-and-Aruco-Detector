import cv2
import numpy as np
import os
from ultralytics import YOLO

# 1. Load Pre-trained YOLOv8
model_path = os.path.join(os.path.dirname(__file__), "yolov8n.pt")
yolo_model = YOLO(model_path) 

# 2. Setup ArUco Dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# 3. Initialize Webcam
cap = cv2.VideoCapture(0)

# 4. Generate Approximated Camera Matrix
ret, frame = cap.read()
if ret:
    h, w, _ = frame.shape
    focal_length = w
    center_x, center_y = w / 2, h / 2
    camera_matrix = np.array([
        [focal_length, 0, center_x],
        [0, focal_length, center_y],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
else:
    print("Error: Could not read from webcam. Exiting.")
    cap.release()
    exit()

# Setup UI Fonts and Colors
font = cv2.FONT_HERSHEY_SIMPLEX
green = (0, 255, 0)
yellow = (0, 255, 255)
red = (0, 0, 255)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
        
    screen_center_x, screen_center_y = w // 2, h // 2
    # Draw a static white crosshair at the center of the screen
    cv2.drawMarker(frame, (screen_center_x, screen_center_y), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
    
    bottle_found = False
    aruco_found = False

    # TRACKER 1: YOLO Object Detection (Target: bottle)
    results = yolo_model(frame, verbose=False)[0]
    
    for box in results.boxes:
        class_id = int(box.cls[0])
        if class_id == 39: # Class 39 is 'bottle'
            bottle_found = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            obj_center_x, obj_center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
            error_x = obj_center_x - screen_center_x
            error_y = obj_center_y - screen_center_y
            
            # Draw the bounding box and a line pointing to the center crosshair
            cv2.rectangle(frame, (x1, y1), (x2, y2), yellow, 2)
            cv2.line(frame, (screen_center_x, screen_center_y), (obj_center_x, obj_center_y), yellow, 2)
            
            # Print the object identity directly above the bounding box
            cv2.putText(frame, "TARGET: bottle", (x1, y1 - 10), font, 0.6, yellow, 2)
            break 

    # TRACKER 2: ArUco Pose Estimation
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is not None:
        aruco_found = True
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)
        x_m, y_m, z_m = tvecs[0][0][0], tvecs[0][0][1], tvecs[0][0][2]
        
        # Draw 3D axis directly on top of the ArUco marker
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[0], tvecs[0], 0.03)

    # VISUAL HUD LOGIC
    if aruco_found and bottle_found:
        cv2.putText(frame, "STATUS: FUSION MODE (YOLO + ArUco)", (20, 40), font, 0.8, green, 2)
        cv2.putText(frame, f"Z (Depth):  {z_m:.2f} m", (20, 80), font, 0.7, green, 2)
        cv2.putText(frame, f"X (Offset): {x_m:.2f} m", (20, 110), font, 0.7, green, 2)
        cv2.putText(frame, f"Y (Offset): {y_m:.2f} m", (20, 140), font, 0.7, green, 2)
        
    elif aruco_found and not bottle_found:
        cv2.putText(frame, "STATUS: PRECISION MODE (ArUco ONLY)", (20, 40), font, 0.8, green, 2)
        cv2.putText(frame, f"Z (Depth):  {z_m:.2f} m", (20, 80), font, 0.7, green, 2)
        cv2.putText(frame, "Target Box: Lost", (20, 110), font, 0.7, red, 2)
        
    elif bottle_found and not aruco_found:
        cv2.putText(frame, "STATUS: COARSE MODE (YOLO ONLY)", (20, 40), font, 0.8, yellow, 2)
        cv2.putText(frame, f"Pixel Error X: {error_x} px", (20, 80), font, 0.7, yellow, 2)
        cv2.putText(frame, f"Pixel Error Y: {error_y} px", (20, 110), font, 0.7, yellow, 2)
        
    else:
        cv2.putText(frame, "STATUS: SEARCHING...", (20, 40), font, 0.8, red, 2)
                
    # Show output
    cv2.imshow("Telemetry HUD", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()