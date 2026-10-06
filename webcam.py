import cv2
import time
from ultralytics import YOLO

# =========================
# SETTINGS
# =========================
MODEL_PATH = "best.pt"
CAMERA_ID = 0

CONFIDENCE = 0.25
IOU = 0.50

# Your YOLO11 classes
CLASS_NAMES = {
    0: "Glass",
    1: "Metal",
    2: "Paper",
    3: "Plastic",
    4: "Waste"
}

# =========================
# LOAD MODELS
# =========================
model = YOLO(MODEL_PATH)

# OpenCV built-in face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam started.")
print("Press Q to quit.")

prev_time = time.time()

# =========================
# MAIN LOOP
# =========================
while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    # --------------------------------
    # 1. DETECT FACES
    # --------------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    # --------------------------------
    # 2. YOLO + BYTETRACK
    # --------------------------------
    results = model.track(
        source=frame,
        tracker="bytetrack.yaml",
        conf=CONFIDENCE,
        iou=IOU,
        persist=True,
        verbose=False
    )

    result = results[0]

    # --------------------------------
    # 3. DRAW FACE AREAS
    # --------------------------------
    # We don't display the face boxes.
    # They are only used for filtering.
    
    # --------------------------------
    # 4. PROCESS YOLO DETECTIONS
    # --------------------------------
    if result.boxes is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        # Track IDs
        if result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = [None] * len(boxes)

        for box, conf, cls, track_id in zip(
            boxes, confidences, classes, track_ids
        ):

            x1, y1, x2, y2 = map(int, box)

            class_id = int(cls)
            class_name = CLASS_NAMES.get(class_id, "Unknown")

            # --------------------------------
            # 5. FACE FILTER
            # --------------------------------
            detection_center_x = (x1 + x2) // 2
            detection_center_y = (y1 + y2) // 2

            is_face_detection = False

            for (fx, fy, fw, fh) in faces:

                face_x2 = fx + fw
                face_y2 = fy + fh

                # Check whether YOLO detection center
                # is inside a detected face
                center_inside_face = (
                    fx <= detection_center_x <= face_x2
                    and
                    fy <= detection_center_y <= face_y2
                )

                # Calculate intersection
                ix1 = max(x1, fx)
                iy1 = max(y1, fy)
                ix2 = min(x2, face_x2)
                iy2 = min(y2, face_y2)

                intersection_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)

                detection_area = max(1, (x2 - x1) * (y2 - y1))

                overlap_ratio = intersection_area / detection_area

                # Ignore detection if it is mostly inside face
                if center_inside_face or overlap_ratio > 0.50:
                    is_face_detection = True
                    break

            # --------------------------------
            # 6. IGNORE FALSE POSITIVE
            # --------------------------------
            if is_face_detection:
                continue

            # --------------------------------
            # 7. DISPLAY VALID DETECTION
            # --------------------------------
            label = f"{class_name} {conf:.2f}"

            if track_id is not None:
                label = f"id:{track_id} {label}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

    # --------------------------------
    # 8. FPS
    # --------------------------------
    current_time = time.time()

    fps = 1 / max(current_time - prev_time, 0.001)

    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # --------------------------------
    # 9. SHOW FRAME
    # --------------------------------
    cv2.imshow("YOLO11 Waste Detection + ByteTrack", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()