# test_plate_output.py
import cv2
from plate_reader import PlateReader
from ultralytics import YOLO

reader   = PlateReader()
detector = YOLO("yolov8n.pt")

frame = cv2.imread("data/test_plate.jpg")

results = detector(frame, verbose=False)
boxes   = results[0].boxes

for box in boxes:
    cls_id = int(box.cls[0])
    if cls_id in [2, 3, 5, 7]:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        # Plaka tespit et
        vehicle_crop = frame[y1:y2, x1:x2]
        plate_bbox   = reader._detect_plate(vehicle_crop)
        plate_text   = reader.read_plate(frame, [x1, y1, x2, y2])

        # Araç bbox çiz
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Arac {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Plaka bbox çiz
        if plate_bbox:
            px1, py1, px2, py2 = plate_bbox
            # Koordinatları tam frame'e çevir
            px1 += x1; px2 += x1
            py1 += y1; py2 += y1
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
            cv2.putText(frame, f"Plaka: {plate_text}", (px1, py1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        print(f" Araç ({conf:.2f}) → Plaka: {plate_text}")

cv2.imwrite("outputs/plate_detection_result.jpg", frame)
print("outputs/plate_detection_result.jpg kaydedildi!")