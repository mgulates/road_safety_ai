
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

        
        vehicle_crop = frame[y1:y2, x1:x2]
        plate_bbox = reader._detect_plate(vehicle_crop)
        print(f"🔍 Plaka bbox tespit: {plate_bbox}")

        plate = reader.read_plate(frame, [x1, y1, x2, y2])
        print(f" Araç ({conf:.2f}) → Plaka: {plate}")