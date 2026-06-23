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

        
        plate = reader.read_plate(frame, [x1, y1, x2, y2])

        
        plate_y1 = y2 - int((y2 - y1) * 0.40)
        crop = frame[plate_y1:y2, x1:x2]
        cv2.imwrite(f"data/plate_crop_{x1}.jpg", crop)

        print(f" Araç ({conf:.2f}) → Plaka: {plate}")