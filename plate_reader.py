import cv2
import re
import os
from ultralytics import YOLO

class PlateReader:
    def __init__(self):
        self.reader      = None
        self.plate_model = None
        self._load_ocr()
        self._load_plate_model()

    def _load_ocr(self):
        try:
            import easyocr
            self.reader = easyocr.Reader(["en"], gpu=False)
            print(" EasyOCR (plaka okuma) yüklendi!")
        except Exception as e:
            print(f"⚠️ EasyOCR yüklenemedi: {e}")

    def _load_plate_model(self):
        model_path = "runs/detect/models/plate_detector/weights/best.pt"
        try:
            self.plate_model = YOLO(model_path)
            print(" Plaka tespit modeli yüklendi!")
        except Exception as e:
            print(f"⚠️ Plaka tespit modeli yüklenemedi: {e}")

    def read_plate(self, frame, bbox):
        if self.reader is None:
            return None

        try:
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            vehicle_crop = frame[y1:y2, x1:x2]
            if vehicle_crop.size == 0:
                return None

            if self.plate_model is not None:
                plate_bbox = self._detect_plate(vehicle_crop)
                if plate_bbox:
                    px1, py1, px2, py2 = plate_bbox
                    plate_img = vehicle_crop[py1:py2, px1:px2]
                else:
                    plate_y1 = int(vehicle_crop.shape[0] * 0.70)
                    plate_img = vehicle_crop[plate_y1:, :]
            else:
                plate_y1 = int(vehicle_crop.shape[0] * 0.70)
                plate_img = vehicle_crop[plate_y1:, :]

            if plate_img.size == 0:
                return None

            
            plate_img_big = cv2.resize(plate_img, None, fx=3, fy=3)

            
            results = self.reader.readtext(
                plate_img_big,
                detail=0,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- '
            )

            if not results:
                return None

            text = " ".join(results).upper().strip()
            text = re.sub(r'[^A-Z0-9 -]', '', text).strip()

            if len(text) >= 4:
                return text

            return None

        except Exception as e:
            return None

    def _detect_plate(self, vehicle_crop):
        try:
            results = self.plate_model(vehicle_crop, verbose=False, conf=0.5)
            boxes   = results[0].boxes

            if boxes is None:
                return None

            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id == 4:  # np = number plate
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    return [x1, y1, x2, y2]

            return None
        except:
            return None