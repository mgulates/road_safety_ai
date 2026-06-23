import cv2
import re

class PlateReader:
    def __init__(self):
        self.reader = None
        self._load()

    def _load(self):
        try:
            import easyocr
            self.reader = easyocr.Reader(["en"], gpu=False)
            print(" EasyOCR (plaka okuma) yüklendi!")
        except Exception as e:
            print(f"⚠️ EasyOCR yüklenemedi: {e}")
            self.reader = None

    def read_plate(self, frame, bbox):
        if self.reader is None:
            return None

        try:
            x1, y1, x2, y2 = bbox

           
            plate_y1 = y2 - int((y2 - y1) * 0.40)
            plate_y2 = y2
            plate_x1 = x1 + int((x2 - x1) * 0.10)
            plate_x2 = x2 - int((x2 - x1) * 0.10)

            h, w = frame.shape[:2]
            plate_x1 = max(0, plate_x1)
            plate_y1 = max(0, plate_y1)
            plate_x2 = min(w, plate_x2)
            plate_y2 = min(h, plate_y2)

          
            plate_img = frame[plate_y1:plate_y2, plate_x1:plate_x2]
            if plate_img.size == 0:
                return None

           
            plate_img_big = cv2.resize(plate_img, None, fx=3, fy=3)

            
            results = self.reader.readtext(plate_img_big, detail=0)

            if not results:
                return None

            text = " ".join(results).upper().strip()
            text = re.sub(r'[^A-Z0-9 ]', '', text).strip()

            if len(text) >= 4:
                return text

            return None

        except Exception as e:
            print(f" Plaka okuma hatası: {e}")
            return None