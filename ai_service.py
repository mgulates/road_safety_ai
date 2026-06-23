
import cv2
import numpy as np
import time
from ultralytics import YOLO
from config.config import Config
from tracker import Tracker
from plate_reader import PlateReader

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except:
    MEDIAPIPE_AVAILABLE = False

class AIService:
    def __init__(self):
        print("🔄 Modeller yükleniyor...")
        self.cfg = Config()

        # Hafif model
        self.light_model = YOLO(self.cfg.LIGHT_MODEL)
        print(" YOLOv8n yüklendi!")

        # Ağır model
        self.heavy_model = YOLO(self.cfg.HEAVY_MODEL)
        print(" RT-DETR yüklendi!")

        # Fine-tuned davranış modeli
        try:
            self.behavior_model = YOLO("runs/classify/models/driver_behavior/weights/best.pt")
            print("Sürücü davranış modeli yüklendi!")
        except Exception as e:
            print(f" Behavior model yüklenemedi: {e}")
            self.behavior_model = None

        # Tracker ve plaka okuma
        self.tracker      = Tracker()
        self.plate_reader = PlateReader()
        self.detected_plates = {}  # track_id → plaka

        # MediaPipe
        self.face_landmarker = None
        self.hand_landmarker = None

        if MEDIAPIPE_AVAILABLE:
            import mediapipe as mp
            self.mp = mp
            try:
                import urllib.request, os
                model_path = "models/face_landmarker.task"
                if not os.path.exists(model_path):
                    print(" Face landmarker indiriliyor...")
                    urllib.request.urlretrieve(
                        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                        model_path
                    )
                options = mp.tasks.vision.FaceLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
                print(" Face landmarker yüklendi!")
            except Exception as e:
                print(f"⚠️ Face landmarker: {e}")

            try:
                model_path = "models/hand_landmarker.task"
                if not os.path.exists(model_path):
                    print(" Hand landmarker indiriliyor...")
                    urllib.request.urlretrieve(
                        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                        model_path
                    )
                options = mp.tasks.vision.HandLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
                    num_hands=2,
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
                print(" Hand landmarker yüklendi!")
            except Exception as e:
                print(f"⚠️ Hand landmarker: {e}")

        
        self.LEFT_EYE  = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]

       
        self.ear_counter = 0
        self.drowsy = False
        self.phone_last_detected = 0

        print("─" * 40)
        print("🚀 Tüm modeller hazır!")

    def run_light(self, frame):
        signals = {
            "vehicle_detected":   False,
            "vehicle_proximity":  0.0,
            "phone_detected":     False,
            "hand_on_wheel":      True,
            "ear_value":          1.0,
            "drowsy":             False,
            "behavior":           "unknown",
            "behavior_conf":      0.0,
            "plates":             [],
            "tracks":             [],
            "tracker_stable":     False,
            "detections":         []
        }

        if frame is None:
            return signals

        h, w = frame.shape[:2]

        results = self.light_model(
            frame,
            conf=self.cfg.VEHICLE_CONF,
            device=self.cfg.DEVICE,
            verbose=False
        )

        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if cls_id in self.cfg.VEHICLE_CLASSES:
                    signals["vehicle_detected"] = True
                    bbox_area  = (x2 - x1) * (y2 - y1)
                    frame_area = h * w
                    signals["vehicle_proximity"] = min(bbox_area / frame_area, 1.0)

                if cls_id == 67 and conf >= self.cfg.PHONE_CONF:
                    now = time.time()
                    if now - self.phone_last_detected > self.cfg.PHONE_COOLDOWN_SEC:
                        signals["phone_detected"] = True
                        self.phone_last_detected  = now

                signals["detections"].append({
                    "class": cls_id,
                    "conf":  conf,
                    "bbox":  [x1, y1, x2, y2]
                })

        
        if self.behavior_model is not None:
            try:
                beh_result = self.behavior_model(
                    frame,
                    verbose=False,
                    device=self.cfg.DEVICE
                )
                probs    = beh_result[0].probs
                top1     = probs.top1
                conf     = float(probs.top1conf)
                names    = beh_result[0].names
                behavior = names[top1]

                signals["behavior"]      = behavior
                signals["behavior_conf"] = conf

                if behavior in ["talking_phone", "texting_phone"] and conf > 0.6:
                    signals["phone_detected"] = True
                    print(f"📱 {behavior} tespit! ({conf:.2f})")

            except Exception as e:
                print(f"⚠️ Behavior model hatası: {e}")

        
        if self.face_landmarker:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = self.mp.Image(
                    image_format=self.mp.ImageFormat.SRGB,
                    data=rgb
                )
                result = self.face_landmarker.detect(mp_image)
                if result.face_landmarks:
                    lm  = result.face_landmarks[0]
                    ear = self._calc_ear(lm, w, h)
                    signals["ear_value"] = ear

                    if ear < self.cfg.EAR_THRESHOLD:
                        self.ear_counter += 1
                        if self.ear_counter >= self.cfg.EAR_CONSEC_FRAMES:
                            signals["drowsy"] = True
                            self.drowsy       = True
                    else:
                        self.ear_counter = 0
                        self.drowsy      = False
            except:
                pass

        
        if self.hand_landmarker:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = self.mp.Image(
                    image_format=self.mp.ImageFormat.SRGB,
                    data=rgb
                )
                result = self.hand_landmarker.detect(mp_image)
                if not result.hand_landmarks:
                    signals["hand_on_wheel"] = False
            except:
                pass

        
        if signals["detections"]:
            tracks, is_stable = self.tracker.update(signals["detections"], frame)
            signals["tracks"]         = tracks
            signals["tracker_stable"] = is_stable

            
            for det in signals["detections"]:
                if det["class"] in self.cfg.VEHICLE_CLASSES and det["conf"] > 0.6:
                    plate = self.plate_reader.read_plate(frame, det["bbox"])
                    if plate:
                        signals["plates"].append(plate)
                        print(f"🚗 Plaka: {plate}")

        return signals

    def _calc_ear(self, landmarks, w, h):
        def get_point(idx):
            lm = landmarks[idx]
            return np.array([lm.x * w, lm.y * h])

        def ear_for_eye(indices):
            p = [get_point(i) for i in indices]
            A = np.linalg.norm(p[1] - p[5])
            B = np.linalg.norm(p[2] - p[4])
            C = np.linalg.norm(p[0] - p[3])
            return (A + B) / (2.0 * C)

        left  = ear_for_eye(self.LEFT_EYE)
        right = ear_for_eye(self.RIGHT_EYE)
        return (left + right) / 2.0

    def run_heavy(self, frame):
        results = self.heavy_model(
            frame,
            conf=0.4,
            device=self.cfg.DEVICE,
            verbose=False
        )

        heavy_detections = []
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                heavy_detections.append({
                    "class": cls_id,
                    "conf":  conf,
                    "bbox":  [x1, y1, x2, y2]
                })

        return heavy_detections