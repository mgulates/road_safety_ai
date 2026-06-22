# preprocessor.py
import cv2
import numpy as np

class Preprocessor:
    def __init__(self, img_size=640):
        self.img_size = img_size

    def process(self, frame, apply_roi=True):
        if frame is None:
            return None, None

        original = frame.copy()

        # 1. Sahne iyileştirme
        frame = self._enhance(frame)

        # 2. ROI — sürücü bölgesi
        if apply_roi:
            frame = self._apply_roi(frame)

        # 3. BGR → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 4. Yeniden boyutlandır
        frame_resized = cv2.resize(frame_rgb, (self.img_size, self.img_size))

        # 5. Normalize (0-255 → 0.0-1.0)
        frame_norm = frame_resized.astype(np.float32) / 255.0

        # 6. Batch boyutu ekle
        frame_batch = np.expand_dims(frame_norm, axis=0)

        return frame_batch, original

    def _enhance(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)

        # Düşük ışık
        if brightness < 60:
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

        # Gürültü
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:
            frame = cv2.GaussianBlur(frame, (3, 3), 0)

        return frame

    def _apply_roi(self, frame):
        h, w = frame.shape[:2]
        mask = np.zeros_like(frame)
        mask[:int(h*0.7), :] = frame[:int(h*0.7), :]
        return mask