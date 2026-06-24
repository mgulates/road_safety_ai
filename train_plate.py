# train_plate.py
from ultralytics import YOLO
import yaml
import os

# data.yaml'ı düzelt — yolları absolute yap
data_yaml = {
    'train': os.path.abspath('data/plate_detection/train/images'),
    'val':   os.path.abspath('data/plate_detection/train/images'),  # val yok, train kullan
    'nc':    6,
    'names': ['NaN', 'front', 'heavy_front', 'heavy_rear', 'np', 'rear']
}

with open('data/plate_detection/data_fixed.yaml', 'w') as f:
    yaml.dump(data_yaml, f)

print(" data.yaml düzeltildi")


model = YOLO("yolov8n.pt")

print("🚀 Plaka tespit modeli eğitimi başlıyor...")

results = model.train(
    data="data/plate_detection/data_fixed.yaml",
    epochs=30,
    imgsz=640,
    batch=16,
    device="mps",
    project="models",
    name="plate_detector",
    patience=5,
    verbose=True
)

print(" Eğitim tamamlandı!")
print(" Model: models/plate_detector/weights/best.pt")