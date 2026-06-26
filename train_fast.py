# train_fast.py
from ultralytics import YOLO
import os, yaml

BASE = os.path.expanduser("~/Downloads")

datasets = [
    {
        "name": "seatbelt",
        "path": f"{BASE}/seatbelt-detection.v4i.yolov8",
    },
    {
        "name": "sigara",
        "path": f"{BASE}/Sigara_Tespit_2.v3i.yolov8",
    },
    {
        "name": "inattention",
        "path": f"{BASE}/Driver Inattention Detection.v3-v3.yolov8",
    },
]

for ds in datasets:
    print(f"\n🚀 {ds['name']} eğitimi başlıyor...")

    # data.yaml düzelt
    with open(f"{ds['path']}/data.yaml", 'r') as f:
        data = yaml.safe_load(f)

    data['train'] = f"{ds['path']}/train/images"
    data['val']   = f"{ds['path']}/valid/images"

    fixed_yaml = f"data/{ds['name']}_data.yaml"
    with open(fixed_yaml, 'w') as f:
        yaml.dump(data, f)

    model = YOLO("yolov8n.pt")
    model.train(
        data=fixed_yaml,
        epochs=10,        # 30 yerine 10
        imgsz=320,        # 640 yerine 320 — 4x hızlı
        batch=8,          # 16 yerine 8
        device="mps",
        project="models",
        name=ds['name'],
        patience=3,       # 5 yerine 3
        verbose=False,
        workers=0         # MPS için önemli
    )

    print(f" {ds['name']} tamamlandı!")

print("\n🎉 Tüm modeller hazır!")