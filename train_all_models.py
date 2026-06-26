# train_all_models.py
from ultralytics import YOLO
import os, yaml, shutil

BASE = os.path.expanduser("~/Downloads")

datasets = [
    {
        "name":    "turkish_plate",
        "path":    f"{BASE}/turkish-license-plate.v8i.yolov8",
        "epochs":  30,
        "imgsz":   640,
    },
    {
        "name":    "seatbelt",
        "path":    f"{BASE}/seatbelt-detection.v4i.yolov8",
        "epochs":  30,
        "imgsz":   640,
    },
    {
        "name":    "sigara",
        "path":    f"{BASE}/Sigara_Tespit_2.v3i.yolov8",
        "epochs":  30,
        "imgsz":   640,
    },
    {
        "name":    "inattention",
        "path":    f"{BASE}/Driver Inattention Detection.v3-v3.yolov8",
        "epochs":  30,
        "imgsz":   640,
    },
]

for ds in datasets:
    print(f"\n{'='*60}")
    print(f" Eğitim başlıyor: {ds['name']}")
    print(f"{'='*60}")

    # data.yaml yollarını absolute yap
    yaml_path = f"{ds['path']}/data.yaml"
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    data['train'] = f"{ds['path']}/train/images"
    data['val']   = f"{ds['path']}/valid/images"

    fixed_yaml = f"data/{ds['name']}_data.yaml"
    with open(fixed_yaml, 'w') as f:
        yaml.dump(data, f)

    # Eğitim
    model = YOLO("yolov8n.pt")
    model.train(
        data=fixed_yaml,
        epochs=ds['epochs'],
        imgsz=ds['imgsz'],
        batch=16,
        device="mps",
        project="models",
        name=ds['name'],
        patience=5,
        verbose=False
    )

    print(f" {ds['name']} tamamlandı!")
    print(f"📁 Model: models/{ds['name']}/weights/best.pt")

print("\n🎉 Tüm modeller eğitildi!")