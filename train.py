# train.py
from ultralytics import YOLO

def main():
    # YOLOv8n classification modelini yükle
    model = YOLO("yolov8n-cls.pt")  # Otomatik indirilir

    print("🚀 Fine-tuning başlıyor...")
    print("Bu işlem 30-60 dakika sürebilir.")
    print("─" * 60)

    results = model.train(
        data="data",           # train/ ve val/ burada
        epochs=20,             # 20 tur yeterli başlangıç için
        imgsz=224,             # Görüntü boyutu
        batch=32,              # Aynı anda işlenen resim sayısı
        device="mps",          # Apple GPU
        project="models",      # Kayıt klasörü
        name="driver_behavior",# Model adı
        patience=5,            # 5 epoch iyileşme olmazsa dur
        verbose=True
    )

    print("─" * 60)
    print("✅ Eğitim tamamlandı!")
    print(f"📁 Model kaydedildi: models/driver_behavior/weights/best.pt")

if __name__ == "__main__":
    main()