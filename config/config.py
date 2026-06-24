# config/config.py

class Config:
    # Model ayarları
    LIGHT_MODEL = "yolov8n.pt"
    HEAVY_MODEL = "rtdetr-l.pt"
    
    # Görüntü boyutu
    IMG_SIZE = 640
    
    # Risk eşikleri
    T1 = 0.40   
    T2 = 0.55   
    # EAR (göz kırpma) eşiği
    EAR_THRESHOLD = 0.25
    EAR_CONSEC_FRAMES = 20
    
    # Tespit güven eşikleri
    VEHICLE_CONF = 0.5
    PHONE_CONF   = 0.20
    PLATE_CONF   = 0.4
    
    # Cihaz — Apple GPU
    DEVICE = "mps"
    
    # COCO sınıf ID'leri
    VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    PERSON_CLASS    = 0
    
    # Telefon cooldown (saniye)
    PHONE_COOLDOWN_SEC = 5