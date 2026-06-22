# main.py
import cv2
import time
import os
from preprocessor import Preprocessor
from ai_service import AIService
from risk_engine import RiskEngine
from decision_layer import DecisionLayer

def main():
    preprocessor = Preprocessor()
    ai_service   = AIService()
    risk_engine  = RiskEngine()
    decision     = DecisionLayer()

    # ── Mod seçimi ──
    # "video"  → data/ klasöründeki .mp4 dosyası
    # "image"  → data/ klasöründeki tek resim
    # "folder" → data/images/ klasöründeki tüm resimler
    MODE       = "folder"
    VIDEO_PATH = "data/test.mp4"
    IMAGE_PATH = "data/test.jpg"
    IMAGE_DIR  = "data/images/data/training_images/Revitsone-5classes/Revitsone-5classes"


    if MODE == "video":
        _run_video(VIDEO_PATH, preprocessor, ai_service, risk_engine, decision)
    elif MODE == "image":
        _run_image(IMAGE_PATH, preprocessor, ai_service, risk_engine, decision)
    elif MODE == "folder":
        _run_folder(IMAGE_DIR, preprocessor, ai_service, risk_engine, decision)


def _run_folder(folder_path, preprocessor, ai_service, risk_engine, decision):
    if not os.path.exists(folder_path):
        print(f"❌ Klasör bulunamadı: {folder_path}")
        return

    # Alt klasörler dahil tüm resimleri topla
    import glob
    all_files = glob.glob(os.path.join(folder_path, "**", "*.jpg"), recursive=True) + \
                glob.glob(os.path.join(folder_path, "**", "*.png"), recursive=True)
    all_files = sorted(all_files)

    if not all_files:
        print(f"❌ Resim bulunamadı!")
        return

    print(f"📁 {len(all_files)} resim bulundu. Analiz başlıyor...")
    print("─" * 60)

    from collections import defaultdict
    category_files = defaultdict(list) 
    for f in all_files:
        cat = os.path.basename(os.path.dirname(f))
        category_files[cat].append(f)

    selected = []
    for cat, flist in category_files.items():
        selected.extend(flist[:10])

    for i, filepath in enumerate(selected):
        filename = os.path.basename(filepath)
        # Hangi klasörden geldiğini göster
        category = os.path.basename(os.path.dirname(filepath))
        frame    = cv2.imread(filepath)

        if frame is None:
            print(f"⚠️  {filename} okunamadı, geçiliyor.")
            continue

        _, original     = preprocessor.process(frame)
        signals         = ai_service.run_light(original)
        score           = risk_engine.calculate(signals)
        decision_result = decision.decide(score)

        heavy_dets = []
        if decision_result["run_heavy"]:
            heavy_dets = ai_service.run_heavy(original)

        print(
            f"[{i+1}] [{category}] {filename}\n"
            f"  Risk: {score:.2f} | Seviye: {decision_result['risk_level']}\n"
            f"  Telefon: {signals['phone_detected']} | "
            f"Uyuklama: {signals['drowsy']} | "
            f"Araç: {signals['vehicle_detected']}\n"
            f"  Ağır model: {'Çalıştı ✅' if decision_result['run_heavy'] else 'Çalışmadı'}"
        )
        print("─" * 60)

        display  = _draw_overlay(original, signals, score, decision_result)
        out_path = os.path.join("outputs", f"{category}_{filename}")
        cv2.imwrite(out_path, display)

    print(f"\n✅ Tamamlandı! Sonuçlar outputs/ klasörüne kaydedildi.")
    
def _run_video(video_path, preprocessor, ai_service, risk_engine, decision):
    """Video analiz et."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Video açılamadı: {video_path}")
        return

    frame_count = 0
    fps_timer   = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        _, original     = preprocessor.process(frame)
        signals         = ai_service.run_light(original)
        score           = risk_engine.calculate(signals)
        decision_result = decision.decide(score)

        if decision_result["run_heavy"]:
            ai_service.run_heavy(original)

        elapsed = time.time() - fps_timer
        fps     = frame_count / elapsed if elapsed > 0 else 0

        if frame_count % 30 == 0:
            print(
                f"[Frame {frame_count}] "
                f"Risk: {score:.2f} | "
                f"Seviye: {decision_result['risk_level']} | "
                f"FPS: {fps:.1f}"
            )

    cap.release()
    print("✅ Video analizi tamamlandı.")


def _draw_overlay(frame, signals, score, decision):
    h, w = frame.shape[:2]

    colors = {
        "LOW":         (0, 255, 0),
        "MEDIUM":      (0, 165, 255),
        "MEDIUM_WAIT": (0, 200, 255),
        "HIGH":        (0, 0, 255),
    }
    color = colors.get(decision["risk_level"], (255, 255, 255))

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    texts = [
        (f"Risk: {score:.2f}  [{decision['risk_level']}]", color),
        (f"EAR: {signals['ear_value']:.2f}  Uyuklama: {'EVET ⚠' if signals['drowsy'] else 'Hayir'}",
         (0, 0, 255) if signals['drowsy'] else (200, 200, 200)),
        (f"Telefon: {'TESPIT ⚠' if signals['phone_detected'] else 'Yok'}  "
         f"El: {'Direksiyonda' if signals['hand_on_wheel'] else 'Yok ⚠'}",
         (0, 0, 255) if signals['phone_detected'] else (200, 200, 200)),
        (f"Arac: {'Var' if signals['vehicle_detected'] else 'Yok'}  "
         f"QoD: {'AKTIF' if decision['trigger_qod'] else 'Pasif'}",
         (200, 200, 200)),
         (f"Davranis: {signals.get('behavior','?')} ({signals.get('behavior_conf',0):.2f})",
        (0, 0, 255) if signals.get('behavior') in ['talking_phone','texting_phone'] else (200, 200, 200)),
    ]

    for i, (text, col) in enumerate(texts):
        cv2.putText(frame, text, (10, 25 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    bar_w = int(200 * score)
    cv2.rectangle(frame, (w - 210, 10), (w - 10, 30), (50, 50, 50), -1)
    cv2.rectangle(frame, (w - 210, 10), (w - 210 + bar_w, 30), color, -1)

    return frame


if __name__ == "__main__":
    main()