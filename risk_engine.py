# risk_engine.py

class RiskEngine:

    def calculate(self, signals: dict) -> float:
        score = 0.0

        behavior      = signals.get("behavior", "unknown")
        behavior_conf = signals.get("behavior_conf", 0.0)

        # Telefon davranışı — direkt HIGH tetikle
        if behavior in ["talking_phone", "texting_phone"]:
            if behavior_conf > 0.90:
                score += 0.80   # Direkt HIGH
            elif behavior_conf > 0.60:
                score += 0.60   # MEDIUM-HIGH

        # YOLOv8 telefon tespiti (davranış modeli yoksa)
        elif signals.get("phone_detected"):
            score += 0.40

        # Uyuklama
        if signals.get("drowsy"):
            score += 0.35

        # El direksiyonda değil
        if not signals.get("hand_on_wheel", True):
            score += 0.15

        # Araç yakınlığı
        proximity = signals.get("vehicle_proximity", 0.0)
        score += proximity * 0.10

        return min(round(score, 4), 1.0)