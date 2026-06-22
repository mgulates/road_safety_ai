# risk_engine.py

class RiskEngine:
    """
    Hafif model sinyallerinden 0.0-1.0 arası risk skoru üretir.
    """

    WEIGHTS = {
        "phone_detected":    0.35,
        "drowsy":            0.30,
        "hand_off_wheel":    0.15,
        "vehicle_proximity": 0.15,
        "no_face":           0.05,
    }

    def calculate(self, signals: dict) -> float:
        score = 0.0
        # Fine-tuned davranış modeli katkısı
        behavior = signals.get("behavior", "unknown")
        behavior_conf = signals.get("behavior_conf", 0.0)
        if behavior in ["talking_phone", "texting_phone"] and behavior_conf > 0.6:
            score += 0.35

        if signals.get("phone_detected"):
            score += self.WEIGHTS["phone_detected"]

        if signals.get("drowsy"):
            score += self.WEIGHTS["drowsy"]

        if not signals.get("hand_on_wheel", True):
            score += self.WEIGHTS["hand_off_wheel"]

        # Araç yakınlığı
        proximity = signals.get("vehicle_proximity", 0.0)
        score += proximity * self.WEIGHTS["vehicle_proximity"]

        # Yüz tespit edilmemiş
        if signals.get("ear_value", 1.0) == 1.0:
            score += self.WEIGHTS["no_face"]
            

        return min(round(score, 4), 1.0)