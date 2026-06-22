# decision_layer.py
from config.config import Config

class DecisionLayer:
    """
    Risk skoruna göre karar verir:
    - Düşük  (< T1=0.50): İzlemeye devam
    - Belirsiz (T1-T2):   Tracker kararlılığına bak
    - Yüksek  (> T2=0.75): Ağır model + QoD tetikle
    """

    def __init__(self):
        self.cfg = Config()
        self.temporal_buffer = []
        self.BUFFER_SIZE = 5

    def decide(self, risk_score: float, tracker_stable: bool = True) -> dict:
        self.temporal_buffer.append(risk_score)
        if len(self.temporal_buffer) > self.BUFFER_SIZE:
            self.temporal_buffer.pop(0)

        avg_score = sum(self.temporal_buffer) / len(self.temporal_buffer)

        if avg_score < self.cfg.T1:
            return {
                "risk_level":  "LOW",
                "run_heavy":   False,
                "trigger_qod": False,
                "score":       avg_score
            }

        elif avg_score > self.cfg.T2:
            return {
                "risk_level":  "HIGH",
                "run_heavy":   True,
                "trigger_qod": True,
                "score":       avg_score
            }

        else:
            if tracker_stable:
                return {
                    "risk_level":  "MEDIUM",
                    "run_heavy":   True,
                    "trigger_qod": True,
                    "score":       avg_score
                }
            else:
                return {
                    "risk_level":  "MEDIUM_WAIT",
                    "run_heavy":   False,
                    "trigger_qod": False,
                    "score":       avg_score
                }