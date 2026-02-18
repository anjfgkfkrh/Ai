import math

EXPONENT_K = 0.08


class EmotionState:
    """AI 감정 상태(valence)를 지수 성장 곡선 기반으로 관리한다.

    valence 범위: -1.0 (매우 부정) ~ 0.0 (중립) ~ +1.0 (매우 긍정)

    update 공식:
        factor  = (e^(k * intensity) - 1) / (e^(k * 100) - 1)
        delta   = (target - valence) * factor
        valence = valence + delta

    -> intensity가 낮으면 factor ≈ 0 이라 변화가 거의 없다.
    -> intensity가 높아질수록 factor가 지수적으로 증가한다.
    -> (target - valence) 덕분에 ±1.0을 절대 넘지 않는다.
    """

    def __init__(self, valence: float = 0.0) -> None:
        self.valence = max(-1.0, min(1.0, valence))

    def update(self, direction: str, intensity: int) -> float:
        delta = self.compute_delta(direction, intensity)
        self.valence = max(-1.0, min(1.0, self.valence + delta))
        return self.valence

    def compute_delta(self, direction: str, intensity: int) -> float:
        if direction == "NEUTRAL" or intensity == 0:
            return 0.0
        target = 1.0 if direction == "POSITIVE" else -1.0
        factor = (math.exp(EXPONENT_K * intensity) - 1) / (math.exp(EXPONENT_K * 100) - 1)
        return (target - self.valence) * factor
