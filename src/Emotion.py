import math

SENSITIVITY = 0.03


class EmotionState:
    """AI 감정 상태(valence)를 지수 곡선 기반으로 관리한다.

    valence 범위: -1.0 (매우 부정) ~ 0.0 (중립) ~ +1.0 (매우 긍정)

    update 공식 (지수 접근):
        target  = +1.0 (POSITIVE) 또는 -1.0 (NEGATIVE)
        delta   = (target - valence) * (1 - e^(-intensity * k))
        valence = valence + delta

    -> intensity가 클수록 target에 빠르게 접근하지만 절대 ±1.0을 넘지 않는다.
    -> 이미 target 근처이면 같은 intensity라도 변화량이 작아진다 (수확 체감).
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
        return (target - self.valence) * (1 - math.exp(-intensity * SENSITIVITY))
