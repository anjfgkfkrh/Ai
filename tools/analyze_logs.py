"""세션 로그 정량 분석.

docs/실험기록.md 와 docs/부록.md 에 인용된 수치를 재현한다.
프로젝트 종료 후 남은 로그를 재집계하는 스크립트이며, 개발 중에 측정한 값이 아니다.

사용법:
    python tools/analyze_logs.py                    # logs_1, logs 자동 탐색
    python tools/analyze_logs.py <디렉터리> ...
"""

from __future__ import annotations

import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

# ── src 의 상수 사본 (원본이 바뀌면 함께 고쳐야 한다) ──────────────
EXPONENT_K = 0.06        # src/Emotion.py
DECAY_RATE = 0.95        # src/Pipeline.py
CLEANUP_THRESHOLD = 0.1
TS_THRESHOLDS = {"분": 1.5, "시": 0.5, "일": 0.15}
MEMORY_ALPHA, MEMORY_BETA, BASE_STRENGTH = 6.0, 3.0, 2  # src/Rag.py

# 기억 수명 계산의 기준이 되는 초기 강도 (감정이 거의 없는 대화 + 랜덤 항 평균)
TYPICAL_STRENGTH = BASE_STRENGTH + 0.5

NEAR_DUPLICATE_RATIO = 0.8   # 이 이상이면 유사 반복으로 센다
VALID_DIRECTIONS = {"POSITIVE", "NEGATIVE", "NEUTRAL"}

ENTRY_HEAD = re.compile(r"^(\d{4}-\d{2}-\d{2}) \d{2}:\d{2}:\d{2},\d+ \| (.*)$")
TAG = re.compile(r"^\[([A-Z][A-Z ]*)(?:\((\d+)\))?\]\s*(.*)$", re.S)
EMOTION = re.compile(
    r"^(\w+)\((\d+)\) \| valence: (-?[\d.]+) → (-?[\d.]+) \(delta: (-?[\d.]+)\)"
)


@dataclass
class Turn:
    date: str
    direction: str = ""
    intensity: int = 0
    valence_before: float = 0.0
    delta: float = 0.0
    memory_logged: bool = False
    memory_failed: bool = False
    utterance: str = ""
    keyword_candidates: int | None = None


@dataclass
class Session:
    path: Path
    date: str = ""
    turns: list[Turn] = field(default_factory=list)
    user_inputs: int = 0
    summaries: int = 0


# ── 파싱 ──────────────────────────────────────────────────────────

def parse_entries(text: str) -> list[tuple[str, str, str, str | None]]:
    """로그를 (날짜, 태그, 본문, 괄호숫자) 단위로 자른다.

    한 항목이 여러 줄일 수 있으므로 타임스탬프로 시작하는 줄을 경계로 삼는다.
    """
    entries: list[tuple[str, str, str, str | None]] = []
    date = body = ""
    for line in text.splitlines():
        head = ENTRY_HEAD.match(line)
        if head:
            if body:
                entries.append(_split_tag(date, body))
            date, body = head.group(1), head.group(2)
        elif body:
            body += "\n" + line
    if body:
        entries.append(_split_tag(date, body))
    return entries


def _split_tag(date: str, body: str) -> tuple[str, str, str, str | None]:
    m = TAG.match(body)
    if not m:
        return date, "", body, None
    return date, m.group(1).strip(), m.group(3), m.group(2)


def load_session(path: Path) -> Session:
    session = Session(path=path)
    pending = Turn(date="")
    open_turn = False

    for date, tag, body, count in parse_entries(path.read_text(encoding="utf-8")):
        if not session.date:
            session.date = date

        if tag == "USER":
            if open_turn:
                session.turns.append(pending)
            pending, open_turn = Turn(date=date), True
            session.user_inputs += 1
        elif tag == "MEMORY":
            pending.memory_logged = True
            pending.memory_failed = "(no relevant memories)" in body
        elif tag == "KEYWORD CANDIDATES":
            pending.keyword_candidates = int(count or 0)
        elif tag == "EMOTION":
            m = EMOTION.match(body)
            if m:
                pending.direction = m.group(1)
                pending.intensity = int(m.group(2))
                pending.valence_before = float(m.group(3))
                pending.delta = float(m.group(5))
        elif tag == "UTTERANCE":
            pending.utterance = body.strip()
        elif tag == "SUMMARY":
            session.summaries += 1

    if open_turn:
        session.turns.append(pending)
    return session


# ── 지표 ──────────────────────────────────────────────────────────

def solve_k(direction: str, intensity: int, valence: float, delta: float) -> float | None:
    """관측된 delta 로부터 감정 곡선의 지수 계수를 역산한다.

        delta = (target - valence) * (e^(k·i) - 1) / (e^(100k) - 1)
    """
    if intensity <= 0 or delta == 0 or direction not in ("POSITIVE", "NEGATIVE"):
        return None
    target = 1.0 if direction == "POSITIVE" else -1.0
    span = target - valence
    if span == 0:
        return None
    want = delta / span
    if not 0 < want < 1:
        return None

    lo, hi = 1e-4, 1.0
    for _ in range(200):                      # 이분법
        mid = (lo + hi) / 2
        factor = (math.exp(mid * intensity) - 1) / (math.exp(mid * 100) - 1)
        if factor > want:                     # factor 는 k 에 대해 단조 감소
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def near_duplicate_count(utterances: list[str]) -> int:
    """앞선 발화와 NEAR_DUPLICATE_RATIO 이상 겹치는 발화의 수."""
    hits = 0
    for i, u in enumerate(utterances):
        if not u:
            continue
        if any(
            SequenceMatcher(None, u, prev).ratio() >= NEAR_DUPLICATE_RATIO
            for prev in utterances[:i] if prev
        ):
            hits += 1
    return hits


def memory_lifetime() -> list[tuple[str, float, float]]:
    """상수만으로 계산한 기억 수명. (설명, 임계값, 예상 턴 수)"""
    rows = []
    for unit, th in TS_THRESHOLDS.items():
        rows.append((f"{unit} 단위 소실", th, math.log(th / TYPICAL_STRENGTH) / math.log(DECAY_RATE)))
    rows.append(
        ("삭제", CLEANUP_THRESHOLD,
         math.log(CLEANUP_THRESHOLD / TYPICAL_STRENGTH) / math.log(DECAY_RATE))
    )
    return rows


# ── 리포트 ────────────────────────────────────────────────────────

def report(sessions: list[Session]) -> None:
    turns = [t for s in sessions for t in s.turns]
    scored = [t for t in turns if t.direction]          # EMOTION 이 찍힌 턴
    dates = sorted({s.date for s in sessions if s.date})

    print("=" * 66)
    print("  대상")
    print("=" * 66)
    print(f"  로그 파일 {len(sessions)}개 (대화가 있는 세션 {len([s for s in sessions if s.turns])}개) "
          f"· 사용자 턴 {sum(s.user_inputs for s in sessions)}개 "
          f"· 발화 {len([t for t in turns if t.utterance])}개")
    print(f"  기간 {dates[0]} ~ {dates[-1]} · 세션 요약 {sum(s.summaries for s in sessions)}건")

    print()
    print("=" * 66)
    print("  기억 검색")
    print("=" * 66)
    attempted = [t for t in turns if t.memory_logged]      # 검색을 시도한 턴
    failed = [t for t in attempted if t.memory_failed]
    print(f"  검색 실패(no relevant memories)  {len(failed)}/{len(attempted)} "
          f"({len(failed) / len(attempted) * 100:.1f}%)")
    kw = [t for t in turns if t.keyword_candidates is not None]
    if kw:
        found = [t for t in kw if t.keyword_candidates]
        print(f"  키워드 검색이 돌아간 턴            {len(kw)}개 "
              f"(후보를 찾은 턴 {len(found)}개)")

    print()
    print("=" * 66)
    print("  감정 신호")
    print("=" * 66)
    zero = [t for t in scored if t.delta == 0]
    print(f"  변화량이 정확히 0인 턴            {len(zero)}/{len(scored)} "
          f"({len(zero) / len(scored) * 100:.1f}%)")
    nonzero = [abs(t.delta) for t in scored if t.delta != 0]
    print(f"  0이 아닌 변화량의 평균 크기       {sum(nonzero) / len(nonzero):.4f}")
    print(f"  관측된 최대 변화량                {max(abs(t.delta) for t in scored):.4f}")

    print("\n  방향 분포")
    by_dir = defaultdict(list)
    for t in scored:
        by_dir[t.direction].append(t)
    for d, group in sorted(by_dir.items(), key=lambda kv: -len(kv[1])):
        mark = "" if d in VALID_DIRECTIONS else "   ← 열거형에 없는 값"
        avg = sum(t.intensity for t in group) / len(group)
        print(f"    {d:10} {len(group):3}턴   평균 intensity {avg:5.1f}{mark}")

    intensities = Counter(t.intensity for t in scored)
    mult5 = sum(v for k, v in intensities.items() if k % 5 == 0)
    print(f"\n  intensity 서로 다른 값 {len(intensities)}개 · "
          f"5의 배수 {mult5}/{len(scored)} ({mult5 / len(scored) * 100:.1f}%) · "
          f"최댓값 {max(intensities)}")
    print("    " + "  ".join(f"{k}:{v}" for k, v in sorted(intensities.items())))

    print()
    print("=" * 66)
    print("  발화 반복")
    print("=" * 66)
    exact = sum(
        len([t for t in s.turns if t.utterance]) - len({t.utterance for t in s.turns if t.utterance})
        for s in sessions
    )
    total_utt = len([t for t in turns if t.utterance])
    near = sum(near_duplicate_count([t.utterance for t in s.turns]) for s in sessions)
    print(f"  세션 내 완전 동일 발화 재출력     {exact}/{total_utt} ({exact / total_utt * 100:.1f}%)")
    print(f"  세션 내 유사 발화(ratio≥{NEAR_DUPLICATE_RATIO})       "
          f"{near}/{total_utt} ({near / total_utt * 100:.1f}%)")

    print()
    print("=" * 66)
    print("  날짜별 방향 분포 (NEUTRAL 제거 시점 확인)")
    print("=" * 66)
    by_date: dict[str, Counter] = defaultdict(Counter)
    for t in scored:
        by_date[t.date][t.direction] += 1
    for d in sorted(by_date):
        c = by_date[d]
        total = sum(c.values())
        neutral = c.get("NEUTRAL", 0)
        print(f"  {d}  총 {total:3}턴  NEUTRAL {neutral:3} "
              f"({neutral / total * 100:4.1f}%)   " +
              " ".join(f"{k}={v}" for k, v in c.most_common() if k != "NEUTRAL"))

    print()
    print("=" * 66)
    print("  감정 곡선 계수 역산")
    print("=" * 66)
    print(f"  현재 코드값 EXPONENT_K = {EXPONENT_K}")
    solved: dict[str, Counter] = defaultdict(Counter)
    for t in scored:
        k = solve_k(t.direction, t.intensity, t.valence_before, t.delta)
        if k is not None:
            solved[t.date][round(k, 2)] += 1          # 소수 둘째 자리로 묶는다
    for d in sorted(solved):
        c = solved[d]
        print(f"  {d}  표본 {sum(c.values()):3}개   " +
              "  ".join(f"k≈{k:.2f} × {n}" for k, n in sorted(c.items())))
    print("  * 로그의 valence·delta 가 소수 4자리로 반올림되어 있어 개별 값에는 오차가 있다.")
    print("  * 같은 날에 두 값이 섞이면 그날 코드가 바뀐 것이다.")

    print()
    print("=" * 66)
    print("  기억 수명 (상수 계산, 로그 집계 아님)")
    print("=" * 66)
    print(f"  초기 강도 {TYPICAL_STRENGTH} · 매 턴 ×{DECAY_RATE} 가정")
    for label, th, n in memory_lifetime():
        print(f"    강도 {th:<5} 도달 → {label:12} 약 {n:.1f}턴 뒤")
    print(f"  회상이 매 턴 이어질 때의 평형 강도  {0.3 / (1 - DECAY_RATE):.1f}")


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parent.parent
    dirs = [Path(a) for a in argv[1:]] or [root / "logs_1", root / "logs"]
    files = sorted(f for d in dirs if d.is_dir() for f in d.glob("*.log"))
    if not files:
        print(f"로그를 찾지 못했습니다: {', '.join(str(d) for d in dirs)}", file=sys.stderr)
        return 1

    report([load_session(f) for f in files])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
