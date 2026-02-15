import json
import logging
import os
from Model import Model
from Rag import Rag
from Emotion import EmotionState


DECAY_RATE = 0.95
RECALL_BOOST = 0.3
CLEANUP_THRESHOLD = 0.1
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")


class ChatPipeline:
    """대화 파이프라인: 입력 → RAG 검색 → 모델 추론 → 감정 갱신 → 저장 → 출력"""

    def __init__(self, max_history: int = 10) -> None:
        self.model = Model()
        self.rag = Rag()
        self.emotion = EmotionState()
        self.max_history = max_history
        self.history: list[dict] = []
        self.logger = logging.getLogger("pipeline")
        self.logger.setLevel(logging.DEBUG)
        sid = self.rag.start_session()
        self._setup_logger(sid)

    def _setup_logger(self, session_id: str) -> None:
        os.makedirs(LOG_DIR, exist_ok=True)
        for h in self.logger.handlers[:]:
            self.logger.removeHandler(h)
            h.close()
        handler = logging.FileHandler(
            os.path.join(LOG_DIR, f"{session_id}.log"), encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        self.logger.addHandler(handler)
        self.logger.info(f"[SESSION START] {session_id}")

    def chat(self, user_input: str) -> str:
        self.logger.info(f"[USER] {user_input}")

        # 1. RAG에서 관련 기억 검색 + 기억 풍화
        memories = self.rag.recall_memory(user_input, limit=3)
        self.rag.decay_memories(DECAY_RATE)
        self.rag.cleanup_weak_memories(CLEANUP_THRESHOLD)
        for m in memories:
            self.rag.reinforce_turn(m["turn_id"], RECALL_BOOST)
        memory_text = self._format_memories(memories)
        self.logger.info(f"[MEMORY]\n{memory_text}")

        # 2. 최근 대화 포함 프롬프트 구성
        prompt = self._build_prompt(user_input)
        self.logger.info(f"[PROMPT]\n{prompt}")

        # 3. 모델 추론 (LoRA 어댑터 사용)
        raw_output = self.model.talk_with_user(
            prompt=prompt,
            valence=self.emotion.valence,
            memory=memory_text,
        )
        self.logger.info(f"[RAW OUTPUT] {raw_output}")

        # 4. JSON 파싱
        parsed = self._parse_response(raw_output)
        direction = parsed.get("emotion_direction", "NEUTRAL")
        intensity = int(parsed.get("emotion_intensity", 0))
        think = parsed.get("think", "")
        utterance = parsed.get("utterance", raw_output)

        # 5. 감정 업데이트 (지수 곡선)
        old_valence = self.emotion.valence
        emotion_delta = self.emotion.compute_delta(direction, intensity)
        self.emotion.update(direction, intensity)
        self.logger.info(
            f"[EMOTION] {direction}({intensity}) | "
            f"valence: {old_valence:.4f} → {self.emotion.valence:.4f} (delta: {emotion_delta:.4f})"
        )

        # 6. RAG에 저장
        self.rag.remember_it(
            user_text=user_input,
            ai_text=utterance,
            thinking=think,
            emotion=self.emotion.valence,
            emotion_delta=emotion_delta,
        )

        # 7. 히스토리 갱신
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": utterance})

        self.logger.info(f"[UTTERANCE] {utterance}")
        return utterance

    # ── Internal ──────────────────────────────────────────

    def _format_memories(self, memories: list[dict]) -> str:
        if not memories:
            return "(no relevant memories)"
        return "\n".join(
            f"- [emotion: {m['emotion']:.2f}] User: {m['user_text']} / AI: {m['ai_text']}"
            for m in memories
        )

    def _build_prompt(self, user_input: str) -> str:
        recent = self.history[-(self.max_history * 2):]
        if not recent:
            return user_input
        history_lines = "\n".join(
            f"{'User' if h['role'] == 'user' else 'AI'}: {h['content']}"
            for h in recent
        )
        return f"[Recent conversation]\n{history_lines}\n\n[Current message]\n{user_input}"

    def _parse_response(self, raw: str) -> dict:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                try:
                    return json.loads(raw[start : end + 1])
                except json.JSONDecodeError:
                    pass
            return {"utterance": raw}

    # ── Session 관리 ──────────────────────────────────────

    def reset_session(self) -> None:
        self.logger.info("[SESSION RESET]")
        self.rag.end_session()
        sid = self.rag.start_session()
        self.emotion = EmotionState()
        self.history.clear()
        self._setup_logger(sid)

    def close_session(self) -> None:
        """세션 종료: 대화 요약 생성 → DB 저장 → 새 세션 준비"""
        if self.history:
            conversation = "\n".join(
                f"{'User' if h['role'] == 'user' else 'AI'}: {h['content']}"
                for h in self.history
            )
            summary = self.model.summary(conversation)
            self.logger.info(f"[SUMMARY] {summary}")
            self.rag.end_session(summary_text=summary)
        else:
            self.rag.end_session()
        self.logger.info("[SESSION END]")
        self.emotion = EmotionState()
        self.history.clear()
        sid = self.rag.start_session()
        self._setup_logger(sid)

    def close(self) -> None:
        self.rag.end_session()
        self.rag.close()
