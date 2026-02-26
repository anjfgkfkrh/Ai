import json
import logging
import os
from kiwipiepy import Kiwi
from Model import Model
from Rag import Rag
from Emotion import EmotionState


DECAY_RATE = 0.95
RECALL_BOOST = 0.3
CLEANUP_THRESHOLD = 0.1
MIN_SIMILARITY = 0.75 # 대화 맥락 검색 최소 유사도
KEYWORD_MIN_SIMILARITY = 0.6 # 키워드 검색 최소 유사도
RECALL_CONTEXT_TURNS = 2
KEYWORD_RECALL_LIMIT = 3       # 명사 기반 키워드 검색 상위 N개
KEYWORD_RERANK_LIMIT = 5       # 키워드 후보를 대화 흐름 기준 재정렬 후 상위 N개
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")


class ChatPipeline:
    """대화 파이프라인: 입력 → RAG 검색 → 모델 추론 → 감정 갱신 → 저장 → 출력"""

    def __init__(self, max_history: int = 10) -> None:
        self.model = Model()
        self.rag = Rag()
        self.emotion = EmotionState()
        self.max_history = max_history
        self.history: list[dict] = []
        self._kiwi = Kiwi()
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
        context = " ".join(h["content"] for h in self.history[-(RECALL_CONTEXT_TURNS * 2):])
        recall_query = f"{context} {user_input}".strip()
        memories = self.rag.recall_memory(recall_query, limit=10, min_score=MIN_SIMILARITY)

        nouns = self._extract_nouns(user_input)
        if nouns:
            seen = {m["turn_id"] for m in memories}
            keyword_candidates: list[dict] = []
            for noun in nouns:
                for m in self.rag.recall_memory(noun, limit=KEYWORD_RECALL_LIMIT, min_score=KEYWORD_MIN_SIMILARITY):
                    if m["turn_id"] not in seen:
                        seen.add(m["turn_id"])
                        keyword_candidates.append(m)
            self.logger.info(
                f"[KEYWORD CANDIDATES({len(keyword_candidates)})] nouns={nouns}\n"
                + "\n".join(f"  {m['user_text'][:50]} / {m['ai_text'][:50]}" for m in keyword_candidates)
            )
            reranked = self.rag.rerank(recall_query, keyword_candidates)[:KEYWORD_RERANK_LIMIT]
            self.logger.info(
                f"[KEYWORD RERANK({len(reranked)})] top {KEYWORD_RERANK_LIMIT} by recall_query\n"
                + "\n".join(f"  {m['user_text'][:50]} / {m['ai_text'][:50]}" for m in reranked)
            )
            memories += reranked

        self.rag.decay_memories(DECAY_RATE)
        self.rag.cleanup_weak_memories(CLEANUP_THRESHOLD)
        for m in memories:
            self.rag.reinforce_turn(m["turn_id"], RECALL_BOOST)
        memory_text = self._format_memories(memories)
        self.logger.info(f"[MEMORY]\n{memory_text}")

        # 2. 최근 대화 히스토리
        recent_history = self.history[-(self.max_history * 2):]
        self.logger.info(f"[HISTORY] {len(recent_history) // 2} turns")

        # 3. 모델 추론 (LoRA 어댑터 사용)
        raw_output = self.model.talk_with_user(
            prompt=user_input,
            valence=self.emotion.valence,
            memory=memory_text,
            history=recent_history,
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

        # 7. 히스토리 갱신 (max_history 턴만 유지)
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": utterance})
        self.history = self.history[-(self.max_history * 2):]

        self.logger.info(f"[UTTERANCE] {utterance}")
        return utterance

    # ── Internal ──────────────────────────────────────────

    def _extract_nouns(self, text: str) -> list[str]:
        tokens = self._kiwi.tokenize(text)  # type: ignore[arg-type]
        return [t.form for t in tokens if t.tag in ("NNG", "NNP")]  # type: ignore[union-attr]

    def _format_memories(self, memories: list[dict]) -> str:
        if not memories:
            return "(no relevant memories)"
        blocks = []
        for m in memories:
            lines = []
            if m.get("prev_user_text"):
                lines.append(f"  (앞 맥락) \"{m['prev_user_text']}\" / \"{m['prev_ai_text']}\"")
            lines.append(
                f"[기억 | 감정: {m['emotion']:.2f}] "
                f"상대방이 \"{m['user_text']}\"라고 했고, 당시 \"{m['ai_text']}\"로 반응했음."
            )
            if m.get("next_user_text"):
                lines.append(f"  (뒤 맥락) \"{m['next_user_text']}\" / \"{m['next_ai_text']}\"")
            blocks.append("\n".join(lines))
        return "\n---\n".join(blocks)

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
