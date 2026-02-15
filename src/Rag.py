import os
import uuid
import random
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

BATCH_MAX_TURNS = 50
MEMORY_ALPHA = 1.0
MEMORY_BETA = 0.5
EMBEDDING_DIM = 1024


class Rag:
    def __init__(self) -> None:
        self.driver = GraphDatabase.driver(
            os.environ['NEO4J_URI'],
            auth=(os.environ['NEO4J_USERNAME'], os.environ['NEO4J_PASSWORD']),
        )
        self.embedder = SentenceTransformer(
            "Qwen/Qwen3-Embedding-0.6B", device="cpu",
        )
        self._init_schema()

        self._session_id: str | None = None
        self._batch_id: str | None = None
        self._batch_turn_count: int = 0
        self._last_turn_id: str | None = None

    def _embed(self, text: str) -> list[float]:
        return self.embedder.encode(text).tolist()

    def _init_schema(self) -> None:
        with self.driver.session() as db:
            db.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Session) REQUIRE n.session_id IS UNIQUE")
            db.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Batch)   REQUIRE n.batch_id IS UNIQUE")
            db.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Turn)    REQUIRE n.turn_id IS UNIQUE")
            db.run("CREATE INDEX IF NOT EXISTS FOR (n:Turn) ON (n.memory_strength)")
            db.run("CREATE INDEX IF NOT EXISTS FOR (n:Turn) ON (n.timestamp)")
            db.run(
                """
                CREATE VECTOR INDEX turn_embedding IF NOT EXISTS
                FOR (t:Turn) ON (t.embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: $dim,
                    `vector.similarity_function`: 'cosine'
                }}
                """,
                dim=EMBEDDING_DIM,
            )

    # ── Session ───────────────────────────────────────────

    def start_session(self) -> str:
        session_id = str(uuid.uuid4())
        with self.driver.session() as db:
            db.run(
                "CREATE (:Session {session_id: $sid, start_time: datetime()})",
                sid=session_id,
            )
        self._session_id = session_id
        self._batch_id = None
        self._batch_turn_count = 0
        self._last_turn_id = None
        return session_id

    def end_session(self, summary_text: str = "") -> None:
        if not self._session_id:
            return
        with self.driver.session() as db:
            db.run(
                "MATCH (s:Session {session_id: $sid}) SET s.end_time = datetime()",
                sid=self._session_id,
            )
            if summary_text:
                db.run(
                    """
                    MATCH (s:Session {session_id: $sid})
                    CREATE (sm:Summary {text: $text, timestamp: datetime()})
                    CREATE (s)-[:HAS_SUMMARY]->(sm)
                    """,
                    sid=self._session_id,
                    text=summary_text,
                )
        self._session_id = None
        self._batch_id = None
        self._batch_turn_count = 0
        self._last_turn_id = None

    # ── Batch (internal) ──────────────────────────────────

    def _ensure_batch(self) -> str:
        if self._batch_id and self._batch_turn_count < BATCH_MAX_TURNS:
            return self._batch_id

        batch_id = str(uuid.uuid4())
        with self.driver.session() as db:
            result = db.run(
                "MATCH (:Session {session_id: $sid})-[:HAS_BATCH]->(b) RETURN count(b) AS cnt",
                sid=self._session_id,
            )
            record = result.single()
            idx = record["cnt"] if record else 0
            db.run(
                """
                MATCH (s:Session {session_id: $sid})
                CREATE (b:Batch {batch_id: $bid, batch_index: $idx})
                CREATE (s)-[:HAS_BATCH]->(b)
                """,
                sid=self._session_id, bid=batch_id, idx=idx,
            )
        self._batch_id = batch_id
        self._batch_turn_count = 0
        return batch_id

    # ── Remember ──────────────────────────────────────────

    def remember_it(
        self,
        user_text: str,
        ai_text: str,
        thinking: str = "",
        emotion: float = 0.0,
        emotion_delta: float = 0.0,
    ) -> str:
        if not self._session_id:
            raise RuntimeError("No active session. Call StartSession() first.")

        batch_id = self._ensure_batch()
        turn_id = str(uuid.uuid4())

        strength = (
            abs(emotion_delta) * MEMORY_ALPHA
            + abs(emotion) * MEMORY_BETA
            + random.uniform(0, 0.1)
        )

        embedding = self._embed(f"{user_text} {ai_text}")

        with self.driver.session() as db:
            db.run(
                """
                MATCH (b:Batch {batch_id: $bid})
                CREATE (t:Turn {
                    turn_id: $tid,
                    user_text: $ut,
                    ai_text: $at,
                    thinking: $th,
                    emotion: $em,
                    emotion_delta: $ed,
                    memory_strength: $ms,
                    embedding: $vec,
                    timestamp: datetime()
                })
                CREATE (b)-[:HAS_TURN]->(t)
                """,
                bid=batch_id, tid=turn_id,
                ut=user_text, at=ai_text, th=thinking,
                em=emotion, ed=emotion_delta, ms=strength,
                vec=embedding,
            )

            if self._last_turn_id:
                db.run(
                    """
                    MATCH (a:Turn {turn_id: $prev})
                    MATCH (b:Turn {turn_id: $curr})
                    CREATE (a)-[:NEXT]->(b)
                    """,
                    prev=self._last_turn_id, curr=turn_id,
                )

        self._last_turn_id = turn_id
        self._batch_turn_count += 1
        return turn_id

    # ── Recall ────────────────────────────────────────────

    def recall_memory(self, query: str, limit: int = 5) -> list[dict]:
        query_vec = self._embed(query)
        with self.driver.session() as db:
            result = db.run(
                """
                CALL db.index.vector.queryNodes('turn_embedding', $lim, $vec)
                YIELD node, score
                RETURN node.turn_id       AS turn_id,
                       node.user_text     AS user_text,
                       node.ai_text       AS ai_text,
                       node.emotion       AS emotion,
                       node.memory_strength AS memory_strength,
                       node.timestamp     AS timestamp,
                       score
                """,
                vec=query_vec, lim=limit,
            )
            return [dict(r) for r in result]

    # ── Memory Maintenance ────────────────────────────────

    def decay_memories(self, decay_rate: float = 0.95) -> None:
        with self.driver.session() as db:
            db.run(
                "MATCH (t:Turn) SET t.memory_strength = t.memory_strength * $r",
                r=decay_rate,
            )

    def cleanup_weak_memories(self, threshold: float = 0.1) -> int:
        with self.driver.session() as db:
            db.run(
                """
                MATCH (prev)-[:NEXT]->(t:Turn)-[:NEXT]->(next)
                WHERE t.memory_strength < $th
                MERGE (prev)-[:NEXT]->(next)
                """,
                th=threshold,
            )
            count_result = db.run(
                "MATCH (t:Turn) WHERE t.memory_strength < $th RETURN count(t) AS cnt",
                th=threshold,
            )
            record = count_result.single()
            removed = record["cnt"] if record else 0
            if removed > 0:
                db.run(
                    "MATCH (t:Turn) WHERE t.memory_strength < $th DETACH DELETE t",
                    th=threshold,
                )
            return removed

    def reinforce_turn(self, turn_id: str, boost: float = 0.3) -> None:
        with self.driver.session() as db:
            db.run(
                "MATCH (t:Turn {turn_id: $tid}) SET t.memory_strength = t.memory_strength + $b",
                tid=turn_id, b=boost,
            )

    # ── Lifecycle ─────────────────────────────────────────

    def close(self) -> None:
        self.driver.close()
