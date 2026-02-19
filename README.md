# Human - AI(아이)

> **이 프로젝트는 현재 개발 중입니다.**

감정과 기억을 가진 대화형 AI 프로토타입.
Graph DB에 대화를 저장하고, 감정 상태에 따라 반응이 달라지는 챗봇.

## 실행

### 사전 준비

- Python 3.12+
- Neo4j (Docker)
- CUDA 지원 GPU

### 1. Neo4j 실행

```bash
docker run -d \
  -p 7687:7687 -p 7474:7474 \
  -v ./DB:/data \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

### 2. 환경 변수 설정

`.env` 파일을 프로젝트 루트에 생성:

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
```

### 3. 의존성 설치 및 실행

```bash
pip install -r requirements.txt
cd src && python app.py
```

브라우저에서 `http://localhost:7860` 접속.
