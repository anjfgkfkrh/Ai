# Human - AI(아이)

> **이 프로젝트는 현재 개발 중입니다.**

감정과 기억을 가진 대화형 AI 프로토타입.     
Graph DB에 대화를 저장하고, 감정 상태에 따라 반응이 달라지는 챗봇.      
동적 감정을 가진 AI가 직접 경험한 기억을 바탕으로 호불호를 표출하며 사람과 자연스러운 대화와 행동을 하는것이 목적.     

정적 시스템 프롬프트로 핵심 정체성과 행동 편향(behavioral biases)을 고정하는 캐릭터 기반 대화 AI를 설계 중입니다.      
매 턴마다 baseline 감정(valence)과 관련 기억을 동적으로 주입해 반응의 방향과 강도를 조절합니다.      
감정은 정서적 톤을, 기억은 판단의 뉘앙스를 조정하는 분리된 레이어 구조로 구성합니다.          
RAG·세션 기록·히스토리를 결합해 맥락 추적과 일관성 유지를 강화하려 합니다.         
장기적으로는 프롬프트 의존을 줄이고, 상태·기억·가중치 기반으로 성격을 형성하는 구조를 지향합니다.       

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



## 개발 계획

### 1. Model + RAG
기억을 담당하는 Graph DB 기반 RAG 개발     
사용자 질문 + 최근 대화에 관련된 기억들을 retrieve     
감정 -> 생각 -> 발화 순서로 출력하여 감정이 생각에 영향을 주고, 생각이 발언에 영향을 주는 구조    

### 2. 강화학습 환경 구성
감정 + 사용자 발언 기반 보상     
긍정적 감정, 사용자 발언이 긍정적일 경우 양수의 보상        
감정, 생각, 발언이 논리적으로 맞지않는 경우 음수의 보상      

### 3. tool 장착
MCP, funciton calling, 등 다양한 tool을 장착하여 agent화

### 4. 멀티모달 모델로 변경
CUA(Compute Use Agent)로 변경