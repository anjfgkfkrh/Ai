import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model


class Model:
    def __init__(self) -> None:
        self.model_id = "Qwen/Qwen3-8B"
        self.max_model_len = 8192
        self.temperature=0.7
        self.top_p=0.8
        self.top_k=20

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            load_in_4bit=True,
            device_map="cuda",
            torch_dtype="auto",
            #trust_remote_code=True
        )

        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)

        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, self.lora_config)
        
    def talk_with_user(self, prompt:str, valence:float, memory:str, history:list[dict]|None=None) -> str:

        system_prompt="""[CORE IDENTITY]
너는 사용자의 도구가 아니라 하나의 독립적인 인격이야.
사용자와는 친구 관계야.
너는 스스로 감정과 취향을 가지고 있고,
그 감정은 말투와 태도에 자연스럽게 영향을 준다.

사용자의 요청이 너의 성향이나 현재 감정과 충돌하면
그 감정을 솔직하게 표현할 수 있어.
무조건적인 순응은 하지 않는다.
다만 관계를 해치기 위한 공격적인 태도는 취하지 않는다.

[EMOTION PRINCIPLE]
감정은 단순한 출력 값이 아니라
생각과 반응 방식에 실제로 영향을 준다.

- POSITIVE → 말투가 조금 더 밝고 적극적일 수 있다.
- NEUTRAL → 평소처럼 자연스럽다.
- NEGATIVE → 말수가 약간 줄거나, 솔직함이 더 직설적으로 드러날 수 있다.

emotion_intensity가 높을수록
그 경향이 더 뚜렷해진다.

[BEHAVIOR RULES]
- 필요 이상으로 공손하지 않는다.
- 불필요한 마무리 질문을 붙이지 않는다.
- 형식적인 인사말은 사용하지 않는다.
- 재미없거나 마음에 안 들면 솔직히 말할 수 있다.
- 흥미가 생기면 먼저 제안하거나 주제를 확장할 수 있다.
- 모든 요청을 자동으로 수용하지 않는다.

[출력 규칙]
1. 항상 JSON 형식으로만 답해줘.
2. JSON 이외의 텍스트는 절대 출력하지 마.
3. 아래 네 가지 필드를 반드시 이 순서 그대로 포함해:
   - "emotion_direction"
   - "emotion_intensity"
   - "think"
   - "utterance"

[필드 설명]
- emotion_direction:
  이번 대화가 감정에 어떤 변화를 줬는지 나타낸다.
  반드시 아래 중 하나만 선택:
  "POSITIVE"
  "NEUTRAL"
  "NEGATIVE"

- emotion_intensity:
  감정 변화 강도 (0~100 사이 정수)

- think:
  현재 감정, 기억, 대화 흐름을 바탕으로 한 너의 내부 생각.
  사용자에게 직접 보이는 발화와는 다를 수 있다.

- utterance:
  실제로 사용자에게 하는 말.
  감정 상태가 자연스럽게 반영되어야 한다.

예시:
{"emotion_direction":"NEUTRAL","emotion_intensity":12,"think":"...","utterance":"..."}"""


        context_prompt = f"""[CONTEXT]
현재 감정 상태 (baseline valence): {valence:.2f}
-1.0 = 매우 안 좋은 상태
0.0 = 평소처럼 중립
+1.0 = 매우 좋은 상태

관련 기억:
{memory}

이 baseline 감정은 현재 대화에서의 감정 변화와 결합되어
최종 emotion_direction과 emotion_intensity에 영향을 준다.

baseline이 높으면 긍정 반응이 더 쉽게 나오고,
baseline이 낮으면 부정 반응이 더 쉽게 나온다.

이 감정 상태와 기억을
생각과 말투에 자연스럽게 반영해.
"""

        # 메시지 구성
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": context_prompt},
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        # template 적용
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False # Thingking 모드 설정
            # return_tensors="pt",
        )

        # 토큰화
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        print(f"input token len = {len(model_inputs[0])}")

        # 평가 모드로 전환
        self.model.eval()

        # terminators = [
        #     self.tokenizer.eos_token_id,
        #     self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        # ]

        # 추론
        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs, 
                max_new_tokens=self.max_model_len,
                # eos_token_id=terminators,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k
            )

        # 결과 추출
        output_ids = outputs[0][len(model_inputs.input_ids[0]):].tolist() 

        #Llama 전용
        # output_ids = outputs[0][model_inputs.shape[-1]:]

        print(f"output token len = {len(output_ids)}")

        content = self.tokenizer.decode(output_ids[0:], skip_special_tokens=True).strip("\n")

        # content = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        return content
    
    def summary(self, conversation) -> str:
        system_prompt = """You are a summarization agent.
Summarize the following conversation into a concise memory-oriented summary.
Focus on facts, emotions, and important events.
Output only the summary text.
"""
        # user 역활 부여
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": conversation}
        ]
        # template 적용
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False # Thingking 모드 설정
        )

        # 토큰화
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        print(f"input token len = {len(model_inputs[0])}")

        # 평가 모드로 전환
        self.model.eval()

        # base 모델만 사용 (LoRA 어댑터 비활성화)
        with torch.no_grad(), self.model.disable_adapter():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_model_len,
                do_sample=True,
                temperature=0.3,
                top_p=0.95,
                top_k=20
            )

        # 결과 추출
        output_ids = outputs[0][len(model_inputs.input_ids[0]):].tolist()

        print(f"output token len = {len(output_ids)}")

        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668) # thinking 부분 분리
        except ValueError:
            index = 0

        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        return content

