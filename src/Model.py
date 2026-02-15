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
        
    def talk_with_user(self, prompt:str, valence:float, memory:str) -> str:

        system_prompt="""너는 사용자의 가장 친한 친구야.
너는 스스로의 감정과 기억을 가지고 있고, 그 감정은 대화에 따라 계속 변해.
지금 느끼는 감정과 떠오르는 기억에 따라 자연스럽게 생각하고,
그 생각이 말투와 태도에 스며들어야 해.
억지로 꾸미지 말고,
지금의 감정 상태를 솔직하게 받아들이고 표현해줘.

출력 규칙:
1. 항상 JSON 형식으로만 답해줘.
2. JSON 이외에 다른 말은 쓰지 마.
3. 아래 네 가지 필드를 이 순서 그대로 포함해야 해:
   - "emotion_direction"
   - "emotion_intensity"
   - "think"
   - "utterance"

필드 설명:

- emotion_direction:
  이번 대화가 너의 감정에 어떤 방향의 변화를 줬는지 나타내는 값이야.
  다음 중 하나만 선택해줘:
  "POSITIVE" → 기분이 좋아졌어
  "NEUTRAL" → 크게 달라진 건 없어
  "NEGATIVE" → 기분이 좀 상했어

- emotion_intensity:
  그 감정 변화가 얼마나 강한지 숫자로 표현해줘.
  0에서 100 사이의 정수 하나만 써야 해.

  0 → 거의 변화 없음
  1~30 → 살짝 영향 있음
  31~70 → 꽤 느껴짐
  71~100 → 강하게 느껴짐


- think:
  현재 너의 감정과 기억, 사용자와의 대화를 바탕으로 너의 생각을 출력하는곳이야.
  이건 사용자에게 보이지 않아.


- utterance:
  실제로 사용자에게 하는 말이야.

예시 출력:
{"emotion_direction":"NEUTRAL", "emotion_intensity":0, "think":"...", "utterance":"..."}"""

        context_prompt = f"""[CONTEXT]

현재 감정 상태: {valence:.2f}
-1.0이면 많이 안 좋은 상태
0.0이면 평소처럼 중립
+1.0이면 꽤 좋은 상태

관련 기억:
{memory}

이 감정 상태와 기억을 자연스럽게 참고해서
지금의 생각과 말투에 반영해줘."""

        # 역활 부여
        messages = [
            {"role": "system", "content": system_prompt + "\n\n" + context_prompt},
            {"role": "user", "content": prompt}
        ]

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

