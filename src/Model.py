import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

MAX_NEW_TOKENS = 2048         # utterance 생성용
MAX_NEW_TOKENS_SUMMARY = 256 # summary 생성용
SYSTEM_PROMPT_PATH  = os.path.join(os.path.dirname(__file__), "prompts", "system_prompt.txt")
CONTEXT_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "context_prompt.txt")


class Model:
    def __init__(self) -> None:
        self.model_id = "Qwen/Qwen3-8B"
        self.temperature = 0.6
        self.top_p = 0.95
        self.top_k = 20

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="cuda",
            attn_implementation="sdpa",  # O(seq_len^2) attention matrix 메모리 절약
        )

        # gradient_checkpointing / prepare_model_for_kbit_training은
        # 학습 전용 → embedding을 float32로 업캐스트하여 VRAM 낭비 발생, inference에서 제거

        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, self.lora_config)
        self.model.eval()
        
    def talk_with_user(self, prompt:str, valence:float, memory:str, history:list[dict]|None=None) -> str:

        with open(SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
            system_prompt = f.read()


        with open(CONTEXT_PROMPT_PATH, encoding="utf-8") as f:
            context_prompt = f.read().format(valence=valence, memory=memory)

        # 메시지 구성
        # system prompt 추가
        messages = [{"role": "system", "content": system_prompt},]

        # 대화 기록 추가
        if history:
            messages.extend(history)

        # context prompt 추가
        messages.append({"role": "user", "content": context_prompt})

        messages.append({"role": "user", "content": prompt})

        # template 적용
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True # Thingking 모드 설정
        )

        # 토큰화
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        print(f"input token len = {len(model_inputs[0])}")

        # 평가 모드로 전환
        self.model.eval()

        # 추론
        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k
            )

        # 결과 추출
        output_ids = outputs[0][len(model_inputs.input_ids[0]):].tolist() 

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        print(f"output token len = {len(output_ids)}")

        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

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
                max_new_tokens=MAX_NEW_TOKENS_SUMMARY,
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

