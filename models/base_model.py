import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoConfig,
)
from peft import get_peft_model


class LLMWrapper:
    def __init__(self, model_name, use_peft=False, peft_config=None, device="cuda"):
        """
        通用语言模型封装器：支持 encoder-decoder 和 decoder-only 架构
        - model_name: HuggingFace 模型名
        - use_peft: 是否使用 PEFT（如 LoRA）
        - peft_config: PEFT 配置对象
        - device: 使用的设备（如 'cuda' 或 'cpu'）
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 初始化 tokenizer（use_fast=True 通常更快）
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # 获取模型配置，判断模型类型
        config = AutoConfig.from_pretrained(model_name)
        model_type = config.model_type.lower()

        # 判断是否为 encoder-decoder 模型（如 T5、BART）
        self.is_encoder_decoder = model_type in [
            "t5", "mt5", "bart", "mbart", "pegasus", "blenderbot"
        ]

        # 根据类型自动加载合适模型
        if self.is_encoder_decoder:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        elif model_type in ["gemma", "llama", "gpt2", "bloom", "opt"]:
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        else:
            self.model = AutoModel.from_pretrained(model_name).to(self.device)

        # 如果启用 PEFT 微调
        if use_peft and peft_config is not None:
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

    def encode_text(self, sentences, max_length=512):
        """
        对输入句子编码为 token ids，输出 tensor 字典
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        batch = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        return {k: v.to(self.device) for k, v in batch.items()}

    def build_prompt(self, sentences, prompt_prefix=""):
        """
        构造 prompt，可拼接固定提示词（硬 prompt）
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        return [f"{prompt_prefix} {s}" for s in sentences]

    def encode_prompt(self, sentences, prompt_prefix="", max_length=512):
        """
        组合 prompt 和原句，进行编码
        """
        prompts = self.build_prompt(sentences, prompt_prefix)
        return self.encode_text(prompts, max_length=max_length)

    def encode_hidden_batch(self, input_ids, attention_mask):
        """
        提取 hidden states，支持 encoder-only 和 encoder-decoder 架构
        用于 token-level 特征抽取（如 BIO 标签任务）
        """
        with torch.no_grad():
            if self.is_encoder_decoder and hasattr(self.model, "encoder"):
                # 对于 T5/BART 等，调用 encoder
                outputs = self.model.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
            else:
                # 对于 GPT/LLaMA/Gemma 等，直接用主模型
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
        return outputs.last_hidden_state, attention_mask

    def generate(self, sentences, prompt_prefix="", max_new_tokens=50):
        """
        对模型进行生成推理（如问答/翻译），要求模型支持 generate()
        """
        if not hasattr(self.model, "generate"):
            raise NotImplementedError("当前模型不支持 generate()")

        batch = self.encode_prompt(sentences, prompt_prefix)

        with torch.no_grad():
            out_ids = self.model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=max_new_tokens,
            )
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
