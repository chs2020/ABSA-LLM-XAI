from transformers import AutoModel, AutoTokenizer, T5ForConditionalGeneration
from peft import get_peft_model
import torch


class LLMWrapper:
    def __init__(self, model_name, use_peft=False, peft_config=None, device="cuda"):
        """
        通用大语言模型封装器（LLM Wrapper）

        主要功能：
        - 封装 tokenizer + 模型
        - 自动判断模型是 encoder-only（如BERT）还是 encoder-decoder（如T5）
        - 支持 PEFT 微调结构（如 LoRA、Adapter）

        参数说明：
        - model_name: 模型名称字符串，如 'bert-base-uncased' 或 't5-base'
        - use_peft: 是否使用 PEFT 微调（True 时应用 peft_config）
        - peft_config: PEFT 配置对象（LoRAConfig / AdapterConfig）
        - device: 使用设备（默认使用 GPU，有则用 GPU，没有用 CPU）
        """

        # 自动选择可用设备
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 加载对应模型的分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 判断模型结构：是否为 encoder-decoder（如 T5）
        if "t5" in model_name.lower():
            self.is_encoder_decoder = True
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        else:
            self.is_encoder_decoder = False
            self.model = AutoModel.from_pretrained(model_name)

        # 如果启用了 PEFT（如 LoRA / Adapter），将其结构注入模型中
        if use_peft and peft_config is not None:
            self.model = get_peft_model(self.model, peft_config)

        # 将模型移动到指定设备（CPU 或 GPU）
        self.model.to(self.device)
        self.model.eval()  # 默认设置为评估模式

    def encode_prompt(self, prompt: str, max_length=512):
        """
        对文本 prompt 进行编码，用于模型输入。

        返回：
        - 一个 dict，包括 input_ids 和 attention_mask，并自动转移到 self.device
        """
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",       # 返回 PyTorch tensor 格式
            max_length=max_length,     # 最大长度限制
            truncation=True,           # 超长截断
            padding=True               # 自动padding至最大长度
        )
        return {k: v.to(self.device) for k, v in encoded.items()}

    def generate(self, prompt: str, max_new_tokens=50):
        """
        文本生成接口，仅用于 encoder-decoder 架构（如 T5）

        参数：
        - prompt: 输入的提示词
        - max_new_tokens: 控制最大生成长度

        返回：
        - 生成的文本结果（字符串）
        """
        if not self.is_encoder_decoder:
            raise NotImplementedError("generate 仅适用于 encoder-decoder 模型（如 T5）")

        # 编码输入
        inputs = self.encode_prompt(prompt)

        # 禁用梯度，进入推理模式
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens
            )

        # 解码输出 token 序列为字符串
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def encode_hidden(self, prompt: str):
        """
        用于分类 / BIO 序列标注任务，返回 encoder 输出。

        参数：
        - prompt: 输入文本（一个句子）

        返回：
        - last_hidden_state: 最后一层隐藏状态（用于分类/BIO）
        - attention_mask: 用于对 padding 部分屏蔽
        """
        # 编码输入文本
        inputs = self.encode_prompt(prompt)

        # 不计算梯度（推理模式）
        with torch.no_grad():
            if self.is_encoder_decoder:
                # T5等模型：直接访问 encoder 模块
                outputs = self.model.encoder(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    return_dict=True
                )
            else:
                # BERT等模型：默认 forward 输出 hidden_state
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    return_dict=True
                )

        # 返回隐藏层表示 + 注意力 mask
        return outputs.last_hidden_state, inputs["attention_mask"]

