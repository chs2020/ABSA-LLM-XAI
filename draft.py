import torch
import re
from transformers import (
    RobertaTokenizer,
    RobertaForTokenClassification,
    LlamaForCausalLM,
    LlamaTokenizer,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#######################################
# 阶段1：RoBERTa方面词检测模型
#######################################

class AspectExtractor:
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.model = RobertaForTokenClassification.from_pretrained("roberta-base", num_labels=3)  # BIO标签
        self.model.to(device)

        # 示例训练数据格式
        self.label_map = {"O": 0, "B-ASP": 1, "I-ASP": 2}

    def train(self, train_data):
        # 数据预处理示例
        def tokenize_and_align_labels(examples):
            tokenized_inputs = self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=128,
                is_split_into_words=True
            )

            labels = []
            for i, label in enumerate(examples["tags"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(self.label_map[label[word_idx]])
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                labels.append(label_ids)

            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        # 转换数据集
        dataset = Dataset.from_dict({
            "text": [d["tokens"] for d in train_data],
            "tags": [d["tags"] for d in train_data]
        })
        dataset = dataset.map(tokenize_and_align_labels, batched=True)

        # 训练参数
        training_args = TrainingArguments(
            output_dir="./aspect_model",
            per_device_train_batch_size=8,
            num_train_epochs=3,
            save_strategy="epoch",
            logging_dir="./logs",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
        )
        trainer.train()

    def predict(self, text):
        tokens = self.tokenizer.tokenize(text)
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()
        predicted_tags = [list(self.label_map.keys())[list(self.label_map.values()).index(p)] for p in predictions]

        # 提取方面词
        aspects = []
        current_aspect = []
        for token, tag in zip(tokens, predicted_tags[1:-1]):  # 跳过特殊token
            if tag == "B-ASP":
                if current_aspect:
                    aspects.append("".join(current_aspect).replace("Ġ", " "))
                    current_aspect = []
                current_aspect.append(token)
            elif tag == "I-ASP":
                current_aspect.append(token)
            else:
                if current_aspect:
                    aspects.append("".join(current_aspect).replace("Ġ", " "))
                    current_aspect = []

        return list(set(aspects))  # 去重


#######################################
# 阶段2：LLaMA-3情感分析模型(LoRA)
#######################################

class ABSAGenerator:
    def __init__(self):
        # 初始化基础模型
        self.tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        # 添加LoRA适配器
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()

    def format_data(self, text, aspects, polarities=None):
        # 构造指令模板
        if polarities:
            output = ";".join([f"{a}:{p}" for a, p in zip(aspects, polarities)])
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            你是一个专业的情感分析系统，需要准确识别文本中的方面词及其情感极性。<|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            分析以下文本并提取方面情感对：
            文本：{text}<|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            {output}<|eot_id|>"""
        else:
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            你是一个专业的情感分析系统，需要准确识别文本中的方面词及其情感极性。<|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            分析以下文本并提取方面情感对：
            文本：{text}<|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>"""

        return prompt

    def train(self, train_data):
        # 准备数据集
        formatted_data = []
        for item in train_data:
            prompt = self.format_data(item["text"], item["aspects"], item["polarities"])
            formatted_data.append({"text": prompt})

        dataset = Dataset.from_list(formatted_data)

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # 训练参数
        training_args = TrainingArguments(
            output_dir="./absa_model",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            num_train_epochs=3,
            learning_rate=2e-5,
            fp16=True,
            save_strategy="epoch",
            logging_steps=10,
            remove_unused_columns=False
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )
        trainer.train()

    def predict(self, text, aspects):
        prompt = self.format_data(text, aspects)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 提取助手的回复部分
        response = result.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        return self.parse_output(response)

    def parse_output(self, output):
        pattern = r"([^:;]+):(positive|negative|neutral)"
        matches = re.findall(pattern, output)
        return dict(matches)


#######################################
# 端到端ABSA系统
#######################################

class ABSASystem:
    def __init__(self):
        self.aspect_extractor = AspectExtractor()
        self.sentiment_analyzer = ABSAGenerator()

    def train(self, aspect_data, sentiment_data):
        print("Training Aspect Extractor...")
        self.aspect_extractor.train(aspect_data)

        print("\nTraining Sentiment Analyzer...")
        self.sentiment_analyzer.train(sentiment_data)

    def analyze(self, text):
        print("Extracting aspects...")
        aspects = self.aspect_extractor.predict(text)
        print(f"Detected aspects: {aspects}")

        print("Analyzing sentiment...")
        sentiment_result = self.sentiment_analyzer.predict(text, aspects)
        return {
            "text": text,
            "aspects": aspects,
            "sentiments": sentiment_result
        }


#######################################
# 示例使用
#######################################

if __name__ == "__main__":
    # 示例数据 (实际使用时需要更大规模数据集)
    aspect_train_data = [
        {
            "tokens": ["The", "steak", "was", "excellent", "but", "service", "was", "slow"],
            "tags": ["O", "B-ASP", "I-ASP", "O", "O", "B-ASP", "O", "O"]
        }
    ]

    sentiment_train_data = [
        {
            "text": "The steak was excellent but service was slow",
            "aspects": ["steak", "service"],
            "polarities": ["positive", "negative"]
        }
    ]

    # 初始化系统
    system = ABSASystem()

    # 训练 (实际使用时需要更多数据)
    print("Starting training...")
    system.train(aspect_train_data, sentiment_train_data)

    # 测试
    test_text = "The phone has great battery life though the camera quality is poor"
    result = system.analyze(test_text)

    print("\nFinal Result:")
    print(f"Input Text: {result['text']}")
    print(f"Detected Aspects: {result['aspects']}")
    print(f"Sentiment Analysis: {result['sentiments']}")