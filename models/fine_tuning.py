import logging
from peft import get_peft_model, LoraConfig, TaskType

class FineTuner:
    def __init__(self, config: dict):
        """
        通用 PEFT 微调控制器，支持 LoRA 微调

        参数：
        - config: 包含如下字段：
            - strategy: 'lora' 或 'none'
            - task_type: 可选 'SEQ_2_SEQ_LM'、'CAUSAL_LM'、'SEQ_CLS'、'TOKEN_CLS'
            - base_model_name: 模型名，用于自动推断 task_type
            - r, lora_alpha, lora_dropout: LoRA 的参数
        """
        self.strategy = config.get("strategy", "lora").lower()
        self.base_model_name = config.get("base_model_name", "t5-base")

        # 自动推断任务类型
        if "task_type" in config:
            task_type_str = config["task_type"]
        else:
            task_type_str = self._infer_task_type(self.base_model_name)

        self.task_type = TaskType[task_type_str]

        # LoRA 参数
        self.r = config.get("r", 8)
        self.alpha = config.get("lora_alpha", 16)
        self.dropout = config.get("lora_dropout", 0.1)

    def _infer_task_type(self, model_name: str) -> str:
        """
        根据模型名自动推断 LoRA 所需的任务类型
        """
        name = model_name.lower()
        if "t5" in name:
            return "SEQ_2_SEQ_LM"
        elif "gemma" in name or "llama" in name or "gpt" in name:
            return "CAUSAL_LM"
        elif "bert" in name or "roberta" in name:
            return "TOKEN_CLS"
        else:
            logging.warning(f"无法自动识别模型类型，默认使用 SEQ_CLS：{model_name}")
            return "SEQ_CLS"

    def apply(self, model):
        """
        注入 LoRA 模块（或跳过）
        """
        if self.strategy == "lora":
            config = LoraConfig(
                task_type=self.task_type,
                inference_mode=False,
                r=self.r,
                lora_alpha=self.alpha,
                lora_dropout=self.dropout
            )
            model = get_peft_model(model, config)
            logging.info(f"应用 LoRA 成功：task_type = {self.task_type.name}")
            try:
                model.print_trainable_parameters()
            except Exception:
                pass
            return model

        elif self.strategy == "none":
            logging.info("未应用 PEFT 微调结构，返回原始模型。")
            return model

        else:
            raise ValueError(f"不支持的微调策略类型: {self.strategy}")
