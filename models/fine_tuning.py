# 引入 PEFT 库中支持的微调策略配置类
from peft import (
    get_peft_model,         # 将轻量微调配置应用到 transformers 模型上
    LoraConfig,             # LoRA 微调配置类
    AdapterConfig,          # Adapter 微调配置类
    TaskType                # PEFT 中任务类型的枚举类（决定注入模块位置）
)

# 引入 logging 模块用于控制台输出日志
import logging


class FineTuner:
    def __init__(self, config: dict):
        """
        FineTuner 类：统一封装 PEFT 的配置入口

        功能：
        - 从 config 字典中提取微调参数
        - 支持多种轻量策略：LoRA / Adapter / None
        - 支持多种模型架构（T5 / BERT / RoBERTa / LLaMA）

        参数:
        - config (dict): 包含模型名称、策略、超参数的配置字典

        推荐 config 字典样例：
        {
            "strategy": "lora",           # 微调策略类型
            "task_type": "SEQ_2_SEQ_LM",  # 模型任务类型
            "base_model_name": "t5-base", # 模型名称（保留字段）
            "r": 8,                       # LoRA 的秩
            "lora_alpha": 16,             # LoRA 缩放因子
            "lora_dropout": 0.1           # LoRA Dropout 概率
        }
        """

        # 微调策略类型：可选 "lora" / "adapter" / "none"
        self.strategy = config.get("strategy", "lora").lower()

        # 模型任务类型（用于 PEFT 插入点定位）
        # 如 "SEQ_2_SEQ_LM" 表示 T5、BART 等 seq2seq 模型
        task_type_str = config.get("task_type", "SEQ_2_SEQ_LM")
        self.task_type = TaskType[task_type_str]  # 转换为枚举类型

        # 模型名称（暂未使用，但可用于 future 结构判断）
        self.base_model_name = config.get("base_model_name", "t5-base")

        # 以下为 LoRA 专属参数
        self.r = config.get("r", 8)                       # 低秩矩阵秩
        self.alpha = config.get("lora_alpha", 16)         # 缩放系数 alpha
        self.dropout = config.get("lora_dropout", 0.1)    # Dropout 概率

        # 以下为 Adapter 专属参数（可拓展为 Pfeiffer、Compacter 等）
        self.adapter_size = config.get("adapter_size", 16)            # adapter 隐层维度
        self.adapter_type = config.get("adapter_type", "houlsby")     # adapter 类型：默认 Houlsby

    def apply(self, model):
        """
        将指定的轻量微调策略应用到传入的 transformers 模型上。

        参数:
        - model: 预训练模型（如 BERT、T5 等）从 transformers 加载

        返回:
        - 应用了 PEFT 的模型（带 adapter 或 lora），可用于训练
        """

        # 如果策略为 LoRA
        if self.strategy == "lora":
            config = LoraConfig(
                task_type=self.task_type,
                inference_mode=False,  # 设置为训练模式
                r=self.r,
                lora_alpha=self.alpha,
                lora_dropout=self.dropout
            )
            model = get_peft_model(model, config)
            logging.info("已应用 LoRA 微调结构。")
            return model

        # 如果策略为 Adapter
        elif self.strategy == "adapter":
            config = AdapterConfig(
                task_type=self.task_type,
                inference_mode=False,  # 设置为训练模式
                adapter_size=self.adapter_size,
                adapter_type=self.adapter_type
            )
            model = get_peft_model(model, config)
            logging.info("已应用 Adapter 微调结构。")
            return model

        # 如果不使用轻量微调策略（原模型全参数训练）
        elif self.strategy == "none":
            logging.info("未启用轻量微调，将进行全参数微调。")
            return model

        # 若用户传入不支持的策略名，则报错
        else:
            raise ValueError(f"不支持的微调策略：{self.strategy}")

