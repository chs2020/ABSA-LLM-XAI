# prompts/template_builder.py

class PromptGenerator:
    def __init__(self,
                 task_type="term_sentiment_classification",
                 prompt_mode="hard",     # hard / soft / hybrid
                 soft_token_count=10,
                 few_shot=False):
        """
        初始化 PromptGenerator 模块

        参数：
        - task_type: ABSA任务类型
            - aspect_term_extraction
            - term_sentiment_classification
            - category_sentiment_classification
        - prompt_mode: 提示类型（hard/soft/hybrid）
        - soft_token_count: 虚拟token数量（用于soft/hybrid）
        - few_shot: 是否启用few-shot示例
        """
        self.task_type = task_type
        self.prompt_mode = prompt_mode
        self.soft_token_count = soft_token_count
        self.few_shot = few_shot

    def generate_prompt(self, sentence: str) -> str:
        if self.prompt_mode == "hard":
            return self._generate_hard_prompt(sentence)
        elif self.prompt_mode == "soft":
            return self._generate_soft_prompt(sentence)
        elif self.prompt_mode == "hybrid":
            return self._generate_hybrid_prompt(sentence)
        else:
            raise ValueError("Unsupported prompt_mode. Choose from ['hard', 'soft', 'hybrid']")

    def _generate_hard_prompt(self, sentence: str) -> str:
        return self._task_template(sentence)

    def _generate_soft_prompt(self, sentence: str) -> str:
        soft_tokens = " ".join([f"<V{i}>" for i in range(self.soft_token_count)])
        return f"{soft_tokens} {sentence}"

    def _generate_hybrid_prompt(self, sentence: str) -> str:
        soft_tokens = " ".join([f"<V{i}>" for i in range(self.soft_token_count)])
        hard_prompt = self._task_template(sentence)
        return f"{soft_tokens} {hard_prompt}"

    def _task_template(self, sentence: str) -> str:
        if self.task_type == "aspect_term_extraction":
            return self._term_extraction_prompt(sentence)
        elif self.task_type == "term_sentiment_classification":
            return self._term_sentiment_prompt(sentence)
        elif self.task_type == "category_sentiment_classification":
            return self._category_sentiment_prompt(sentence)
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

    def _term_extraction_prompt(self, sentence: str) -> str:
        prefix = "Please extract all aspect terms in the following sentence:\n"
        if self.few_shot:
            example = (
                "Example:\n"
                "Sentence: \"The food was great but the service was awful.\"\n"
                "Output: [\"food\", \"service\"]\n\n"
            )
            return example + prefix + f'"{sentence}"'
        return prefix + f'"{sentence}"'

    def _term_sentiment_prompt(self, sentence: str) -> str:
        prefix = "For each aspect term in the sentence below, identify its sentiment polarity (positive, negative, or neutral):\n"
        if self.few_shot:
            example = (
                "Example:\n"
                "Sentence: \"The food was great but the service was awful.\"\n"
                "Output:\nfood \u2192 positive\nservice \u2192 negative\n\n"
            )
            return example + prefix + f'"{sentence}"'
        return prefix + f'"{sentence}"'

    def _category_sentiment_prompt(self, sentence: str) -> str:
        prefix = (
            "Given the sentence below, classify the sentiment polarity for each aspect category (e.g., FOOD#QUALITY, SERVICE#GENERAL):\n"
        )
        if self.few_shot:
            example = (
                "Example:\n"
                "Sentence: \"The food was great but the service was awful.\"\n"
                "Output:\nFOOD#QUALITY \u2192 positive\nSERVICE#GENERAL \u2192 negative\n\n"
            )
            return example + prefix + f'"{sentence}"'
        return prefix + f'"{sentence}"'
