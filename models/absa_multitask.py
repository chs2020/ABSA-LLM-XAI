import torch
import torch.nn as nn


class AspectSentimentModel(nn.Module):
    def __init__(self, base_model, hidden_size=768, num_bio_labels=3, num_sentiment_labels=3, lambda_weight=1.0):
        """
        ABSA 多任务模型：使用 BERT 作为 backbone，执行方面词抽取（BIO）+ 情感极性分类

        参数：
        - base_model: LLMWrapper 封装的 BERT 模型
        - hidden_size: 隐层维度（BERT 默认 768）
        - num_bio_labels: BIO 标签数（如 B=1, I=2, O=0）
        - num_sentiment_labels: 情感类别（如 正/负/中 = 3）
        - lambda_weight: 联合损失中情感任务的权重
        """

        super().__init__()
        self.base_model = base_model
        self.lambda_weight = lambda_weight

        # BIO分类头：token-level 的线性层
        self.bio_classifier = nn.Linear(hidden_size, num_bio_labels)
        self.bio_loss_fn = nn.CrossEntropyLoss()

        # 情感分类头：句向量输入，多分类输出
        self.sentiment_classifier = nn.Linear(hidden_size, num_sentiment_labels)
        self.sentiment_loss_fn = nn.CrossEntropyLoss()

    def forward(self, sentence: str, bio_labels=None, sentiment_label=None):
        """
        前向传播逻辑：先编码句子 → 获取 token 表示和 pooled 表示 → 输出 BIO 和情感 logits → 可计算联合 loss

        输入：
        - sentence: 文本输入（单句）
        - bio_labels: token 级别 BIO 标签（[batch, seq_len]）
        - sentiment_label: 情感极性标签（[batch]）

        返回：
        - 包含 bio_logits, sentiment_logits, loss（可选） 的字典
        """
        # 获取模型的输出（hidden states 和 attention mask）
        hidden_states, attention_mask = self.base_model.encode_hidden(sentence)

        # [CLS] 位作为句向量（BERT 架构）
        pooled = hidden_states[:, 0, :]  # [batch, hidden_size]

        # BIO：每个 token 的分类输出
        bio_logits = self.bio_classifier(hidden_states)  # [batch, seq_len, bio_class]

        # 情感：句向量的分类输出
        sentiment_logits = self.sentiment_classifier(pooled)  # [batch, sentiment_class]

        output = {
            "bio_logits": bio_logits,
            "sentiment_logits": sentiment_logits
        }

        # 如果标签存在，计算两个子任务的 loss
        if bio_labels is not None and sentiment_label is not None:
            # BIO 任务的 token 级别交叉熵
            loss_bio = self.bio_loss_fn(bio_logits.view(-1, bio_logits.shape[-1]), bio_labels.view(-1))

            # 情感任务的句子级交叉熵
            loss_sent = self.sentiment_loss_fn(sentiment_logits, sentiment_label)

            # 联合损失：BIO + λ * 情感分类
            total_loss = loss_bio + self.lambda_weight * loss_sent

            output.update({
                "loss": total_loss,
                "loss_bio": loss_bio,
                "loss_sentiment": loss_sent
            })

        return output
