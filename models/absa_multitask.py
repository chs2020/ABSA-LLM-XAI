# -*- coding: utf-8 -*-
"""
ABSA 多任务模型:
- BiLSTM + CRF 做方面词抽取 (BIO)
- Linear 做情感极性分类
"""
import torch
import torch.nn as nn
from models.bio_head import BiLSTMCRF


class AspectSentimentModel(nn.Module):
    def __init__(
        self,
        base_model,
        hidden_size: int = 768,
        num_bio_labels: int = 3,
        num_sentiment_labels: int = 3,
        lambda_weight: float = 1.0,
    ):
        super().__init__()
        self.base_model = base_model
        self.lambda_weight = lambda_weight

        # ---- BIO 头：BiLSTM + CRF ----
        self.bio_head = BiLSTMCRF(hidden_size, num_bio_labels)

        # ---- 句子情感头 ----
        self.sentiment_classifier = nn.Linear(hidden_size, num_sentiment_labels)
        self.sentiment_loss_fn = nn.CrossEntropyLoss()

    # -------------------------------------------------
    def _pool_sentence(self, hidden, mask):
        """
        BERT 用 [CLS]；其它模型用 mean pooling
        """
        if (
            hasattr(self.base_model.model.config, "model_type")
            and "bert" in self.base_model.model.config.model_type.lower()
        ):
            return hidden[:, 0, :]  # CLS
        # mean pooling
        mask = mask.unsqueeze(-1).expand(hidden.size()).float()
        return (hidden * mask).sum(1) / mask.sum(1)

    # -------------------------------------------------
    def forward(self, input_ids, attention_mask, bio_labels=None, sentiment_label=None):
        hidden_states, _ = self.base_model.encode_hidden_batch(input_ids, attention_mask)

        # -------- BIO 分支 --------
        loss_bio, bio_preds = self.bio_head(hidden_states, attention_mask, bio_labels)

        # -------- 情感分支 --------
        pooled = self._pool_sentence(hidden_states, attention_mask)
        sentiment_logits = self.sentiment_classifier(pooled)

        output = {
            "bio_preds": bio_preds,                 # tensor(B, L)
            "sentiment_logits": sentiment_logits,   # tensor(B, C)
        }

        # -------- Loss 计算 --------
        if (bio_labels is not None) and (sentiment_label is not None):
            loss_sent = self.sentiment_loss_fn(sentiment_logits, sentiment_label)
            total_loss = loss_bio + self.lambda_weight * loss_sent
            output.update(
                {
                    "loss": total_loss,
                    "loss_bio": loss_bio.item(),
                    "loss_sentiment": loss_sent.item(),
                }
            )

        return output
