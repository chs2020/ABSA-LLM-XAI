# -*- coding: utf-8 -*-
"""
BiLSTM + CRF 序列标注头
"""
import torch
import torch.nn as nn
from torchcrf import CRF            # pip install pytorch-crf


class BiLSTMCRF(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        lstm_hidden: int = 256,
        dropout: float = 0.3,
    ):
        """
        参数:
        - hidden_size : Transformer 输出维度
        - num_labels  : BIO 标签数
        - lstm_hidden : BiLSTM 隐层大小
        - dropout     : Dropout 概率
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, num_labels)
        self.crf = CRF(num_tags=num_labels, batch_first=True)

    def forward(self, feats, mask, labels=None):
        """
        feats  : (B, L, H)  —— Transformer hidden_states
        mask   : (B, L)     —— attention_mask (1=valid, 0=pad)
        labels : (B, L)     —— BIO 真值标签; pad 处应为 -100

        返回:
        - loss (若 labels 给定)
        - preds: (B, L) 预测标签, pad 处为 -100
        """
        # 将 pad 位标签临时置 0（CRF 会用 mask 忽略）
        if labels is not None:
            labels = labels.clone()
            labels[labels == -100] = 0

        lstm_out, _ = self.lstm(feats)             # (B, L, 2H')
        emissions = self.fc(self.dropout(lstm_out))

        loss = None
        if labels is not None:
            loss = -self.crf(
                emissions,
                labels,
                mask=mask.bool(),
                reduction="mean",
            )

        # 解码最佳路径
        best_paths = self.crf.decode(emissions, mask=mask.bool())  # List[List[int]]
        max_len = emissions.size(1)
        preds = torch.full(
            (len(best_paths), max_len),
            -100,
            dtype=torch.long,
            device=emissions.device,
        )
        for b, path in enumerate(best_paths):
            preds[b, : len(path)] = torch.tensor(path, device=emissions.device)

        return loss, preds
