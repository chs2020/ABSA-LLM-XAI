# controller_train.py
# -----------------------------------------------------------
# 训练 ABSA: 方面词抽取(BiLSTM+CRF) + 句子情感分类
# -----------------------------------------------------------
import os, csv, glob
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from tqdm import trange, tqdm

# === 项目内模块 ======
from config.config import train_args                    # 学习率 / batch_size / 设备等
from data_loader.preprocessor import DataLoaderPreprocessor
from data_loader.data_loader import ABSADataset
from models.base_model import LLMWrapper
from models.fine_tuning import FineTuner
from models.absa_multitask import AspectSentimentModel  # 已改为 BiLSTM+CRF 版本
# =====================


# -----------------------------------------------------------
def load_data(csv_paths):
    """读取多 CSV → DataFrame → records 列表"""
    pre = DataLoaderPreprocessor()
    df = pre.load_files(csv_paths)

    df["from"] = df.apply(
        lambda r: r["Sentence"].lower().find(r["Aspect Term"].lower()), axis=1
    )
    df["to"] = df["from"] + df["Aspect Term"].str.len()

    records = []
    for idx, row in df.iterrows():
        if row["from"] >= 0:
            records.append(
                {
                    "id": idx,
                    "Sentence": row["Sentence"],
                    "Aspect_Term": row["Aspect Term"],
                    "polarity": row["polarity"],
                    "from": int(row["from"]),
                    "to": int(row["to"]),
                }
            )
    return records


# -----------------------------------------------------------
def train_one_experiment(backbone, finetune_stg, task_type, tag):
    # ① 数据
    csv_files = glob.glob("E:/ABSA-LLM-XAI/dataset/SemEval2014/**/*.csv", recursive=True)
    data = load_data(csv_files)
    dataset = ABSADataset(
        data,
        tokenizer_name="bert-base-uncased",
        max_length=train_args["max_length"]
    )
    loader = DataLoader(dataset, batch_size=train_args["batch_size"], shuffle=True)

    # ② Backbone + PEFT
    wrapper = LLMWrapper(backbone, device=train_args["device"])
    wrapper.model = FineTuner(
        {
            "strategy": finetune_stg,
            "task_type": task_type,
            "base_model_name": backbone,
        }
    ).apply(wrapper.model)

    # ③ 多任务模型
    model = AspectSentimentModel(wrapper).to(train_args["device"])
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_args["lr"])

    # ④ 日志 / checkpoint
    ckpt_dir = "./checkpoints"; os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = f"{ckpt_dir}/model_{tag}.pth"
    log_path  = f"{ckpt_dir}/log_{tag}.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "bio_f1", "sent_acc"])

    best_f1, patience = 0.0, 0
    model.train()

    for epoch in trange(train_args["num_epochs"], desc=f"[{tag}] Epoch"):
        sum_loss = 0
        true_bio, pred_bio = [], []
        true_sent, pred_sent = [], []

        for batch in tqdm(loader, desc="  Batches", leave=False):
            ids = batch["input_ids"].to(train_args["device"])
            mask = batch["attention_mask"].to(train_args["device"])
            lbl_bio = batch["bio_labels"].to(train_args["device"])
            lbl_sent = batch["sentiment_label"].to(train_args["device"])

            out = model(ids, mask, lbl_bio, lbl_sent)
            loss = out["loss"]

            optimizer.zero_grad(); loss.backward(); optimizer.step()
            sum_loss += loss.item()

            # ---- 收集预测 ----
            bio_pred = out["bio_preds"].view(-1).cpu().numpy()
            bio_true = lbl_bio.view(-1).cpu().numpy()
            keep = bio_true != -100                      # 去掉 pad
            pred_bio.extend(bio_pred[keep])
            true_bio.extend(bio_true[keep])

            pred_sent.extend(out["sentiment_logits"].argmax(-1).cpu().numpy())
            true_sent.extend(lbl_sent.cpu().numpy())

        # ---- Epoch 指标 ----
        ep_loss = sum_loss / len(loader)
        ep_f1   = f1_score(true_bio, pred_bio, average="macro")
        ep_acc  = accuracy_score(true_sent, pred_sent)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, ep_loss, ep_f1, ep_acc])

        print(f"[{tag}] Epoch {epoch+1} | Loss {ep_loss:.4f} | BIO F1 {ep_f1:.4f} | SentACC {ep_acc:.4f}")

        if ep_f1 > best_f1:
            best_f1, patience = ep_f1, 0
            torch.save(model.state_dict(), ckpt_path)
            print("  ↳ New best model saved.")
        else:
            patience += 1
            print(f"  ↳ No improvement ({patience}/{train_args['patience']})")
            if patience >= train_args["patience"]:
                print("  ↳ Early stopping.")
                break

    return best_f1, ep_acc


# -----------------------------------------------------------
if __name__ == "__main__":
    exp_cfgs = [
        ("google/gemma-1.1-2b-it",   "lora", "CAUSAL_LM",    "gemma_lora"),
        ("t5-base",           "lora", "SEQ_2_SEQ_LM", "t5_lora"),
    ]

    results = []
    for backbone, stg, ttype, tag in exp_cfgs:
        print(f"\n训练：{backbone} + {stg}")
        f1, acc = train_one_experiment(backbone, stg, ttype, tag)
        results.append({"Backbone": backbone, "Finetune": stg, "BIO F1": f1, "SentACC": acc})

    pd.DataFrame(results).to_csv("./checkpoints/final_results.csv", index=False)
    print("\n实验对比：\n", pd.DataFrame(results))

