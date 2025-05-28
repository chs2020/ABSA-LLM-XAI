
import glob
import warnings
import torch
import pandas as pd
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import AutoTokenizer, T5ForConditionalGeneration
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


# ---------------- Global Parameters ----------------
CSV_GLOB       = r"E:/ABSA-LLM-XAI/dataset/SemEval2014/**/*.csv"
TEXT_COL       = "Sentence"
ASPECT_COL     = "Aspect Term"
LABEL_COL      = "polarity"
MAX_LEN        = 128
BATCH_SIZE     = 8
EPOCHS         = 15
LR             = 2e-4
BACKBONE       = "t5-base"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
ENABLE_XAI_VIS = True
warnings.filterwarnings("ignore")

# ---------------- 1. Data Loading ----------------
class DataLoaderPreprocessor:
    def load_files(self, paths):
        dfs = []
        for p in paths:
            try:
                df = pd.read_csv(p)
                if not all(c in df.columns for c in [TEXT_COL, ASPECT_COL, LABEL_COL]):
                    continue
                dfs.append(df[[TEXT_COL, ASPECT_COL, LABEL_COL]])
            except:
                continue
        if not dfs:
            raise FileNotFoundError("No valid CSV files.")
        return pd.concat(dfs, ignore_index=True)


def load_records():
    df = DataLoaderPreprocessor().load_files(glob.glob(CSV_GLOB, recursive=True))
    df["from"] = df.apply(lambda r: r[TEXT_COL].lower().find(r[ASPECT_COL].lower()), axis=1)
    df["to"]   = df["from"] + df[ASPECT_COL].str.len()
    rec = []
    for i, r in df.iterrows():
        if r["from"] >= 0:
            rec.append({
                "Sentence": r[TEXT_COL],
                "from_": int(r["from"]),
                "to": int(r["to"]),
                "polarity": r[LABEL_COL].lower()
            })
    return rec

records   = load_records()
label_set = sorted({r["polarity"] for r in records})
pol2id    = {lab: idx for idx, lab in enumerate(label_set)}
NUM_SENT  = len(pol2id)

# ---------------- 2. Dataset ----------------
tok = AutoTokenizer.from_pretrained(BACKBONE)
class ABSADataset(Dataset):
    def __init__(self, recs): self.data = [self.encode(r) for r in recs]
    def encode(self, r):
        enc = tok(r["Sentence"], return_offsets_mapping=True,
                  truncation=True, padding="max_length", max_length=MAX_LEN)
        offsets = enc.pop("offset_mapping")
        bio = []
        for s, e in offsets:
            if s == e == 0:
                bio.append(-100)
            elif s >= r["from_"] and e <= r["to"]:
                bio.append(1 if s == r["from_"] else 2)
            else:
                bio.append(0)
        enc["labels_bio"]  = bio
        enc["labels_sent"] = pol2id[r["polarity"]]
        return enc
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return {k: torch.tensor(v) for k, v in self.data[i].items()}

dataset = ABSADataset(records)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---------------- 3. Model & LoRA ----------------
base      = T5ForConditionalGeneration.from_pretrained(BACKBONE)
lora_cfg  = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=8, lora_alpha=16, lora_dropout=0.1)
base      = get_peft_model(base, lora_cfg).to(DEVICE)
class ABSAHead(nn.Module):
    def __init__(self, dim, num_cls):
        super().__init__()
        self.bio = nn.Linear(dim, 3)
        self.sent = nn.Linear(dim, num_cls)
        self.lb = nn.CrossEntropyLoss(ignore_index=-100)
        self.ls = nn.CrossEntropyLoss()
    def forward(self, hid, mask, yb=None, ys=None):
        pooled = (hid * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        bio_logits = self.bio(hid)
        sent_logits = self.sent(pooled)
        out = {"bio_logits": bio_logits, "sent_logits": sent_logits}
        if yb is not None and ys is not None:
            loss_b = self.lb(bio_logits.view(-1, 3), yb.view(-1))
            loss_s = self.ls(sent_logits, ys)
            out["loss"] = loss_b + loss_s
        return out
head = ABSAHead(base.config.d_model, NUM_SENT).to(DEVICE)
optim = torch.optim.Adam(list(base.parameters()) + list(head.parameters()), lr=LR)

# ---------------- 4. Training ----------------
h_losses, h_f1s, h_accs = [], [], []
true_s_all, pred_s_all = [], []
true_b_all, pred_b_all = [], []
for _ in trange(EPOCHS, desc="Epoch"):
    for batch in tqdm(loader, leave=False):
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        yb = batch["labels_bio"].to(DEVICE)
        ys = batch["labels_sent"].to(DEVICE)
        enc = base.encoder(input_ids=ids, attention_mask=mask, return_dict=True)
        out = head(enc.last_hidden_state, mask, yb, ys)
        optim.zero_grad(); out["loss"].backward(); optim.step()
        pred_s_all.extend(out["sent_logits"].argmax(-1).cpu().numpy())
        true_s_all.extend(ys.cpu().numpy())
        pb = out["bio_logits"].argmax(-1).view(-1).cpu().numpy()
        tb = yb.view(-1).cpu().numpy()
        mask_keep = tb != -100
        pred_b_all.extend(pb[mask_keep]); true_b_all.extend(tb[mask_keep])
    h_losses.append(out["loss"].item())
    h_f1s.append(f1_score(true_b_all, pred_b_all, average='macro'))
    h_accs.append(accuracy_score(true_s_all, pred_s_all))

# ---------------- 5. Evaluation Report Visualization ----------------
def plot_classification_report(y_true, y_pred, labels, title):
    report = classification_report(y_true, y_pred, labels=labels,
                                   target_names=labels, output_dict=True)
    df = pd.DataFrame(report).T.iloc[:-1, :3]  # drop accuracy row
    df.plot(kind='bar', figsize=(8, 4))
    plt.title(title)
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

plot_classification_report(true_s_all, pred_s_all, labels=label_set,
                           title='Sentiment Classification Metrics')
plot_classification_report(true_b_all, pred_b_all,
                           labels=['O', 'B', 'I'],
                           title='BIO Tag Classification Metrics')

# ---------------- 6. Training Curves ----------------
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1); plt.plot(h_losses); plt.title('Loss')
plt.subplot(1, 3, 2); plt.plot(h_f1s);    plt.title('BIO F1')
plt.subplot(1, 3, 3); plt.plot(h_accs);   plt.title('Sent ACC')
plt.tight_layout(); plt.show()

# ---------------- 7. XAI Module ----------------
try:
    from lime.lime_text import LimeTextExplainer
except ImportError:
    LimeTextExplainer = None
    print("WARNING: lime module not installed. LIME visualization skipped.", flush=True)

class WrappedSentModel(nn.Module):
    def __init__(self, base, head):
        super().__init__()
        self.base = base.encoder
        self.head = head
    def forward(self, input_ids, attention_mask):
        enc = self.base(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = (enc.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        logits = self.head.sent(pooled)
        return torch.softmax(logits, dim=-1)
wrapped = WrappedSentModel(base, head).to(DEVICE)

def visualize_lime(text, cls=0, num_features=10):
    if LimeTextExplainer is None:
        print("Skipping LIME visualization: module not found.")
        return
    explainer = LimeTextExplainer(class_names=label_set)
    def predict_proba(texts):
        toks = tok(texts, return_tensors='pt', truncation=True, padding=True, max_length=MAX_LEN).to(DEVICE)
        with torch.no_grad():
            out = wrapped(toks['input_ids'], attention_mask=toks['attention_mask'])
        return out.cpu().numpy()
    exp = explainer.explain_instance(text, predict_proba, num_features=num_features, labels=[cls])
    fig = exp.as_pyplot_figure(label=cls)
    plt.title(f"LIME Explanation (class={label_set[cls]})")
    plt.show()

def visualize_saliency(text, cls=0):
    toked = tok(text, return_tensors='pt', truncation=True, max_length=MAX_LEN).to(DEVICE)
    input_ids = toked['input_ids']
    attn_mask = toked['attention_mask']
    embeds = base.encoder.embed_tokens(input_ids)
    embeds.requires_grad_()
    enc = base.encoder(inputs_embeds=embeds, attention_mask=attn_mask, return_dict=True)
    pooled = (enc.last_hidden_state * attn_mask.unsqueeze(-1)).sum(1) / attn_mask.sum(1, keepdim=True)
    logits = head.sent(pooled)
    probs = torch.softmax(logits, dim=-1)
    score = probs[0, cls]
    base.zero_grad(); head.zero_grad()
    score.backward()
    grads = embeds.grad[0].abs().sum(dim=-1).cpu().numpy()
    tokens = tok.convert_ids_to_tokens(input_ids[0])
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(tokens)), grads)
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.title(f"Saliency for class={label_set[cls]}")
    plt.tight_layout()
    plt.show()

def visualize_attention(text):
    toked = tok(text, return_tensors='pt', truncation=True, max_length=MAX_LEN).to(DEVICE)
    with torch.no_grad():
        out = base.encoder(**toked, output_attentions=True, return_dict=True)
        attn = out.attentions[-1][0]
    tokens = tok.convert_ids_to_tokens(toked['input_ids'][0])
    matrix = attn.mean(0).cpu().numpy()
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, interpolation='nearest')
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)
    plt.title("Encoder Attention")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

if ENABLE_XAI_VIS:
    print("[ XAI Sample Analysis ]")
    test_sentence = "The camera is amazing but the battery is bad"
    visualize_lime(test_sentence, cls=0)
    visualize_saliency(test_sentence, cls=0)
    visualize_attention(test_sentence)

macro_f1 = f1_score(true_s_all, pred_s_all, average='macro')
plt.figure(figsize=(4, 4))
plt.bar(['Macro F1 (Sentiment)'], [macro_f1])
plt.ylim(0, 1)
plt.ylabel('Score')
plt.title('Macro F1 Score for Classification')
plt.grid(axis='y')
plt.show()

print("=== Sentiment Classification Report ===")
report = classification_report(true_s_all, pred_s_all, digits=4)
print(report)
print(f"\nMacro F1 Score (Sentiment): {macro_f1:.4f}")