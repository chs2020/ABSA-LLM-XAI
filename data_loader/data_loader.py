from transformers import BertTokenizerFast
from torch.utils.data import Dataset
import torch

class ABSADataset(Dataset):
    def __init__(self, data, tokenizer_name='bert-base-uncased', max_length=128):
        self.data = data
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.polarity2id = {"negative": 0, "neutral": 1, "positive": 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        sentence = row['Sentence']
        aspect = row['Aspect_Term']
        aspect_start = sentence.lower().find(aspect.lower())
        aspect_end = aspect_start + len(aspect)


        encoding = self.tokenizer(
            sentence,
            return_offsets_mapping=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        offset_mapping = encoding['offset_mapping'].squeeze()

        # 构造BIO标签
        bio_labels = torch.zeros_like(input_ids)
        for i, (start, end) in enumerate(offset_mapping.tolist()):
            if start >= aspect_end:
                break
            if end <= aspect_start:
                continue
            if start == aspect_start:
                bio_labels[i] = 1  # B
            elif start > aspect_start:
                bio_labels[i] = 2  # I

        # 情感标签
        polarity = row['polarity'].lower()
        sentiment_label = torch.tensor(self.polarity2id.get(polarity, 1))  # 默认neutral

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'bio_labels': bio_labels,
            'sentiment_label': sentiment_label
        }


