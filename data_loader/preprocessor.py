import pandas as pd
import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from typing import List, Tuple

class DataLoaderPreprocessor:
    def __init__(self, text_col="Sentence", aspect_col="Aspect Term", label_col="polarity"):
        self.text_col = text_col
        self.aspect_col = aspect_col
        self.label_col = label_col
        self.stop_words = set(ENGLISH_STOP_WORDS)

    def load_files(self, file_paths: List[str]) -> pd.DataFrame:
        all_dfs = []
        total_count = 0
        print(" loading...\n")
        for path in file_paths:
            try:
                df = pd.read_csv(path)
                if not all(col in df.columns for col in [self.text_col, self.aspect_col, self.label_col]):
                    print(f"文件缺少必要列: {path}")
                    continue
                row_count = len(df)
                total_count += row_count
                print(f"{path.split('/')[-1]}: {row_count} 条数据")
                all_dfs.append(df[[self.text_col, self.aspect_col, self.label_col]])
            except Exception as e:
                print(f"无法读取 {path}: {e}")
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"\n数据集加载完成，共读取 {total_count} 条记录。\n")
        return combined_df

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        tokens = text.split()
        tokens = [word for word in tokens if word not in self.stop_words]
        return " ".join(tokens)

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df["processed_text"] = df.apply(
            lambda row: self.clean_text(f"{row[self.aspect_col]} {row[self.text_col]}"), axis=1
        )
        return df

    def generate_bio_tags(self, sentence: str, aspect_terms: List[Tuple[int, int]]) -> List[Tuple[str, str]]:
        tokens = sentence.split()
        char_offsets = self._get_token_char_offsets(sentence, tokens)
        tags = ["O"] * len(tokens)
        for start, end in aspect_terms:
            for i, (tok_start, tok_end) in enumerate(char_offsets):
                if tok_start >= end:
                    break
                if tok_end <= start:
                    continue
                if start <= tok_start < end:
                    tags[i] = "B" if tags[i] == "O" else "I"
        return list(zip(tokens, tags))

    def _get_token_char_offsets(self, sentence: str, tokens: List[str]) -> List[Tuple[int, int]]:
        offsets = []
        idx = 0
        for token in tokens:
            while idx < len(sentence) and sentence[idx].isspace():
                idx += 1
            start = idx
            for char in token:
                if idx < len(sentence) and sentence[idx] == char:
                    idx += 1
            end = idx
            offsets.append((start, end))
        return offsets

    def convert_to_bio_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        all_data = []
        for _, row in df.iterrows():
            sentence = row[self.text_col]
            aspect = row[self.aspect_col]
            start = sentence.lower().find(aspect.lower())
            if start == -1:
                continue
            end = start + len(aspect)
            bio_pairs = self.generate_bio_tags(sentence, [(start, end)])
            for token, tag in bio_pairs:
                all_data.append({"token": token, "label": tag, "sentence": sentence})
        return pd.DataFrame(all_data)

    def load_and_preprocess(self, file_paths: List[str]) -> pd.DataFrame:
        df = self.load_files(file_paths)
        return self.preprocess_dataframe(df)

    def generate_bio_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.convert_to_bio_dataframe(df)

    def split_train_val(self, df: pd.DataFrame, test_size: float = 0.1, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        用于 ASC 训练集划分：按句子行划分，不影响 ATE 的 BIO 格式
        """
        return train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[self.label_col])