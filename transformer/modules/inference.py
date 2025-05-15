from __future__ import annotations
from typing import List
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ModelPredictor:
    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        top_k: int = 1,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()
        self.device    = device
        self.top_k     = top_k
        self.id2label  = self.model.config.id2label

    def predict_single(self, text: str) -> str | List[str]:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: List[str]) -> List[str] | List[List[str]]:
        enc = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=512,
        ).to(self.device)

        with torch.inference_mode():
            logits = self.model(**enc).logits   # [B, C]

        topk_ids = logits.topk(self.top_k, dim=1).indices.cpu().tolist()
        if self.top_k == 1:
            return [self.id2label[i[0]] for i in topk_ids]
        else:
            return [[self.id2label[j] for j in row] for row in topk_ids]
