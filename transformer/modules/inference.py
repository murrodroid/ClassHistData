# inference.py
from __future__ import annotations
from pathlib import Path
from typing import List

import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
)

# -----------------------------------------------------------
class ModelPredictor:
    def __init__(
        self,
        run_dir: str | Path,         # e.g. runs/Bio_ClinicalBERT_20250515_215342
        *,
        device: str | torch.device | None = None,
        top_k: int = 1,
    ):
        run_dir = Path(run_dir)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # --- tokenizer ----------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(run_dir.parent)  
        # ^ points to the *base* model folder so your special tokens match.

        # --- model body & label maps --------------------------------------
        cfg = AutoConfig.from_pretrained(
            run_dir.parent,                       # same base model
            id2label=run_dir.joinpath("config.json").read_text() and None
        )
        self.model = AutoModelForSequenceClassification.from_config(cfg)

        state = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu")
        # unwrap if you saved {"model_state": ...}
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        self.model.load_state_dict(state)
        self.model.to(device).eval()

        self.id2label = cfg.id2label
        self.device   = device
        self.top_k    = top_k

    # -----------------------------------------------------------
    @torch.inference_mode()
    def _forward(self, texts: List[str]):
        enc = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        return self.model(**enc).logits       # [B, C]

    def predict_batch(self, texts: List[str]):
        logits = self._forward(texts)
        topk = logits.topk(self.top_k, dim=1).indices.cpu().tolist()
        if self.top_k == 1:
            return [self.id2label[i[0]] for i in topk]
        return [[self.id2label[j] for j in row] for row in topk]

    def predict_single(self, text: str):
        return self.predict_batch([text])[0]
