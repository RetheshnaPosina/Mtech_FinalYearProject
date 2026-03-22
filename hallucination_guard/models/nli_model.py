"""DeBERTa-v3-base NLI wrapper — entailment / contradiction / neutral."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class NLIResult:
    entailment: float
    contradiction: float
    neutral: float

    @property
    def label(self) -> str:
        d = {
            "entailment": self.entailment,
            "contradiction": self.contradiction,
            "neutral": self.neutral,
        }
        return max(d, key=d.get)


class NLIModel:
    """Wraps a HuggingFace CrossEncoder-style NLI model for entailment scoring."""

    def __init__(self, model_id: str, cache_dir: str) -> None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        AutoTokenizer = AutoTokenizer  # noqa: F841 — imported for clarity
        AutoModelForSequenceClassification = AutoModelForSequenceClassification

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, cache_dir=cache_dir
        )
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        # Build label lookup: {int(id): str(label).lower()}
        self.id2label = {
            int(k): str(v).lower()
            for k, v in self.model.config.id2label.items()
        }

    def predict(self, premise: str, hypothesis: str) -> NLIResult:
        """Score a single premise-hypothesis pair."""
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = logits.softmax(dim=-1).squeeze().tolist()

        if not isinstance(probs, list):
            probs = [probs]

        label_map = {
            self.id2label.get(str(i), self.id2label.get(i, str(i))): p
            for i, p in enumerate(probs)
        }

        return NLIResult(
            entailment=label_map.get("entailment", 0.33),
            contradiction=label_map.get("contradiction", 0.33),
            neutral=label_map.get("neutral", 0.34),
        )

    def predict_batch(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 8,
    ) -> List[NLIResult]:
        """Score a list of (premise, hypothesis) pairs in batches."""
        results: List[NLIResult] = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            premises = [p for p, _ in batch]
            hypotheses = [h for _, h in batch]

            inputs = self.tokenizer(
                premises,
                hypotheses,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs_batch = logits.softmax(dim=-1).tolist()

            for p in probs_batch:
                label_map = {
                    self.id2label.get(str(idx), self.id2label.get(idx, str(idx))): v
                    for idx, v in enumerate(p)
                }
                results.append(
                    NLIResult(
                        entailment=label_map.get("entailment", 0.33),
                        contradiction=label_map.get("contradiction", 0.33),
                        neutral=label_map.get("neutral", 0.34),
                    )
                )
        return results
