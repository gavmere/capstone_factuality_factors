# toxicity_model.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Dict, Tuple

class ToxicityDetector:
    """
    RoBERTa-based multi-class toxicity detector.
    Labels:
        0 -> friendly
        1 -> neutral
        2 -> rude
        3 -> toxic
        4 -> super_toxic
    """

    _instance = None  # Singleton cache

    LABELS = [
        "friendly",
        "neutral",
        "rude",
        "toxic",
        "super_toxic"
    ]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

            model_name = "roberta-base"  # or path to your fine-tuned model
            cls._instance.tokenizer = AutoTokenizer.from_pretrained(model_name)
            cls._instance.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(cls.LABELS)
            )

            cls._instance.model.eval()
            cls._instance.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            cls._instance.model.to(cls._instance.device)

        return cls._instance

    @torch.no_grad()
    def score(self, text: str) -> Tuple[Dict[str, float], str]:
        """
        Returns:
            - Dict[str, float]: probability per toxicity class
            - str: predicted label
        """

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).squeeze()

        scores = {
            label: round(probs[idx].item(), 4)
            for idx, label in enumerate(self.LABELS)
        }

        predicted_label = self.LABELS[probs.argmax().item()]

        return scores, predicted_label
