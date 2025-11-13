from typing import Dict
import os
import re

import numpy as np
import torch
from transformers import AutoConfig, pipeline

from models.factuality_factor import FactualityFactor



class Sensationalism(FactualityFactor):
    def __init__(self):
        super().__init__(
            "Sensationalism",
            "Assesses how exaggerated or sensational the article's framing is",
        )
        #download model from hugging face
        self.model_id = "Sami92/XLM-R-Large-Sensationalism-Classifier"

        # Choose device: GPU if available, else CPU
        device = 0 if torch.cuda.is_available() else -1

        # Load config to get id2label mapping
        cfg = AutoConfig.from_pretrained(self.model_id)
        id2label = getattr(cfg, "id2label", {0: "LABEL_0", 1: "LABEL_1"})
        # Normalize keys to int
        self.id2label = {int(k): v for k, v in id2label.items()}

        # Assume label id 1 is the "sensational" class if present;
        # otherwise take the highest id as the positive class.
        self.pos_id = 1 if 1 in self.id2label else max(self.id2label)

        # Hugging Face pipeline; will download the model on first use, then cache.
        self.clf = pipeline(
            "text-classification",
            model=self.model_id,
            tokenizer=self.model_id,
            truncation=True,
            padding=True,
            max_length=512,
            top_k=None, 
            device=device,
        )


    #helpers to score the text
    def _sentences(self, text: str):
        """
        Lightweight sentence splitter (no NLTK dependency).
        """
        text = (text or "").strip()
        if not text:
            return []
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [s for s in parts if len(s.split()) >= 2]


    def _prob_pos_from_out(self, out):
        """
        Convert Hugging Face pipeline output (list of dicts) into
        the probability of the positive (sensational) class.
        out example: [{'label': 'LABEL_0', 'score': ...}, {'label': 'LABEL_1', 'score': ...}]
        """
        by_id = {}
        for d in out:
            lbl = d["label"]
            if lbl.startswith("LABEL_"):
                k = int(lbl.split("_")[-1])
            else:
                k = next((i for i, name in self.id2label.items() if name == lbl), None)
                if k is None:
                    continue
            by_id[k] = float(d["score"])
        if not by_id:
            return 0.0
        return float(by_id.get(self.pos_id, by_id[max(by_id)]))


    def _score_text(self, text: str, weight_headline: float = 0.6):
        """
        Given a raw text string, treat the first sentence as a 'headline' and
        the rest as body sentences, then compute:
          - headline_prob
          - body_mean
          - final_prob (weighted blend)
        """
        sents = self._sentences(text)
        if not sents:
            return 0.0, 0.0, 0.0

        headline = sents[0]
        body_sents = sents[1:]

        head_prob = self._prob_pos_from_out(self.clf([headline])[0])

        if body_sents:
            outs = self.clf(body_sents)
            body_probs = [self._prob_pos_from_out(o) for o in outs]
            body_mean = float(np.mean(body_probs))
        else:
            body_mean = head_prob

        final_prob = float(weight_headline * head_prob + (1.0 - weight_headline) * body_mean)
        return head_prob, body_mean, final_prob


    def probability(self, text: str) -> Dict[str, float]:
        if text is None:
            text = ""
        text = str(text)

        _, _, final_prob = self._score_text(text)
        return {"sensationalism": float(final_prob)}



if __name__ == "__main__":
    # Run sensationalism specific checks here
    factor = Sensationalism()

    boring = "Weather expected to clear up soon. Rain that has been ongoing for the past week should go away soon."
    spicy = "You WON'T BELIEVE what scientists found! A new paper reports a modest 2% increase in signal."

    print("Boring:", factor.probability(boring))
    print("Spicy:", factor.probability(spicy))