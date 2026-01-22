from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class ToxicityDetector:
    # Model that supports multi-class toxicity classification
    MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
    TOXICITY_LABELS = ["friendly", "neutral", "rude", "toxic", "super_toxic"]
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
    
    def score(self, text: str):
        """
        Detects toxicity level in text.
        Returns probabilities for each toxicity level:
        - friendly: kind, respectful text
        - neutral: objective, non-toxic text
        - rude: impolite but not severely toxic
        - toxic: toxic language present
        - super_toxic: extremely toxic or hateful
        """
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1).squeeze()
        
        # Create probability dict for each class
        probabilities = {}
        for i, label in enumerate(self.TOXICITY_LABELS):
            probabilities[label] = float(probs[i]) if i < len(probs) else 0.0
        
        # Get the class with highest probability
        max_idx = torch.argmax(probs).item()
        predicted_label = self.TOXICITY_LABELS[min(max_idx, len(self.TOXICITY_LABELS) - 1)]
        
        return {
            "probabilities": probabilities,
            "predicted_level": predicted_label,
            "confidence": float(probs[max_idx])
        }
