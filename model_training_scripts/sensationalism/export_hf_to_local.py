import os
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

CHECKPOINT = "Sami92/XLM-R-Large-Sensationalism-Classifier"
OUT_DIR = os.path.join("models", "sensationalism", "hf_model")

os.makedirs(OUT_DIR, exist_ok=True)

print(f"Downloading {CHECKPOINT} ...")
tok = AutoTokenizer.from_pretrained(CHECKPOINT)
cfg = AutoConfig.from_pretrained(CHECKPOINT)
mdl = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT)

print(f"Saving to {OUT_DIR} ...")
tok.save_pretrained(OUT_DIR)
cfg.save_pretrained(OUT_DIR)
mdl.save_pretrained(OUT_DIR)

print("Done. Files written to:", OUT_DIR)
