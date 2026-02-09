# toxicity_model.py
from detoxify import Detoxify
from typing import Dict, Tuple
import torch
import os

class ToxicityDetector:
    """
    Detoxify-based toxicity detector using unitary/toxic-bert model.
    
    Detects multiple toxicity categories:
        - toxicity: overall toxicity score
        - severe_toxicity: severe toxic content
        - obscene: obscene language
        - threat: threatening content
        - insult: insulting content
        - identity_attack: attacks on identity groups
    
    More info: https://github.com/unitaryai/detoxify
    """

    _instance = None  # Singleton cache
    _initialized = False  # Track initialization status

    LABELS = [
        "toxicity",
        "severe_toxicity",
        "obscene",
        "threat",
        "insult",
        "identity_attack"
    ]

    def __new__(cls):
        if cls._instance is None:
            print("[ToxicityDetector] Creating new instance...")
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Only initialize once, even if __init__ is called multiple times
        if not ToxicityDetector._initialized:
            import sys
            print("[ToxicityDetector] Initializing model (this may take a moment on first run)...")
            print("[ToxicityDetector] First-time download: ~500MB model will be cached to ~/.cache/huggingface/")
            
            # Set device - use MPS for Apple Silicon, CUDA for NVIDIA, otherwise CPU
            if torch.backends.mps.is_available():
                self.device = 'mps'
                print(f"[ToxicityDetector] Using Apple Silicon GPU (MPS)")
            elif torch.cuda.is_available():
                self.device = 'cuda'
                print(f"[ToxicityDetector] Using NVIDIA GPU (CUDA)")
            else:
                self.device = 'cpu'
                print(f"[ToxicityDetector] Using CPU")
            
            try:
                # Load model with explicit device
                # The model will be cached in ~/.cache/huggingface/ after first download
                print("[ToxicityDetector] Loading model from cache or downloading...")
                sys.stdout.flush()  # Force output to appear immediately
                
                self.model = Detoxify('original', device=self.device)
                
                ToxicityDetector._initialized = True
                print("[ToxicityDetector] ✓ Model loaded successfully!")
            except Exception as e:
                print(f"[ToxicityDetector] ✗ Error loading model: {e}")
                print("[ToxicityDetector] Troubleshooting:")
                print("  1. Check internet connection (needed for first download)")
                print("  2. Ensure ~1GB free disk space")
                print("  3. Try: python models/toxicity/download_model.py")
                raise

    def score(self, text: str) -> Tuple[Dict[str, float], str]:
        """
        Returns:
            - Dict[str, float]: score per toxicity category (0-1 range)
            - str: predicted label (category with highest score)
        """
        
        # Get predictions from detoxify
        results = self.model.predict(text)
        
        # Round scores to 4 decimal places
        scores = {
            label: round(float(results[label]), 4)
            for label in self.LABELS
        }
        
        # Get the label with highest score
        predicted_label = max(scores, key=scores.get)
        
        return scores, predicted_label
