#!/usr/bin/env python3
"""
Pre-download the Detoxify model to cache.

This script downloads the Detoxify 'original' model (~500MB) and caches it locally.
Run this once to avoid waiting for download during actual model usage.

Usage:
    python models/toxicity/download_model.py
    or
    cd models/toxicity && python download_model.py
"""

import time
import sys

print("=" * 60)
print("Detoxify Model Download Script")
print("=" * 60)
print("\nThis will download the 'original' Detoxify model (~500MB)")
print("to your HuggingFace cache directory: ~/.cache/huggingface/\n")

# Import detoxify
try:
    from detoxify import Detoxify
except ImportError:
    print("✗ Error: detoxify not installed")
    print("\nPlease install it first:")
    print("  pip install detoxify")
    sys.exit(1)

print("Starting download...")
start_time = time.time()

try:
    # Load the model - this will download it if not already cached
    model = Detoxify('original')
    
    download_time = time.time() - start_time
    print(f"\n✓ Success! Model downloaded and cached in {download_time:.2f} seconds")
    
    # Test the model
    print("\nTesting model...")
    test_result = model.predict("This is a test.")
    print(f"✓ Model is working correctly!")
    print(f"  Test toxicity score: {test_result['toxicity']:.4f}")
    
    print("\n" + "=" * 60)
    print("Setup complete! The model is now cached and ready to use.")
    print("Future loads will be much faster (2-5 seconds).")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Check your internet connection")
    print("2. Ensure you have ~1GB free disk space")
    print("3. Try running: pip install --upgrade detoxify transformers torch")
    exit(1)
