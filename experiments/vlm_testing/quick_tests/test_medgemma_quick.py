"""Quick test of MedGemma model loading and inference"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

print("="*70)
print("MedGemma 4B Quick Test")
print("="*70)

model_id = "google/medgemma-4b-it"

print(f"\nLoading model: {model_id}")
print("This may take a few minutes on first run (downloading ~8GB)...")

start = time.time()

try:
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("   [OK] Tokenizer loaded")

    # Load model
    print("\n2. Loading model (8-bit quantization)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=True
    )
    print("   [OK] Model loaded")

    load_time = time.time() - start
    print(f"\n   Loading time: {load_time:.1f} seconds")
    print(f"   Model parameters: {model.num_parameters() / 1e9:.1f}B")
    print(f"   Memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")

    # Test inference
    print("\n3. Testing inference...")
    test_prompt = "What structures are typically visible in a fetal ultrasound?"

    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            num_beams=3
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"\n   Prompt: {test_prompt}")
    print(f"\n   Response:")
    # Handle Unicode for Windows console
    try:
        print(f"   {response}")
    except UnicodeEncodeError:
        print(f"   {response.encode('ascii', 'ignore').decode('ascii')}")
        print(f"\n   [Note: Some characters removed for Windows console compatibility]")

    print("\n" + "="*70)
    print("[SUCCESS] MedGemma is working!")
    print("="*70)
    print("\nModel is ready for VQA training on fetal ultrasound images.")
    print("\nNext steps:")
    print("  1. Create training notebook: notebooks/blip2_training/train_medgemma_vqa.ipynb")
    print("  2. Adapt data loading and training loop")
    print("  3. Train on Non_standard_NT first (487 images)")
    print("  4. Compare performance with BLIP-2")

except Exception as e:
    print(f"\n[ERROR] {type(e).__name__}: {e}")
    print("\nPossible issues:")
    print("  - Insufficient GPU memory (need ~8GB)")
    print("  - Missing dependencies")
    print("  - Model download failed")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
