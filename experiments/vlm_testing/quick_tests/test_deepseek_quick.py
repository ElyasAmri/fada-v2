"""Quick test of DeepSeek-VL-1.3B on FADA dataset"""

import torch
from pathlib import Path
import time
import sys

print("="*70)
print("DeepSeek-VL-1.3B Quick Test on FADA Dataset")
print("="*70)

# Configuration
MODEL_ID = "deepseek-ai/deepseek-vl-1.3b-chat"
DATA_DIR = "data/Fetal Ultrasound"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n1. Loading DeepSeek-VL-1.3B model...")
print(f"   Model: {MODEL_ID}")
print(f"   Device: {device}")

start = time.time()

try:
    # Import DeepSeek-VL components
    try:
        from transformers import AutoModelForCausalLM
        from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
        from deepseek_vl.utils.io import load_pil_images
    except ImportError as e:
        print(f"\n[ERROR] Missing DeepSeek-VL package")
        print(f"   {e}")
        print("\nInstallation required:")
        print("   git clone https://github.com/deepseek-ai/DeepSeek-VL")
        print("   cd DeepSeek-VL")
        print("   pip install -e .")
        print("\nEstimated setup time: 10-15 minutes")
        sys.exit(1)

    # Load processor and model
    vl_chat_processor = VLChatProcessor.from_pretrained(MODEL_ID)
    vl_gpt = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    vl_gpt = vl_gpt.cuda() if torch.cuda.is_available() else vl_gpt
    vl_gpt.eval()

    load_time = time.time() - start
    print(f"   [OK] Model loaded in {load_time:.1f} seconds")

    # Get model size
    param_count = sum(p.numel() for p in vl_gpt.parameters())
    print(f"   Parameters: {param_count / 1e9:.2f}B")

    if torch.cuda.is_available():
        memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"   GPU Memory: {memory:.2f} GB")

    print("\n2. Testing VQA on sample images...")

    # Collect test images (3 per category, first 5 categories)
    test_images = []
    category_dirs = [d for d in Path(DATA_DIR).iterdir() if d.is_dir()]

    for cat_dir in category_dirs[:5]:
        cat_name = cat_dir.name
        images = list(cat_dir.glob("*.png")) + list(cat_dir.glob("*.jpg"))

        if not images:
            continue

        for img_path in images[:3]:
            test_images.append({
                'path': img_path,
                'category': cat_name
            })

    print(f"   Testing {len(test_images)} images...")

    # Test questions for ultrasound images
    questions = [
        "What anatomical structures can you see in this ultrasound image?",
        "Describe this fetal ultrasound image.",
        "What part of the fetus is shown in this image?"
    ]

    results = []

    for i, item in enumerate(test_images[:5]):  # Test first 5 for quick test
        img_path = item['path']
        true_cat = item['category']

        print(f"\n   Image {i+1}/{5}: {img_path.name}")
        print(f"   True category: {true_cat}")

        # Ask question
        question = questions[0]  # Use first question

        # Prepare conversation
        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder>{question}",
                "images": [str(img_path)]
            },
            {
                "role": "Assistant",
                "content": ""
            }
        ]

        # Load images
        pil_images = load_pil_images(conversation)

        # Prepare inputs
        prepare_inputs = vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(vl_gpt.device)

        # Generate response
        start_gen = time.time()
        with torch.no_grad():
            inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
            outputs = vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=vl_chat_processor.tokenizer.eos_token_id,
                bos_token_id=vl_chat_processor.tokenizer.bos_token_id,
                eos_token_id=vl_chat_processor.tokenizer.eos_token_id,
                max_new_tokens=100,
                do_sample=False,
                use_cache=True
            )
        gen_time = time.time() - start_gen

        # Decode response
        response = vl_chat_processor.tokenizer.decode(
            outputs[0].cpu().tolist(),
            skip_special_tokens=True
        )

        print(f"   Question: {question}")
        print(f"   Response: {response}")
        print(f"   Generation time: {gen_time:.2f}s")

        results.append({
            'image': img_path.name,
            'category': true_cat,
            'question': question,
            'response': response,
            'time': gen_time
        })

    # Summary
    print("\n" + "="*70)
    print("Quick Test Summary")
    print("="*70)
    print(f"\nModel: DeepSeek-VL-1.3B-chat")
    print(f"Parameters: {param_count / 1e9:.2f}B")
    print(f"Images tested: {len(results)}")

    avg_time = sum(r['time'] for r in results) / len(results)
    print(f"Average generation time: {avg_time:.2f}s")

    if torch.cuda.is_available():
        print(f"GPU Memory used: {memory:.2f} GB")

    print("\n" + "="*70)
    print("Comparison with Other Models")
    print("="*70)
    print(f"\nDeepSeek-VL-1.3B: {param_count / 1e9:.2f}B params, ~{memory:.1f}GB memory")
    print(f"SmolVLM-500M:     0.51B params, ~1.0GB memory")
    print(f"BLIP-2:           3.4B params, ~4.2GB memory")

    print("\nNote: DeepSeek-VL trained on 400B vision-language tokens")
    print("Check response quality for scientific/medical reasoning capabilities")

    # Save results
    import json
    with open("deepseek_quick_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: deepseek_quick_test_results.json")

except Exception as e:
    print(f"\n[ERROR] {type(e).__name__}: {e}")
    print("\nPossible issues:")
    print("  - DeepSeek-VL package not installed")
    print("  - Incompatible transformers version")
    print("  - Insufficient GPU memory")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
