"""Test MiniCPM-V vision-language model"""

import torch
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from PIL import Image
from pathlib import Path
import time

print("="*70)
print("MiniCPM-V Test")
print("="*70)

# Check CUDA
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

DATA_DIR = "data/Fetal Ultrasound"

# Try different MiniCPM-V models
model_ids = [
    "openbmb/MiniCPM-V-2",
    "openbmb/MiniCPM-V-2_5",
    "openbmb/MiniCPM-Llama3-V-2_5",
]

for MODEL_ID in model_ids:
    print(f"\n1. Trying {MODEL_ID}...")
    try:
        start_time = time.time()

        # Load model and tokenizer
        model = AutoModel.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

        if torch.cuda.is_available():
            model = model.cuda()

        model.eval()

        load_time = time.time() - start_time
        param_count = sum(p.numel() for p in model.parameters())
        print(f"   [OK] Model loaded: {param_count / 1e9:.2f}B params")
        print(f"   Load time: {load_time:.1f}s")

        if torch.cuda.is_available():
            memory = torch.cuda.max_memory_allocated() / 1e9
            print(f"   GPU Memory: {memory:.2f} GB")

        print("\n2. Testing on fetal ultrasound images...")

        # Get test images
        test_images = []
        categories = ["Abodomen", "Aorta", "Brain", "Femur", "Heart"]
        for cat in categories:
            cat_dir = Path(DATA_DIR) / cat
            if cat_dir.exists():
                images = list(cat_dir.glob("*.png"))[:2]
                test_images.extend(images)

        if not test_images:
            print("   No test images found!")
            continue

        correct_anatomy = 0
        fetal_context = 0

        for i, img_path in enumerate(test_images[:5]):  # Test 5 images
            print(f"\n   Image {i+1}/5: {img_path.name}")
            print(f"   Category: {img_path.parent.name}")

            image = Image.open(img_path).convert('RGB')

            # Test with medical VQA
            question = "What anatomical structures can you see in this ultrasound image?"

            # Format input based on model
            if "2_5" in MODEL_ID:
                # MiniCPM-V-2.5 format
                msgs = [
                    {'role': 'user', 'content': question}
                ]

                # Generate
                start = time.time()
                res = model.chat(
                    image=image,
                    msgs=msgs,
                    tokenizer=tokenizer,
                    sampling=True,
                    temperature=0.7,
                    stream=False
                )
                gen_time = time.time() - start

            else:
                # MiniCPM-V-2 format
                start = time.time()
                res, context, _ = model.chat(
                    image=image,
                    question=question,
                    context=None,
                    tokenizer=tokenizer,
                    do_sample=False
                )
                gen_time = time.time() - start

            print(f"   Response: {res[:300]}...")
            print(f"   Time: {gen_time:.2f}s")

            # Check for medical understanding
            response_lower = res.lower()
            if any(term in response_lower for term in ['fetal', 'fetus', 'baby', 'pregnancy', 'ultrasound']):
                fetal_context += 1

            category_lower = img_path.parent.name.lower()
            if category_lower in response_lower or \
               (category_lower == "abodomen" and "abdomen" in response_lower):
                correct_anatomy += 1

        # Calculate accuracy
        print(f"\n3. Accuracy Assessment:")
        print(f"   Fetal context recognition: {fetal_context}/5 ({fetal_context/5*100:.1f}%)")
        print(f"   Anatomy identification: {correct_anatomy}/5 ({correct_anatomy/5*100:.1f}%)")

        # Test more VQA capabilities
        print("\n4. Testing additional VQA capabilities...")

        if test_images:
            img_path = test_images[0]
            image = Image.open(img_path).convert('RGB')

            test_prompts = [
                "Is this a normal or abnormal ultrasound?",
                "What organ system is visible?",
                "Describe any potential abnormalities you see.",
            ]

            for prompt in test_prompts:
                print(f"\n   Q: {prompt}")

                if "2_5" in MODEL_ID:
                    msgs = [{'role': 'user', 'content': prompt}]
                    res = model.chat(
                        image=image,
                        msgs=msgs,
                        tokenizer=tokenizer,
                        sampling=True,
                        temperature=0.7,
                        stream=False
                    )
                else:
                    res, _, _ = model.chat(
                        image=image,
                        question=prompt,
                        context=None,
                        tokenizer=tokenizer,
                        do_sample=False
                    )

                print(f"   A: {res[:200]}...")

        print("\n" + "="*70)
        print("SUCCESS! MiniCPM-V works!")
        print("="*70)

        print(f"\nModel Summary:")
        print(f"  - Model: {MODEL_ID}")
        print(f"  - Size: {param_count / 1e9:.2f}B params")
        print(f"  - Memory: {memory:.2f} GB")
        print(f"  - Fetal context: {'YES' if fetal_context >= 3 else 'LIMITED'}")
        print(f"  - Medical accuracy: {'HIGH' if correct_anatomy >= 3 else 'MODERATE' if correct_anatomy >= 2 else 'LOW'}")
        print(f"  - Better than BLIP-2? {'POSSIBLY' if fetal_context >= 4 and correct_anatomy >= 3 else 'NEEDS TESTING'}")

        break  # Success, don't try other models

    except Exception as e:
        print(f"   [ERROR] {type(e).__name__}: {str(e)[:200]}")
        continue

else:
    print("\nNo MiniCPM-V models could be loaded successfully")

print("\n" + "="*70)