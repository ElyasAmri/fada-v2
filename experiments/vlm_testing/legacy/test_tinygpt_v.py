"""Test TinyGPT-V vision-language model"""

import torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image
from pathlib import Path
import time

print("="*70)
print("TinyGPT-V Test")
print("="*70)

# Check CUDA
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

DATA_DIR = "data/Fetal Ultrasound"
MODEL_ID = "Tyrannosaurus/TinyGPT-V"

try:
    print(f"\n1. Loading TinyGPT-V from {MODEL_ID}...")
    start_time = time.time()

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    load_time = time.time() - start_time
    print(f"   [OK] Model loaded in {load_time:.1f}s")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"   Size: {param_count / 1e9:.2f}B params")

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
        exit(1)

    correct_anatomy = 0
    fetal_context = 0
    detailed_responses = 0

    for i, img_path in enumerate(test_images[:6]):  # Test 6 images
        print(f"\n   Image {i+1}/6: {img_path.name}")
        print(f"   Category: {img_path.parent.name}")

        image = Image.open(img_path).convert('RGB')

        # TinyGPT-V query format
        query = "Describe this medical ultrasound image in detail. What anatomical structures are visible?"

        # Generate response
        start = time.time()

        # Try different inference methods
        if hasattr(model, 'chat'):
            response = model.chat(image, query)
        elif hasattr(model, 'generate'):
            # Standard generation
            inputs = tokenizer(query, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            print("   [ERROR] Model does not have chat() or generate() method")
            break

        gen_time = time.time() - start

        print(f"   Response: {response[:300]}...")
        print(f"   Time: {gen_time:.2f}s")

        # Check for medical understanding
        response_lower = response.lower()

        # Check fetal context
        if any(term in response_lower for term in ['fetal', 'fetus', 'baby', 'pregnancy', 'prenatal', 'ultrasound']):
            fetal_context += 1

        # Check anatomy identification
        category_lower = img_path.parent.name.lower()
        anatomy_terms = {
            "abodomen": ["abdomen", "stomach", "liver", "kidney", "bowel"],
            "aorta": ["aorta", "vessel", "artery", "cardiac", "heart"],
            "brain": ["brain", "skull", "cerebral", "ventricle", "head"],
            "femur": ["femur", "bone", "leg", "limb", "extremity"],
            "heart": ["heart", "cardiac", "ventricle", "atrium", "valve"]
        }

        if category_lower in anatomy_terms:
            if any(term in response_lower for term in anatomy_terms[category_lower]):
                correct_anatomy += 1

        # Check response detail
        if len(response) > 100:
            detailed_responses += 1

    # Calculate accuracy
    total_tested = min(6, len(test_images))
    print(f"\n3. Accuracy Assessment:")
    print(f"   Fetal context recognition: {fetal_context}/{total_tested} ({fetal_context/total_tested*100:.1f}%)")
    print(f"   Anatomy identification: {correct_anatomy}/{total_tested} ({correct_anatomy/total_tested*100:.1f}%)")
    print(f"   Detailed responses: {detailed_responses}/{total_tested} ({detailed_responses/total_tested*100:.1f}%)")

    # Test diagnostic questions
    print("\n4. Testing diagnostic capabilities...")

    if test_images:
        img_path = test_images[0]
        image = Image.open(img_path).convert('RGB')

        test_questions = [
            "Is this a normal fetal ultrasound?",
            "What trimester might this be from?",
            "Are there any abnormalities visible?",
        ]

        for question in test_questions:
            print(f"\n   Q: {question}")

            if hasattr(model, 'chat'):
                response = model.chat(image, question)
            else:
                inputs = tokenizer(question, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            print(f"   A: {response[:200]}...")

    print("\n" + "="*70)
    print("TinyGPT-V Test Complete")
    print("="*70)

    print(f"\nModel Summary:")
    print(f"  - TinyGPT-V: {param_count / 1e9:.2f}B params")
    print(f"  - Memory: {memory:.2f} GB")
    print(f"  - Fetal context: {'EXCELLENT' if fetal_context >= 5 else 'GOOD' if fetal_context >= 3 else 'LIMITED'}")
    print(f"  - Anatomy accuracy: {'HIGH' if correct_anatomy >= 4 else 'MODERATE' if correct_anatomy >= 2 else 'LOW'}")
    print(f"  - Response quality: {'DETAILED' if detailed_responses >= 4 else 'ADEQUATE'}")

    # Compare to BLIP-2
    overall_score = (fetal_context + correct_anatomy + detailed_responses/2) / total_tested
    print(f"\n  - Overall performance: {overall_score*100:.1f}%")
    print(f"  - Better than BLIP-2? {'YES!' if overall_score > 0.8 else 'COMPARABLE' if overall_score >= 0.6 else 'NO'}")

except Exception as e:
    print(f"\n[ERROR] {type(e).__name__}: {str(e)[:300]}")

    if "out of memory" in str(e).lower():
        print("   TinyGPT-V exceeded 8GB VRAM")
    elif "trust_remote_code" in str(e).lower():
        print("   Model requires trust_remote_code=True")
    else:
        print("   TinyGPT-V may require special setup or dependencies")

print("\n" + "="*70)
