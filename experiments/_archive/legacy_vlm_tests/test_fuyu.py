"""Test Fuyu-8B vision-language model"""

import torch
from transformers import FuyuProcessor, FuyuForCausalLM, BitsAndBytesConfig
from PIL import Image
from pathlib import Path
import time

print("="*70)
print("Fuyu-8B Test")
print("="*70)

# Check CUDA
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

DATA_DIR = "data/Fetal Ultrasound"
MODEL_ID = "adept/fuyu-8b"

try:
    print(f"\n1. Loading Fuyu-8B with quantization...")
    start_time = time.time()

    # Configure 4-bit quantization for 8B model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # Load processor and model
    processor = FuyuProcessor.from_pretrained(MODEL_ID)

    model = FuyuForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    load_time = time.time() - start_time
    print(f"   [OK] Model loaded in {load_time:.1f}s")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"   Size: {param_count / 1e9:.2f}B params")

    if torch.cuda.is_available():
        memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"   GPU Memory: {memory:.2f} GB (with 4-bit)")

    print("\n2. Testing on fetal ultrasound images...")

    # Get test images
    test_images = []
    categories = ["Abodomen", "Aorta", "Brain", "Femur", "Heart"]
    for cat in categories:
        cat_dir = Path(DATA_DIR) / cat
        if cat_dir.exists():
            images = list(cat_dir.glob("*.png"))[:2]
            test_images.extend(images)

    correct_anatomy = 0
    fetal_context = 0
    medical_quality = 0

    for i, img_path in enumerate(test_images[:6]):  # Test 6 images
        print(f"\n   Image {i+1}/6: {img_path.name}")
        print(f"   Category: {img_path.parent.name}")

        image = Image.open(img_path).convert('RGB')

        # Fuyu expects specific prompt format
        text_prompt = "Generate a detailed medical description of this fetal ultrasound image. What anatomical structures are visible?"

        # Process inputs
        inputs = processor(text=text_prompt, images=image, return_tensors="pt")

        # Move to device
        for k, v in inputs.items():
            if hasattr(v, 'to'):
                inputs[k] = v.to(model.device)

        # Generate
        start = time.time()
        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False
            )
        gen_time = time.time() - start

        # Decode
        generation_text = processor.batch_decode(generation_output, skip_special_tokens=True)[0]

        # Clean response (remove prompt from output)
        response = generation_text.replace(text_prompt, "").strip()

        print(f"   Response: {response[:300]}...")
        print(f"   Time: {gen_time:.2f}s")

        # Evaluate response
        response_lower = response.lower()

        # Check fetal context
        fetal_terms = ['fetal', 'fetus', 'baby', 'pregnancy', 'prenatal', 'gestational', 'ultrasound']
        if any(term in response_lower for term in fetal_terms):
            fetal_context += 1

        # Check anatomy identification
        category_lower = img_path.parent.name.lower()
        anatomy_checks = {
            "abodomen": ["abdomen", "stomach", "liver", "kidney"],
            "aorta": ["aorta", "vessel", "heart", "cardiac"],
            "brain": ["brain", "skull", "head", "cerebral"],
            "femur": ["femur", "bone", "leg", "limb"],
            "heart": ["heart", "cardiac", "chamber", "ventricle"]
        }

        if category_lower in anatomy_checks:
            if any(term in response_lower for term in anatomy_checks[category_lower]):
                correct_anatomy += 1

        # Check medical quality (detailed response)
        if len(response) > 50 and any(term in response_lower for term in ['structure', 'visible', 'appears', 'shows']):
            medical_quality += 1

    # Calculate accuracy
    total_tested = min(6, len(test_images))
    print(f"\n3. Accuracy Assessment:")
    print(f"   Fetal context recognition: {fetal_context}/{total_tested} ({fetal_context/total_tested*100:.1f}%)")
    print(f"   Anatomy identification: {correct_anatomy}/{total_tested} ({correct_anatomy/total_tested*100:.1f}%)")
    print(f"   Medical quality responses: {medical_quality}/{total_tested} ({medical_quality/total_tested*100:.1f}%)")

    # Test specific medical questions
    print("\n4. Testing medical VQA capabilities...")

    if test_images:
        img_path = test_images[0]
        image = Image.open(img_path).convert('RGB')

        medical_questions = [
            "Is this a normal fetal ultrasound?",
            "What trimester does this appear to be from?",
            "Are there any visible abnormalities?",
        ]

        for question in medical_questions:
            print(f"\n   Q: {question}")

            inputs = processor(text=question, images=image, return_tensors="pt")
            for k, v in inputs.items():
                if hasattr(v, 'to'):
                    inputs[k] = v.to(model.device)

            with torch.no_grad():
                generation_output = model.generate(
                    **inputs,
                    max_new_tokens=80,
                    do_sample=False
                )

            generation_text = processor.batch_decode(generation_output, skip_special_tokens=True)[0]
            response = generation_text.replace(question, "").strip()

            print(f"   A: {response[:200]}...")

    print("\n" + "="*70)
    print("Fuyu-8B Test Complete")
    print("="*70)

    # Final evaluation
    print(f"\nModel Summary:")
    print(f"  - Fuyu-8B: {param_count / 1e9:.2f}B params")
    print(f"  - Memory: {memory:.2f} GB (with 4-bit)")
    print(f"  - Designed for: Multimodal understanding")
    print(f"  - Fetal context: {'YES' if fetal_context >= 4 else 'PARTIAL' if fetal_context >= 2 else 'LIMITED'}")
    print(f"  - Anatomy accuracy: {'HIGH' if correct_anatomy >= 4 else 'MODERATE' if correct_anatomy >= 2 else 'LOW'}")

    # Compare to BLIP-2
    overall_score = (fetal_context + correct_anatomy + medical_quality/2) / total_tested
    print(f"  - Overall performance: {overall_score*100:.1f}%")
    print(f"  - Better than BLIP-2? {'POTENTIALLY!' if overall_score > 0.6 else 'COMPARABLE' if overall_score > 0.4 else 'NO'}")

except Exception as e:
    print(f"\n[ERROR] {type(e).__name__}: {str(e)[:300]}")

    if "out of memory" in str(e).lower():
        print("   Memory exceeded even with 4-bit quantization")
    elif "fuyu" in str(e).lower():
        print("   Fuyu model may not be available or compatible")

print("\n" + "="*70)