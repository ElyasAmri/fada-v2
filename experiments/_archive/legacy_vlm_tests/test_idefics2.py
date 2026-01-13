"""Test IDEFICS2 vision-language model"""

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from PIL import Image
from pathlib import Path
import time

print("="*70)
print("IDEFICS2 Test")
print("="*70)

# Check CUDA
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

DATA_DIR = "data/Fetal Ultrasound"
MODEL_ID = "HuggingFaceM4/idefics2-8b"  # 8B parameter version

try:
    print(f"\n1. Loading IDEFICS2-8B...")
    start_time = time.time()

    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    print("   Loading with 4-bit quantization...")

    # Load processor and model
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
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
    categories = ["Abodomen", "Aorta", "Brain", "Femur", "Heart", "Cervical"]
    for cat in categories:
        cat_dir = Path(DATA_DIR) / cat
        if cat_dir.exists():
            images = list(cat_dir.glob("*.png"))[:2]
            test_images.extend(images)

    correct_anatomy = 0
    fetal_context = 0
    medical_terminology = 0

    for i, img_path in enumerate(test_images[:8]):  # Test 8 images
        print(f"\n   Image {i+1}/8: {img_path.name}")
        print(f"   Category: {img_path.parent.name}")

        image = Image.open(img_path).convert('RGB')

        # Create conversation
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "This is a fetal ultrasound image. What anatomical structures can you identify? Please provide a detailed medical description."},
                ]
            }
        ]

        # Process inputs
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        start = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False
            )
        gen_time = time.time() - start

        # Decode
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        response = generated_texts[0].split("Assistant:")[-1].strip() if "Assistant:" in generated_texts[0] else generated_texts[0]

        print(f"   Response: {response[:300]}...")
        print(f"   Time: {gen_time:.2f}s")

        # Evaluate response
        response_lower = response.lower()

        # Check fetal context
        if any(term in response_lower for term in ['fetal', 'fetus', 'baby', 'pregnancy', 'prenatal', 'gestational', 'maternal']):
            fetal_context += 1

        # Check medical terminology
        medical_terms = ['ultrasound', 'echogenic', 'hypoechoic', 'anechoic', 'doppler', 'gestational', 'trimester',
                        'amniotic', 'placenta', 'umbilical', 'ventricle', 'chamber', 'vessel', 'organ']
        if sum(1 for term in medical_terms if term in response_lower) >= 2:
            medical_terminology += 1

        # Check anatomy identification
        category_lower = img_path.parent.name.lower()
        anatomy_mapping = {
            "abodomen": ["abdomen", "stomach", "liver", "kidney", "bowel", "bladder"],
            "aorta": ["aorta", "vessel", "artery", "cardiac", "heart", "circulation"],
            "brain": ["brain", "skull", "cerebral", "ventricle", "head", "cranial"],
            "femur": ["femur", "bone", "leg", "limb", "extremity", "long bone"],
            "heart": ["heart", "cardiac", "ventricle", "atrium", "valve", "chamber"],
            "cervical": ["cervix", "cervical", "uterus", "birth canal"]
        }

        if category_lower in anatomy_mapping:
            if any(term in response_lower for term in anatomy_mapping[category_lower]):
                correct_anatomy += 1

    # Calculate accuracy
    total_tested = min(8, len(test_images))
    print(f"\n3. Accuracy Assessment:")
    print(f"   Fetal context recognition: {fetal_context}/{total_tested} ({fetal_context/total_tested*100:.1f}%)")
    print(f"   Anatomy identification: {correct_anatomy}/{total_tested} ({correct_anatomy/total_tested*100:.1f}%)")
    print(f"   Medical terminology usage: {medical_terminology}/{total_tested} ({medical_terminology/total_tested*100:.1f}%)")

    # Test diagnostic questions
    print("\n4. Testing diagnostic capabilities...")

    if test_images:
        img_path = test_images[0]
        image = Image.open(img_path).convert('RGB')

        diagnostic_prompts = [
            "Based on this fetal ultrasound, what is your assessment of fetal development?",
            "Are there any abnormalities or concerns visible in this ultrasound?",
            "What measurements would you take from this ultrasound view?",
        ]

        for prompt_text in diagnostic_prompts:
            print(f"\n   Q: {prompt_text}")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_text},
                    ]
                }
            ]

            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt, images=[image], return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False
                )

            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            response = generated_texts[0].split("Assistant:")[-1].strip() if "Assistant:" in generated_texts[0] else generated_texts[0]

            print(f"   A: {response[:250]}...")

    print("\n" + "="*70)
    print("IDEFICS2 Test Complete")
    print("="*70)

    # Final evaluation
    print(f"\nModel Summary:")
    print(f"  - IDEFICS2-8B: {param_count / 1e9:.2f}B params")
    print(f"  - Memory: {memory:.2f} GB (with 4-bit)")
    print(f"  - Fetal context: {'EXCELLENT' if fetal_context >= 6 else 'GOOD' if fetal_context >= 4 else 'LIMITED'}")
    print(f"  - Anatomy accuracy: {'HIGH' if correct_anatomy >= 6 else 'MODERATE' if correct_anatomy >= 3 else 'LOW'}")
    print(f"  - Medical language: {'PROFESSIONAL' if medical_terminology >= 6 else 'ADEQUATE' if medical_terminology >= 3 else 'BASIC'}")

    # Overall assessment
    overall_score = (fetal_context + correct_anatomy + medical_terminology) / (total_tested * 3) * 100
    print(f"  - Overall accuracy: {overall_score:.1f}%")
    print(f"  - Better than BLIP-2? {'LIKELY YES!' if overall_score > 60 else 'POSSIBLY' if overall_score > 40 else 'NEEDS MORE TESTING'}")

except Exception as e:
    print(f"\n[ERROR] {type(e).__name__}: {str(e)[:300]}")
    print("\nIDEFICS2 may be too large or incompatible")

print("\n" + "="*70)