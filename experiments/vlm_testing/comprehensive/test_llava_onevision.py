"""Test LLaVA-OneVision-Qwen2-7B - Latest LLaVA variant"""

import torch
from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from PIL import Image
from pathlib import Path
import time

print("="*70)
print("LLaVA-OneVision-Qwen2-7B Test")
print("="*70)

# Check CUDA
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

DATA_DIR = "data/Fetal Ultrasound"
MODEL_ID = "llava-hf/llava-onevision-qwen2-7b-ov-hf"

try:
    print(f"\n1. Loading LLaVA-OneVision-Qwen2-7B...")
    start_time = time.time()

    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # Load processor and model
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        _attn_implementation="eager"
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
    medical_quality = 0

    for i, img_path in enumerate(test_images[:6]):
        print(f"\n   Image {i+1}/6: {img_path.name}")
        print(f"   Category: {img_path.parent.name}")

        image = Image.open(img_path).convert('RGB')

        # LLaVA-OneVision format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "This is a fetal ultrasound image. What anatomical structures can you identify? Provide a detailed medical description."}
                ]
            }
        ]

        # Process inputs
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

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
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Clean response
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()
        elif "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
            response = response.lstrip(":").strip()

        print(f"   Response: {response[:300]}...")
        print(f"   Time: {gen_time:.2f}s")

        # Evaluate
        response_lower = response.lower()

        # Check fetal context
        if any(term in response_lower for term in ['fetal', 'fetus', 'ultrasound', 'pregnancy', 'prenatal']):
            fetal_context += 1

        # Check anatomy
        category_lower = img_path.parent.name.lower()
        anatomy_mapping = {
            "abodomen": ["abdomen", "stomach", "liver", "kidney"],
            "aorta": ["aorta", "vessel", "artery", "heart", "cardiac"],
            "brain": ["brain", "skull", "cerebral", "ventricle", "head"],
            "femur": ["femur", "bone", "leg", "limb"],
            "heart": ["heart", "cardiac", "ventricle", "atrium"]
        }

        if category_lower in anatomy_mapping:
            if any(term in response_lower for term in anatomy_mapping[category_lower]):
                correct_anatomy += 1

        # Check medical quality
        medical_terms = ['structure', 'anatomical', 'visible', 'measurement', 'normal', 'abnormal']
        if any(term in response_lower for term in medical_terms):
            medical_quality += 1

    # Results
    total_tested = min(6, len(test_images))
    print(f"\n3. Accuracy Assessment:")
    print(f"   Fetal context: {fetal_context}/{total_tested} ({fetal_context/total_tested*100:.1f}%)")
    print(f"   Anatomy accuracy: {correct_anatomy}/{total_tested} ({correct_anatomy/total_tested*100:.1f}%)")
    print(f"   Medical quality: {medical_quality}/{total_tested} ({medical_quality/total_tested*100:.1f}%)")

    print("\n" + "="*70)
    print("LLaVA-OneVision-Qwen2-7B Test Complete")
    print("="*70)

    overall_score = (fetal_context + correct_anatomy + medical_quality) / (total_tested * 3) * 100
    print(f"\nOverall: {overall_score:.1f}%")
    print(f"Better than SmolVLM (~66.7%)? {'YES!' if overall_score > 66.7 else 'NO'}")

except Exception as e:
    print(f"\n[ERROR] {type(e).__name__}: {str(e)[:300]}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
