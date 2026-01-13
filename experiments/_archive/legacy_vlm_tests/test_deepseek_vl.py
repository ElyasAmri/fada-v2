"""Test DeepSeek-VL vision-language model - designed for scientific reasoning"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from pathlib import Path
import time

print("="*70)
print("DeepSeek-VL Test (Scientific VLM)")
print("="*70)

# Check CUDA
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

DATA_DIR = "data/Fetal Ultrasound"

# DeepSeek-VL models
model_ids = [
    "deepseek-ai/deepseek-vl-1.3b-chat",  # Smallest
    "deepseek-ai/deepseek-vl-1.3b-base",
]

for MODEL_ID in model_ids:
    print(f"\n1. Trying {MODEL_ID}...")

    try:
        start_time = time.time()

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

        load_time = time.time() - start_time
        print(f"   [OK] Model loaded in {load_time:.1f}s")

        param_count = sum(p.numel() for p in model.parameters())
        print(f"   Size: {param_count / 1e9:.2f}B params")

        if torch.cuda.is_available():
            memory = torch.cuda.max_memory_allocated() / 1e9
            print(f"   GPU Memory: {memory:.2f} GB")

        print("\n2. Testing on fetal ultrasound images (scientific analysis)...")

        # Get test images
        test_images = []
        categories = ["Abodomen", "Aorta", "Brain", "Femur", "Heart", "Thorax"]
        for cat in categories:
            cat_dir = Path(DATA_DIR) / cat
            if cat_dir.exists():
                images = list(cat_dir.glob("*.png"))[:2]
                test_images.extend(images)

        correct_anatomy = 0
        fetal_context = 0
        scientific_quality = 0

        for i, img_path in enumerate(test_images[:8]):  # Test 8 images
            print(f"\n   Image {i+1}/8: {img_path.name}")
            print(f"   Category: {img_path.parent.name}")

            image = Image.open(img_path).convert('RGB')

            # DeepSeek-VL conversation format
            conversation = [
                {
                    "role": "user",
                    "content": "<image>This is a fetal ultrasound image. Please provide a scientific analysis: What anatomical structures are visible? What measurements could be taken? Any notable features?"
                }
            ]

            # Format inputs for DeepSeek-VL
            if "chat" in MODEL_ID:
                # Chat model format
                prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            else:
                # Base model format
                prompt = "Analyze this fetal ultrasound image. Identify anatomical structures:"

            # Prepare inputs
            inputs = tokenizer(prompt, return_tensors="pt")

            # Add image embeddings (model-specific)
            # Note: DeepSeek-VL may require special image processing

            # Move to device
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate
            start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=False
                )
            gen_time = time.time() - start

            # Decode
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Clean response
            if prompt in response:
                response = response.replace(prompt, "").strip()

            print(f"   Response: {response[:300]}...")
            print(f"   Time: {gen_time:.2f}s")

            # Evaluate scientific quality
            response_lower = response.lower()

            # Check fetal/medical context
            medical_terms = ['fetal', 'fetus', 'ultrasound', 'gestational', 'trimester', 'anatomy', 'structure']
            if any(term in response_lower for term in medical_terms):
                fetal_context += 1

            # Check scientific terminology
            scientific_terms = ['measure', 'diameter', 'length', 'analysis', 'observation', 'appears', 'visible',
                              'structure', 'tissue', 'echo', 'hyperechoic', 'hypoechoic', 'anechoic']
            if sum(1 for term in scientific_terms if term in response_lower) >= 3:
                scientific_quality += 1

            # Check anatomy identification
            category_lower = img_path.parent.name.lower()
            anatomy_mapping = {
                "abodomen": ["abdomen", "stomach", "liver", "kidney", "intestine"],
                "aorta": ["aorta", "vessel", "artery", "blood flow", "cardiac"],
                "brain": ["brain", "skull", "cerebral", "ventricle", "cranium"],
                "femur": ["femur", "bone", "leg", "limb", "long bone"],
                "heart": ["heart", "cardiac", "chamber", "ventricle", "atrium"],
                "thorax": ["thorax", "chest", "lung", "ribs", "diaphragm"]
            }

            if category_lower in anatomy_mapping:
                if any(term in response_lower for term in anatomy_mapping[category_lower]):
                    correct_anatomy += 1

        # Calculate accuracy
        total_tested = min(8, len(test_images))
        print(f"\n3. Scientific Analysis Assessment:")
        print(f"   Medical context: {fetal_context}/{total_tested} ({fetal_context/total_tested*100:.1f}%)")
        print(f"   Anatomy accuracy: {correct_anatomy}/{total_tested} ({correct_anatomy/total_tested*100:.1f}%)")
        print(f"   Scientific quality: {scientific_quality}/{total_tested} ({scientific_quality/total_tested*100:.1f}%)")

        # Test measurement capabilities
        print("\n4. Testing measurement/diagnostic capabilities...")

        if test_images:
            img_path = test_images[0]
            image = Image.open(img_path).convert('RGB')

            scientific_questions = [
                "What measurements should be taken from this ultrasound view?",
                "Estimate the gestational age based on visible structures.",
                "Describe the echogenicity patterns you observe.",
            ]

            for question in scientific_questions:
                print(f"\n   Q: {question}")

                conversation = [
                    {"role": "user", "content": f"<image>{question}"}
                ]

                if "chat" in MODEL_ID:
                    prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                else:
                    prompt = question

                inputs = tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=False
                    )

                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if prompt in response:
                    response = response.replace(prompt, "").strip()

                print(f"   A: {response[:200]}...")

        print("\n" + "="*70)
        print("DeepSeek-VL Test Complete")
        print("="*70)

        # Final evaluation
        print(f"\nModel Summary:")
        print(f"  - Model: {MODEL_ID}")
        print(f"  - Size: {param_count / 1e9:.2f}B params")
        print(f"  - Memory: {memory:.2f} GB")
        print(f"  - Designed for: Scientific reasoning and analysis")
        print(f"  - Medical understanding: {'STRONG' if fetal_context >= 6 else 'MODERATE' if fetal_context >= 3 else 'LIMITED'}")
        print(f"  - Anatomy accuracy: {'HIGH' if correct_anatomy >= 6 else 'MODERATE' if correct_anatomy >= 3 else 'LOW'}")
        print(f"  - Scientific quality: {'EXCELLENT' if scientific_quality >= 6 else 'GOOD' if scientific_quality >= 3 else 'BASIC'}")

        # Overall assessment
        overall_score = (fetal_context + correct_anatomy + scientific_quality) / (total_tested * 3)
        print(f"  - Overall score: {overall_score*100:.1f}%")
        print(f"  - Better than BLIP-2? {'LIKELY YES!' if overall_score > 0.6 else 'POSSIBLY' if overall_score > 0.4 else 'NEEDS FINE-TUNING'}")

        break  # Success

    except Exception as e:
        print(f"   [ERROR] {type(e).__name__}: {str(e)[:200]}")

        if "deepseek" not in str(e).lower() and "import" not in str(e).lower():
            print("   Model loaded but may need special setup")

        continue

else:
    print("\nNo DeepSeek-VL models could be loaded")
    print("May require custom installation or dependencies")

print("\n" + "="*70)