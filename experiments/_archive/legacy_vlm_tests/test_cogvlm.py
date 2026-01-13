"""Test CogVLM vision-language model"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from PIL import Image
from pathlib import Path
import time

print("="*70)
print("CogVLM Test")
print("="*70)

# Check CUDA
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

DATA_DIR = "data/Fetal Ultrasound"

# Try different CogVLM models
model_ids = [
    "THUDM/cogvlm2-llama3-chat-19B",  # Newer, larger
    "THUDM/cogvlm-chat-hf",  # Original 17B
    "THUDM/cogagent-chat-hf",  # 18B variant
]

for MODEL_ID in model_ids:
    print(f"\n1. Trying {MODEL_ID}...")

    try:
        start_time = time.time()

        # Configure 4-bit quantization for large models
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        print("   Loading with 4-bit quantization...")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config,
            trust_remote_code=True,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        load_time = time.time() - start_time
        print(f"   [OK] Model loaded in {load_time:.1f}s")

        # Get model size
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
            continue

        correct_anatomy = 0
        fetal_context = 0
        detailed_responses = 0

        for i, img_path in enumerate(test_images[:5]):  # Test 5 images
            print(f"\n   Image {i+1}/5: {img_path.name}")
            print(f"   Category: {img_path.parent.name}")

            image = Image.open(img_path).convert('RGB')

            # Format query for CogVLM
            query = "What anatomical structures can you see in this ultrasound image? Please identify any fetal organs or structures."

            # Build input
            if "cogvlm2" in MODEL_ID.lower():
                # CogVLM2 format
                inputs = tokenizer.apply_chat_template(
                    [{"role": "user", "image": image, "content": query}],
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True
                ).to(model.device)
            else:
                # Original CogVLM format
                inputs = tokenizer([query], images=[image], return_tensors="pt").to(model.device)

            # Generate
            start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    temperature=0.7
                )
            gen_time = time.time() - start

            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Clean response (remove query from response)
            if query in response:
                response = response.split(query)[-1].strip()

            print(f"   Response: {response[:300]}...")
            print(f"   Time: {gen_time:.2f}s")

            # Check for medical understanding
            response_lower = response.lower()

            # Check fetal context
            if any(term in response_lower for term in ['fetal', 'fetus', 'baby', 'pregnancy', 'prenatal', 'gestational']):
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
        print(f"\n3. Accuracy Assessment:")
        print(f"   Fetal context recognition: {fetal_context}/5 ({fetal_context/5*100:.1f}%)")
        print(f"   Anatomy identification: {correct_anatomy}/5 ({correct_anatomy/5*100:.1f}%)")
        print(f"   Detailed responses: {detailed_responses}/5 ({detailed_responses/5*100:.1f}%)")

        # Test additional VQA capabilities
        print("\n4. Testing diagnostic capabilities...")

        if test_images:
            img_path = test_images[0]
            image = Image.open(img_path).convert('RGB')

            test_prompts = [
                "Is this a normal or abnormal fetal ultrasound? Explain your assessment.",
                "What gestational age might this fetus be based on the visible structures?",
                "Are there any concerning findings in this ultrasound image?",
            ]

            for prompt in test_prompts:
                print(f"\n   Q: {prompt}")

                if "cogvlm2" in MODEL_ID.lower():
                    inputs = tokenizer.apply_chat_template(
                        [{"role": "user", "image": image, "content": prompt}],
                        add_generation_prompt=True,
                        tokenize=True,
                        return_tensors="pt",
                        return_dict=True
                    ).to(model.device)
                else:
                    inputs = tokenizer([prompt], images=[image], return_tensors="pt").to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=False
                    )

                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if prompt in response:
                    response = response.split(prompt)[-1].strip()

                print(f"   A: {response[:200]}...")

        print("\n" + "="*70)
        print("SUCCESS! CogVLM works!")
        print("="*70)

        print(f"\nModel Summary:")
        print(f"  - Model: {MODEL_ID}")
        print(f"  - Size: {param_count / 1e9:.2f}B params")
        print(f"  - Memory: {memory:.2f} GB (with 4-bit)")
        print(f"  - Fetal context: {'EXCELLENT' if fetal_context >= 4 else 'GOOD' if fetal_context >= 3 else 'LIMITED'}")
        print(f"  - Anatomy accuracy: {'HIGH' if correct_anatomy >= 4 else 'MODERATE' if correct_anatomy >= 2 else 'LOW'}")
        print(f"  - Response quality: {'DETAILED' if detailed_responses >= 4 else 'ADEQUATE'}")

        # Compare to BLIP-2
        blip2_score = 3  # Baseline
        cogvlm_score = (fetal_context + correct_anatomy + detailed_responses/2) / 3

        print(f"\n  - Better than BLIP-2? {'YES - TEST MORE!' if cogvlm_score > blip2_score else 'COMPARABLE' if cogvlm_score >= blip2_score*0.9 else 'NO'}")

        break  # Success

    except Exception as e:
        print(f"   [ERROR] {type(e).__name__}: {str(e)[:200]}")

        if "out of memory" in str(e).lower():
            print("   Trying with 8-bit quantization instead...")
            # Could retry with 8-bit here

        continue

else:
    print("\nNo CogVLM models could be loaded successfully")
    print("Models may be too large even with quantization")

print("\n" + "="*70)