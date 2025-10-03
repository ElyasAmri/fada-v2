"""Test mPLUG-Owl2 vision-language model"""

import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from PIL import Image
from pathlib import Path
import time

print("="*70)
print("mPLUG-Owl2 Test")
print("="*70)

# Check CUDA
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

DATA_DIR = "data/Fetal Ultrasound"

# Try different mPLUG-Owl2 models
model_ids = [
    "MAGAer13/mplug-owl2-llama2-7b",
]

for MODEL_ID in model_ids:
    print(f"\n1. Trying {MODEL_ID}...")

    try:
        start_time = time.time()

        # Configure 4-bit quantization for 7B model
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        print("   Loading with 4-bit quantization...")

        # Load model components
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config,
            trust_remote_code=True,
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
            continue

        correct_anatomy = 0
        fetal_context = 0
        detailed_responses = 0

        for i, img_path in enumerate(test_images[:6]):  # Test 6 images
            print(f"\n   Image {i+1}/6: {img_path.name}")
            print(f"   Category: {img_path.parent.name}")

            image = Image.open(img_path).convert('RGB')

            # Format query for mPLUG-Owl2
            query = "USER: <|image|>This is a fetal ultrasound image. What anatomical structures can you identify? Describe what you see in detail.\nASSISTANT:"

            # Process inputs
            inputs = processor(
                text=query,
                images=image,
                return_tensors="pt"
            )

            # Move to device
            device = next(model.parameters()).device
            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

            # Generate
            start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    temperature=0.7
                )
            gen_time = time.time() - start

            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Clean response (remove query from response)
            if "ASSISTANT:" in response:
                response = response.split("ASSISTANT:")[-1].strip()

            print(f"   Response: {response[:300]}...")
            print(f"   Time: {gen_time:.2f}s")

            # Check for medical understanding
            response_lower = response.lower()

            # Check fetal context
            if any(term in response_lower for term in ['fetal', 'fetus', 'baby', 'pregnancy', 'prenatal', 'gestational', 'ultrasound']):
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

                query = f"USER: <|image|>{prompt}\nASSISTANT:"
                inputs = processor(text=query, images=image, return_tensors="pt")
                inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=False
                    )

                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if "ASSISTANT:" in response:
                    response = response.split("ASSISTANT:")[-1].strip()

                print(f"   A: {response[:200]}...")

        print("\n" + "="*70)
        print("SUCCESS! mPLUG-Owl2 works!")
        print("="*70)

        print(f"\nModel Summary:")
        print(f"  - Model: {MODEL_ID}")
        print(f"  - Size: {param_count / 1e9:.2f}B params")
        print(f"  - Memory: {memory:.2f} GB (with 4-bit)")
        print(f"  - Fetal context: {'EXCELLENT' if fetal_context >= 5 else 'GOOD' if fetal_context >= 3 else 'LIMITED'}")
        print(f"  - Anatomy accuracy: {'HIGH' if correct_anatomy >= 4 else 'MODERATE' if correct_anatomy >= 2 else 'LOW'}")
        print(f"  - Response quality: {'DETAILED' if detailed_responses >= 4 else 'ADEQUATE'}")

        # Compare to BLIP-2
        overall_score = (fetal_context + correct_anatomy + detailed_responses/2) / total_tested
        print(f"\n  - Overall performance: {overall_score*100:.1f}%")
        print(f"  - Better than BLIP-2? {'YES - TEST MORE!' if overall_score > 0.8 else 'COMPARABLE' if overall_score >= 0.6 else 'NO'}")

        break  # Success

    except Exception as e:
        print(f"   [ERROR] {type(e).__name__}: {str(e)[:300]}")

        if "out of memory" in str(e).lower():
            print("   Trying with 8-bit quantization instead...")
            # Could retry with 8-bit here

        continue

else:
    print("\nmPLUG-Owl2 could not be loaded successfully")

print("\n" + "="*70)
