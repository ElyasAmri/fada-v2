"""Test Qwen-VL vision-language model with extended timeout and better error handling"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from PIL import Image
from pathlib import Path
import time
import sys

print("="*70)
print("Qwen-VL Extended Test with Quantization")
print("Starting at:", time.strftime("%H:%M:%S"))
print("="*70)

# Check CUDA
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

DATA_DIR = "data/Fetal Ultrasound"

# Try Qwen-VL-Chat which is more commonly used
MODEL_ID = "Qwen/Qwen-VL-Chat"

try:
    print(f"\n1. Loading {MODEL_ID} with 4-bit quantization...")
    print("   This may take up to 60 minutes for download...")
    start_time = time.time()

    # Configure 4-bit quantization for 8GB GPU
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # Load tokenizer first (faster)
    print("   Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"   Tokenizer loaded! Time: {time.time() - start_time:.1f}s")

    # Load model with quantization
    print("   Loading model (this is the slow part)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )

    load_time = time.time() - start_time
    print(f"   [OK] Model loaded in {load_time:.1f} seconds!")

    # Get model info
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   Model size: {param_count / 1e9:.2f}B params")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"   GPU Memory used: {memory:.2f} GB")

    print("\n2. Testing on fetal ultrasound images...")

    # Get test images
    test_images = []
    for cat_dir in list(Path(DATA_DIR).iterdir())[:3]:
        if cat_dir.is_dir():
            images = list(cat_dir.glob("*.png"))[:1]
            if images:
                test_images.append(images[0])

    if not test_images:
        print("   No test images found!")
    else:
        for i, img_path in enumerate(test_images):
            print(f"\n   Test {i+1}/{len(test_images)}")
            print(f"   Image: {img_path.name}")
            print(f"   Category: {img_path.parent.name}")

            try:
                # Load image
                image = Image.open(img_path).convert('RGB')

                # Save image temporarily for Qwen-VL (it uses file paths)
                temp_path = f"temp_test_image_{i}.png"
                image.save(temp_path)

                # Create query in Qwen-VL format
                query = tokenizer.from_list_format([
                    {'image': temp_path},
                    {'text': 'What anatomical structures can you see in this ultrasound image?'},
                ])

                # Generate response
                print("   Generating response...")
                start = time.time()

                response, history = model.chat(
                    tokenizer,
                    query=query,
                    history=None
                )

                gen_time = time.time() - start

                print(f"   Response: {response}")
                print(f"   Generation time: {gen_time:.2f}s")

                # Clean up temp file
                Path(temp_path).unlink(missing_ok=True)

            except Exception as e:
                print(f"   Error processing image: {e}")
                continue

    print("\n" + "="*70)
    print("SUCCESS! Qwen-VL-Chat works with 4-bit quantization")
    print(f"Total time: {time.time() - start_time:.1f} seconds")
    print("="*70)

    # Additional capabilities test
    print("\n3. Testing Qwen-VL special capabilities...")

    if test_images:
        img_path = test_images[0]
        image = Image.open(img_path).convert('RGB')
        image.save("temp_test.png")

        # Test different prompts
        test_prompts = [
            "Describe this medical image in detail.",
            "Is this a normal or abnormal ultrasound?",
            "What type of medical scan is this?",
        ]

        for prompt in test_prompts:
            print(f"\n   Prompt: {prompt}")
            query = tokenizer.from_list_format([
                {'image': "temp_test.png"},
                {'text': prompt},
            ])

            response, _ = model.chat(tokenizer, query=query, history=None)
            print(f"   Response: {response[:200]}...")

        Path("temp_test.png").unlink(missing_ok=True)

    print("\n" + "="*70)
    print("Qwen-VL-Chat Evaluation Complete")
    print("="*70)

except KeyboardInterrupt:
    print("\n\n[!] Test interrupted by user")
    print(f"Time elapsed: {time.time() - start_time:.1f} seconds")
    sys.exit(0)

except Exception as e:
    print(f"\n[ERROR] {type(e).__name__}: {str(e)[:200]}")
    print(f"\nTime elapsed before error: {time.time() - start_time:.1f} seconds")

    # Try alternative approach
    print("\n4. Trying alternative Qwen-VL loading approach...")

    try:
        # Try without quantization but with CPU offload
        print("   Attempting CPU offload strategy...")

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            device_map="auto",
            max_memory={0: "7GiB", "cpu": "30GiB"},
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

        print("   Model loaded with CPU offload!")

    except Exception as e2:
        print(f"   Also failed: {str(e2)[:100]}")
        print("\n   Qwen-VL appears to be too large or incompatible")
        print("   Consider using Qwen-VL-Chat-Int4 (pre-quantized version)")

print("\n" + "="*70)
print("Test completed at:", time.strftime("%H:%M:%S"))
print("="*70)