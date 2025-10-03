"""Check availability of medical-specific VLM models"""

from transformers import AutoModel, AutoTokenizer
import torch

print("="*70)
print("Checking Medical VLM Model Availability")
print("="*70)

medical_models = [
    # Tier 3: Medical Models
    ("StanfordAIMI/RadFM", "RadFM - Radiology Foundation Model"),
    ("microsoft/BiomedVLP-CXR-BERT", "BiomedVLP for Chest X-rays"),
    ("UCSD-AI4H/rad-dino", "RadDINO - Radiology DINO"),
    ("microsoft/BiomedVLP", "BiomedVLP - General Medical"),
    ("pathchat/pathchat", "PathChat - Pathology VQA"),
    ("radialog/radialog", "RaDialog - Radiology Dialog"),

    # Medical CLIP variants
    ("flaviagiammarino/pubmed-clip-vit-base-patch32", "PubMedCLIP"),
    ("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", "BiomedCLIP"),
    ("vinid/plip", "PLIP - Pathology Language Image Pre-training"),
]

available_models = []
unavailable_models = []

for model_id, description in medical_models:
    try:
        print(f"\nChecking: {model_id}")
        print(f"  Description: {description}")

        # Try to load config/model info
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

        # Get model size
        param_count = sum(p.numel() for p in model.parameters())
        memory = param_count * 4 / 1e9  # Rough estimate in GB

        print(f"  [OK] Available!")
        print(f"  Parameters: {param_count / 1e6:.1f}M")
        print(f"  Estimated Memory: {memory:.2f} GB")

        available_models.append((model_id, description, memory))

        # Clean up
        del model
        torch.cuda.empty_cache()

    except Exception as e:
        error_msg = str(e)
        if "does not appear to have a file named" in error_msg:
            print(f"  [X] Model not found on HuggingFace")
        elif "Cannot instantiate" in error_msg:
            print(f"  [!] Model exists but requires special loading")
        elif "is not a valid model identifier" in error_msg:
            print(f"  [X] Invalid model identifier")
        else:
            print(f"  [X] Error: {error_msg[:100]}...")

        unavailable_models.append((model_id, description, error_msg))

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

if available_models:
    print(f"\n[OK] Available Models ({len(available_models)}):")
    for model_id, desc, memory in available_models:
        if memory < 8:
            print(f"  - {model_id}: {memory:.2f}GB (Fits 8GB GPU)")
        else:
            print(f"  - {model_id}: {memory:.2f}GB (Needs quantization)")
else:
    print("\n[X] No medical VLM models readily available")

if unavailable_models:
    print(f"\n[X] Unavailable Models ({len(unavailable_models)}):")
    for model_id, desc, _ in unavailable_models[:5]:
        print(f"  - {model_id}: {desc}")

print("\nNOTE: Most medical VLMs are either:")
print("1. Private/gated models requiring special access")
print("2. Research models not released publicly")
print("3. CLIP-style models (classification, not VQA)")
print("4. Too large for consumer GPUs")

print("\n" + "="*70)