"""Quick test of FetalCLIP zero-shot classification on FADA dataset"""

import os
import json
import torch
import open_clip
from PIL import Image
from pathlib import Path
import pandas as pd
from collections import defaultdict

print("="*70)
print("FetalCLIP Quick Test on FADA Dataset")
print("="*70)

# Paths
FETALCLIP_DIR = "FetalCLIP"
CONFIG_PATH = os.path.join(FETALCLIP_DIR, "FetalCLIP_config.json")
WEIGHTS_PATH = "FetalCLIP_weights.pt"  # Weights in project root
DATA_DIR = "data/Fetal Ultrasound"

# Check if weights exist
if not os.path.exists(WEIGHTS_PATH):
    print(f"\n[ERROR] FetalCLIP weights not found at: {WEIGHTS_PATH}")
    print("\nPlease download weights from:")
    print("https://mbzuaiac-my.sharepoint.com/:f:/g/personal/fadillah_maani_mbzuai_ac_ae/EspGREsyuOtEpxt36RoEUBoB6jtlsvPeoiDTBC1qX8WdZQ?e=uAbuyv")
    print("\nSave as: FetalCLIP/FetalCLIP_weights.pt")
    exit(1)

print(f"\n1. Loading FetalCLIP model...")

# Load config
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = config

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Device: {device}")

model, preprocess_train, preprocess_test = open_clip.create_model_and_transforms(
    "FetalCLIP",
    pretrained=WEIGHTS_PATH
)
tokenizer = open_clip.get_tokenizer("FetalCLIP")
model.eval()
model.to(device)

print("   [OK] Model loaded")

# Define FADA categories and prompts
print("\n2. Preparing text prompts for FADA categories...")

# Main categories (matching your 12-class model)
FADA_CATEGORIES = {
    "Abdomen": "Ultrasound image focusing on the fetal abdominal area, highlighting structural development.",
    "Brain": "Fetal ultrasound image showing the brain region with visible anatomical structures.",
    "Femur": "Ultrasound image displaying the fetal femur bone, used for measuring fetal growth.",
    "Thorax": "Fetal ultrasound image focusing on the thoracic region, showing the chest cavity.",
    "Heart": "Fetal ultrasound image focusing on the heart, highlighting detailed cardiac structures.",
    "Cervix": "Ultrasound image showing the cervical region during pregnancy.",
    "Trans-cerebellum": "Fetal ultrasound image of the brain at the trans-cerebellar plane.",
    "Trans-thalamic": "Fetal ultrasound image of the brain at the trans-thalamic plane.",
    "Trans-ventricular": "Fetal ultrasound image of the brain at the trans-ventricular plane.",
    "Standard_NT": "Ultrasound image measuring the nuchal translucency in standard view.",
    "Non_standard_NT": "Ultrasound image showing nuchal translucency in non-standard view.",
    "Lips": "Fetal ultrasound image focusing on the facial region, particularly the lips."
}

# Prepare text prompts
categories = list(FADA_CATEGORIES.keys())
prompts = list(FADA_CATEGORIES.values())

print(f"   Categories: {len(categories)}")
for cat, prompt in FADA_CATEGORIES.items():
    print(f"   - {cat}")

# Tokenize prompts
text_tokens = tokenizer(prompts).to(device)

# Encode text features once (they don't change)
print("\n3. Encoding text features...")
with torch.no_grad(), torch.cuda.amp.autocast():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)
print("   [OK] Text features encoded")

# Test on sample images from each category
print("\n4. Testing zero-shot classification...")

# Collect images from each category (max 10 per category)
test_images = []
category_dirs = [d for d in Path(DATA_DIR).iterdir() if d.is_dir()]

for cat_dir in category_dirs[:5]:  # Test first 5 categories for quick test
    cat_name = cat_dir.name
    images = list(cat_dir.glob("*.png")) + list(cat_dir.glob("*.jpg"))

    if not images:
        continue

    # Take first 3 images from this category
    for img_path in images[:3]:
        test_images.append({
            'path': img_path,
            'true_category': cat_name
        })

print(f"   Testing {len(test_images)} images...")

# Run inference
results = []
correct = 0
total = 0

for item in test_images:
    img_path = item['path']
    true_cat = item['true_category']

    # Load and preprocess image
    image = preprocess_test(Image.open(img_path)).unsqueeze(0).to(device)

    # Encode image
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Compute similarity
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # Get prediction
        pred_idx = similarity.argmax().item()
        pred_cat = categories[pred_idx]
        confidence = similarity[0, pred_idx].item()

    # Check if correct
    is_correct = (pred_cat == true_cat) or (true_cat.startswith(pred_cat))
    if is_correct:
        correct += 1
    total += 1

    results.append({
        'image': img_path.name,
        'true': true_cat,
        'predicted': pred_cat,
        'confidence': confidence,
        'correct': is_correct
    })

    status = '[OK]' if is_correct else '[X]'
    print(f"   {img_path.name}: {pred_cat} ({confidence:.2%}) {status} (True: {true_cat})")

# Calculate accuracy
accuracy = correct / total if total > 0 else 0

print("\n" + "="*70)
print("Results Summary")
print("="*70)
print(f"\nTotal images tested: {total}")
print(f"Correct predictions: {correct}")
print(f"Accuracy: {accuracy:.2%}")

# Per-category breakdown
category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
for r in results:
    cat = r['true']
    category_stats[cat]['total'] += 1
    if r['correct']:
        category_stats[cat]['correct'] += 1

print("\nPer-category accuracy:")
for cat, stats in sorted(category_stats.items()):
    cat_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    print(f"  {cat}: {cat_acc:.2%} ({stats['correct']}/{stats['total']})")

print("\n" + "="*70)
print("Comparison with FADA Phase 1 Classifier")
print("="*70)
print(f"\nFetalCLIP (zero-shot):     {accuracy:.2%}")
print(f"EfficientNet-B0 (trained): 88.0%")
print("\nNote: This is a quick test on limited images.")
print("For full comparison, run evaluation on complete test set.")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("fetalclip_quick_test_results.csv", index=False)
print(f"\nResults saved to: fetalclip_quick_test_results.csv")

print("\n" + "="*70)
print("Next Steps")
print("="*70)
print("\n1. Run full evaluation on entire test set")
print("2. Compare with Phase 1 classification results")
print("3. Test FetalCLIP as vision encoder for BLIP-2")
print("4. Document findings for research paper")
