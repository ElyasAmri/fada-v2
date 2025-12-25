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
FETALCLIP_DIR = "experiments/external_models/FetalCLIP"
CONFIG_PATH = os.path.join(FETALCLIP_DIR, "FetalCLIP_config.json")
WEIGHTS_PATH = "artifacts/weights/FetalCLIP_weights.pt"
DATA_DIR = "data/Fetal Ultrasound"

# Folder name to FetalCLIP category mapping
# Handles typos and semantic equivalents
FOLDER_TO_CATEGORY = {
    "Abodomen": "Abdomen",      # Typo in folder name
    "Aorta": "Heart",           # Aorta is cardiac structure
    "Cervical": "Cervix",       # Semantic equivalent
    "Cervix": "Cervix",
    "Femur": "Femur",
    "Non_standard_NT": "Non_standard_NT",
    "Standard_NT": "Standard_NT",
    "Thorax": "Thorax",
    "Trans-cerebellum": "Trans-cerebellum",
    "Trans-thalamic": "Trans-thalamic",
    "Trans-ventricular": "Trans-ventricular",
    # Excluded: Public_Symphysis_fetal_head (no matching FetalCLIP category)
}

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
with torch.no_grad(), torch.amp.autocast('cuda'):
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)
print("   [OK] Text features encoded")

# Test on proper test set
print("\n4. Loading test set from dataset splits...")

# Import dataset splits
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.data.dataset_splits import get_split_with_labels, DATA_ROOT

# Get test split images
test_split = get_split_with_labels('test')
print(f"   Total test images: {len(test_split)}")

# Filter and map to FetalCLIP categories
test_images = []
skipped_categories = set()

for img_rel_path, folder_name in test_split:
    # Skip folders not in mapping
    if folder_name not in FOLDER_TO_CATEGORY:
        skipped_categories.add(folder_name)
        continue

    # Map folder name to FetalCLIP category
    mapped_category = FOLDER_TO_CATEGORY[folder_name]
    img_path = DATA_ROOT / img_rel_path

    test_images.append({
        'path': img_path,
        'folder': folder_name,
        'true_category': mapped_category
    })

if skipped_categories:
    print(f"   Skipped categories (no mapping): {skipped_categories}")

print(f"   Testing {len(test_images)} images...")

# Run inference with progress bar
from tqdm import tqdm

results = []
correct = 0
total = 0

for item in tqdm(test_images, desc="FetalCLIP inference"):
    img_path = item['path']
    true_cat = item['true_category']

    # Load and preprocess image
    image = preprocess_test(Image.open(img_path)).unsqueeze(0).to(device)

    # Encode image
    with torch.no_grad(), torch.amp.autocast('cuda'):
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
        'folder': item['folder'],
        'true': true_cat,
        'predicted': pred_cat,
        'confidence': confidence,
        'correct': is_correct
    })

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
output_file = "experiments/vlm_testing/quick_tests/fetalclip_test_results.csv"
results_df.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")

# 5-class coarse-grained analysis
print("\n" + "="*70)
print("5-CLASS COARSE-GRAINED ANALYSIS")
print("="*70)

COARSE_MAP = {
    'Abdomen': 'Abdomen', 'Thorax': 'Thorax', 'Femur': 'Femur',
    'Cervix': 'Cervix', 'Heart': 'Heart', 'Brain': 'Brain',
    'Trans-cerebellum': 'Brain', 'Trans-thalamic': 'Brain', 'Trans-ventricular': 'Brain',
    'Standard_NT': None, 'Non_standard_NT': None,
}

results_df['true_coarse'] = results_df['true'].map(COARSE_MAP)
results_df['pred_coarse'] = results_df['predicted'].map(COARSE_MAP)
df_filtered = results_df[results_df['true_coarse'].notna()].copy()
df_filtered['correct_coarse'] = df_filtered['true_coarse'] == df_filtered['pred_coarse']

print(f"\n12-class accuracy: {accuracy:.2%} ({correct}/{total})")
coarse_acc = df_filtered['correct_coarse'].mean()
coarse_correct = df_filtered['correct_coarse'].sum()
print(f"5-class accuracy:  {coarse_acc:.2%} ({coarse_correct}/{len(df_filtered)})")

print("\n5-class per-category:")
for cat in sorted(df_filtered['true_coarse'].unique()):
    cat_df = df_filtered[df_filtered['true_coarse'] == cat]
    acc = cat_df['correct_coarse'].mean() * 100
    print(f"  {cat:12s}: {acc:.1f}% ({cat_df['correct_coarse'].sum()}/{len(cat_df)})")

print("\n" + "="*70)
print("Next Steps")
print("="*70)
print("\n1. Run full evaluation on entire test set")
print("2. Compare with Phase 1 classification results")
print("3. Test FetalCLIP as vision encoder for BLIP-2")
print("4. Document findings for research paper")
