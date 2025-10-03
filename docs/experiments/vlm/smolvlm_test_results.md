# SmolVLM-500M Test Results

**Date**: October 2, 2025
**Model**: HuggingFaceTB/SmolVLM-500M-Instruct
**Dataset**: FADA Fetal Ultrasound (5 sample images)

## Model Specifications

- **Parameters**: 0.51B (510M)
- **GPU Memory**: 1.02 GB
- **Device**: CUDA (RTX 4070)
- **Precision**: bfloat16
- **Load Time**: 8.6 seconds

## Performance Metrics

- **Images Tested**: 5 (2 categories)
- **Average Generation Time**: 4.53 seconds/image
- **Question Used**: "What anatomical structures can you see in this ultrasound image?"

## Detailed Results

### Abodomen Category (3 images)

**Image 1: Abodomen_001.png**
- **Response**: "The anatomical structures visible in this ultrasound image include the uterus, cervix, and fallopian tubes."
- **Generation Time**: 2.87s
- **Assessment**: ❌ Incorrect - Not fetal abdomen structures

**Image 2: Abodomen_002.png**
- **Response**: "The anatomical structures visible in this ultrasound image include the heart, lungs, and other organs."
- **Generation Time**: 1.69s
- **Assessment**: ❌ Incorrect - Generic adult anatomy

**Image 3: Abodomen_003.png**
- **Response**: "The anatomical structures visible in this ultrasound image include the uterus, cervix, and fallopian tubes."
- **Generation Time**: 1.85s
- **Assessment**: ❌ Incorrect - Maternal structures, not fetal

### Aorta Category (2 images)

**Image 4: Aorta_001.png**
- **Response**: "The image contains a human heart. The heart is depicted in a cross-sectional view, showing the different chambers and structures within it. The heart is centrally located in the image, with the superior vena cava (SVC) vein and the aorta (the main artery) visible on the left side of the image. The right atrium (RA) and the right ventricle (RV) are also visible on the right side of the image. The superior vena cava and the aorta"
- **Generation Time**: 8.03s
- **Assessment**: ✅ Partially correct - Identified cardiac structures and aorta, detailed anatomical terminology

**Image 5: Aorta_002.png**
- **Response**: "The image contains a human heart. The heart is depicted in a cross-sectional view, showing the different chambers and structures within it. The heart is depicted in a black and white color scheme, which makes it easier to distinguish the various parts of the heart.\n\n### Detailed Description:\n1. **Heart Structure**:\n   - **Atria**: The heart has two atria, which are the upper chambers.\n   - **Ventricles**: The heart has two ventricles, which are"
- **Generation Time**: 8.24s
- **Assessment**: ✅ Partially correct - Identified cardiac anatomy with structured description

## Analysis

### Strengths
1. **Efficiency**: Very small model (0.51B params) with low memory footprint (1.02GB)
2. **Speed**: Fast inference (4.53s average)
3. **Anatomical Knowledge**: Demonstrates understanding of cardiac anatomy
4. **Response Structure**: Can generate detailed, structured descriptions
5. **Technical Accuracy**: Uses correct anatomical terminology (atria, ventricles, SVC)

### Weaknesses
1. **No Fetal Context**: Does not recognize these are fetal ultrasounds
2. **Generic Responses**: Describes adult anatomy instead of fetal structures
3. **Inconsistent Accuracy**: Good on cardiac images, poor on abdominal images
4. **Domain Mismatch**: Not trained on medical/obstetric ultrasound data
5. **Maternal vs Fetal Confusion**: Identifies maternal structures (uterus, cervix) instead of fetal organs

### Comparison with Other Models

| Model | Size | Memory | Speed | Fetal Context | Quality |
|-------|------|--------|-------|---------------|---------|
| **SmolVLM-500M** | 0.51B | 1.0GB | 4.5s | ❌ None | Mixed (good anatomy, wrong context) |
| **BLIP-2** | 3.4B | 4.2GB | ~5-6s | ✅ Yes | Good |
| **FetalCLIP** | 0.4B | 3.0GB | Fast | ✅ Yes | 40% accuracy (classification only) |

## Conclusions

### Suitability for FADA
- ❌ **Not Recommended**: SmolVLM-500M lacks domain-specific knowledge for fetal ultrasound
- The model generates anatomically correct descriptions but misses the fetal context entirely
- Would require fine-tuning on fetal ultrasound data to be useful

### Use Cases Where SmolVLM Excels
- General image captioning
- Document understanding
- Visual question answering on common objects
- Scenarios requiring small model size and low memory

### Research Value
- Demonstrates importance of domain-specific training data
- Shows that model size alone doesn't determine medical imaging performance
- 6.7x smaller than BLIP-2 but significantly worse on specialized medical task
- Validates choice of BLIP-2 for medical VQA

## Technical Notes

### API Usage
- Model uses `Idefics3Processor` via `AutoProcessor`
- Images passed as `{"type": "image", "path": str(path)}` in chat messages
- `apply_chat_template` handles image loading automatically
- Supports multi-turn conversations and multiple images per turn

### Installation Requirements
```bash
pip install transformers torch pillow num2words
```

### Code Example
```python
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
model = AutoModelForImageTextToText.from_pretrained(
    "HuggingFaceTB/SmolVLM-500M-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "path": "image.png"},
        {"type": "text", "text": "What do you see?"}
    ]
}]

inputs = processor.apply_chat_template(messages, add_generation_prompt=True,
                                      tokenize=True, return_dict=True,
                                      return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}

generated_ids = model.generate(**inputs, max_new_tokens=100)
response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

## Recommendation

**Continue with BLIP-2** as the primary VQA model for FADA. SmolVLM's efficiency gains do not compensate for the lack of medical domain knowledge. For production deployment requiring smaller models, consider fine-tuning SmolVLM on fetal ultrasound dataset rather than using zero-shot.

## Files Generated
- `test_smolvlm_quick.py` - Test script
- `smolvlm_quick_test_results.json` - Raw results data
- `info/smolvlm_test_results.md` - This documentation

---

**Status in VLM Testing List**: ✅ Item #4 - Tested and documented
**Next Model**: DeepSeek-VL-1.3B (Item #5) or skip to BLIP-VQA-base (Item #7)
