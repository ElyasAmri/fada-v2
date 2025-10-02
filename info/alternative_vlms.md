# Alternative Vision-Language Models for FADA

**Date**: October 2, 2025
**Hardware**: NVIDIA RTX 4070 (8GB VRAM)
**Current Model**: BLIP-2 (working, 5 categories trained)

## Viable Alternatives Found

### 1. TinyGPT-V ⭐ RECOMMENDED
**Status**: Best alternative for 8GB GPU

#### Specifications
- **Type**: Efficient multimodal large language model
- **Parameters**: 2.8B (after quantization)
- **Architecture**:
  - Vision Encoder: Pre-trained (EVA-CLIP compatible)
  - Language Model: Phi-2
  - Mapping Module: Custom visual-linguistic fusion
- **Memory Requirements**:
  - Training: 24GB
  - Inference: **8GB** (with quantization)
  - Much lower than competitors (LLaVA requires 8 A100 GPUs)

#### Performance
- **VQA Performance**: 98% of InstructBLIP performance
- **Tasks**: Image captioning, VQA, instruction following
- **Speed**: Optimized for real-time inference on edge devices
- **Quantization**: 8-bit quantization built-in

#### Availability
- **Repository**: https://github.com/DLYuanGod/TinyGPT-V
- **Paper**: arXiv:2312.16862
- **HuggingFace**: Available with pre-trained weights
- **License**: Open-source

#### For FADA
- ✅ Fits in 8GB VRAM for inference
- ✅ Native VQA support
- ✅ Can be fine-tuned like BLIP-2 (LoRA)
- ✅ Similar architecture (vision encoder + LLM)
- ⚠️ Not medical-specialized (like BLIP-2)
- ⚠️ Training requires 24GB (would need to rent GPU)

#### Integration Strategy
1. Use pre-trained TinyGPT-V for inference testing
2. Compare zero-shot performance with BLIP-2
3. If promising, rent cloud GPU for fine-tuning
4. Train on same ultrasound Q&A pairs
5. Compare performance vs BLIP-2

---

### 2. Medical CLIP Variants (CLIP-based, not full VQA)

These are CLIP-style models (image-text embedding) rather than full VQA models, but can be adapted:

#### 2a. PubMedCLIP ✅ STABLE
**Status**: Most stable for ultrasound (research finding)

- **Type**: Medical vision-language foundation model
- **Training Data**: ROCO dataset (radiology images + captions)
- **Architecture**:
  - Text: Transformer (CLIP-style)
  - Vision: ViT-B/32, ResNet-50, or ResNet-50x4
- **Memory**: ~2-4GB (smaller than BLIP-2)
- **Performance**:
  - Breast ultrasound: Most stable VLM (research study)
  - Improved accuracy up to 3% over MAML
  - Best for tumor classification in ultrasound
- **VQA**: Requires adapter/decoder for generation
- **Paper**: EACL 2023 Findings
- **Repository**: HuggingFace available

#### 2b. BiomedCLIP ✅ BEST PERFORMANCE
**Status**: Best performance on medical VQA benchmarks

- **Type**: Multimodal biomedical foundation model
- **Training Data**: PMC-15M (15 million figure-caption pairs from PubMed)
- **Architecture**: CLIP-based (PubMedBERT text + ViT vision)
- **Memory**: ~3-5GB
- **Performance**:
  - VQA-RAD: +1 point over PubMedCLIP
  - SLAKE: +6 points over PubMedCLIP
  - Outperforms MedCLIP in low-resource settings
- **Tasks**: Classification, retrieval, zero-shot
- **VQA**: Requires generative head
- **Repository**: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
- **Paper**: arXiv:2303.00915

#### 2c. PLIP (Pathology-specialized)
**Status**: Specialized for pathology, not ultrasound

- **Type**: Pathology language-image pretraining
- **Training Data**: OpenPath (208K pathology images from medical Twitter)
- **Architecture**: Fine-tuned CLIP
- **Memory**: ~2-3GB
- **Performance**: F1 0.565-0.832 vs CLIP 0.030-0.481 on pathology
- **Use Case**: Primarily for microscopy/pathology
- **VQA**: Used as encoder in PathologyVLM, PA-LLaVA
- **Repository**: PathologyFoundation/plip
- **Paper**: Nature Medicine 2023

**Note**: PLIP is pathology-specific, not suitable for ultrasound

---

### 3. Recent Medical VLMs (May require >8GB)

#### 3a. FetalCLIP (February 2025) 🎯 DOMAIN-SPECIFIC
**Status**: HIGHLY RELEVANT - Fetal ultrasound specialized!

- **Type**: Visual-language foundation model for FETAL ULTRASOUND
- **Domain**: Fetal development monitoring, congenital abnormalities
- **Status**: Recently published (Feb 2025)
- **Memory**: Unknown (paper in arXiv:2502.14807)
- **Relevance**: PERFECT domain match for FADA
- **Action Required**: Check paper for model availability

#### 3b. LLAUS (2025) 🎯 ULTRASOUND-SPECIFIC
**Status**: HIGHLY RELEVANT - Ultrasound VQA!

- **Type**: Large Vision-Language Model for Ultrasound
- **Tasks**: Question-answering, caption generation for ultrasound
- **Training**: High-quality instruction-following data
- **Performance**: "Exceptional multimodal ultrasound communication"
- **Status**: ACM ICMR 2025
- **Memory**: Unknown
- **Action Required**: Check if model is released

#### 3c. U2-BENCH Evaluated Models (May 2025)
**Status**: Recent benchmark for ultrasound understanding

- **Paper**: arXiv:2505.17779
- **Benchmark**: 7,241 cases, 15 anatomical regions, 8 tasks
- **Models Tested**: 20 state-of-the-art LVLMs
- **Findings**:
  - Strong on image-level classification
  - Weak on spatial reasoning
  - Weak on clinical language generation
- **Action Required**: Check paper for model recommendations

---

## Comparison Matrix

| Model | Memory | VQA Ready | Medical | Ultrasound | Availability | Priority |
|-------|--------|-----------|---------|------------|--------------|----------|
| **BLIP-2** | 4.2 GB | ✅ Yes | ❌ No | ❌ No | ✅ Trained | **CURRENT** |
| **TinyGPT-V** | 8 GB | ✅ Yes | ❌ No | ❌ No | ✅ Open | **HIGH** |
| **PubMedCLIP** | 2-4 GB | ⚠️ Adapter | ✅ Yes | ✅ Stable | ✅ Open | **MEDIUM** |
| **BiomedCLIP** | 3-5 GB | ⚠️ Adapter | ✅ Yes | ❌ No | ✅ Open | **MEDIUM** |
| **FetalCLIP** | Unknown | Unknown | ✅ Yes | ✅ FETAL | ⚠️ Check | **INVESTIGATE** |
| **LLAUS** | Unknown | ✅ Yes | ✅ Yes | ✅ US | ⚠️ Check | **INVESTIGATE** |
| **MedGemma** | 5.0 GB | ❌ No | ✅ Yes | ❌ No | ✅ Accessible | **SKIP** |
| **Florence-2** | Unknown | ❌ Error | ❌ No | ❌ No | ❌ Broken | **SKIP** |
| **LLaVA-1.5** | >8 GB | ✅ Yes | ❌ No | ❌ No | ❌ Too large | **SKIP** |

---

## Recommendations by Priority

### Priority 1: Investigate Domain-Specific Models
**Action**: Check if available and fits hardware

1. **FetalCLIP** (arXiv:2502.14807)
   - PERFECT domain match (fetal ultrasound)
   - Published Feb 2025
   - Check: Model weights available? Memory requirements?

2. **LLAUS** (ACM ICMR 2025)
   - Ultrasound-specialized VQA
   - Instruction-tuned
   - Check: Open-source? Hardware requirements?

### Priority 2: Test TinyGPT-V
**Action**: Quick test on FADA dataset

- Download pre-trained TinyGPT-V
- Test zero-shot on fetal ultrasound images
- Compare with BLIP-2 zero-shot
- If promising: rent GPU for fine-tuning

### Priority 3: Test Medical CLIP Models
**Action**: Use as feature extractors or with adapters

- **PubMedCLIP**: Best for ultrasound stability
- **BiomedCLIP**: Best overall medical performance
- Options:
  - Use as vision encoder (replace BLIP-2's encoder)
  - Add generative head for VQA
  - Use for zero-shot classification + template responses

---

## Implementation Paths

### Option A: Quick Comparison (This Week)
1. Test TinyGPT-V zero-shot on 5 categories
2. Test PubMedCLIP + BiomedCLIP zero-shot
3. Compare with BLIP-2
4. Document results for paper

**Effort**: 1-2 days
**Benefit**: Comprehensive model comparison

### Option B: Domain-Specific Model (If Available)
1. Investigate FetalCLIP availability
2. Investigate LLAUS availability
3. If available: test and compare
4. If better: consider switching

**Effort**: 3-5 days (if models available)
**Benefit**: Domain-matched model

### Option C: TinyGPT-V Fine-Tuning (Requires GPU)
1. Rent cloud GPU (24GB, e.g., A100)
2. Fine-tune TinyGPT-V on ultrasound Q&A
3. Compare performance with BLIP-2
4. Evaluate efficiency vs accuracy trade-off

**Effort**: 1 week + cloud costs
**Benefit**: Potentially more efficient model

### Option D: Continue with BLIP-2 (Safest)
1. Already working
2. Already trained on 5 categories
3. Focus on full-scale training
4. Document as baseline

**Effort**: 0 days
**Benefit**: No risk, proven solution

---

## Next Steps

### Immediate (Today)
1. ✅ Search for alternative VLMs
2. ⬜ Check FetalCLIP paper and availability
3. ⬜ Check LLAUS availability
4. ⬜ Test TinyGPT-V zero-shot (if time permits)

### Short-term (This Week)
1. Download and test TinyGPT-V
2. Download and test PubMedCLIP/BiomedCLIP
3. Compare zero-shot performance
4. Document findings for research paper

### Long-term (Optional)
1. Fine-tune TinyGPT-V if significantly better
2. Try FetalCLIP/LLAUS if available
3. Ensemble methods (combine multiple models)

---

## Key Findings from Research

### Medical VLM Landscape (2024-2025)
- **Rapid Development**: Multiple medical VLMs published in last year
- **Specialization**: Models increasingly domain-specific (ultrasound, pathology)
- **Benchmarks**: New ultrasound benchmarks (U2-BENCH) show VLMs still struggle
- **Challenges**:
  - Spatial reasoning weak
  - Clinical language generation weak
  - Hallucinations common
  - Need for domain-specific training

### Best Practices
- **Visual Prompting**: Improves performance on medical images
- **Few-shot Adaptation**: Critical for medical domains
- **Prompt Engineering**: Significant impact on results
- **Fine-tuning**: Parameter-efficient methods (LoRA) effective

### Research Insights
- Most general VLMs underperform on medical tasks
- Medical-specific VLMs (BiomedCLIP) outperform by significant margins
- Ultrasound is particularly challenging for VLMs
- Domain-specific models (FetalCLIP) show promise

---

## Resources

### Papers to Read
1. FetalCLIP: arXiv:2502.14807
2. LLAUS: ACM ICMR 2025
3. TinyGPT-V: arXiv:2312.16862
4. BiomedCLIP: arXiv:2303.00915
5. U2-BENCH: arXiv:2505.17779
6. PubMedCLIP for VQA: EACL 2023 Findings

### Model Repositories
1. TinyGPT-V: https://github.com/DLYuanGod/TinyGPT-V
2. BiomedCLIP: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
3. PLIP: PathologyFoundation/plip

### Benchmarks
1. U2-BENCH: Ultrasound understanding (2025)
2. VQA-RAD: Radiology VQA
3. SLAKE: Medical VQA
4. Med-HallMark: Medical hallucination detection

---

## Conclusion

**For Immediate Use**: Continue with BLIP-2 (working, proven)

**For Research Comparison**: Test TinyGPT-V, PubMedCLIP, BiomedCLIP

**For Future Work**: Investigate FetalCLIP and LLAUS (domain-specific)

**For Paper**: Document all comparisons to show thorough model evaluation

The discovery of FetalCLIP (fetal ultrasound-specific) and LLAUS (ultrasound VQA) is particularly exciting and warrants immediate investigation.
