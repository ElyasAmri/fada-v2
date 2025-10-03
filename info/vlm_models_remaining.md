# Remaining VLM Models to Test

**Date**: October 3, 2025
**Status**: After comprehensive testing session

## Models Already Tested (14 total)

### Fully Working (8)
1. ✅ **BLIP-2** - Best overall
2. ✅ **Moondream2** - Good, fast
3. ✅ **SmolVLM-500M** - No fetal context
4. ✅ **SmolVLM-256M** - World's smallest
5. ✅ **BLIP-VQA-base** - Too brief
6. ✅ **VILT-b32** - Nonsensical
7. ✅ **LLaVA-NeXT-7B** (4-bit) - Excellent
8. ✅ **InstructBLIP-7B** (4-bit) - Very good

### Working with Issues (3)
9. ✅ **Florence-2** - Requires special setup
10. ✅ **PaliGemma-3B** (8-bit) - Works well
11. ⚠️ **FetalCLIP** - Category mismatch

### Loaded but Failed (3)
12. ❌ **Qwen2-VL-7B** - Too large (9.14GB)
13. ❌ **Qwen-VL-Chat** - Visual encoder issue
14. ❌ **Qwen-VL-Chat-Int4** - Visual encoder issue

## Remaining Models NOT Tested

### Potentially Testable (Could Work)

#### 1. **DeepSeek-VL-1.3B**
- Size: 1.3B params
- Memory: ~3-4GB
- Why not tested: Requires custom package installation
- Effort: 10-15 minutes setup
- Worth trying: YES - Small size, strong reasoning

#### 2. **TinyGPT-V**
- Size: 2.8B params
- Memory: ~8GB
- Why not tested: Complex conda setup + Phi-2 weights
- Effort: 2-3 hours
- Worth trying: MAYBE - If time permits

#### 3. **MiniGPT-4**
- Size: ~7B params
- Why not tested: Not on HuggingFace, requires GitHub clone
- Effort: 1 hour setup
- Worth trying: NO - Too complex

#### 4. **CogVLM**
- Size: 17B/9B variants
- Not on original list
- Worth trying: YES - Has smaller 9B variant

#### 5. **Fuyu-8B**
- Size: 8B params
- By Adept AI
- Worth trying: YES - Might fit with quantization

#### 6. **Kosmos-2**
- Size: 1.6B params
- Microsoft model
- Worth trying: YES - Small size

#### 7. **IDEFICS/IDEFICS2**
- Size: 9B/8B params
- HuggingFace's open alternative to Flamingo
- Worth trying: YES - Newer version might work

#### 8. **mPLUG-Owl/Owl2**
- Size: 7B params
- Alibaba's VLM
- Worth trying: YES - Good for general VQA

#### 9. **Otter**
- Size: 7B params
- Based on OpenFlamingo
- Worth trying: MAYBE

#### 10. **MiniCPM-V**
- Size: 2.5B params
- Lightweight Chinese VLM
- Worth trying: YES - Very small

### Medical Models (Likely Unavailable)

11. **RaDialog** - Not found on HuggingFace
12. **PathChat** - Not found on HuggingFace
13. **Flamingo-CXR** - Not found on HuggingFace
14. **MedGemma** - Tested earlier, failed
15. **BiomedCLIP** - CLIP only (no VQA)
16. **PubMedCLIP** - CLIP only (no VQA)
17. **PLIP** - CLIP only (no VQA)

## Recommended Next Models to Test

### Priority 1 (Most Likely to Work)
1. **Kosmos-2** (1.6B) - Small, Microsoft
2. **MiniCPM-V** (2.5B) - Very lightweight
3. **CogVLM-9B** (with quantization)

### Priority 2 (Worth Trying)
4. **DeepSeek-VL-1.3B** - If willing to do setup
5. **Fuyu-8B** - With heavy quantization
6. **IDEFICS2-8B** - HuggingFace model

### Priority 3 (Only if Time)
7. **mPLUG-Owl2** (7B with quantization)
8. **TinyGPT-V** - Complex setup

## Summary

**Tested**: 14 models
**Remaining viable**: 8-10 models
**Recommended to test**: 3-5 more models

Given that BLIP-2 is already validated as the best choice and we've tested most major alternatives, further testing would be for:
1. Research completeness
2. Finding a lightweight backup option
3. Discovering unexpectedly good models

The most promising untested models are the smaller ones (Kosmos-2, MiniCPM-V) that might offer good efficiency.