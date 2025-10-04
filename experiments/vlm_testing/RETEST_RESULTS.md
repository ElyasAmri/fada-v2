# Failed Models Retest Results
**Date**: October 4, 2025
**Reason**: MedGemma was incorrectly marked as failed - systematic retest of all failed models

---

## Models Successfully Recovered ✅

### 1. MedGemma-4B ✅
- **Status**: NOW WORKING
- **Performance**: ~65% (High Tier)
- **Memory**: 3.2GB (4-bit)
- **Fix**: Required `AutoModelForCausalLM` + chat template format
- **Code**:
```python
from transformers import AutoProcessor, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "google/medgemma-4b-it",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    load_in_4bit=True
)

messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": question}
]}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
```

### 2. Qwen2.5-VL-3B ✅
- **Status**: NOW WORKING
- **Performance**: TBD (needs accuracy testing)
- **Memory**: 2.35GB (4-bit)
- **Previous Error**: AssertionError on inference
- **Fix**: Required `Qwen2_5_VLForConditionalGeneration` + chat template
- **Code**:
```python
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    load_in_4bit=True
)

# Use chat template format
messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": question}
]}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
```

---

## Models Still Broken ❌

### 3. DeepSeek-VL-1.3B ❌
- **Status**: STILL BROKEN
- **Error**: `The checkpoint you are trying to load has model type 'multi_modality' but Transformers does not recognize this architecture`
- **Reason**: Custom architecture not supported by transformers library
- **Attempted Fixes**:
  - AutoModel with trust_remote_code
  - AutoModelForCausalLM
- **Conclusion**: Requires custom loading code from DeepSeek repo

### 4. CheXagent-8b ⚠️
- **Status**: LOADS BUT USELESS
- **Memory**: 4.58GB (4-bit)
- **Issue**: Gives empty/minimal responses (0% accuracy)
- **Reason**: Trained exclusively on chest X-rays, severe domain mismatch with fetal ultrasound
- **Conclusion**: Technically "works" but provides no value for this task

---

## Models Not Retested (Skipped)

The following models were not retested due to known structural issues that cannot be fixed with updated transformers:

### 5. Qwen-VL-Chat ❌
- **Reason**: Visual encoder compatibility issues, superseded by Qwen2-VL and Qwen2.5-VL

### 6. mPLUG-Owl2 ❌
- **Reason**: Custom architecture `mplug_owl2` not in transformers

### 7. TinyGPT-V ❌
- **Reason**: Missing `model_type` in config.json

### 8. CogVLM-chat-hf ❌
- **Reason**: Custom `CogVLMConfig` not recognized by transformers

### 9. CogAgent-chat-hf ❌
- **Reason**: Custom `CogAgentConfig` not recognized by transformers

### 10. Fuyu-8B ❌
- **Reason**: Exceeds 8GB VRAM even with 4-bit quantization

---

## Summary Statistics

**Total Failed Models Retested**: 4/10
**Successfully Recovered**: 2 (MedGemma-4B, Qwen2.5-VL-3B)
**Still Broken**: 2 (DeepSeek-VL-1.3B, CheXagent-8b)
**Skipped** (known unfixable): 6

**Impact**: 2 additional working models added to test suite

---

## Recommendations

1. **Add to Model Selection UI**:
   - MedGemma-4B (medical-domain model, 65% accuracy)
   - Qwen2.5-VL-3B (efficient 2.35GB model)

2. **Update VLM Interface**:
   - Add support for `Qwen2_5_VLForConditionalGeneration`
   - Document chat template requirements

3. **Documentation Updates**:
   - Move MedGemma-4B from "Failed" to "High Tier (60-79%)"
   - Move Qwen2.5-VL-3B from "Failed" to appropriate tier after accuracy testing
   - Keep CheXagent-8b in failed (domain mismatch = unusable)

4. **Future Work**:
   - Run full accuracy test on Qwen2.5-VL-3B with all 8 questions
   - Compare performance against Qwen2-VL-2B (83.3%)
   - Test Qwen2.5-VL-7B variant if VRAM budget allows
