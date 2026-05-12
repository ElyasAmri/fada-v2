import os

import gradio as gr
import spaces
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


BASE_MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-4-E2B-it")
ADAPTER_REPO = os.environ.get("ADAPTER_REPO", "elyasamri/gemma-4-e2b-fada-adapter")
HF_TOKEN = os.environ.get("HF_TOKEN") or None
SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are FADA, a research prototype for fetal ultrasound image understanding. "
    "Provide concise, useful observations and do not claim clinical certainty.",
)

processor_source = ADAPTER_REPO or BASE_MODEL_ID
processor = AutoProcessor.from_pretrained(processor_source, token=HF_TOKEN, trust_remote_code=True)
base_model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL_ID,
    token=HF_TOKEN,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to("cuda")

if ADAPTER_REPO:
    model = PeftModel.from_pretrained(base_model, ADAPTER_REPO, token=HF_TOKEN).to("cuda")
else:
    model = base_model

model.eval()


def _decode_generated_text(output_ids, input_len: int) -> str:
    raw = processor.decode(output_ids[0][input_len:], skip_special_tokens=False)
    if "<channel|>" in raw:
        raw = raw.split("<channel|>", 1)[1]
    return raw.strip()


@spaces.GPU(duration=180)
def analyze_image(
    image: Image.Image | None,
    question: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    prompt = question.strip()
    if not prompt:
        raise gr.Error("Enter a question or instruction.")

    user_content: list[dict[str, object]] = []
    processor_kwargs: dict[str, object] = {}

    if image is not None:
        user_content.append({"type": "image"})
        processor_kwargs["images"] = [image.convert("RGB")]

    user_content.append({"type": "text", "text": prompt})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = processor(
        text=text,
        return_tensors="pt",
        **processor_kwargs,
    ).to(model.device)

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.bfloat16)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    return _decode_generated_text(output_ids, int(inputs["input_ids"].shape[1]))


with gr.Blocks(title="FADA Gemma 4 ZeroGPU") as demo:
    gr.Markdown(
        "# FADA Gemma 4 ZeroGPU\n"
        "Research prototype for fetal ultrasound image understanding on Hugging Face Spaces ZeroGPU.\n\n"
        f"Base model: `{BASE_MODEL_ID}`  \n"
        f"Adapter: `{ADAPTER_REPO or 'none'}`"
    )

    with gr.Row():
        image_input = gr.Image(
            type="pil",
            label="Ultrasound image",
            sources=["upload", "clipboard"],
        )
        response_output = gr.Textbox(label="Response", lines=18)

    question_input = gr.Textbox(
        label="Question",
        lines=3,
        value="Describe the visible anatomy and key observations in this ultrasound image.",
    )

    with gr.Row():
        max_tokens_input = gr.Slider(
            minimum=64,
            maximum=1024,
            value=256,
            step=32,
            label="Max new tokens",
        )
        temperature_input = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.2,
            step=0.1,
            label="Temperature",
        )

    submit_button = gr.Button("Analyze", variant="primary")
    submit_button.click(
        fn=analyze_image,
        inputs=[image_input, question_input, max_tokens_input, temperature_input],
        outputs=response_output,
    )


if __name__ == "__main__":
    demo.launch()
