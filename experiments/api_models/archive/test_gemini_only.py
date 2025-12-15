"""
Quick test to find working Gemini model for medical image analysis
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from PIL import Image

# Single test question
TEST_QUESTION = "What anatomical structures are visible in this fetal ultrasound image? List the main structures you can identify."


def main():
    """Test different Gemini models"""
    print("Gemini Model Quick Test")
    print("="*60)

    image_path = project_root / "data" / "Fetal Ultrasound" / "Abodomen" / "Abodomen_001.png"
    image = Image.open(image_path).convert('RGB')
    print(f"Test image: {image_path.name}")

    # Models to try
    models = [
        "gemini-2.5-pro-preview-06-05",
        "gemini-2.5-flash-preview-05-20",
        "gemini-2.5-pro-exp-03-25",
        "gemini-2.0-flash",
        "gemini-2.0-flash-exp",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ]

    from src.inference.gemini_vlm import GeminiVLM

    for model_name in models:
        print(f"\n--- Testing: {model_name} ---")

        try:
            vlm = GeminiVLM(model_name=model_name, max_retries=1)
            vlm.load()

            start = time.time()
            response = vlm.answer_question(image, TEST_QUESTION)
            elapsed = time.time() - start

            vlm.unload()

            if "No response generated" in response or "blocked" in response.lower():
                print(f"  BLOCKED ({elapsed:.1f}s): {response[:100]}")
            else:
                print(f"  SUCCESS ({elapsed:.1f}s)!")
                print(f"  Response preview: {response[:300]}...")
                print(f"\n>>> WORKING MODEL FOUND: {model_name}")
                return model_name

        except Exception as e:
            print(f"  ERROR: {str(e)[:100]}")

    print("\nNo working Gemini model found!")
    return None


if __name__ == "__main__":
    main()
