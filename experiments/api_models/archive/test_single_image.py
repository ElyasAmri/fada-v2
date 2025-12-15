"""
Test Gemini 3 and Grok 4 APIs on a single ultrasound image with 8 clinical questions
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import time
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from PIL import Image


# The 8 clinical questions
QUESTIONS = [
    "Anatomical Structures Identification: Identify and describe all anatomical structures visible in the image.",
    "Fetal Orientation: Determine the orientation of the fetus based on the image (e.g., head up/down, front/back view).",
    "Plane Evaluation: Assess if the image is taken at a standard diagnostic plane and describe its diagnostic relevance.",
    "Biometric Measurements: Identify any measurable biometric parameters (e.g., femur length, head circumference) from the image.",
    "Gestational Age: Estimate the gestational age of the fetus based on the visible features.",
    "Image Quality: Assess the quality of the ultrasound image, mentioning any factors that might affect its interpretation (e.g., clarity, artifacts).",
    "Normality / Abnormality: Determine whether the observed structures appear normal or identify any visible abnormalities or concerns.",
    "Clinical Recommendations: Provide any relevant clinical recommendations or suggested next steps based on your interpretation."
]

QUESTION_SHORT_NAMES = [
    "Q1: Anatomical Structures",
    "Q2: Fetal Orientation",
    "Q3: Plane Evaluation",
    "Q4: Biometric Measurements",
    "Q5: Gestational Age",
    "Q6: Image Quality",
    "Q7: Normality/Abnormality",
    "Q8: Clinical Recommendations"
]


def test_gemini_3(image_path: Path, output_dir: Path):
    """Test Gemini 3 Pro Preview on the image"""
    print("\n" + "="*60)
    print("TESTING GEMINI 3 PRO PREVIEW")
    print("="*60)

    try:
        from src.inference.gemini_vlm import create_gemini_vlm

        # Try different Gemini model variants (2.0-flash confirmed working)
        models_to_try = [
            "gemini-2.0-flash",
            "gemini-2.0-flash-exp",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ]

        image = Image.open(image_path).convert('RGB')
        results = {"model": None, "responses": [], "errors": []}

        for model_name in models_to_try:
            print(f"\n--- Trying model: {model_name} ---")

            try:
                from src.inference.gemini_vlm import GeminiVLM
                vlm = GeminiVLM(model_name=model_name, max_retries=1, retry_delay=0.5)
                vlm.load()
                results["model"] = model_name

                for i, question in enumerate(QUESTIONS):
                    print(f"\n{QUESTION_SHORT_NAMES[i]}...")
                    start = time.time()

                    try:
                        response = vlm.answer_question(image, question)
                        elapsed = time.time() - start

                        results["responses"].append({
                            "question": QUESTION_SHORT_NAMES[i],
                            "response": response,
                            "time": elapsed
                        })

                        print(f"  Response ({elapsed:.1f}s): {response[:200]}...")

                    except Exception as e:
                        elapsed = time.time() - start
                        error_msg = str(e)
                        results["errors"].append({
                            "question": QUESTION_SHORT_NAMES[i],
                            "error": error_msg,
                            "time": elapsed
                        })
                        print(f"  ERROR ({elapsed:.1f}s): {error_msg[:100]}")

                vlm.unload()

                # Check if we got real responses (not just "No response generated")
                real_responses = [r for r in results["responses"] if "No response generated" not in r.get("response", "")]
                if real_responses:
                    print(f"\n  Model {model_name} produced {len(real_responses)} real responses!")
                    break
                else:
                    print(f"\n  Model {model_name} returned empty responses, trying next...")
                    results["responses"] = []  # Clear and try next model

            except Exception as e:
                print(f"  Model failed: {e}")
                results["errors"].append({"model": model_name, "error": str(e)})
                continue

        # Save results
        output_file = output_dir / f"gemini3_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

        return results

    except Exception as e:
        print(f"Gemini setup failed: {e}")
        return {"error": str(e)}


def test_grok_4(image_path: Path, output_dir: Path):
    """Test Grok 4 on the image"""
    print("\n" + "="*60)
    print("TESTING GROK 4")
    print("="*60)

    try:
        from src.inference.grok_vlm import create_grok_vlm

        # Try different Grok 4 model variants
        models_to_try = [
            "grok-4",
            "grok-4-fast",
            "grok-4-0709",
            "grok-vision-beta",
        ]

        image = Image.open(image_path).convert('RGB')
        results = {"model": None, "responses": [], "errors": []}

        for model_name in models_to_try:
            print(f"\n--- Trying model: {model_name} ---")

            try:
                from src.inference.grok_vlm import GrokVLM
                vlm = GrokVLM(model_name=model_name, max_retries=1, retry_delay=0.5)
                vlm.load()
                results["model"] = model_name

                for i, question in enumerate(QUESTIONS):
                    print(f"\n{QUESTION_SHORT_NAMES[i]}...")
                    start = time.time()

                    try:
                        response = vlm.answer_question(image, question)
                        elapsed = time.time() - start

                        results["responses"].append({
                            "question": QUESTION_SHORT_NAMES[i],
                            "response": response,
                            "time": elapsed
                        })

                        print(f"  Response ({elapsed:.1f}s): {response[:200]}...")

                    except Exception as e:
                        elapsed = time.time() - start
                        error_msg = str(e)
                        results["errors"].append({
                            "question": QUESTION_SHORT_NAMES[i],
                            "error": error_msg,
                            "time": elapsed
                        })
                        print(f"  ERROR ({elapsed:.1f}s): {error_msg[:100]}")

                vlm.unload()

                # If we got responses, we're done
                if results["responses"]:
                    break

            except Exception as e:
                print(f"  Model failed: {e}")
                results["errors"].append({"model": model_name, "error": str(e)})
                continue

        # Save results
        output_file = output_dir / f"grok4_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

        return results

    except Exception as e:
        print(f"Grok setup failed: {e}")
        return {"error": str(e)}


def main():
    """Main entry point"""
    print("API VLM Single Image Test - Gemini 3 and Grok 4")
    print("="*60)

    # Test image
    image_path = project_root / "data" / "Fetal Ultrasound" / "Abodomen" / "Abodomen_001.png"

    if not image_path.exists():
        print(f"Test image not found: {image_path}")
        return

    print(f"Test image: {image_path}")

    # Output directory
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    # Test both models
    gemini_results = test_gemini_3(image_path, output_dir)
    grok_results = test_grok_4(image_path, output_dir)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print("\nGemini 3:")
    if "error" in gemini_results:
        print(f"  FAILED: {gemini_results['error']}")
    else:
        print(f"  Model: {gemini_results.get('model', 'N/A')}")
        print(f"  Successful responses: {len(gemini_results.get('responses', []))}/8")
        print(f"  Errors: {len(gemini_results.get('errors', []))}")

    print("\nGrok 4:")
    if "error" in grok_results:
        print(f"  FAILED: {grok_results['error']}")
    else:
        print(f"  Model: {grok_results.get('model', 'N/A')}")
        print(f"  Successful responses: {len(grok_results.get('responses', []))}/8")
        print(f"  Errors: {len(grok_results.get('errors', []))}")


if __name__ == "__main__":
    main()
