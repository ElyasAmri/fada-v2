"""
Optimized Gemini 3 Pro Preview test with robust response extraction
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import time
import json
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv(project_root / '.env.local')

QUESTIONS = [
    "Identify and describe all anatomical structures visible in this fetal ultrasound image.",
    "Determine the orientation of the fetus (head up/down, front/back view).",
    "Assess if this is a standard diagnostic plane and describe its diagnostic relevance.",
    "Identify any measurable biometric parameters from this image.",
    "Estimate the gestational age based on visible features.",
    "Assess the quality of this ultrasound image.",
    "Determine whether structures appear normal or identify any abnormalities.",
    "Provide clinical recommendations based on your interpretation."
]

Q_SHORT = ["Q1:Anatomy", "Q2:Orientation", "Q3:Plane", "Q4:Biometry",
           "Q5:Age", "Q6:Quality", "Q7:Normal/Abnormal", "Q8:Recommendations"]


def extract_response_text(response):
    """
    Robust extraction of text from Gemini 3 response.
    Handles thinking mode responses where response.text may fail.
    """
    # Method 1: Direct text accessor (fastest when it works)
    try:
        if response.text:
            return response.text
    except (ValueError, AttributeError):
        pass

    # Method 2: Extract from candidates -> content -> parts
    try:
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                texts = []
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        texts.append(part.text)
                if texts:
                    return "\n".join(texts)
    except (AttributeError, IndexError):
        pass

    # Method 3: Check for blocked content
    try:
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            return f"[BLOCKED: {response.prompt_feedback.block_reason}]"
    except AttributeError:
        pass

    return "[NO RESPONSE]"


def get_test_images(n=10):
    data_dir = project_root / "data" / "Fetal Ultrasound"
    categories = ["Abodomen", "Femur", "Thorax", "Aorta", "Trans-cerebellum"]
    images = []
    for cat in categories:
        cat_dir = data_dir / cat
        if cat_dir.exists():
            for img in sorted(cat_dir.glob("*.png"))[:2]:
                images.append({"path": img, "category": cat, "name": img.name})
                if len(images) >= n:
                    return images
    return images


def main():
    print("GEMINI 3 PRO PREVIEW - Optimized Test")
    print("="*70)

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    model = genai.GenerativeModel('gemini-3-pro-preview')

    images = get_test_images(10)
    print(f"Test images: {len(images)}")

    results = {"model": "gemini-3-pro-preview", "images": [], "summary": {}}
    total_time = 0
    all_word_counts = []
    failed_count = 0

    for idx, img_info in enumerate(images):
        print(f"\n[{idx+1}/10] {img_info['name']} ({img_info['category']})")
        image = Image.open(img_info['path'])
        img_result = {"image": img_info['name'], "category": img_info['category'], "questions": []}

        for q_idx, question in enumerate(QUESTIONS):
            start = time.time()
            try:
                response = model.generate_content([image, question])
                text = extract_response_text(response)
                elapsed = time.time() - start

                word_count = len(text.split())

                if text.startswith("["):  # [NO RESPONSE] or [BLOCKED]
                    failed_count += 1
                    status = "FAIL"
                else:
                    all_word_counts.append(word_count)
                    status = "OK"

                img_result["questions"].append({
                    "q": Q_SHORT[q_idx],
                    "words": word_count,
                    "time": round(elapsed, 1),
                    "status": status
                })
                print(f"  {Q_SHORT[q_idx]}: {word_count} words, {elapsed:.1f}s [{status}]")
                total_time += elapsed

            except Exception as e:
                elapsed = time.time() - start
                failed_count += 1
                img_result["questions"].append({
                    "q": Q_SHORT[q_idx],
                    "error": str(e)[:100],
                    "time": round(elapsed, 1),
                    "status": "ERROR"
                })
                print(f"  {Q_SHORT[q_idx]}: ERROR - {str(e)[:50]}")
                total_time += elapsed

        results["images"].append(img_result)

    # Calculate summary
    total_questions = len(images) * 8
    success_count = total_questions - failed_count
    avg_words = sum(all_word_counts) / len(all_word_counts) if all_word_counts else 0

    results["summary"] = {
        "total_questions": total_questions,
        "successful": success_count,
        "failed": failed_count,
        "success_rate": round(success_count / total_questions, 3),
        "avg_words": round(avg_words, 1),
        "total_time": round(total_time, 1),
        "avg_time_per_q": round(total_time / total_questions, 1)
    }

    output_file = output_dir / f"gemini3_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Model: gemini-3-pro-preview")
    print(f"Success Rate: {success_count}/{total_questions} ({results['summary']['success_rate']:.1%})")
    print(f"Avg Words: {avg_words:.0f}")
    print(f"Total Time: {total_time:.0f}s")
    print(f"Avg per Question: {total_time/total_questions:.1f}s")
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
