"""
Test Gemini 2.0 Flash (no thinking mode) on 10 ultrasound images
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
    print("GEMINI 2.0 FLASH - 10 Images x 8 Questions (No Thinking Mode)")
    print("="*70)

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    model = genai.GenerativeModel('gemini-2.0-flash')

    images = get_test_images(10)
    print(f"Test images: {len(images)}")

    results = {"model": "gemini-2.0-flash", "images": [], "summary": {}}
    total_time = 0
    all_scores = []

    for idx, img_info in enumerate(images):
        print(f"\n[{idx+1}/10] {img_info['name']} ({img_info['category']})")
        image = Image.open(img_info['path'])
        img_result = {"image": img_info['name'], "category": img_info['category'], "questions": []}

        for q_idx, question in enumerate(QUESTIONS):
            start = time.time()
            try:
                response = model.generate_content([image, question])
                text = response.text if response.text else "No response"
                elapsed = time.time() - start

                word_count = len(text.split())
                score = min(1.0, word_count / 100)

                img_result["questions"].append({
                    "q": Q_SHORT[q_idx],
                    "response": text[:300],
                    "words": word_count,
                    "time": round(elapsed, 1),
                    "score": round(score, 2)
                })
                all_scores.append(score)
                print(f"  {Q_SHORT[q_idx]}: {word_count} words, {elapsed:.1f}s")
                total_time += elapsed

            except Exception as e:
                elapsed = time.time() - start
                img_result["questions"].append({
                    "q": Q_SHORT[q_idx],
                    "error": str(e)[:100],
                    "time": round(elapsed, 1),
                    "score": 0
                })
                all_scores.append(0)
                print(f"  {Q_SHORT[q_idx]}: ERROR - {str(e)[:50]}")
                total_time += elapsed

        results["images"].append(img_result)

    results["summary"] = {
        "total_questions": len(images) * 8,
        "avg_score": round(sum(all_scores) / len(all_scores), 3) if all_scores else 0,
        "total_time": round(total_time, 1),
        "avg_time_per_q": round(total_time / (len(images) * 8), 1)
    }

    output_file = output_dir / f"gemini2flash_10img_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Model: gemini-2.0-flash")
    print(f"Score: {results['summary']['avg_score']:.1%}")
    print(f"Total Time: {total_time:.0f}s")
    print(f"Avg per Question: {total_time/(len(images)*8):.1f}s")
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
