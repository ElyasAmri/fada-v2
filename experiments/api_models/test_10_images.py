"""
Test Gemini 3 and Grok 4 APIs on 10 ultrasound images with 8 clinical questions each
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
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(project_root / '.env.local')

# 8 clinical questions
QUESTIONS = [
    "Anatomical Structures Identification: Identify and describe all anatomical structures visible in the image.",
    "Fetal Orientation: Determine the orientation of the fetus based on the image.",
    "Plane Evaluation: Assess if the image is taken at a standard diagnostic plane.",
    "Biometric Measurements: Identify any measurable biometric parameters from the image.",
    "Gestational Age: Estimate the gestational age of the fetus based on visible features.",
    "Image Quality: Assess the quality of the ultrasound image.",
    "Normality / Abnormality: Determine whether the observed structures appear normal or identify any abnormalities.",
    "Clinical Recommendations: Provide relevant clinical recommendations based on your interpretation."
]

Q_SHORT = ["Q1:Anatomy", "Q2:Orientation", "Q3:Plane", "Q4:Biometry",
           "Q5:Age", "Q6:Quality", "Q7:Normal/Abnormal", "Q8:Recommendations"]


def get_test_images(n=10):
    """Get n test images from different categories"""
    data_dir = project_root / "data" / "Fetal Ultrasound"
    categories = ["Abodomen", "Femur", "Thorax", "Aorta", "Trans-cerebellum"]

    images = []
    for cat in categories:
        cat_dir = data_dir / cat
        if cat_dir.exists():
            for img in sorted(cat_dir.glob("*.png"))[:2]:  # 2 per category = 10 total
                images.append({"path": img, "category": cat, "name": img.name})
                if len(images) >= n:
                    return images
    return images


def extract_gemini_text(response):
    """Extract text from Gemini response, handling thinking mode"""
    try:
        return response.text
    except ValueError:
        pass
    # Extract from parts
    if hasattr(response, 'candidates') and response.candidates:
        parts = response.candidates[0].content.parts
        texts = [p.text for p in parts if hasattr(p, 'text') and p.text]
        if texts:
            return "\n".join(texts)
    return "No response"


def test_gemini3(images, output_dir):
    """Test Gemini 3 Pro Preview on all images"""
    print("\n" + "="*70)
    print("GEMINI 3 PRO PREVIEW - 10 Images x 8 Questions")
    print("="*70)

    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    model = genai.GenerativeModel('gemini-3-pro-preview')

    results = {"model": "gemini-3-pro-preview", "images": [], "summary": {}}
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
                text = extract_gemini_text(response)
                elapsed = time.time() - start

                # Simple quality score based on response length
                word_count = len(text.split())
                score = min(1.0, word_count / 100)  # 100+ words = 1.0

                img_result["questions"].append({
                    "q": Q_SHORT[q_idx],
                    "response": text[:500],  # Truncate for storage
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
                print(f"  {Q_SHORT[q_idx]}: ERROR - {str(e)[:50]}")
                total_time += elapsed

        results["images"].append(img_result)

        # Save incrementally
        with open(output_dir / "gemini3_10img_progress.json", 'w') as f:
            json.dump(results, f, indent=2)

    results["summary"] = {
        "total_questions": len(images) * 8,
        "avg_score": round(sum(all_scores) / len(all_scores), 3) if all_scores else 0,
        "total_time": round(total_time, 1),
        "avg_time_per_q": round(total_time / (len(images) * 8), 1)
    }

    # Save final
    output_file = output_dir / f"gemini3_10img_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nGemini 3 Summary: Avg Score={results['summary']['avg_score']:.1%}, "
          f"Total Time={total_time:.0f}s, Avg={total_time/(len(images)*8):.1f}s/question")

    return results


def test_grok4(images, output_dir):
    """Test Grok 4 on all images"""
    print("\n" + "="*70)
    print("GROK 4 - 10 Images x 8 Questions")
    print("="*70)

    client = OpenAI(api_key=os.getenv('XAI_API_KEY'), base_url="https://api.x.ai/v1")

    results = {"model": "grok-4", "images": [], "summary": {}}
    total_time = 0
    all_scores = []

    for idx, img_info in enumerate(images):
        print(f"\n[{idx+1}/10] {img_info['name']} ({img_info['category']})")

        # Convert image to base64
        image = Image.open(img_info['path']).convert('RGB')
        import io, base64
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        img_url = f"data:image/jpeg;base64,{img_b64}"

        img_result = {"image": img_info['name'], "category": img_info['category'], "questions": []}

        for q_idx, question in enumerate(QUESTIONS):
            start = time.time()
            try:
                response = client.chat.completions.create(
                    model="grok-4",
                    messages=[
                        {"role": "system", "content": "You are a medical imaging expert analyzing fetal ultrasound images."},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": img_url}},
                            {"type": "text", "text": question}
                        ]}
                    ],
                    max_tokens=1024
                )
                text = response.choices[0].message.content if response.choices else "No response"
                elapsed = time.time() - start

                word_count = len(text.split())
                score = min(1.0, word_count / 100)

                img_result["questions"].append({
                    "q": Q_SHORT[q_idx],
                    "response": text[:500],
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
                print(f"  {Q_SHORT[q_idx]}: ERROR - {str(e)[:50]}")
                total_time += elapsed

        results["images"].append(img_result)

        # Save incrementally
        with open(output_dir / "grok4_10img_progress.json", 'w') as f:
            json.dump(results, f, indent=2)

    results["summary"] = {
        "total_questions": len(images) * 8,
        "avg_score": round(sum(all_scores) / len(all_scores), 3) if all_scores else 0,
        "total_time": round(total_time, 1),
        "avg_time_per_q": round(total_time / (len(images) * 8), 1)
    }

    output_file = output_dir / f"grok4_10img_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nGrok 4 Summary: Avg Score={results['summary']['avg_score']:.1%}, "
          f"Total Time={total_time:.0f}s, Avg={total_time/(len(images)*8):.1f}s/question")

    return results


def main():
    print("API VLM Analysis - 10 Images x 8 Questions")
    print("="*70)

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    images = get_test_images(10)
    print(f"Test images: {len(images)}")
    for img in images:
        print(f"  - {img['name']} ({img['category']})")

    # Run tests
    gemini_results = test_gemini3(images, output_dir)
    grok_results = test_grok4(images, output_dir)

    # Final comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"\n{'Model':<25} {'Avg Score':<12} {'Total Time':<12} {'Avg/Question'}")
    print("-"*60)
    print(f"{'Gemini 3 Pro Preview':<25} {gemini_results['summary']['avg_score']:.1%}        "
          f"{gemini_results['summary']['total_time']:.0f}s         "
          f"{gemini_results['summary']['avg_time_per_q']:.1f}s")
    print(f"{'Grok 4':<25} {grok_results['summary']['avg_score']:.1%}        "
          f"{grok_results['summary']['total_time']:.0f}s         "
          f"{grok_results['summary']['avg_time_per_q']:.1f}s")


if __name__ == "__main__":
    main()
