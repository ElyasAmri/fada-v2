"""
Test Gemini 3 Pro Preview with LOW thinking level via REST API
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import time
import json
import requests
import base64
import io
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from PIL import Image
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


def encode_image(image_path):
    """Encode image to base64"""
    img = Image.open(image_path).convert('RGB')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=95)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def query_gemini3_low(api_key, img_b64, question, timeout=60):
    """Query Gemini 3 with LOW thinking level"""
    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-preview:generateContent?key={api_key}'

    payload = {
        'contents': [{
            'parts': [
                {'inline_data': {'mime_type': 'image/jpeg', 'data': img_b64}},
                {'text': question}
            ]
        }],
        'generationConfig': {
            'temperature': 0.4,
            'maxOutputTokens': 1024,
            'thinkingConfig': {
                'thinkingLevel': 'LOW'
            }
        }
    }

    resp = requests.post(url, json=payload, timeout=timeout)

    if resp.status_code == 200:
        data = resp.json()
        if 'candidates' in data and data['candidates']:
            parts = data['candidates'][0].get('content', {}).get('parts', [])
            texts = [p.get('text', '') for p in parts if 'text' in p]
            return '\n'.join(texts) if texts else '[NO TEXT]'

    return f'[ERROR {resp.status_code}]: {resp.text[:100]}'


def main():
    print("GEMINI 3 PRO PREVIEW - LOW Thinking Mode (REST API)")
    print("="*70)

    api_key = os.getenv('GEMINI_API_KEY')
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    images = get_test_images(10)
    print(f"Test images: {len(images)}")

    results = {"model": "gemini-3-pro-preview (LOW thinking)", "images": [], "summary": {}}
    total_time = 0
    all_word_counts = []
    failed_count = 0

    for idx, img_info in enumerate(images):
        print(f"\n[{idx+1}/10] {img_info['name']} ({img_info['category']})")
        img_b64 = encode_image(img_info['path'])
        img_result = {"image": img_info['name'], "category": img_info['category'], "questions": []}

        for q_idx, question in enumerate(QUESTIONS):
            start = time.time()
            try:
                text = query_gemini3_low(api_key, img_b64, question)
                elapsed = time.time() - start

                word_count = len(text.split())

                if text.startswith('['):
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
                print(f"  {Q_SHORT[q_idx]}: ERROR ({elapsed:.1f}s) - {str(e)[:50]}")
                total_time += elapsed

        results["images"].append(img_result)

        # Save progress
        with open(output_dir / "gemini3_low_progress.json", 'w') as f:
            json.dump(results, f, indent=2)

    # Summary
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

    output_file = output_dir / f"gemini3_low_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Model: gemini-3-pro-preview (LOW thinking)")
    print(f"Success Rate: {success_count}/{total_questions} ({results['summary']['success_rate']:.1%})")
    print(f"Avg Words: {avg_words:.0f}")
    print(f"Total Time: {total_time:.0f}s")
    print(f"Avg per Question: {total_time/total_questions:.1f}s")
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
