"""Convert Windows paths to Linux relative paths in JSONL files."""
import json

# Convert paths
for split in ['train', 'val']:
    input_file = f'data/vlm_training/gemini_complete_{split}.jsonl'
    output_file = f'experiments/fine_tuning/vastai_upload/gemini_complete_{split}.jsonl'

    count = 0
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line)
            # Convert Windows path to relative Linux path
            if 'images' in data:
                new_images = []
                for img_path in data['images']:
                    # Extract relative path from 'data/Fetal Ultrasound/...'
                    if 'Fetal Ultrasound' in img_path:
                        # Split on 'Fetal Ultrasound' and take the rest
                        parts = img_path.split('Fetal Ultrasound')
                        if len(parts) > 1:
                            remainder = parts[1].lstrip('/').lstrip('\\')
                            remainder = remainder.replace('\\', '/')
                            rel_path = 'images/' + remainder
                            new_images.append(rel_path)
                        else:
                            new_images.append(img_path)
                    else:
                        new_images.append(img_path)
                data['images'] = new_images
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
            count += 1

    print(f'Converted {split}: {count} samples -> {output_file}')

# Verify
with open('experiments/fine_tuning/vastai_upload/gemini_complete_train.jsonl', 'r') as f:
    sample = json.loads(f.readline())
    print(f'Sample image path: {sample["images"][0]}')
