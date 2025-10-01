# Fetal Ultrasound Vision LLM Project: Fine-tuning Proposal

We need to fine-tune a small vision LLM model to interpret fetal ultrasound images. Our goal is to upload fetal images and have the model provide accurate interpretations based on clinical criteria.

## Dataset Overview

I have a comprehensive dataset of 12 fetal organ types:

- Aorta
- Cervical region
- Abdomen
- Femur
- Standard and non-standard NT
- Cervix
- Thorax
- Pubic symphysis/fetal head
- Trans-cerebellum, trans-thalamic, and trans-ventricular views

This dataset is available at [Fetal-Ultrasound], though it currently lacks annotations (please download it on your machine do not make change on the uploaded file)

## Annotation Status

A sonographer has annotated 50 images for each of five organ categories:

- Abdomen
- Femur
- Non-standard NT
- Standard NT
- Thorax

For each image, the sonographer answered 8 standardized questions covering:

1. Anatomical structures identification
2. Fetal orientation
3. Plane evaluation
4. Biometric measurements
5. Gestational age estimation
6. Image quality assessment
7. Normality/abnormality determination
8. Clinical recommendations

These annotations are available in Excel format at [Fetal-Ultrasound-labeled]. (please download it on your machine do not make change on the uploaded file)

## Technical Challenge

Our primary challenge is achieving accurate interpretations with limited annotated data. While doctors can distinguish subtle differences between images of the same organ type, our model needs to learn these distinctions with only 50 examples per category.

## Recommended Approach for Limited Data

Based on recent research in medical imaging AI, I recommend implementing:

1. Hierarchical contrastive learning techniques to maximize learning efficiency from limited examples by aligning visual and text features at multiple representation levels.

2. Data augmentation strategies specific to ultrasound imaging, including controlled transformations that preserve clinical relevance.

3. Transfer learning from larger medical imaging models, with selective fine-tuning of critical layers.

## Recommended Framework

After evaluating the options, Florence-2 Vision Language Model is the most suitable framework for our project because:

1. It's specifically designed for vision-language tasks and has demonstrated strong performance on medical imaging with limited data.

2. It offers efficient fine-tuning capabilities through selective layer training and parameter-efficient techniques like LoRA (Low-Rank Adaptation) using the peft library.

3. It provides notebook examples for medical image fine-tuning, simplifying implementation.

4. Recent research shows Florence-2 outperforms other models in zero-shot and few-shot learning scenarios for similar medical imaging tasks.

Implementation can begin with the repository at: https://github.com/anyantudre/Florence-2-Vision-Language-Model

## Some helpful videos

### How to Run Microsoft Florence-2 with Ultralytics for Visual Reasoning, OCR & Object Detection Tasks
Learn how to use Microsoft's Florence-2 model with Ultralytics utilities for vision-language tasks. In this tutorial, you'll walk through the complete workflow...  
www.youtube.com

### Florence-2: Fine-tune Microsoft's Multimodal Model
Learn how to fine-tune Microsoft's Florence-2, a powerful open-source Vision Language Model, for custom object detection tasks. This in-depth tutorial guides you through setting up your environment in Google Colab, preparing datasets, and optimizing the model using LoRA.  
www.youtube.com

### How to Fine-tune Florence 2: The Best Small Vision Model

You can take your time for exploration, is not necessary to follow this plan 100%, we are welcoming any new idea.

Please let me know if you have any questions.
