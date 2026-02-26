# FADA Android App

Native Android application for offline fetal ultrasound classification using TensorFlow Lite.

**DISCLAIMER: This is a research prototype for educational purposes only. NOT intended for clinical use or medical diagnosis.**

## Features

- Offline on-device inference (no internet required)
- Camera capture for ultrasound images
- Gallery picker for existing images
- 12-class ultrasound view classification
- Material 3 UI with dark mode support
- GPU acceleration (when available)

## Prerequisites

1. **Android Studio** (Hedgehog 2023.1.1 or later)
2. **JDK 17** or later
3. **Android SDK 34**
4. **TFLite model file** (see Model Conversion below)

## Project Structure

```
app/android/
|-- build.gradle.kts           # Root build config
|-- settings.gradle.kts        # Project settings
|-- gradle.properties          # Gradle properties
|-- app/
|   |-- build.gradle.kts       # App module config
|   |-- proguard-rules.pro     # ProGuard rules
|   +-- src/main/
|       |-- AndroidManifest.xml
|       |-- assets/
|       |   +-- fada_classifier.tflite  # Model file (generate first!)
|       |-- java/com/fada/ultrasound/
|       |   |-- MainActivity.kt
|       |   |-- FADAApplication.kt
|       |   |-- data/
|       |   |   +-- UltrasoundClasses.kt
|       |   |-- inference/
|       |   |   |-- TFLiteClassifier.kt
|       |   |   |-- ImagePreprocessor.kt
|       |   |   +-- ClassificationResult.kt
|       |   |-- viewmodel/
|       |   |   +-- InferenceViewModel.kt
|       |   +-- ui/
|       |       |-- Navigation.kt
|       |       |-- MainScreen.kt
|       |       |-- CameraScreen.kt
|       |       |-- ResultsScreen.kt
|       |       +-- theme/
|       |           +-- Theme.kt
|       +-- res/
|           |-- values/
|           |-- drawable/
|           +-- mipmap-anydpi-v26/
+-- gradle/
    |-- libs.versions.toml     # Version catalog
    +-- wrapper/
        +-- gradle-wrapper.properties
```

## Model Conversion

Before building the app, you need to convert the PyTorch model to TFLite format:

```bash
# From project root
cd fada-v3

# Activate virtual environment
source venv/Scripts/activate  # Git Bash
# or
./venv/Scripts/activate       # PowerShell

# Run conversion script
python scripts/export_model_tflite.py

# This creates: app/android/app/src/main/assets/fada_classifier.tflite
```

### Conversion Options

```bash
# Custom checkpoint path
python scripts/export_model_tflite.py --checkpoint path/to/model.pth

# Different quantization (none, float16, int8)
python scripts/export_model_tflite.py --quantize int8

# Skip validation
python scripts/export_model_tflite.py --skip-validation
```

### Requirements for Conversion

```bash
pip install onnx onnx-tf tensorflow
```

## Building the App

### Using Android Studio

1. Open `app/android/` in Android Studio
2. Wait for Gradle sync to complete
3. Ensure TFLite model is in `app/src/main/assets/`
4. Build > Make Project (or Ctrl+F9)
5. Run > Run 'app' (or Shift+F10)

### Using Command Line

```bash
cd app/android

# Debug build
./gradlew assembleDebug
# Output: app/build/outputs/apk/debug/app-debug.apk

# Release build (requires signing config)
./gradlew assembleRelease
```

## 12-Class Classification

The model classifies ultrasound images into these categories:

| Index | Class Name | Display Name |
|-------|------------|--------------|
| 0 | Abodomen | Abdomen |
| 1 | Aorta | Aortic Arch |
| 2 | Cervical | Cervical View |
| 3 | Cervix | Cervix |
| 4 | Femur | Femur |
| 5 | Non_standard_NT | Non-standard NT |
| 6 | Public_Symphysis_fetal_head | Fetal Head Position |
| 7 | Standard_NT | Standard NT |
| 8 | Thorax | Thorax |
| 9 | Trans-cerebellum | Transcerebellar Plane |
| 10 | Trans-thalamic | Transthalamic Plane |
| 11 | Trans-ventricular | Transventricular Plane |

## Technical Details

### Model Input

- Size: 224x224 pixels
- Format: RGB (3 channels)
- Normalization: ImageNet statistics
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]

### Model Output

- 12 logits (one per class)
- Softmax applied in post-processing
- Top-5 predictions displayed

### Performance

- Model size: ~11MB (float16 quantized)
- Inference time: ~50-100ms (depends on device)
- GPU acceleration available on supported devices

## Permissions

The app requires:

- **Camera** - For capturing ultrasound images
- **Storage** - For reading images from gallery (Android 12 and below)
- **Media Images** - For reading images (Android 13+)

## Troubleshooting

### Model not loading

1. Verify `fada_classifier.tflite` exists in `assets/`
2. Check Logcat for TFLiteClassifier errors
3. Try rebuilding with clean: `./gradlew clean assembleDebug`

### Camera not working

1. Grant camera permission in device settings
2. Check if camera is used by another app
3. Restart the app

### Slow inference

1. Enable GPU delegation (automatic on supported devices)
2. Use smaller model (int8 quantization)
3. Reduce image preprocessing overhead

## Dependencies

- TensorFlow Lite 2.14.0
- CameraX 1.3.4
- Jetpack Compose (2024.06.00 BOM)
- Material 3
- Navigation Compose 2.7.7
- Coil 2.6.0 (image loading)

## License

Research prototype - see main project LICENSE.
