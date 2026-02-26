package com.fada.ultrasound.viewmodel

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.fada.ultrasound.inference.ClassificationResult
import com.fada.ultrasound.inference.TFLiteClassifier
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.InputStream

/**
 * ViewModel for managing model inference state.
 *
 * DISCLAIMER: This is a research prototype for educational purposes only.
 * NOT intended for clinical use or medical diagnosis.
 */
class InferenceViewModel(application: Application) : AndroidViewModel(application) {

    private val classifier = TFLiteClassifier(application.applicationContext)

    // UI state
    private val _uiState = MutableStateFlow<InferenceUiState>(InferenceUiState.Idle)
    val uiState: StateFlow<InferenceUiState> = _uiState.asStateFlow()

    // Selected image
    private val _selectedImage = MutableStateFlow<Bitmap?>(null)
    val selectedImage: StateFlow<Bitmap?> = _selectedImage.asStateFlow()

    // Classification result
    private val _classificationResult = MutableStateFlow<ClassificationResult?>(null)
    val classificationResult: StateFlow<ClassificationResult?> = _classificationResult.asStateFlow()

    // Model initialization state
    private val _isModelReady = MutableStateFlow(false)
    val isModelReady: StateFlow<Boolean> = _isModelReady.asStateFlow()

    init {
        initializeModel()
    }

    /**
     * Initialize the TFLite model in the background.
     */
    private fun initializeModel() {
        viewModelScope.launch(Dispatchers.IO) {
            _uiState.value = InferenceUiState.Loading("Initializing model...")
            val success = classifier.initialize()
            _isModelReady.value = success

            _uiState.value = if (success) {
                InferenceUiState.Idle
            } else {
                InferenceUiState.Error("Failed to initialize model")
            }
        }
    }

    /**
     * Load an image from URI (gallery picker).
     */
    fun loadImageFromUri(uri: Uri) {
        viewModelScope.launch(Dispatchers.IO) {
            _uiState.value = InferenceUiState.Loading("Loading image...")

            try {
                val inputStream: InputStream? = getApplication<Application>()
                    .contentResolver
                    .openInputStream(uri)

                val bitmap = BitmapFactory.decodeStream(inputStream)
                inputStream?.close()

                if (bitmap != null) {
                    withContext(Dispatchers.Main) {
                        _selectedImage.value = bitmap
                        _classificationResult.value = null
                        _uiState.value = InferenceUiState.ImageLoaded
                    }
                } else {
                    _uiState.value = InferenceUiState.Error("Failed to decode image")
                }
            } catch (e: Exception) {
                _uiState.value = InferenceUiState.Error("Failed to load image: ${e.message}")
            }
        }
    }

    /**
     * Set a captured image from camera.
     */
    fun setCapturedImage(bitmap: Bitmap) {
        _selectedImage.value = bitmap
        _classificationResult.value = null
        _uiState.value = InferenceUiState.ImageLoaded
    }

    /**
     * Run classification on the currently selected image.
     */
    fun classifyCurrentImage() {
        val bitmap = _selectedImage.value
        if (bitmap == null) {
            _uiState.value = InferenceUiState.Error("No image selected")
            return
        }

        if (!_isModelReady.value) {
            _uiState.value = InferenceUiState.Error("Model not ready")
            return
        }

        viewModelScope.launch(Dispatchers.Default) {
            _uiState.value = InferenceUiState.Classifying

            val result = classifier.classify(bitmap)

            withContext(Dispatchers.Main) {
                if (result != null) {
                    _classificationResult.value = result
                    _uiState.value = InferenceUiState.ClassificationComplete(result)
                } else {
                    _uiState.value = InferenceUiState.Error("Classification failed")
                }
            }
        }
    }

    /**
     * Classify an image directly from a bitmap.
     */
    fun classifyBitmap(bitmap: Bitmap) {
        _selectedImage.value = bitmap

        if (!_isModelReady.value) {
            _uiState.value = InferenceUiState.Error("Model not ready")
            return
        }

        viewModelScope.launch(Dispatchers.Default) {
            _uiState.value = InferenceUiState.Classifying

            val result = classifier.classify(bitmap)

            withContext(Dispatchers.Main) {
                if (result != null) {
                    _classificationResult.value = result
                    _uiState.value = InferenceUiState.ClassificationComplete(result)
                } else {
                    _uiState.value = InferenceUiState.Error("Classification failed")
                }
            }
        }
    }

    /**
     * Clear the current selection and results.
     */
    fun clearSelection() {
        _selectedImage.value?.recycle()
        _selectedImage.value = null
        _classificationResult.value = null
        _uiState.value = InferenceUiState.Idle
    }

    /**
     * Get model information for debugging.
     */
    fun getModelInfo(): String {
        return classifier.getModelInfo()
    }

    override fun onCleared() {
        super.onCleared()
        classifier.close()
        _selectedImage.value?.recycle()
    }
}

/**
 * UI state for inference operations.
 */
sealed class InferenceUiState {
    data object Idle : InferenceUiState()
    data class Loading(val message: String) : InferenceUiState()
    data object ImageLoaded : InferenceUiState()
    data object Classifying : InferenceUiState()
    data class ClassificationComplete(val result: ClassificationResult) : InferenceUiState()
    data class Error(val message: String) : InferenceUiState()
}
