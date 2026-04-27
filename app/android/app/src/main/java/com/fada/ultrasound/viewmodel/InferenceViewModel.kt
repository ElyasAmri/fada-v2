package com.fada.ultrasound.viewmodel

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.fada.ultrasound.llm.LlmModelOption
import com.fada.ultrasound.llm.LlmModels
import com.fada.ultrasound.llm.LlmResponseClient
import com.fada.ultrasound.llm.LlmResponseGenerator
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.InputStream

class InferenceViewModel(
    application: Application,
    private val responseClient: LlmResponseClient = LlmResponseGenerator
) : AndroidViewModel(application) {

    constructor(application: Application) : this(
        application = application,
        responseClient = LlmResponseGenerator
    )

    // UI state
    private val _uiState = MutableStateFlow<InferenceUiState>(InferenceUiState.Idle)
    val uiState: StateFlow<InferenceUiState> = _uiState.asStateFlow()

    // Selected image
    private val _selectedImage = MutableStateFlow<Bitmap?>(null)
    val selectedImage: StateFlow<Bitmap?> = _selectedImage.asStateFlow()

    private val _selectedModel = MutableStateFlow(LlmModels.default)
    val selectedModel: StateFlow<LlmModelOption> = _selectedModel.asStateFlow()

    private val _modelOptions = MutableStateFlow(LlmModels.options)
    val modelOptions: StateFlow<List<LlmModelOption>> = _modelOptions.asStateFlow()

    private val _llmResponse = MutableStateFlow<LlmResponse?>(null)
    val llmResponse: StateFlow<LlmResponse?> = _llmResponse.asStateFlow()

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
                        _llmResponse.value = null
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
        _llmResponse.value = null
        _uiState.value = InferenceUiState.ImageLoaded
    }

    /**
     * Select the current multimodal model.
     */
    fun selectModel(modelId: String) {
        val model = _modelOptions.value.firstOrNull { it.id == modelId } ?: return
        _selectedModel.value = model
    }

    /**
     * Generate a multimodal response on the selected image.
     */
    fun generateResponseForCurrentImage() {
        val bitmap = _selectedImage.value
        if (bitmap == null) {
            _uiState.value = InferenceUiState.Error("No image selected")
            return
        }

        viewModelScope.launch(Dispatchers.Default) {
            _uiState.value = InferenceUiState.GeneratingResponse
            val startTime = System.currentTimeMillis()
            val selected = _selectedModel.value
            try {
                val output = responseClient.generate(
                    context = getApplication(),
                    model = selected,
                    image = bitmap,
                    onStatus = { status ->
                        _uiState.value = InferenceUiState.Loading(status)
                    }
                )
                val elapsed = System.currentTimeMillis() - startTime

                withContext(Dispatchers.Main) {
                    val response = LlmResponse(
                        modelId = selected.id,
                        modelName = selected.displayName,
                        content = output,
                        latencyMs = elapsed
                    )
                    _llmResponse.value = response
                    _uiState.value = InferenceUiState.ResponseReady(response)
                }
            } catch (e: IllegalStateException) {
                _uiState.value = InferenceUiState.Error(e.message ?: "Model runtime failed")
            } catch (e: RuntimeException) {
                _uiState.value = InferenceUiState.Error(e.message ?: "Inference failed")
            }
        }
    }

    /**
     * Clear the current selection and results.
     */
    fun clearSelection() {
        _selectedImage.value?.recycle()
        _selectedImage.value = null
        _llmResponse.value = null
        _uiState.value = InferenceUiState.Idle
    }

    override fun onCleared() {
        super.onCleared()
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
    data object GeneratingResponse : InferenceUiState()
    data class ResponseReady(val response: LlmResponse) : InferenceUiState()
    data class Error(val message: String) : InferenceUiState()
}

data class LlmResponse(
    val modelId: String,
    val modelName: String,
    val content: String,
    val latencyMs: Long
)
