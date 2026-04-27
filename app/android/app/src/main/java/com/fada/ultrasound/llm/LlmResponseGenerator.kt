package com.fada.ultrasound.llm

import android.content.Context
import android.graphics.Bitmap
import android.os.Build
import android.system.Os
import android.system.OsConstants
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import java.io.File
import java.io.FileOutputStream

/**
 * On-device Gemma response generator using LiteRT-LM runtime.
 */
interface LlmResponseClient {
    suspend fun generate(
        context: Context,
        model: LlmModelOption,
        image: Bitmap,
        prompt: String,
        onStatus: (String) -> Unit = {}
    ): String
}

object LlmResponseGenerator : LlmResponseClient {
    private const val DEFAULT_PROMPT =
        "You are a fetal ultrasound research assistant. Describe visible anatomy, likely view, and uncertainty. " +
            "Do not provide clinical diagnosis."

    override suspend fun generate(
        context: Context,
        model: LlmModelOption,
        image: Bitmap,
        prompt: String,
        onStatus: (String) -> Unit
    ): String {
        enforceSupportedRuntime()
        val downloadManager = ModelDownloadManager(context)
        var modelFile = downloadManager.getOrDownloadModel(
            option = model,
            onStatus = onStatus
        )

        return try {
            onStatus("Initializing ${model.displayName}...")
            runInference(context, modelFile, image, prompt)
        } catch (e: RuntimeException) {
            if (!isPrefillDecodeMissingError(e)) {
                throw e
            }

            onStatus("Refreshing ${model.displayName} model file...")
            downloadManager.invalidateModel(model)
            modelFile = downloadManager.getOrDownloadModel(
                option = model,
                onStatus = onStatus
            )
            onStatus("Retrying ${model.displayName}...")
            runInference(context, modelFile, image, prompt)
        }
    }

    private fun runInference(
        context: Context,
        modelFile: File,
        image: Bitmap,
        prompt: String
    ): String {
        val tempImageFile = File.createTempFile("ultrasound_input_", ".jpg", context.cacheDir)
        FileOutputStream(tempImageFile).use { out ->
            image.compress(Bitmap.CompressFormat.JPEG, 95, out)
        }

        val response = try {
            runWithBackend(
                modelFile = modelFile,
                imageFile = tempImageFile,
                backend = Backend.GPU(),
                visionBackend = Backend.GPU(),
                cacheDir = context.cacheDir.absolutePath,
                prompt = prompt.ifBlank { DEFAULT_PROMPT }
            )
        } catch (_: RuntimeException) {
            runWithBackend(
                modelFile = modelFile,
                imageFile = tempImageFile,
                backend = Backend.CPU(),
                visionBackend = Backend.CPU(),
                cacheDir = context.cacheDir.absolutePath,
                prompt = prompt.ifBlank { DEFAULT_PROMPT }
            )
        }

        tempImageFile.delete()
        return response
    }

    private fun runWithBackend(
        modelFile: File,
        imageFile: File,
        backend: Backend,
        visionBackend: Backend,
        cacheDir: String,
        prompt: String
    ): String {
        val engineConfig = EngineConfig(
            modelPath = modelFile.absolutePath,
            backend = backend,
            visionBackend = visionBackend,
            cacheDir = cacheDir
        )
        return Engine(engineConfig).use { engine ->
            engine.initialize()
            engine.createConversation().use { conversation ->
                val message = conversation.sendMessage(
                    Contents.of(
                        Content.ImageFile(imageFile.absolutePath),
                        Content.Text(prompt)
                    )
                )
                message.toString()
            }
        }
    }

    private fun isPrefillDecodeMissingError(e: RuntimeException): Boolean {
        val message = e.message ?: return false
        return message.contains("TF_LITE_PREFILL_DECODE", ignoreCase = true) &&
            message.contains("not found", ignoreCase = true)
    }

    private fun enforceSupportedRuntime() {
        val pageSize = runCatching { Os.sysconf(OsConstants._SC_PAGESIZE).toInt() }.getOrDefault(4096)
        val primaryAbi = Build.SUPPORTED_ABIS.firstOrNull().orEmpty()
        val isX86_64 = primaryAbi.contains("x86_64", ignoreCase = true)
        val is16Kb = pageSize >= 16_384

        if (isX86_64 && is16Kb) {
            throw IllegalStateException(
                "This emulator image is not supported for on-device model runtime " +
                    "(x86_64 with 16 KB page size). Use an ARM64 image/device."
            )
        }
    }
}

