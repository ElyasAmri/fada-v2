package com.fada.ultrasound.llm

import android.content.Context
import android.graphics.Bitmap
import android.os.Build
import android.system.Os
import android.system.OsConstants
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.flow.collect
import java.io.File
import java.io.FileOutputStream

data class LlmChatTurn(
    val role: LlmChatRole,
    val content: String,
    val imageFileName: String? = null
)

enum class LlmChatRole {
    User,
    Assistant
}

/**
 * On-device Gemma response generator using LiteRT-LM runtime.
 */
interface LlmResponseClient {
    suspend fun prepare(
        context: Context,
        model: LlmModelOption,
        onStatus: (String) -> Unit = {}
    )

    suspend fun generate(
        context: Context,
        model: LlmModelOption,
        conversationId: String,
        history: List<LlmChatTurn>,
        image: Bitmap?,
        imageFileName: String?,
        prompt: String,
        onStatus: (String) -> Unit = {},
        onPartialResponse: (String) -> Unit = {}
    ): String

    fun release()
}

object LlmResponseGenerator : LlmResponseClient {
    private const val DEFAULT_SYSTEM_PROMPT =
        "You are a fetal ultrasound assistant. Describe visible anatomy, likely view, and uncertainty. " +
            "Do not provide clinical diagnosis."

    private val mutex = Mutex()
    private val runtimes = mutableMapOf<String, ModelRuntime>()

    override suspend fun prepare(
        context: Context,
        model: LlmModelOption,
        onStatus: (String) -> Unit
    ) {
        enforceSupportedRuntime()
        mutex.withLock {
            getOrCreateRuntime(context, model, onStatus)
        }
    }

    override suspend fun generate(
        context: Context,
        model: LlmModelOption,
        conversationId: String,
        history: List<LlmChatTurn>,
        image: Bitmap?,
        imageFileName: String?,
        prompt: String,
        onStatus: (String) -> Unit,
        onPartialResponse: (String) -> Unit
    ): String {
        enforceSupportedRuntime()
        val normalizedPrompt = prompt.ifBlank { "Continue." }
        val conversation = mutex.withLock {
            getOrCreateRuntime(context, model, onStatus)
                .getOrCreateConversation(conversationId, history)
        }

        val imageFile = image?.let {
            val suffix = imageFileName
                ?.substringAfterLast('.', "")
                ?.takeIf { extension -> extension.isNotBlank() }
                ?.let { extension -> ".$extension" }
                ?: ".jpg"
            File.createTempFile("ultrasound_input_", suffix, context.cacheDir).also { file ->
                FileOutputStream(file).use { out ->
                    image.compress(Bitmap.CompressFormat.JPEG, 95, out)
                }
            }
        }

        return try {
            val contents = buildCurrentContents(normalizedPrompt, imageFile)
            var accumulated = ""
            conversation.sendMessageAsync(contents).collect { message ->
                accumulated = mergeStreamText(accumulated, message.toString())
                onPartialResponse(accumulated)
            }
            accumulated
        } finally {
            imageFile?.delete()
        }
    }

    override fun release() {
        runtimes.values.forEach { runtime ->
            runtime.close()
        }
        runtimes.clear()
    }

    private fun getOrCreateRuntime(
        context: Context,
        model: LlmModelOption,
        onStatus: (String) -> Unit
    ): ModelRuntime {
        runtimes[model.id]?.let { return it }

        val downloadManager = ModelDownloadManager(context)
        val modelFile = downloadManager.getOrDownloadModel(
            option = model,
            onStatus = onStatus
        )
        onStatus("Loading ${model.displayName}...")

        val engine = createInitializedEngine(context, modelFile, useGpu = true)
            ?: createInitializedEngine(context, modelFile, useGpu = false)
            ?: throw IllegalStateException("Failed to initialize ${model.displayName}")

        return ModelRuntime(engine).also {
            runtimes[model.id] = it
        }
    }

    private fun createInitializedEngine(
        context: Context,
        modelFile: File,
        useGpu: Boolean
    ): Engine? {
        return try {
            val backend = if (useGpu) Backend.GPU() else Backend.CPU()
            Engine(
                EngineConfig(
                    modelPath = modelFile.absolutePath,
                    backend = backend,
                    visionBackend = backend,
                    cacheDir = context.cacheDir.absolutePath
                )
            ).also { it.initialize() }
        } catch (_: RuntimeException) {
            null
        }
    }

    private fun buildCurrentContents(prompt: String, imageFile: File?): Contents {
        return if (imageFile != null) {
            Contents.of(
                Content.ImageFile(imageFile.absolutePath),
                Content.Text(prompt)
            )
        } else {
            Contents.of(prompt)
        }
    }

    private fun historyToInitialMessages(history: List<LlmChatTurn>): List<Message> {
        return history
            .filter { it.content.isNotBlank() }
            .map { turn ->
                val text = buildString {
                    if (!turn.imageFileName.isNullOrBlank()) {
                        append("[Image attached: ")
                        append(turn.imageFileName)
                        append("]\n")
                    }
                    append(turn.content)
                }
                when (turn.role) {
                    LlmChatRole.User -> Message.user(text)
                    LlmChatRole.Assistant -> Message.model(text)
                }
            }
    }

    private fun mergeStreamText(current: String, next: String): String {
        return when {
            next.isBlank() -> current
            next.startsWith(current) -> next
            current.endsWith(next) -> current
            else -> current + next
        }
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

    private class ModelRuntime(
        private val engine: Engine
    ) : AutoCloseable {
        private val conversations = mutableMapOf<String, Conversation>()

        fun getOrCreateConversation(
            conversationId: String,
            history: List<LlmChatTurn>
        ): Conversation {
            conversations[conversationId]?.let { return it }
            val config = ConversationConfig(
                systemInstruction = Contents.of(DEFAULT_SYSTEM_PROMPT),
                initialMessages = historyToInitialMessages(history)
            )
            return engine.createConversation(config).also {
                conversations[conversationId] = it
            }
        }

        override fun close() {
            conversations.values.forEach { conversation ->
                conversation.close()
            }
            conversations.clear()
            engine.close()
        }
    }
}
