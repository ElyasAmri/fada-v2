package com.fada.ultrasound.llm

/**
 * Supported on-device Gemma variants in LiteRT-LM format.
 */
data class LlmModelOption(
    val id: String,
    val displayName: String,
    val version: String,
    val downloadUrl: String,
    val expectedSha256: String? = null,
    val localFileName: String
)

object LlmModels {
    val default = LlmModelOption(
        id = "gemma-4-e2b",
        displayName = "Gemma 4 E2B",
        version = "1",
        downloadUrl = "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm",
        localFileName = "gemma-4-E2B-it.litertlm"
    )

    val options = listOf(
        default,
        LlmModelOption(
            id = "gemma-4-e4b",
            displayName = "Gemma 4 E4B",
            version = "1",
            downloadUrl = "https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it.litertlm",
            localFileName = "gemma-4-E4B-it.litertlm"
        )
    )
}

