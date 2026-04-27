package com.fada.ultrasound.llm

import android.content.Context
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/**
 * Downloads and versions LiteRT-LM model files in phone storage under FADA/models.
 */
class ModelDownloadManager(private val context: Context) {

    fun getOrDownloadModel(
        option: LlmModelOption,
        onStatus: (String) -> Unit
    ): File {
        val targetFile = getModelFile(option)
        if (targetFile.exists()) {
            return targetFile
        }

        targetFile.parentFile?.mkdirs()
        val tempFile = File(targetFile.parentFile, "${targetFile.name}.download")

        onStatus("Downloading ${option.displayName}...")
        downloadFile(option.downloadUrl, tempFile, onStatus)

        if (!option.expectedSha256.isNullOrBlank()) {
            onStatus("Verifying model integrity...")
            val downloadedSha = sha256(tempFile)
            if (!downloadedSha.equals(option.expectedSha256, ignoreCase = true)) {
                tempFile.delete()
                throw IllegalStateException(
                    "Checksum mismatch for ${option.displayName}. Expected ${option.expectedSha256}, got $downloadedSha"
                )
            }
        }

        if (targetFile.exists()) {
            targetFile.delete()
        }
        if (!tempFile.renameTo(targetFile)) {
            tempFile.delete()
            throw IllegalStateException("Failed to finalize model file: ${targetFile.absolutePath}")
        }

        return targetFile
    }

    fun invalidateModel(option: LlmModelOption) {
        getModelFile(option).delete()
    }

    private fun getModelFile(option: LlmModelOption): File {
        val externalBase = context.getExternalFilesDir(null)
        val modelRoot = if (externalBase != null) {
            File(externalBase, "FADA/models")
        } else {
            File(context.filesDir, "models")
        }
        return File(modelRoot, "${option.id}/${option.version}/${option.localFileName}")
    }

    private fun downloadFile(
        fileUrl: String,
        destination: File,
        onStatus: (String) -> Unit
    ) {
        val connection = (URL(fileUrl).openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = 30_000
            readTimeout = 120_000
            instanceFollowRedirects = true
        }

        connection.connect()
        if (connection.responseCode !in 200..299) {
            connection.disconnect()
            throw IllegalStateException("Model download failed with HTTP ${connection.responseCode}")
        }

        val totalBytes = connection.contentLengthLong
        connection.inputStream.use { input ->
            FileOutputStream(destination).use { output ->
                val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                var bytesCopied = 0L
                var bytes = input.read(buffer)
                while (bytes >= 0) {
                    output.write(buffer, 0, bytes)
                    bytesCopied += bytes

                    if (totalBytes > 0L) {
                        val percent = ((bytesCopied * 100) / totalBytes).toInt()
                        onStatus("Downloading ${percent}%")
                    }
                    bytes = input.read(buffer)
                }
            }
        }

        connection.disconnect()
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            var bytes = input.read(buffer)
            while (bytes >= 0) {
                digest.update(buffer, 0, bytes)
                bytes = input.read(buffer)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}

