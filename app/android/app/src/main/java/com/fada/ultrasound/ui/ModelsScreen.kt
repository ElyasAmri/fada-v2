package com.fada.ultrasound.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Storage
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.fada.ultrasound.viewmodel.InferenceViewModel
import com.fada.ultrasound.viewmodel.ModelStorageInfo

@Composable
fun ModelsScreen(viewModel: InferenceViewModel) {
    val modelStorage by viewModel.modelStorage.collectAsState()
    val selectedModel by viewModel.selectedModel.collectAsState()

    LaunchedEffect(Unit) {
        viewModel.refreshModelStorage()
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "Models",
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = "Download, delete, and choose on-device models",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            IconButton(onClick = { viewModel.refreshModelStorage() }) {
                Icon(Icons.Default.Refresh, contentDescription = "Refresh")
            }
        }

        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            items(
                items = modelStorage,
                key = { it.model.id }
            ) { info ->
                ModelCard(
                    info = info,
                    isSelected = info.model.id == selectedModel.id,
                    onSelect = { viewModel.selectModel(info.model.id) },
                    onDownload = { viewModel.downloadModel(info.model.id) },
                    onDelete = { viewModel.deleteStoredModel(info.model.id) }
                )
            }
        }
    }
}

@Composable
private fun ModelCard(
    info: ModelStorageInfo,
    isSelected: Boolean,
    onSelect: () -> Unit,
    onDownload: () -> Unit,
    onDelete: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = if (isSelected) {
                MaterialTheme.colorScheme.primaryContainer
            } else {
                MaterialTheme.colorScheme.surfaceVariant
            }
        )
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Icon(
                    imageVector = if (info.isStored) Icons.Default.CheckCircle else Icons.Default.Storage,
                    contentDescription = null,
                    tint = if (info.isStored) {
                        MaterialTheme.colorScheme.primary
                    } else {
                        MaterialTheme.colorScheme.onSurfaceVariant
                    }
                )
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = info.model.displayName,
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold
                    )
                    Text(
                        text = "Version ${info.model.version} - ${if (info.isStored) formatBytes(info.sizeBytes) else "Not downloaded"}",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                if (info.isBusy) {
                    CircularProgressIndicator()
                }
            }

            Text(
                text = info.status ?: shortModelPath(info.filePath),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Button(
                    onClick = onSelect,
                    enabled = !isSelected,
                    modifier = Modifier.weight(1f)
                ) {
                    Text(if (isSelected) "Selected" else "Select")
                }
                OutlinedButton(
                    onClick = onDownload,
                    enabled = !info.isBusy,
                    modifier = Modifier.weight(1f)
                ) {
                    Text(if (info.isStored) "Update" else "Get")
                }
                OutlinedButton(
                    onClick = onDelete,
                    enabled = info.isStored && !info.isBusy,
                    modifier = Modifier.weight(1f)
                ) {
                    Text("Remove")
                }
            }
        }
    }
}

private fun formatBytes(bytes: Long): String {
    if (bytes < 1_024) return "$bytes B"
    val kib = bytes / 1_024.0
    if (kib < 1_024) return "%.1f KiB".format(kib)
    val mib = kib / 1_024.0
    if (mib < 1_024) return "%.1f MiB".format(mib)
    val gib = mib / 1_024.0
    return "%.2f GiB".format(gib)
}

private fun shortModelPath(path: String): String {
    val normalized = path.replace('\\', '/')
    val marker = "FADA/"
    val index = normalized.indexOf(marker)
    return if (index >= 0) {
        normalized.substring(index)
    } else {
        normalized.substringAfterLast('/')
    }
}
