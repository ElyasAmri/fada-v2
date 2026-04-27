package com.fada.ultrasound.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Storage
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
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
        modifier = Modifier.fillMaxSize()
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(start = 24.dp, end = 12.dp, top = 20.dp, bottom = 8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "Models",
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = "Choose and manage on-device model files",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            IconButton(onClick = { viewModel.refreshModelStorage() }) {
                Icon(Icons.Default.Refresh, contentDescription = "Refresh")
            }
        }

        LazyColumn(
            modifier = Modifier.fillMaxSize()
        ) {
            items(
                items = modelStorage,
                key = { it.model.id }
            ) { info ->
                ModelRow(
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
private fun ModelRow(
    info: ModelStorageInfo,
    isSelected: Boolean,
    onSelect: () -> Unit,
    onDownload: () -> Unit,
    onDelete: () -> Unit
) {
    Column {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .clickable(enabled = !isSelected, onClick = onSelect)
                .padding(horizontal = 24.dp, vertical = 14.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Icon(
                imageVector = when {
                    isSelected -> Icons.Default.CheckCircle
                    info.isStored -> Icons.Default.Storage
                    else -> Icons.Default.Download
                },
                contentDescription = null,
                modifier = Modifier.size(28.dp),
                tint = if (isSelected) {
                    MaterialTheme.colorScheme.primary
                } else {
                    MaterialTheme.colorScheme.onSurfaceVariant
                }
            )

            Column(modifier = Modifier.weight(1f)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Text(
                        text = info.model.displayName,
                        modifier = Modifier.weight(1f),
                        style = MaterialTheme.typography.bodyLarge,
                        fontWeight = FontWeight.SemiBold,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                    if (info.isBusy) {
                        CircularProgressIndicator(modifier = Modifier.size(20.dp))
                    } else if (isSelected) {
                        Text(
                            text = "Selected",
                            style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.primary
                        )
                    }
                }

                Text(
                    text = "Version ${info.model.version} - ${if (info.isStored) formatBytes(info.sizeBytes) else "Not downloaded"}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
                Text(
                    text = info.status ?: shortModelPath(info.filePath),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.End
                ) {
                    TextButton(
                        onClick = onDownload,
                        enabled = !info.isBusy
                    ) {
                        Text(if (info.isStored) "Update" else "Download")
                    }
                    TextButton(
                        onClick = onDelete,
                        enabled = info.isStored && !info.isBusy
                    ) {
                        Text("Remove")
                    }
                }
            }
        }
        HorizontalDivider(modifier = Modifier.padding(start = 68.dp))
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
