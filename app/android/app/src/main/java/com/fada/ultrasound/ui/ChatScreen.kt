package com.fada.ultrasound.ui

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.Forum
import androidx.compose.material.icons.filled.Image
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.fada.ultrasound.llm.LlmModelOption
import com.fada.ultrasound.viewmodel.ChatConversation
import com.fada.ultrasound.viewmodel.ChatMessage
import com.fada.ultrasound.viewmodel.ChatRole
import com.fada.ultrasound.viewmodel.InferenceUiState
import com.fada.ultrasound.viewmodel.InferenceViewModel
import java.text.DateFormat
import java.util.Date

@Composable
fun ChatScreen(
    viewModel: InferenceViewModel,
    onNavigateToCamera: () -> Unit,
    onNavigateToConversations: () -> Unit
) {
    val uiState by viewModel.uiState.collectAsState()
    val selectedImage by viewModel.selectedImage.collectAsState()
    val selectedImageFileName by viewModel.selectedImageFileName.collectAsState()
    val selectedModel by viewModel.selectedModel.collectAsState()
    val modelOptions by viewModel.modelOptions.collectAsState()
    val conversations by viewModel.conversations.collectAsState()
    val currentConversationId by viewModel.currentConversationId.collectAsState()
    val currentConversation = conversations.firstOrNull { it.id == currentConversationId }
    val isGenerating = uiState is InferenceUiState.GeneratingResponse || uiState is InferenceUiState.Loading

    var draft by remember(currentConversationId) { mutableStateOf("") }
    var isAttachMenuExpanded by remember { mutableStateOf(false) }
    var isModelPickerOpen by remember { mutableStateOf(false) }

    val galleryLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { viewModel.loadImageFromUri(it) }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 12.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        ConversationStrip(
            currentConversation = currentConversation,
            onNavigateToConversations = onNavigateToConversations
        )

        Box(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
        ) {
            if (currentConversation?.messages.isNullOrEmpty()) {
                EmptyChatState()
            } else {
                LazyColumn(
                    modifier = Modifier.fillMaxSize(),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                    contentPadding = PaddingValues(vertical = 4.dp)
                ) {
                    items(
                        items = currentConversation.messages,
                        key = { it.id }
                    ) { message ->
                        ChatMessageBubble(message = message)
                    }
                }
            }
        }

        if (uiState is InferenceUiState.Error) {
            ErrorCard(message = (uiState as InferenceUiState.Error).message)
        }

        if (isGenerating) {
            LoadingCard(uiState = uiState)
        }

        selectedImage?.let { bitmap ->
            AttachmentPreview(
                fileName = selectedImageFileName ?: "Attached image",
                thumbnail = {
                    Image(
                        bitmap = bitmap.asImageBitmap(),
                        contentDescription = "Selected image",
                        modifier = Modifier
                            .size(42.dp)
                            .clip(RoundedCornerShape(10.dp)),
                        contentScale = ContentScale.Crop
                    )
                }
            )
        }

        MessageComposer(
            draft = draft,
            onDraftChange = { draft = it },
            selectedModel = selectedModel,
            isSendEnabled = draft.isNotBlank() && !isGenerating,
            isAttachMenuExpanded = isAttachMenuExpanded,
            onAttachMenuExpandedChange = { isAttachMenuExpanded = it },
            onAddImage = {
                isAttachMenuExpanded = false
                galleryLauncher.launch("image/*")
            },
            onTakePhoto = {
                isAttachMenuExpanded = false
                onNavigateToCamera()
            },
            onChangeModel = {
                isAttachMenuExpanded = false
                isModelPickerOpen = true
            },
            onSend = {
                viewModel.sendChatMessage(draft)
                draft = ""
            }
        )
    }

    if (isModelPickerOpen) {
        ModelPickerDialog(
            modelOptions = modelOptions,
            selectedModel = selectedModel,
            onSelectModel = { model ->
                viewModel.selectModel(model.id)
                isModelPickerOpen = false
            },
            onDismiss = { isModelPickerOpen = false }
        )
    }
}

@Composable
private fun ConversationStrip(
    currentConversation: ChatConversation?,
    onNavigateToConversations: () -> Unit
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        IconButton(onClick = onNavigateToConversations) {
            Icon(
                imageVector = Icons.Default.Forum,
                contentDescription = "Threads"
            )
        }
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = currentConversation?.title ?: "New conversation",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            Text(
                text = conversationSubtitle(currentConversation),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
    }
}

@Composable
private fun MessageComposer(
    draft: String,
    onDraftChange: (String) -> Unit,
    selectedModel: LlmModelOption,
    isSendEnabled: Boolean,
    isAttachMenuExpanded: Boolean,
    onAttachMenuExpandedChange: (Boolean) -> Unit,
    onAddImage: () -> Unit,
    onTakePhoto: () -> Unit,
    onChangeModel: () -> Unit,
    onSend: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(28.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(start = 4.dp, end = 4.dp, top = 2.dp, bottom = 2.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box {
                IconButton(
                    onClick = { onAttachMenuExpandedChange(true) },
                    modifier = Modifier.size(44.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.Add,
                        contentDescription = "Add attachment"
                    )
                }
                DropdownMenu(
                    expanded = isAttachMenuExpanded,
                    onDismissRequest = { onAttachMenuExpandedChange(false) }
                ) {
                    DropdownMenuItem(
                        text = { Text("Add image") },
                        leadingIcon = { Icon(Icons.Default.Image, contentDescription = null) },
                        onClick = onAddImage
                    )
                    DropdownMenuItem(
                    text = { Text("Take a photo") },
                        leadingIcon = { Icon(Icons.Default.CameraAlt, contentDescription = null) },
                        onClick = onTakePhoto
                    )
                    DropdownMenuItem(
                        text = { Text("Current model: ${selectedModel.displayName}") },
                        leadingIcon = { Icon(Icons.Default.Tune, contentDescription = null) },
                        onClick = onChangeModel
                    )
                }
            }

            TextField(
                value = draft,
                onValueChange = onDraftChange,
                modifier = Modifier.weight(1f),
                placeholder = { Text("Message") },
                minLines = 1,
                maxLines = 5,
                colors = TextFieldDefaults.colors(
                    focusedContainerColor = Color.Transparent,
                    unfocusedContainerColor = Color.Transparent,
                    disabledContainerColor = Color.Transparent,
                    focusedIndicatorColor = Color.Transparent,
                    unfocusedIndicatorColor = Color.Transparent,
                    disabledIndicatorColor = Color.Transparent
                )
            )

            IconButton(
                onClick = onSend,
                enabled = isSendEnabled,
                modifier = Modifier.size(44.dp)
            ) {
                Icon(
                    imageVector = Icons.AutoMirrored.Filled.Send,
                    contentDescription = "Send",
                    tint = if (isSendEnabled) {
                        MaterialTheme.colorScheme.primary
                    } else {
                        MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.45f)
                    }
                )
            }
        }
    }
}

@Composable
private fun AttachmentPreview(
    fileName: String,
    thumbnail: @Composable () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(14.dp))
            .padding(horizontal = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        thumbnail()
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = fileName,
                style = MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.SemiBold,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            Text(
                text = "Ready to send",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
private fun ModelPickerDialog(
    modelOptions: List<LlmModelOption>,
    selectedModel: LlmModelOption,
    onSelectModel: (LlmModelOption) -> Unit,
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Choose model") },
        text = {
            Column {
                modelOptions.forEach { model ->
                    TextButton(
                        onClick = { onSelectModel(model) },
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(
                            text = if (model.id == selectedModel.id) {
                                "${model.displayName} (current)"
                            } else {
                                model.displayName
                            },
                            modifier = Modifier.fillMaxWidth()
                        )
                    }
                }
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss) {
                Text("Close")
            }
        }
    )
}

@Composable
private fun EmptyChatState() {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp)
            .verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Icon(
            imageVector = Icons.Default.Forum,
            contentDescription = null,
            modifier = Modifier.size(56.dp),
            tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.65f)
        )
        Spacer(modifier = Modifier.height(12.dp))
        Text(
            text = "Start a conversation",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.SemiBold
        )
        Text(
            text = "Use + to add an image, take a photo, or change the current model.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

@Composable
private fun ChatMessageBubble(message: ChatMessage) {
    val isUser = message.role == ChatRole.User
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start
    ) {
        Card(
            modifier = Modifier.fillMaxWidth(0.86f),
            shape = RoundedCornerShape(
                topStart = 18.dp,
                topEnd = 18.dp,
                bottomStart = if (isUser) 18.dp else 4.dp,
                bottomEnd = if (isUser) 4.dp else 18.dp
            ),
            colors = CardDefaults.cardColors(
                containerColor = if (isUser) {
                    MaterialTheme.colorScheme.primaryContainer
                } else {
                    MaterialTheme.colorScheme.surfaceVariant
                }
            )
        ) {
            Column(
                modifier = Modifier.padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                Text(
                    text = if (isUser) "You" else message.modelName ?: "Assistant",
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.SemiBold
                )
                if (isUser) {
                    Text(
                        text = message.content,
                        style = MaterialTheme.typography.bodyMedium
                    )
                } else {
                    MarkdownMessageText(
                        markdown = message.content,
                        isStreaming = message.isStreaming
                    )
                }
                if (message.hasImage || message.latencyMs != null) {
                    Text(
                        text = buildString {
                            message.imageFileName?.let { append(it) }
                            if (message.hasImage && message.imageFileName.isNullOrBlank()) append("Image attached")
                            message.latencyMs?.let {
                                if (isNotEmpty()) append(" - ")
                                append("${it}ms")
                            }
                        },
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                }
            }
        }
    }
}

@Composable
private fun MarkdownMessageText(markdown: String, isStreaming: Boolean) {
    val blocks = remember(markdown, isStreaming) {
        MarkdownTextFormatter.parseBlocks(markdown)
    }

    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        blocks.forEach { block ->
            when (block) {
                is MarkdownBlock.Blank -> Spacer(modifier = Modifier.height(2.dp))
                is MarkdownBlock.CodeBlock -> MarkdownCodeBlock(block.text)
                is MarkdownBlock.Heading -> MarkdownLine(
                    text = block.text,
                    weight = if (block.level <= 2) FontWeight.Bold else FontWeight.SemiBold,
                    isStreaming = isStreaming
                )
                is MarkdownBlock.ListItem -> MarkdownListItem(
                    marker = block.marker,
                    text = block.text,
                    isStreaming = isStreaming
                )
                is MarkdownBlock.Paragraph -> MarkdownLine(
                    text = block.text,
                    weight = FontWeight.Normal,
                    isStreaming = isStreaming
                )
            }
        }
    }
}

@Composable
private fun MarkdownLine(
    text: String,
    weight: FontWeight,
    isStreaming: Boolean,
    modifier: Modifier = Modifier
) {
    Text(
        text = MarkdownTextFormatter.buildAnnotatedText(
            text = text,
            streaming = isStreaming,
            baseWeight = weight
        ),
        modifier = modifier,
        style = MaterialTheme.typography.bodyMedium,
        fontWeight = weight
    )
}

@Composable
private fun MarkdownListItem(
    marker: String,
    text: String,
    isStreaming: Boolean
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Text(
            text = marker,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        MarkdownLine(
            text = text,
            weight = FontWeight.Normal,
            isStreaming = isStreaming,
            modifier = Modifier.weight(1f)
        )
    }
}

@Composable
private fun MarkdownCodeBlock(text: String) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(10.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        )
    ) {
        Text(
            text = text,
            modifier = Modifier.padding(10.dp),
            style = MaterialTheme.typography.bodySmall,
            fontFamily = FontFamily.Monospace
        )
    }
}

@Composable
private fun ErrorCard(message: String) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.errorContainer
        )
    ) {
        Text(
            text = message,
            modifier = Modifier.padding(12.dp),
            color = MaterialTheme.colorScheme.onErrorContainer
        )
    }
}

@Composable
private fun LoadingCard(uiState: InferenceUiState) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        )
    ) {
        Row(
            modifier = Modifier.padding(12.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            CircularProgressIndicator(modifier = Modifier.size(24.dp))
            Text(
                text = when (uiState) {
                    is InferenceUiState.Loading -> uiState.message
                    InferenceUiState.GeneratingResponse -> "Generating response..."
                    else -> "Working..."
                },
                style = MaterialTheme.typography.bodyMedium
            )
        }
    }
}

private fun conversationSubtitle(conversation: ChatConversation?): String {
    if (conversation == null) return "New thread"
    val createdAt = DateFormat.getDateTimeInstance(DateFormat.SHORT, DateFormat.SHORT)
        .format(Date(conversation.createdAt))
    val attachment = conversation.messages
        .lastOrNull { !it.imageFileName.isNullOrBlank() }
        ?.imageFileName
    return if (attachment.isNullOrBlank()) {
        "Created $createdAt"
    } else {
        "Created $createdAt - $attachment"
    }
}
