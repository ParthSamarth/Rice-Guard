package com.riceguard.ai.ui.screens.historydetail

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.DeleteOutline
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import com.riceguard.ai.data.local.ScanRecord
import com.riceguard.ai.data.repository.ScanRepository
import com.riceguard.ai.di.rememberViewModel
import com.riceguard.ai.domain.model.ScanSource
import com.riceguard.ai.ui.components.ConfidenceBar
import com.riceguard.ai.ui.components.ConfirmDialog
import com.riceguard.ai.ui.components.PrimaryButton
import com.riceguard.ai.ui.components.RiceGuardCard
import com.riceguard.ai.ui.components.ScreenTopBar
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun HistoryDetailScreen(
    scanId: String,
    scanRepository: ScanRepository,
    onBack: () -> Unit,
    onDeleted: () -> Unit,
) {
    val viewModel = rememberViewModel { HistoryDetailViewModel(scanRepository, scanId) }
    val uiState by viewModel.uiState.collectAsState()
    var showDeleteConfirm by remember { mutableStateOf(false) }

    LaunchedEffect(uiState.deleted) {
        if (uiState.deleted) onDeleted()
    }

    Scaffold(
        topBar = {
            ScreenTopBar(
                title = "Scan Detail",
                onBack = onBack,
                actions = {
                    IconButton(onClick = { showDeleteConfirm = true }) {
                        Icon(Icons.Filled.DeleteOutline, contentDescription = "Delete Scan")
                    }
                },
            )
        },
    ) { padding ->
        val record = uiState.record
        when {
            uiState.loading -> Box(Modifier.fillMaxSize().padding(padding), contentAlignment = Alignment.Center) {
                CircularProgressIndicator()
            }
            record == null -> Box(Modifier.fillMaxSize().padding(padding), contentAlignment = Alignment.Center) {
                Text("This scan is no longer available.", style = MaterialTheme.typography.bodyMedium)
            }
            else -> LazyColumn(modifier = Modifier.fillMaxSize().padding(padding).padding(20.dp)) {
                item {
                    Text(record.disease ?: "Unresolved", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.SemiBold)
                    record.finalConfidence?.let {
                        ConfidenceBar(confidence = it, modifier = Modifier.fillMaxWidth().padding(top = 12.dp, bottom = 4.dp))
                    }
                    ScanMetadataCard(record = record, modifier = Modifier.fillMaxWidth().padding(top = 16.dp))
                }

                item {
                    NotesCard(
                        initialNotes = record.notes.orEmpty(),
                        onSave = viewModel::saveNotes,
                        modifier = Modifier.fillMaxWidth().padding(top = 16.dp),
                    )
                }

                item {
                    Text("Original photo", style = MaterialTheme.typography.titleSmall, modifier = Modifier.padding(bottom = 8.dp))
                    AsyncImage(
                        model = File(record.originalImagePath),
                        contentDescription = "Original photo",
                        modifier = Modifier.fillMaxWidth().aspectRatio(1f).clip(MaterialTheme.shapes.large),
                        contentScale = ContentScale.Crop,
                    )
                }

                record.detectionImagePath?.let { path ->
                    item {
                        Text("Detection", style = MaterialTheme.typography.titleSmall, modifier = Modifier.padding(top = 20.dp, bottom = 8.dp))
                        AsyncImage(
                            model = File(path),
                            contentDescription = "Detection visualization",
                            modifier = Modifier.fillMaxWidth().aspectRatio(1f).clip(MaterialTheme.shapes.large),
                            contentScale = ContentScale.Fit,
                        )
                    }
                }

                record.gradcamImagePath?.let { path ->
                    item {
                        Text("Model attention (Grad-CAM)", style = MaterialTheme.typography.titleSmall, modifier = Modifier.padding(top = 20.dp, bottom = 8.dp))
                        Text(
                            "Original — Model attention — Overlay",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.padding(bottom = 2.dp),
                        )
                        Text(
                            "Areas that influenced the model's classification.",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.padding(bottom = 6.dp),
                        )
                        AsyncImage(
                            model = File(path),
                            contentDescription = "Model attention: original photo, heatmap, and overlay side by side",
                            // The saved file is a wide 3-panel comparison figure (see
                            // explainability/gradcam.py), not a square image -- fit it
                            // in full rather than cropping two of the three panels away.
                            modifier = Modifier.fillMaxWidth().aspectRatio(1800f / 630f).clip(MaterialTheme.shapes.large),
                            contentScale = ContentScale.Fit,
                        )
                    }
                }

                uiState.recommendation?.let { rec ->
                    if (rec.available) {
                        item {
                            RiceGuardCard(modifier = Modifier.fillMaxWidth().padding(top = 20.dp)) {
                                Text("Treatment guidance", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
                                rec.management.forEach {
                                    Text("• $it", style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(top = 6.dp))
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (showDeleteConfirm) {
        ConfirmDialog(
            title = "Delete this scan?",
            message = "This will permanently remove the stored image, explanation, result, and history entry.",
            confirmLabel = "Delete",
            onConfirm = {
                showDeleteConfirm = false
                viewModel.deleteScan()
            },
            onDismiss = { showDeleteConfirm = false },
        )
    }
}

private val DATE_FORMAT = SimpleDateFormat("d MMM yyyy", Locale.getDefault())
private val TIME_FORMAT = SimpleDateFormat("HH:mm", Locale.getDefault())

@Composable
private fun ScanMetadataCard(record: ScanRecord, modifier: Modifier = Modifier) {
    val date = Date(record.timestampMillis)
    val isHealthy = record.disease?.equals("Healthy", ignoreCase = true) == true
    RiceGuardCard(modifier = modifier) {
        MetadataRow("SCAN ID", record.id)
        MetadataRow("DATE", DATE_FORMAT.format(date))
        MetadataRow("TIME", TIME_FORMAT.format(date))
        MetadataRow("SOURCE", ScanSource.fromWire(record.source).label)
        MetadataRow("RESULT", record.disease ?: "Unresolved")
        record.finalConfidence?.let { MetadataRow("CONFIDENCE", "${"%.1f".format(it * 100)}%") }
        MetadataRow("STATUS", if (isHealthy) "Healthy" else "Diseased")
        record.pipelineProcessingTimeSec?.let { MetadataRow("INFERENCE TIME", "${"%.1f".format(it)}s") }
    }
}

@Composable
private fun MetadataRow(label: String, value: String) {
    Row(modifier = Modifier.fillMaxWidth().padding(vertical = 6.dp), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(
            label,
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(value, style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.SemiBold)
    }
}

@Composable
private fun NotesCard(initialNotes: String, onSave: (String) -> Unit, modifier: Modifier = Modifier) {
    var text by remember(initialNotes) { mutableStateOf(initialNotes) }
    RiceGuardCard(modifier = modifier) {
        Text("Notes", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
        OutlinedTextField(
            value = text,
            onValueChange = { text = it },
            placeholder = { Text("Add your own notes about this scan…") },
            modifier = Modifier.fillMaxWidth().padding(top = 10.dp),
            minLines = 2,
        )
        if (text != initialNotes) {
            PrimaryButton(
                text = "Save Notes",
                onClick = { onSave(text) },
                modifier = Modifier.fillMaxWidth().padding(top = 10.dp),
            )
        }
    }
}
