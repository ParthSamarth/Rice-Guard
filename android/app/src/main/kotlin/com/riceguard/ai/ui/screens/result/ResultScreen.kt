package com.riceguard.ai.ui.screens.result

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Spa
import androidx.compose.material.icons.filled.Visibility
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import com.riceguard.ai.domain.model.ScanResult
import com.riceguard.ai.ui.components.ConfidenceBar
import com.riceguard.ai.ui.components.PrimaryButton
import com.riceguard.ai.ui.components.RiceGuardCard
import com.riceguard.ai.ui.components.StatusChip
import com.riceguard.ai.ui.navigation.ScanSessionViewModel
import com.riceguard.ai.ui.navigation.ScanUploadState
import com.riceguard.ai.ui.theme.riceGuardStatus

/** Below this, a result is shown as "Uncertain" rather than "Confident" even
 * though the server's own (lower) cnn_confidence threshold already let it
 * reach this screen at all (anything below THAT threshold routes to
 * LowConfidenceScreen instead, see RiceGuardNavGraph). This is a stricter,
 * purely-display threshold -- a UX choice about labeling, not a claim about
 * model calibration -- so a borderline-passing prediction is never labeled
 * exactly the same as a high-confidence one. */
private const val HIGH_CONFIDENCE_DISPLAY_THRESHOLD = 0.75f

@Composable
fun ResultScreen(
    scanSession: ScanSessionViewModel,
    onViewExplanation: () -> Unit,
    onViewRecommendation: () -> Unit,
    onDone: () -> Unit,
) {
    val uploadState by scanSession.uploadState.collectAsState()
    val capturedPhoto by scanSession.capturedPhoto.collectAsState()
    val result = (uploadState as? ScanUploadState.Success)?.result ?: run {
        // Guarded by navigation (Result is only reached on Success), but if
        // this state is ever hit directly, fail safe rather than crash.
        onDone()
        return
    }

    Scaffold { padding ->
        LazyColumn(
            modifier = Modifier.fillMaxSize().padding(padding),
            contentPadding = PaddingValues(bottom = 24.dp),
        ) {
            item {
                AsyncImage(
                    model = capturedPhoto,
                    contentDescription = "Scanned rice leaf",
                    modifier = Modifier.fillMaxWidth().aspectRatio(1.1f)
                        .clip(RoundedCornerShape(bottomStart = 28.dp, bottomEnd = 28.dp)),
                    contentScale = ContentScale.Crop,
                )
            }

            item {
                Column(modifier = Modifier.padding(20.dp)) {
                    val isHighConfidence = (result.finalConfidence ?: 0f) >= HIGH_CONFIDENCE_DISPLAY_THRESHOLD
                    StatusChip(
                        label = if (isHighConfidence) "Confident result" else "Uncertain — review carefully",
                        color = if (isHighConfidence) riceGuardStatus.good else riceGuardStatus.warning,
                        containerColor = if (isHighConfidence) riceGuardStatus.goodContainer else riceGuardStatus.warningContainer,
                    )
                    Text(
                        "PREDICTION",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(top = 16.dp),
                    )
                    Text(
                        result.finalDisease ?: "Unresolved",
                        style = MaterialTheme.typography.displayMedium,
                        fontWeight = FontWeight.SemiBold,
                        modifier = Modifier.padding(top = 4.dp),
                    )
                    Text(
                        decisionPathLabel(result),
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(top = 4.dp),
                    )

                    result.finalConfidence?.let {
                        ConfidenceBar(confidence = it, modifier = Modifier.fillMaxWidth().padding(top = 20.dp))
                    }

                    if (!result.cnnProbabilities.isNullOrEmpty()) {
                        Text(
                            "Other possibilities",
                            style = MaterialTheme.typography.titleMedium,
                            modifier = Modifier.padding(top = 28.dp, bottom = 10.dp),
                        )
                        RiceGuardCard(modifier = Modifier.fillMaxWidth()) {
                            TopPredictions(probabilities = result.cnnProbabilities)
                        }
                    }

                    if (result.detectionUrl != null && result.detections.isNotEmpty()) {
                        Text(
                            "Detected regions",
                            style = MaterialTheme.typography.titleMedium,
                            modifier = Modifier.padding(top = 28.dp, bottom = 10.dp),
                        )
                        RiceGuardCard(modifier = Modifier.fillMaxWidth(), contentPadding = PaddingValues(0.dp)) {
                            AsyncImage(
                                model = result.detectionUrl,
                                contentDescription = "YOLO detection visualization",
                                modifier = Modifier.fillMaxWidth().aspectRatio(1f).clip(MaterialTheme.shapes.large),
                                contentScale = ContentScale.Fit,
                            )
                        }
                    }

                    Row(
                        modifier = Modifier.fillMaxWidth().padding(top = 24.dp),
                        horizontalArrangement = Arrangement.spacedBy(12.dp),
                    ) {
                        ResultActionCard(
                            icon = Icons.Filled.Visibility,
                            label = "Why this result",
                            onClick = onViewExplanation,
                            modifier = Modifier.weight(1f),
                        )
                        ResultActionCard(
                            icon = Icons.Filled.Spa,
                            label = "Treatment guidance",
                            onClick = onViewRecommendation,
                            modifier = Modifier.weight(1f),
                        )
                    }

                    PrimaryButton(text = "Done", onClick = onDone, modifier = Modifier.fillMaxWidth().padding(top = 28.dp))
                }
            }
        }
    }
}

private fun decisionPathLabel(result: ScanResult): String {
    if (result.detections.isEmpty()) return "No specific region detected — classified from the full image"
    val base = "YOLO detected ${result.detections.size} region(s), verified by the CNN"
    return if (result.agreement) base else "$base — models disagreed on the top region"
}

/** project brief section 25 (never claim certainty beyond what the model
 * supports): the CNN's full probability distribution already comes back on
 * every response (regions[].cnn_probabilities) but was previously discarded
 * client-side -- showing the top 3 lets the user see what the model
 * considered instead of only the single winning class. */
@Composable
private fun TopPredictions(probabilities: Map<String, Float>) {
    val top3 = probabilities.entries.sortedByDescending { it.value }.take(3)
    val maxValue = top3.maxOfOrNull { it.value } ?: 1f
    Column {
        top3.forEachIndexed { index, (className, confidence) ->
            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth().padding(vertical = 6.dp)) {
                Text(
                    className,
                    style = MaterialTheme.typography.bodyMedium,
                    modifier = Modifier.weight(1f),
                )
                Box(
                    modifier = Modifier
                        .width((60 * (confidence / maxValue).coerceIn(0.05f, 1f)).dp)
                        .height(6.dp)
                        .clip(RoundedCornerShape(100))
                        .background(if (index == 0) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outlineVariant),
                )
                Text(
                    "${(confidence * 100).toInt()}%",
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.padding(start = 10.dp).width(44.dp),
                )
            }
        }
    }
}

@Composable
private fun ResultActionCard(icon: ImageVector, label: String, onClick: () -> Unit, modifier: Modifier = Modifier) {
    RiceGuardCard(modifier = modifier.height(96.dp).clickable(onClick = onClick)) {
        Icon(icon, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
        Text(label, style = MaterialTheme.typography.labelLarge, modifier = Modifier.padding(top = 10.dp))
    }
}
