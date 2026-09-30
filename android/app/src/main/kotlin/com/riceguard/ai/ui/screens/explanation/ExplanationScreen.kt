package com.riceguard.ai.ui.screens.explanation

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Info
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import com.riceguard.ai.ui.components.RiceGuardCard
import com.riceguard.ai.ui.components.ScreenTopBar
import com.riceguard.ai.ui.navigation.ScanSessionViewModel
import com.riceguard.ai.ui.navigation.ScanUploadState

/** The server saves this as one wide comparison figure -- Original |
 * Grad-CAM heatmap | Overlay, side by side (see explainability/gradcam.py:
 * explain_image, matplotlib figsize=(12, 4.2) @ 150dpi = 1800x630px) -- so it
 * must be displayed at (close to) its own aspect ratio. A square crop here
 * previously cut off most of the original/overlay panels, leaving only a
 * sliver of the bare heatmap visible with no leaf to compare it against,
 * which is what made attention look "unaligned" regardless of the
 * underlying Grad-CAM computation. */
private const val GRADCAM_PANEL_ASPECT_RATIO = 1800f / 630f

/** project brief section 12, screen 7. Grad-CAM explains the CNN's decision
 * only -- this screen is written to never imply it proves the prediction is
 * correct (see PROJECT_README.md §7.3 for why: this project's own Grad-CAM
 * review found high-confidence CORRECT predictions whose attention was not
 * actually on the lesion). */
@Composable
fun ExplanationScreen(scanSession: ScanSessionViewModel, onBack: () -> Unit) {
    val uploadState by scanSession.uploadState.collectAsState()
    val capturedPhoto by scanSession.capturedPhoto.collectAsState()
    val result = (uploadState as? ScanUploadState.Success)?.result

    Scaffold(topBar = { ScreenTopBar(title = "Explanation", onBack = onBack) }) { padding ->
        LazyColumn(modifier = Modifier.fillMaxSize().padding(padding).padding(20.dp)) {
            item {
                Text("Original photo", style = MaterialTheme.typography.titleSmall, modifier = Modifier.padding(bottom = 8.dp))
                AsyncImage(
                    model = capturedPhoto,
                    contentDescription = "Original photo",
                    modifier = Modifier.fillMaxWidth().aspectRatio(1f).clip(MaterialTheme.shapes.large),
                    contentScale = ContentScale.Crop,
                )
            }

            item {
                Text(
                    "Model attention (Grad-CAM)",
                    style = MaterialTheme.typography.titleSmall,
                    modifier = Modifier.padding(top = 24.dp, bottom = 8.dp),
                )
                if (result?.gradcamUrl != null) {
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
                    val panelAlpha = remember { Animatable(0f) }
                    LaunchedEffect(result.gradcamUrl) {
                        panelAlpha.snapTo(0f)
                        panelAlpha.animateTo(1f, animationSpec = tween(durationMillis = 400))
                    }
                    AsyncImage(
                        model = result.gradcamUrl,
                        contentDescription = "Model attention: original photo, heatmap, and overlay side by side",
                        modifier = Modifier
                            .fillMaxWidth()
                            .aspectRatio(GRADCAM_PANEL_ASPECT_RATIO)
                            .clip(MaterialTheme.shapes.large)
                            .alpha(panelAlpha.value),
                        contentScale = ContentScale.Fit,
                    )
                    AttentionLegend(modifier = Modifier.fillMaxWidth().padding(top = 12.dp))
                } else {
                    RiceGuardCard(modifier = Modifier.fillMaxWidth()) {
                        Text("No explanation image is available for this scan.", style = MaterialTheme.typography.bodyMedium)
                    }
                }
            }

            item {
                RiceGuardCard(modifier = Modifier.fillMaxWidth().padding(top = 20.dp)) {
                    Text(
                        "The highlighted areas are model attention, not a diagnosis of the disease region.",
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.Medium,
                    )
                    Row(modifier = Modifier.padding(top = 12.dp)) {
                        Icon(
                            Icons.Filled.Info,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.padding(end = 8.dp),
                        )
                        Text(
                            "This is a diagnostic aid, not proof the result is correct. The model can be confident, " +
                                "and even right, without its attention being on the actual lesion — treat this as " +
                                "supporting context, not a guarantee.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
            }
        }
    }
}

/** The server's heatmap panel uses OpenCV's COLORMAP_JET (see
 * explainability/gradcam.py:overlay_heatmap) -- blue at the low end through
 * red at the high end. This mirrors that exact ramp so the legend actually
 * matches what's drawn, rather than a generic gradient. */
@Composable
private fun AttentionLegend(modifier: Modifier = Modifier) {
    val jetRamp = Brush.horizontalGradient(
        listOf(
            Color(0xFF00008F), Color(0xFF0000FF), Color(0xFF00FFFF),
            Color(0xFF7FFF7F), Color(0xFFFFFF00), Color(0xFFFF0000), Color(0xFF800000),
        ),
    )
    Column(modifier = modifier) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(10.dp)
                .clip(RoundedCornerShape(100))
                .background(jetRamp),
        )
        Row(modifier = Modifier.fillMaxWidth().padding(top = 4.dp), horizontalArrangement = Arrangement.SpaceBetween) {
            Text("Low attention", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Text("High attention", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}
