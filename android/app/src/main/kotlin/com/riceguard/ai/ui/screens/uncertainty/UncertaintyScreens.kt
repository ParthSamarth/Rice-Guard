package com.riceguard.ai.ui.screens.uncertainty

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.HelpOutline
import androidx.compose.material.icons.filled.ImageNotSupported
import androidx.compose.material.icons.filled.RuleFolder
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.riceguard.ai.ui.components.ConfidenceBar
import com.riceguard.ai.ui.components.PrimaryButton
import com.riceguard.ai.ui.components.RiceGuardCard
import com.riceguard.ai.ui.components.SecondaryButton
import com.riceguard.ai.ui.navigation.ScanSessionViewModel
import com.riceguard.ai.ui.navigation.ScanUploadState
import com.riceguard.ai.ui.theme.riceGuardStatus

/** project brief section 12, screen 9. Reached only when
 * ScanResult.status == LOW_CONFIDENCE -- the CNN gave an answer, but below
 * this project's own deployment confidence threshold, so it is shown as an
 * explicit uncertainty state rather than a normal result (project brief
 * section 25: never claim certainty the model doesn't have). */
@Composable
fun LowConfidenceScreen(scanSession: ScanSessionViewModel, onRetake: () -> Unit, onDone: () -> Unit) {
    val uploadState by scanSession.uploadState.collectAsState()
    val result = (uploadState as? ScanUploadState.Success)?.result

    Scaffold { padding ->
        Column(
            modifier = Modifier.fillMaxSize().padding(padding).padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            Icon(
                Icons.Filled.HelpOutline,
                contentDescription = null,
                modifier = Modifier.size(56.dp),
                tint = riceGuardStatus.warning,
            )
            Text(
                "Low confidence",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.SemiBold,
                modifier = Modifier.padding(top = 16.dp),
            )
            Text(
                "The system is not confident enough to provide a reliable result.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(top = 8.dp, bottom = 24.dp),
            )

            if (result?.finalDisease != null) {
                RiceGuardCard(modifier = Modifier.fillMaxWidth()) {
                    Text(
                        "Best guess: ${result.finalDisease}",
                        style = MaterialTheme.typography.titleMedium,
                    )
                    result.finalConfidence?.let {
                        ConfidenceBar(confidence = it, modifier = Modifier.fillMaxWidth().padding(top = 14.dp))
                    }
                }
                Spacer(modifier = Modifier.height(20.dp))
            }

            Row(modifier = Modifier.fillMaxWidth().padding(top = 24.dp), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                SecondaryButton(text = "Not Now", onClick = onDone, modifier = Modifier.weight(1f))
                PrimaryButton(text = "Retake Photo", onClick = onRetake, modifier = Modifier.weight(1f))
            }
        }
    }
}

/** project brief section 12, screen 10. Reached only when
 * ScanResult.status == MODEL_DISAGREEMENT -- YOLO and the CNN disagreed on
 * the top region, and the system deliberately does not pick a winner
 * (PROJECT_README.md §5: "YOLO/CNN disagreement is never silently
 * resolved"). Both predictions are shown, unresolved. */
@Composable
fun ModelDisagreementScreen(scanSession: ScanSessionViewModel, onRetake: () -> Unit, onDone: () -> Unit) {
    val uploadState by scanSession.uploadState.collectAsState()
    val result = (uploadState as? ScanUploadState.Success)?.result

    Scaffold { padding ->
        Column(
            modifier = Modifier.fillMaxSize().padding(padding).padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            Icon(
                Icons.Filled.RuleFolder,
                contentDescription = null,
                modifier = Modifier.size(56.dp),
                tint = riceGuardStatus.caution,
            )
            Text(
                "Needs Review",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.SemiBold,
                modifier = Modifier.padding(top = 16.dp),
            )
            Text(
                "The detection and verification models disagree.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(top = 8.dp, bottom = 24.dp),
            )

            RiceGuardCard(modifier = Modifier.fillMaxWidth()) {
                DisagreementRow(label = "YOLO", prediction = result?.yoloPrediction, confidence = result?.yoloConfidence)
                HorizontalDivider(modifier = Modifier.padding(vertical = 14.dp))
                DisagreementRow(label = "CNN", prediction = result?.cnnPrediction, confidence = result?.cnnConfidence)
            }

            Row(modifier = Modifier.fillMaxWidth().padding(top = 24.dp), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                SecondaryButton(text = "Not Now", onClick = onDone, modifier = Modifier.weight(1f))
                PrimaryButton(text = "Retake Photo", onClick = onRetake, modifier = Modifier.weight(1f))
            }
        }
    }
}

/** project brief safety fix: reached when the domain gate (pipeline/
 * domain_gate.py) rejected the image before disease classification ever
 * ran -- status == NOT_RECOGNIZED. Deliberately shows NOTHING from the
 * result: no disease name, no confidence, no Grad-CAM, no detected regions,
 * no treatment guidance. Unlike LowConfidenceScreen (a real disease guess,
 * just an uncertain one), there is no "best guess" here to show -- the
 * pipeline itself never produced one for this image. */
@Composable
fun NotRecognizedScreen(onRetake: () -> Unit, onDone: () -> Unit) {
    Scaffold { padding ->
        Column(
            modifier = Modifier.fillMaxSize().padding(padding).padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            Icon(
                Icons.Filled.ImageNotSupported,
                contentDescription = null,
                modifier = Modifier.size(56.dp),
                tint = riceGuardStatus.warning,
            )
            Text(
                "Unable to analyze this image",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.SemiBold,
                textAlign = TextAlign.Center,
                modifier = Modifier.padding(top = 16.dp),
            )
            Text(
                "RiceGuard could not identify a clear rice leaf.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center,
                modifier = Modifier.padding(top = 8.dp),
            )
            Text(
                "Please capture a close, well-lit photo of a rice leaf.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center,
                modifier = Modifier.padding(top = 4.dp, bottom = 24.dp),
            )

            Row(modifier = Modifier.fillMaxWidth().padding(top = 24.dp), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                SecondaryButton(text = "Not Now", onClick = onDone, modifier = Modifier.weight(1f))
                PrimaryButton(text = "Retake Photo", onClick = onRetake, modifier = Modifier.weight(1f))
            }
        }
    }
}

@Composable
private fun DisagreementRow(label: String, prediction: String?, confidence: Float?) {
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(label, style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Column(horizontalAlignment = Alignment.End) {
            Text(prediction ?: "—", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            confidence?.let {
                Text(
                    "${(it * 100).toInt()}% confidence",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}
