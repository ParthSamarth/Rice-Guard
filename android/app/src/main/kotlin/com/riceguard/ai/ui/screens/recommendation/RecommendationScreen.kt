package com.riceguard.ai.ui.screens.recommendation

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Shield
import androidx.compose.material.icons.filled.WarningAmber
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.riceguard.ai.ui.components.EmptyState
import com.riceguard.ai.ui.components.RiceGuardCard
import com.riceguard.ai.ui.components.ScreenTopBar
import com.riceguard.ai.ui.navigation.ScanSessionViewModel
import com.riceguard.ai.ui.navigation.ScanUploadState
import com.riceguard.ai.ui.theme.riceGuardStatus

/** project brief section 12, screen 8 / section 24: pure pass-through of the
 * rule-based recommendation engine's output. NEEDS_AUTHORITATIVE_VALIDATION
 * is never hidden or paraphrased away -- it is shown as its own explicit
 * warning banner whenever requiresExpertValidation is true. Deliberately
 * separate from ExplanationScreen (Grad-CAM/model attention): this screen
 * explains the DISEASE, not what image regions influenced the prediction. */
@Composable
fun RecommendationScreen(scanSession: ScanSessionViewModel, onBack: () -> Unit) {
    val uploadState by scanSession.uploadState.collectAsState()
    val result = (uploadState as? ScanUploadState.Success)?.result
    val recommendation = result?.recommendation

    Scaffold(topBar = { ScreenTopBar(title = "Disease Information", onBack = onBack) }) { padding ->
        if (recommendation == null || !recommendation.available) {
            EmptyState(
                icon = Icons.Filled.Shield,
                title = "No recommendation available",
                message = recommendation?.unavailableReason
                    ?: "This disease doesn't have a matching entry in the treatment guide.",
                modifier = Modifier.padding(padding),
            )
            return@Scaffold
        }

        LazyColumn(modifier = Modifier.fillMaxSize().padding(padding).padding(20.dp)) {
            item {
                Text(recommendation.disease.orEmpty(), style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.SemiBold)
                result.finalConfidence?.let { confidence ->
                    Row(modifier = Modifier.padding(top = 6.dp)) {
                        Text(
                            "Prediction confidence: ",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        Text(
                            "${(confidence * 100).toInt()}%",
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = FontWeight.SemiBold,
                        )
                    }
                }
                Text(
                    "This is a model prediction, not a confirmed diagnosis.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 6.dp),
                )
            }

            if (recommendation.requiresExpertValidation) {
                item {
                    RiceGuardCard(
                        modifier = Modifier.fillMaxWidth().padding(top = 20.dp),
                        containerColor = riceGuardStatus.warningContainer,
                    ) {
                        Row {
                            Icon(Icons.Filled.WarningAmber, contentDescription = null, tint = riceGuardStatus.warning)
                            Column(modifier = Modifier.padding(start = 10.dp)) {
                                Text(
                                    "NEEDS AUTHORITATIVE VALIDATION",
                                    style = MaterialTheme.typography.titleSmall,
                                    color = riceGuardStatus.warning,
                                    fontWeight = FontWeight.Bold,
                                )
                                Text(
                                    "Treatment guidance below has not been validated by an agricultural authority. " +
                                        "Consult a licensed extension officer before applying any treatment.",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurface,
                                    modifier = Modifier.padding(top = 4.dp),
                                )
                            }
                        }
                    }
                }
            }

            items(buildDiseaseInfoSections(recommendation)) { section ->
                DiseaseInfoCard(section, modifier = Modifier.animateItem())
            }

            recommendation.disclaimer?.let { disclaimer ->
                item {
                    Text(
                        disclaimer,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(top = 24.dp, bottom = 12.dp),
                    )
                }
            }
        }
    }
}

@Composable
private fun DiseaseInfoCard(section: DiseaseInfoSection, modifier: Modifier = Modifier) {
    RiceGuardCard(modifier = modifier.fillMaxWidth().padding(top = 16.dp)) {
        Row {
            Icon(section.icon, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
            Text(
                section.title,
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
                modifier = Modifier.padding(start = 10.dp),
            )
        }
        if (!section.available) {
            Text(
                unavailableMessage(),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(top = 10.dp),
            )
            return@RiceGuardCard
        }
        section.paragraph?.let {
            Text(it, style = MaterialTheme.typography.bodyMedium, modifier = Modifier.padding(top = 10.dp))
        }
        section.bullets.forEach { line ->
            Row(modifier = Modifier.fillMaxWidth().padding(top = 8.dp)) {
                Text("•", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.primary)
                Text(line, style = MaterialTheme.typography.bodyMedium, modifier = Modifier.padding(start = 10.dp))
            }
        }
    }
}
