package com.riceguard.ai.ui.screens.recommendation

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Build
import androidx.compose.material.icons.filled.BugReport
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Shield
import androidx.compose.material.icons.filled.Sync
import androidx.compose.material.icons.filled.Visibility
import androidx.compose.material.icons.filled.WarningAmber
import androidx.compose.material.icons.filled.WbCloudy
import androidx.compose.ui.graphics.vector.ImageVector
import com.riceguard.ai.domain.model.Recommendation

private const val UNAVAILABLE_MESSAGE = "Not available in the current knowledge base."

/** One card in the Disease Information screen. [bullets] is empty and
 * [available] is false for a category this project's knowledge base
 * (recommendation/knowledge_base.json) does not currently carry data for --
 * shown honestly as unavailable rather than inferred or invented. */
data class DiseaseInfoSection(
    val icon: ImageVector,
    val title: String,
    val paragraph: String? = null,
    val bullets: List<String> = emptyList(),
    val available: Boolean = true,
)

/** Maps the 7 fields the knowledge base actually has (causal_agent,
 * description, symptoms, management, treatment, prevention,
 * requires_expert_validation) onto the requested 8-category layout.
 * favorableConditions/transmission/cropImpact have no dedicated field in the
 * knowledge base today -- they are shown as explicitly unavailable, never
 * backfilled with invented agronomic content. */
fun buildDiseaseInfoSections(recommendation: Recommendation): List<DiseaseInfoSection> = listOf(
    DiseaseInfoSection(
        icon = Icons.Filled.Info,
        title = "What is this?",
        paragraph = recommendation.description,
        available = !recommendation.description.isNullOrBlank(),
    ),
    DiseaseInfoSection(
        icon = Icons.Filled.BugReport,
        title = "Probable Cause",
        paragraph = recommendation.causalAgent,
        available = !recommendation.causalAgent.isNullOrBlank(),
    ),
    DiseaseInfoSection(
        icon = Icons.Filled.Visibility,
        title = "Symptoms",
        bullets = recommendation.symptoms,
        available = recommendation.symptoms.isNotEmpty(),
    ),
    DiseaseInfoSection(icon = Icons.Filled.WbCloudy, title = "Favorable Conditions", available = false),
    DiseaseInfoSection(icon = Icons.Filled.Sync, title = "Spread / Transmission", available = false),
    DiseaseInfoSection(icon = Icons.Filled.WarningAmber, title = "Risk & Crop Impact", available = false),
    DiseaseInfoSection(
        icon = Icons.Filled.Build,
        title = "Management",
        bullets = recommendation.management + recommendation.treatment,
        available = recommendation.management.isNotEmpty() || recommendation.treatment.isNotEmpty(),
    ),
    DiseaseInfoSection(
        icon = Icons.Filled.Shield,
        title = "Prevention",
        bullets = recommendation.prevention,
        available = recommendation.prevention.isNotEmpty(),
    ),
).let { sections ->
    if (recommendation.disease == "Healthy") sections.filter { it.title != "Risk & Crop Impact" } else sections
}

internal fun unavailableMessage() = UNAVAILABLE_MESSAGE
