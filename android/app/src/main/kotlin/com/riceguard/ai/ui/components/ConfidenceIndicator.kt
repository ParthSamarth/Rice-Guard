package com.riceguard.ai.ui.components

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.riceguard.ai.ui.theme.riceGuardStatus

/** Confidence as a labeled bar, never a bare unlabeled number -- the color
 * itself communicates the confidence tier at a glance (project brief
 * section 25: never claim certainty; low values must visibly look uncertain,
 * not just say so in text).
 *
 * Animates from 0 on first appearance (not just on later value changes,
 * which a plain animateFloatAsState wouldn't do since Result composes once
 * with the final value already known) -- both the bar width and the percent
 * label count up together, so a scan reads as "the result arriving" rather
 * than a static number just appearing. The tier color is fixed to the final
 * value throughout the count-up (not recomputed per animated frame) so it
 * settles once and never flickers between tiers while it plays. */
@Composable
fun ConfidenceBar(confidence: Float, modifier: Modifier = Modifier, label: String = "Confidence") {
    val status = riceGuardStatus
    val target = confidence.coerceIn(0f, 1f)
    val color = when {
        target >= 0.75f -> MaterialTheme.colorScheme.primary
        target >= 0.5f -> status.warning
        else -> status.critical
    }
    val animated = remember { Animatable(0f) }

    LaunchedEffect(target) {
        animated.animateTo(target, animationSpec = tween(durationMillis = 700, easing = FastOutSlowInEasing))
    }

    Column(modifier = modifier) {
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text(label, style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Text(
                "${(animated.value * 100).toInt()}%",
                style = MaterialTheme.typography.labelLarge,
                color = color,
                fontWeight = FontWeight.SemiBold,
            )
        }
        Spacer(Modifier.height(6.dp))
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(8.dp)
                .clip(RoundedCornerShape(100))
                .background(MaterialTheme.colorScheme.surfaceVariant),
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth(animated.value)
                    .height(8.dp)
                    .clip(RoundedCornerShape(100))
                    .background(color),
            )
        }
    }
}
