package com.riceguard.ai.ui.components

import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.height
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

/** ~0.97x -- present on every tap without reading as a "squish" (project
 * brief animation pass: subtle tactile feedback, never bouncy). Shared by
 * all three button styles below so the whole app's buttons feel consistent. */
private const val PRESS_SCALE = 0.97f

@Composable
private fun rememberPressScale(interactionSource: MutableInteractionSource): Float {
    val pressed by interactionSource.collectIsPressedAsState()
    val scale by animateFloatAsState(
        targetValue = if (pressed) PRESS_SCALE else 1f,
        animationSpec = spring(dampingRatio = Spring.DampingRatioNoBouncy, stiffness = Spring.StiffnessMedium),
        label = "buttonPressScale",
    )
    return scale
}

@Composable
fun PrimaryButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    loading: Boolean = false,
    containerColor: Color = MaterialTheme.colorScheme.primary,
    /** Callers representing a stronger confirmation (e.g. "Use Photo",
     * saving a scan) can pass SUCCESS; everything else defaults to the same
     * light tap every button in the app uses. */
    hapticStyle: RiceGuardHapticStyle = RiceGuardHapticStyle.LIGHT,
) {
    val performHaptic = rememberHapticPerformer()
    val interactionSource = remember { MutableInteractionSource() }
    val scale = rememberPressScale(interactionSource)

    Button(
        onClick = {
            performHaptic(hapticStyle)
            onClick()
        },
        enabled = enabled && !loading,
        modifier = modifier.height(56.dp).scale(scale),
        shape = MaterialTheme.shapes.large,
        colors = ButtonDefaults.buttonColors(containerColor = containerColor),
        contentPadding = PaddingValues(horizontal = 24.dp),
        interactionSource = interactionSource,
    ) {
        if (loading) {
            CircularProgressIndicator(
                modifier = Modifier.height(20.dp),
                strokeWidth = 2.dp,
                color = MaterialTheme.colorScheme.onPrimary,
            )
        } else {
            Text(text, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
        }
    }
}

@Composable
fun SecondaryButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
) {
    val performHaptic = rememberHapticPerformer()
    val interactionSource = remember { MutableInteractionSource() }
    val scale = rememberPressScale(interactionSource)

    OutlinedButton(
        onClick = {
            performHaptic(RiceGuardHapticStyle.LIGHT)
            onClick()
        },
        enabled = enabled,
        modifier = modifier.height(56.dp).scale(scale),
        shape = MaterialTheme.shapes.large,
        contentPadding = PaddingValues(horizontal = 24.dp),
        interactionSource = interactionSource,
    ) {
        Text(text, style = MaterialTheme.typography.titleMedium)
    }
}

@Composable
fun TextOnlyButton(text: String, onClick: () -> Unit, modifier: Modifier = Modifier) {
    val performHaptic = rememberHapticPerformer()
    TextButton(
        onClick = {
            performHaptic(RiceGuardHapticStyle.LIGHT)
            onClick()
        },
        modifier = modifier,
    ) {
        Text(text, style = MaterialTheme.typography.titleSmall)
    }
}
