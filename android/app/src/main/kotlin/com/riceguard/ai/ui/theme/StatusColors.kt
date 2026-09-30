package com.riceguard.ai.ui.theme

import androidx.compose.runtime.Immutable
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.graphics.Color

/**
 * Extended semantic colors Material3's default ColorScheme has no slot for
 * (it only has "error"). RiceGuard needs THREE distinct non-brand states --
 * good/ok, low_confidence, model_disagreement -- that must never be confused
 * with each other or with the brand accent (project brief sections 21/25:
 * uncertainty must be visually unmistakable, not a subtle tint).
 */
@Immutable
data class RiceGuardStatusColors(
    val good: Color,
    val goodContainer: Color,
    val warning: Color,
    val warningContainer: Color,
    val caution: Color,
    val cautionContainer: Color,
    val critical: Color,
    val criticalContainer: Color,
)

val LightStatusColors = RiceGuardStatusColors(
    good = StatusGood, goodContainer = ForestGreen95,
    warning = StatusWarning, warningContainer = StatusWarningBg,
    caution = StatusCaution, cautionContainer = StatusCautionBg,
    critical = StatusError, criticalContainer = StatusErrorBg,
)

val DarkStatusColors = RiceGuardStatusColors(
    good = ForestGreen70, goodContainer = ForestGreen20,
    warning = Color(0xFFE3B667), warningContainer = Color(0xFF3A2E12),
    caution = Color(0xFFD79BC4), cautionContainer = Color(0xFF3A1F32),
    critical = Color(0xFFE0897C), criticalContainer = Color(0xFF3A1712),
)

val LocalRiceGuardStatusColors = staticCompositionLocalOf { LightStatusColors }
