package com.riceguard.ai.ui.components

import android.view.HapticFeedbackConstants
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.LocalView

/** Compose's own `HapticFeedbackType` only reliably exposes `LongPress` on
 * this project's Compose BOM (2024.09.00) -- there's no built-in way to
 * distinguish "light tap" from "success" from "warning" through it. This
 * goes straight to the platform's `View.performHapticFeedback`, which has
 * offered that distinction since `HapticFeedbackConstants.CONFIRM`/`REJECT`
 * (API 30) -- calling those on an older device is a documented no-op, not a
 * crash, so no SDK_INT guard is needed even though minSdk is 26. */
enum class RiceGuardHapticStyle(val constant: Int) {
    /** Every ordinary button/toggle tap. */
    LIGHT(HapticFeedbackConstants.VIRTUAL_KEY),

    /** A meaningful positive confirmation -- e.g. "Use Photo", a completed
     * analysis, saving a scan. Used sparingly, never for routine taps. */
    SUCCESS(HapticFeedbackConstants.CONFIRM),

    /** A gentle heads-up, not an alarm -- e.g. a low-confidence result, the
     * server going unavailable. */
    WARNING(HapticFeedbackConstants.REJECT),
}

@Composable
fun rememberHapticPerformer(): (RiceGuardHapticStyle) -> Unit {
    val view = LocalView.current
    return { style -> view.performHapticFeedback(style.constant) }
}
