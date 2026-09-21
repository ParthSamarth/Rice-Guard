package com.riceguard.ai.ui.theme

import android.app.Activity
import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

private val LightColors = lightColorScheme(
    primary = ForestGreen40,
    onPrimary = Ivory99,
    primaryContainer = ForestGreen90,
    onPrimaryContainer = ForestGreen20,
    secondary = RiceGold60,
    onSecondary = Ivory99,
    secondaryContainer = RiceGold90,
    onSecondaryContainer = RiceGold40,
    tertiary = ForestGreen60,
    onTertiary = Ivory99,
    background = Ivory95,
    onBackground = Neutral10,
    surface = Ivory99,
    onSurface = Neutral10,
    surfaceVariant = Ivory90,
    onSurfaceVariant = Neutral40,
    outline = Neutral60,
    outlineVariant = Neutral80,
    error = StatusError,
    onError = Ivory99,
    errorContainer = StatusErrorBg,
    onErrorContainer = StatusError,
)

private val DarkColors = darkColorScheme(
    primary = ForestGreen70,
    onPrimary = ForestGreen10,
    primaryContainer = ForestGreen30,
    onPrimaryContainer = ForestGreen90,
    secondary = RiceGold70,
    onSecondary = Color(0xFF241B02),
    secondaryContainer = RiceGold40,
    onSecondaryContainer = RiceGold90,
    tertiary = ForestGreen80,
    onTertiary = ForestGreen10,
    background = Ivory10,
    onBackground = Neutral95,
    surface = Ivory20,
    onSurface = Neutral95,
    surfaceVariant = Neutral20,
    onSurfaceVariant = Neutral80,
    outline = Neutral50,
    outlineVariant = Neutral40,
    error = Color(0xFFE0897C),
    onError = Color(0xFF3A1712),
    errorContainer = Color(0xFF3A1712),
    onErrorContainer = Color(0xFFE0897C),
)

/** Extended status-color accessor -- MaterialTheme.colorScheme has no
 * "warning"/"caution" role, so uncertainty states (project brief sections
 * 21/25) read this instead of repurposing a brand color. */
val riceGuardStatus: RiceGuardStatusColors
    @Composable get() = LocalRiceGuardStatusColors.current

@Composable
fun RiceGuardTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = false, // off by default -- this app has a deliberate brand palette, not device-wallpaper theming
    content: @Composable () -> Unit,
) {
    val context = LocalContext.current
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S ->
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        darkTheme -> DarkColors
        else -> LightColors
    }
    val statusColors = if (darkTheme) DarkStatusColors else LightStatusColors

    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            window.statusBarColor = colorScheme.background.toArgb()
            WindowCompat.getInsetsController(window, view).isAppearanceLightStatusBars = !darkTheme
        }
    }

    CompositionLocalProvider(LocalRiceGuardStatusColors provides statusColors) {
        MaterialTheme(
            colorScheme = colorScheme,
            typography = RiceGuardTypography,
            shapes = RiceGuardShapes,
            content = content,
        )
    }
}
