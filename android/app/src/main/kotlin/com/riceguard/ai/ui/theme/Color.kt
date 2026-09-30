package com.riceguard.ai.ui.theme

import androidx.compose.ui.graphics.Color

/**
 * RiceGuard AI palette (project brief section 11): forest green primary,
 * warm ivory surface, restrained rice-gold accent. Deliberately not a
 * generic "neon AI" or high-saturation dashboard palette -- every hue here
 * is desaturated enough to read as a considered agri-tech product, not a
 * demo. Semantic (status) colors are kept visually distinct from the brand
 * accent so a confidence/warning color never gets mistaken for a themed UI
 * element.
 */

// ---- Forest green (primary) ----
val ForestGreen10 = Color(0xFF04150E)
val ForestGreen20 = Color(0xFF0B2A1C)
val ForestGreen30 = Color(0xFF0F3D2A)
val ForestGreen40 = Color(0xFF15503A)
val ForestGreen50 = Color(0xFF1B6B4C)
val ForestGreen60 = Color(0xFF2E8B66)
val ForestGreen70 = Color(0xFF52A784)
val ForestGreen80 = Color(0xFF8FC7AA)
val ForestGreen90 = Color(0xFFD3EADD)
val ForestGreen95 = Color(0xFFE9F5EF)

// ---- Warm ivory (surface/background) ----
val Ivory10 = Color(0xFF1B1811)
val Ivory20 = Color(0xFF2A2519)
val Ivory90 = Color(0xFFF3EEE0)
val Ivory95 = Color(0xFFFAF6EC)
val Ivory99 = Color(0xFFFFFDF8)

// ---- Rice gold (accent -- used sparingly: highlights, confidence, CTAs) ----
val RiceGold40 = Color(0xFF7A5E14)
val RiceGold50 = Color(0xFF9C7A1D)
val RiceGold60 = Color(0xFFC9A227)
val RiceGold70 = Color(0xFFDDBB55)
val RiceGold80 = Color(0xFFEBD48F)
val RiceGold90 = Color(0xFFF6EBCC)

// ---- Neutrals (text/outline, green-biased so they never read as flat gray) ----
val Neutral10 = Color(0xFF191D1A)
val Neutral20 = Color(0xFF2D332F)
val Neutral40 = Color(0xFF5B6459)
val Neutral50 = Color(0xFF747D71)
val Neutral60 = Color(0xFF929B8E)
val Neutral80 = Color(0xFFC6CCC2)
val Neutral90 = Color(0xFFE2E6DE)
val Neutral95 = Color(0xFFF0F2EC)

// ---- Status (reserved -- never doubles as the brand accent) ----
val StatusGood = Color(0xFF2E8B66)        // "ok" -- reuses forest green, it IS the good state
val StatusWarningBg = Color(0xFFFBEFD8)
val StatusWarning = Color(0xFF9C6B0E)     // low_confidence
val StatusCautionBg = Color(0xFFF3E6EE)
val StatusCaution = Color(0xFF8A4B78)     // model_disagreement -- distinct hue from warning, same weight
val StatusErrorBg = Color(0xFFFBE7E4)
val StatusError = Color(0xFFB0392A)       // hard errors only
