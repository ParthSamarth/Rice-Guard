package com.riceguard.ai.ui.theme

import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Shapes
import androidx.compose.ui.unit.dp

// "Large rounded cards, refined iconography, restrained shadows" (project
// brief section 11) -- one consistent rounding scale used everywhere rather
// than ad hoc corner radii per screen.
val RiceGuardShapes = Shapes(
    extraSmall = RoundedCornerShape(8.dp),
    small = RoundedCornerShape(12.dp),
    medium = RoundedCornerShape(16.dp),
    large = RoundedCornerShape(24.dp),
    extraLarge = RoundedCornerShape(32.dp),
)

val CardCornerRadius = 24.dp
val ChipCornerRadius = 100.dp // fully rounded pill
