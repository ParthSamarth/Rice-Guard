package com.riceguard.ai.ui.screens.splash

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.scale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.riceguard.ai.R
import kotlinx.coroutines.delay

/** project brief section 12, screen 1: logo + tagline, then straight to
 * Home -- connection status and any server setup are Home/Settings'
 * responsibility (section 18), not a gate the splash screen blocks on. */
@Composable
fun SplashScreen(onFinished: () -> Unit) {
    val alpha = remember { Animatable(0f) }
    // Starts very slightly enlarged and settles to 1x as it fades in -- a
    // small, premium "arriving" reveal rather than a plain fade, still
    // restrained enough to read as a mark settling into place, not a bounce.
    val logoScale = remember { Animatable(1.08f) }

    LaunchedEffect(Unit) {
        alpha.animateTo(1f, animationSpec = tween(500, easing = FastOutSlowInEasing))
    }
    LaunchedEffect(Unit) {
        logoScale.animateTo(1f, animationSpec = tween(650, easing = FastOutSlowInEasing))
        delay(750)
        onFinished()
    }

    Box(modifier = Modifier.fillMaxSize().padding(24.dp)) {
        Column(
            modifier = Modifier.fillMaxSize(),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            Image(
                painter = painterResource(id = R.drawable.ic_launcher_foreground),
                contentDescription = "RiceGuard",
                modifier = Modifier.size(120.dp).alpha(alpha.value).scale(logoScale.value),
            )
            Text(
                "RiceGuard",
                style = MaterialTheme.typography.headlineLarge,
                color = MaterialTheme.colorScheme.onBackground,
                fontWeight = FontWeight.SemiBold,
                modifier = Modifier.padding(top = 20.dp).alpha(alpha.value),
            )
            Text(
                "Intelligent rice health at your fingertips",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(top = 8.dp).alpha(alpha.value),
            )
        }
        Text(
            "Developed by: Parth Samarth",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.align(Alignment.BottomCenter).alpha(alpha.value),
        )
    }
}
