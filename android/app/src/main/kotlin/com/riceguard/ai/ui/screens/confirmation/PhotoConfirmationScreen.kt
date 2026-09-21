package com.riceguard.ai.ui.screens.confirmation

import android.graphics.BitmapFactory
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.WarningAmber
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import com.riceguard.ai.domain.model.ScanSource
import com.riceguard.ai.ui.components.PrimaryButton
import com.riceguard.ai.ui.components.RiceGuardCard
import com.riceguard.ai.ui.components.RiceGuardHapticStyle
import com.riceguard.ai.ui.components.ScreenTopBar
import com.riceguard.ai.ui.components.SecondaryButton
import com.riceguard.ai.ui.navigation.ScanSessionViewModel
import com.riceguard.ai.ui.theme.riceGuardStatus
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

private const val DEFAULT_ASPECT_RATIO = 3f / 4f
private const val MIN_ASPECT_RATIO = 0.5f   // 9:16-ish, tall portrait
private const val MAX_ASPECT_RATIO = 1.6f   // wider than 3:2, a generous landscape cap

// Budget left for the quality card + privacy line below the photo, so the
// photo itself can be sized to fill whatever vertical room remains instead
// of sitting short (full-width but aspect-ratio-short for wide/landscape
// shots) with a dead gap above the buttons.
private val BELOW_IMAGE_RESERVED_HEIGHT = 190.dp

/** project brief section 8: nothing is sent to the server before "Use
 * Photo" -- this screen only ever reads the already-local staged file. The
 * card below is sized to the photo's own aspect ratio (read via a
 * bounds-only decode, same technique as ImageQualityChecker) rather than
 * forced into a fixed box, so a portrait capture is never cropped just to
 * fill a square. */
@Composable
fun PhotoConfirmationScreen(
    scanSession: ScanSessionViewModel,
    onRetake: () -> Unit,
    onUsePhoto: () -> Unit,
) {
    val photo by scanSession.capturedPhoto.collectAsState()
    val source by scanSession.photoSource.collectAsState()
    var quality by remember { mutableStateOf<ImageQualityResult?>(null) }
    var aspectRatio by remember { mutableFloatStateOf(DEFAULT_ASPECT_RATIO) }
    val imageAlpha = remember { Animatable(0f) }

    LaunchedEffect(photo) {
        val file = photo ?: return@LaunchedEffect
        imageAlpha.snapTo(0f)
        quality = withContext(Dispatchers.Default) { ImageQualityChecker.check(file) }
        aspectRatio = withContext(Dispatchers.Default) { readAspectRatio(file) }
        imageAlpha.animateTo(1f, animationSpec = tween(durationMillis = 350))
    }

    Scaffold(topBar = { ScreenTopBar(title = "Review Photo", onBack = onRetake) }) { padding ->
        Column(modifier = Modifier.fillMaxSize().padding(padding)) {
            BoxWithConstraints(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .padding(horizontal = 20.dp, vertical = 16.dp),
            ) {
                // Fit the photo to the available box (bounded by screen width
                // and the room left after reserving space for the card below)
                // instead of always going full-width -- a wide/landscape shot
                // then grows to use the vertical space it has, rather than
                // sitting short and leaving a dead gap above the buttons.
                val maxImageHeight = (maxHeight - BELOW_IMAGE_RESERVED_HEIGHT).coerceAtLeast(160.dp)
                val imageWidth = (maxImageHeight * aspectRatio).coerceAtMost(maxWidth)
                val imageHeight = imageWidth / aspectRatio

                Column(
                    modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center,
                ) {
                    Surface(
                        modifier = Modifier.width(imageWidth).height(imageHeight).alpha(imageAlpha.value),
                        shape = RoundedCornerShape(20.dp),
                        color = MaterialTheme.colorScheme.surfaceVariant,
                        shadowElevation = 1.dp,
                        border = BorderStroke(0.75.dp, MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.4f)),
                    ) {
                        if (photo != null) {
                            AsyncImage(
                                model = photo,
                                contentDescription = "Captured rice leaf photo",
                                modifier = Modifier.fillMaxSize(),
                                // Safe: the container's own aspect ratio already
                                // matches the source image's, so Crop here fills
                                // the rounded card edge-to-edge without actually
                                // trimming any of the frame.
                                contentScale = ContentScale.Crop,
                            )
                        }
                    }

                    QualityGuidanceCard(quality = quality, modifier = Modifier.fillMaxWidth().padding(top = 16.dp))

                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.fillMaxWidth().padding(top = 16.dp),
                    ) {
                        Icon(
                            Icons.Filled.Lock,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.size(14.dp),
                        )
                        Text(
                            "Your photo stays on this device until you confirm.",
                            style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.padding(start = 6.dp),
                        )
                    }
                }
            }

            Row(
                modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp).padding(bottom = 20.dp, top = 8.dp),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                SecondaryButton(
                    text = if (source == ScanSource.GALLERY) "Choose Different" else "Retake",
                    onClick = onRetake,
                    modifier = Modifier.weight(1f),
                )
                PrimaryButton(
                    text = if (quality?.warnings.isNullOrEmpty()) "Use Photo" else "Use Anyway",
                    onClick = onUsePhoto,
                    modifier = Modifier.weight(1f),
                    enabled = photo != null,
                    hapticStyle = RiceGuardHapticStyle.SUCCESS,
                )
            }
        }
    }
}

@Composable
private fun QualityGuidanceCard(quality: ImageQualityResult?, modifier: Modifier = Modifier) {
    val warnings = quality?.warnings.orEmpty()
    RiceGuardCard(modifier = modifier, contentPadding = PaddingValues(16.dp)) {
        if (warnings.isEmpty()) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Filled.CheckCircle, contentDescription = null, tint = riceGuardStatus.good)
                Column(modifier = Modifier.padding(start = 10.dp)) {
                    Text("Looks good", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
                    Text(
                        "Clear, well-lit, and ready to analyze.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(top = 2.dp),
                    )
                }
            }
        } else {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Filled.WarningAmber, contentDescription = null, tint = riceGuardStatus.warning)
                Text(
                    "Double-check before continuing",
                    style = MaterialTheme.typography.titleSmall,
                    color = riceGuardStatus.warning,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.padding(start = 10.dp),
                )
            }
            warnings.forEach {
                Text(
                    "• $it",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurface,
                    modifier = Modifier.padding(top = 6.dp, start = 28.dp),
                )
            }
        }
    }
}

private fun readAspectRatio(file: File): Float {
    val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    BitmapFactory.decodeFile(file.absolutePath, options)
    if (options.outWidth <= 0 || options.outHeight <= 0) return DEFAULT_ASPECT_RATIO
    return (options.outWidth.toFloat() / options.outHeight).coerceIn(MIN_ASPECT_RATIO, MAX_ASPECT_RATIO)
}
