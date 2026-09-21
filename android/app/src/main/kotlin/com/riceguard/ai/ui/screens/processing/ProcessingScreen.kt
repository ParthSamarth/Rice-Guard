package com.riceguard.ai.ui.screens.processing

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.RadioButtonUnchecked
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.riceguard.ai.domain.model.AppError
import com.riceguard.ai.domain.model.ScanStatus
import com.riceguard.ai.ui.components.PrimaryButton
import com.riceguard.ai.ui.components.SecondaryButton
import com.riceguard.ai.ui.navigation.ScanSessionViewModel
import com.riceguard.ai.ui.navigation.ScanUploadState
import kotlinx.coroutines.delay

private val STAGES = listOf(
    "Image uploaded",
    "Detecting disease",
    "Verifying diagnosis",
    "Generating explanation",
    "Preparing guidance",
)

/** project brief section 12, screen 5: named stages, no fake 0-100%
 * animation. The single /predict call has no server-side progress channel,
 * so stage advancement here is a PACED, HONEST approximation of the real
 * pipeline order (which really is detect -> verify -> explain -> recommend)
 * -- but it is bounded by the actual network result: it never advances past
 * the last stage, and never shows "done," until scanSession.uploadState
 * genuinely becomes Success/Failed. */
@Composable
fun ProcessingScreen(
    scanSession: ScanSessionViewModel,
    onDone: (ScanStatus) -> Unit,
    onRetryFailed: () -> Unit,
    onGiveUp: () -> Unit,
) {
    val uploadState by scanSession.uploadState.collectAsState()
    var stageIndex by remember { mutableIntStateOf(0) }

    LaunchedEffect(Unit) {
        if (scanSession.uploadState.value is ScanUploadState.Idle) {
            scanSession.submitPhoto()
        }
    }

    LaunchedEffect(uploadState) {
        if (uploadState is ScanUploadState.Uploading) {
            stageIndex = 0
            while (stageIndex < STAGES.lastIndex) {
                delay(1100)
                stageIndex++
            }
        }
    }

    LaunchedEffect(uploadState) {
        val state = uploadState
        if (state is ScanUploadState.Success) {
            stageIndex = STAGES.lastIndex
            delay(400) // let the last stage register visually before navigating
            onDone(state.result.status)
        }
    }

    Scaffold { padding ->
        when (val state = uploadState) {
            is ScanUploadState.Failed -> ProcessingFailed(state.error, onRetryFailed, onGiveUp, Modifier.padding(padding))
            else -> ProcessingInProgress(stageIndex, Modifier.padding(padding))
        }
    }
}

@Composable
private fun ProcessingInProgress(stageIndex: Int, modifier: Modifier) {
    Column(
        modifier = modifier.fillMaxSize().padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        CircularProgressIndicator(modifier = Modifier.size(48.dp), strokeWidth = 3.dp)
        Text(
            "Analyzing your photo",
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.SemiBold,
            modifier = Modifier.padding(top = 24.dp, bottom = 28.dp),
        )
        STAGES.forEachIndexed { i, label ->
            Row(
                modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                val done = i < stageIndex
                val current = i == stageIndex
                Icon(
                    imageVector = if (done) Icons.Filled.CheckCircle else Icons.Filled.RadioButtonUnchecked,
                    contentDescription = null,
                    tint = if (done || current) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outlineVariant,
                    modifier = Modifier.size(20.dp),
                )
                Text(
                    label,
                    style = MaterialTheme.typography.bodyLarge,
                    color = if (done || current) MaterialTheme.colorScheme.onBackground else MaterialTheme.colorScheme.onSurfaceVariant,
                    fontWeight = if (current) FontWeight.SemiBold else FontWeight.Normal,
                    modifier = Modifier.padding(start = 12.dp),
                )
            }
        }
    }
}

@Composable
private fun ProcessingFailed(
    error: AppError,
    onRetry: () -> Unit,
    onGiveUp: () -> Unit,
    modifier: Modifier,
) {
    val (title, message) = errorCopy(error)
    Column(
        modifier = modifier.fillMaxSize().padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text(title, style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.SemiBold)
        Text(
            message,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 8.dp, bottom = 28.dp),
        )
        if (error is AppError.Unknown && error.requestId != null) {
            Text(
                "Request ID: ${error.requestId}",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(bottom = 20.dp),
            )
        }
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            SecondaryButton(text = "Cancel", onClick = onGiveUp, modifier = Modifier.weight(1f))
            PrimaryButton(text = "Retry", onClick = onRetry, modifier = Modifier.weight(1f))
        }
    }
}

private fun errorCopy(error: AppError): Pair<String, String> = when (error) {
    is AppError.NoConnection ->
        "Cannot connect to the server." to error.reason
    is AppError.ServerUnavailable ->
        "Server unavailable." to "The server is starting up or its models aren't ready yet. Try again in a moment."
    is AppError.InvalidImage ->
        "This image could not be read." to error.reason
    is AppError.UploadFailed ->
        "Upload failed." to error.reason
    is AppError.ServerProcessingFailed ->
        "The server could not complete the analysis." to error.reason
    is AppError.Unknown ->
        "Something went wrong." to error.reason
}
