package com.riceguard.ai.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.riceguard.ai.data.repository.ConnectionState
import com.riceguard.ai.ui.theme.riceGuardStatus

@Composable
fun StatusChip(
    label: String,
    color: Color,
    containerColor: Color,
    modifier: Modifier = Modifier,
    showDot: Boolean = true,
) {
    Row(
        modifier = modifier
            .clip(RoundedCornerShape(100))
            .background(containerColor)
            .padding(horizontal = 12.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (showDot) {
            Box(modifier = Modifier.size(8.dp).clip(CircleShape).background(color))
            Spacer(Modifier.size(6.dp))
        }
        Text(label, style = MaterialTheme.typography.labelMedium, color = color)
    }
}

@Composable
fun ConnectionStatusChip(state: ConnectionState, modifier: Modifier = Modifier) {
    val status = riceGuardStatus
    val (label, color, container) = when (state) {
        ConnectionState.CONNECTED -> Triple("Connected", MaterialTheme.colorScheme.primary, MaterialTheme.colorScheme.primaryContainer)
        ConnectionState.SEARCHING -> Triple("Searching…", MaterialTheme.colorScheme.onSurfaceVariant, MaterialTheme.colorScheme.surfaceVariant)
        ConnectionState.FOUND -> Triple("Server found", MaterialTheme.colorScheme.onSurfaceVariant, MaterialTheme.colorScheme.surfaceVariant)
        ConnectionState.CHECKING -> Triple("Connecting…", MaterialTheme.colorScheme.onSurfaceVariant, MaterialTheme.colorScheme.surfaceVariant)
        ConnectionState.OFFLINE -> Triple("Server unavailable", status.critical, status.criticalContainer)
        ConnectionState.NOT_CONFIGURED -> Triple("Server Not Configured", status.warning, status.warningContainer)
        ConnectionState.UNKNOWN -> Triple("Connecting…", MaterialTheme.colorScheme.onSurfaceVariant, MaterialTheme.colorScheme.surfaceVariant)
    }
    StatusChip(label = label, color = color, containerColor = container, modifier = modifier)
}
