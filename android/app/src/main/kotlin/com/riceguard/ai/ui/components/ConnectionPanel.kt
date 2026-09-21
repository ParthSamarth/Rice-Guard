package com.riceguard.ai.ui.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.riceguard.ai.data.repository.ConnectionDetails
import com.riceguard.ai.data.repository.ConnectionState
import java.text.DateFormat
import java.util.Date

/** Tapping the Home screen's connection indicator opens this (project brief
 * section 5) -- the one place a user can see exactly what RiceGuard found
 * (or didn't) without ever having to know or type an IP address themselves. */
@Composable
fun ConnectionPanel(
    state: ConnectionState,
    details: ConnectionDetails,
    errorDetail: String?,
    onRediscover: () -> Unit,
    onDismiss: () -> Unit,
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("RiceGuard Server", style = MaterialTheme.typography.titleLarge) },
        text = {
            Column {
                InfoRow("Status", connectionStatusLabel(state))
                if (!errorDetail.isNullOrBlank()) {
                    Text(
                        errorDetail,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(top = 2.dp, bottom = 8.dp),
                    )
                }
                InfoRow("Server name", details.serverName ?: "—")
                InfoRow("IP address", details.host ?: "—")
                InfoRow("Port", details.port?.toString() ?: "—")
                InfoRow(
                    "Last successful connection",
                    details.lastSuccessfulConnection?.let {
                        DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.SHORT).format(Date(it.timestampMillis))
                    } ?: "Never",
                )
                if (!details.autoDiscoveryEnabled) {
                    Text(
                        "Automatic discovery is off. Turn it on, or update the server address, from Settings.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(top = 10.dp),
                    )
                }
            }
        },
        confirmButton = {
            TextButton(onClick = onRediscover, enabled = details.autoDiscoveryEnabled) {
                Text("Rediscover", style = MaterialTheme.typography.titleSmall)
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) { Text("Close", style = MaterialTheme.typography.titleSmall) }
        },
    )
}

@Composable
private fun InfoRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Text(label, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(value, style = MaterialTheme.typography.bodyMedium)
    }
}

/** Shared by the compact Home-screen chip and this panel so the two never
 * drift out of sync with different wording for the same state. */
fun connectionStatusLabel(state: ConnectionState): String = when (state) {
    ConnectionState.CONNECTED -> "Connected"
    ConnectionState.SEARCHING -> "Searching…"
    ConnectionState.FOUND -> "Server found"
    ConnectionState.CHECKING -> "Connecting…"
    ConnectionState.OFFLINE -> "Server unavailable"
    ConnectionState.NOT_CONFIGURED -> "Not configured"
    ConnectionState.UNKNOWN -> "Connecting…"
}
