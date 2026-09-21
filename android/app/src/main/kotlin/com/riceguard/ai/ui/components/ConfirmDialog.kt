package com.riceguard.ai.ui.components

import androidx.compose.material3.AlertDialog
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

/** Used by Delete Scan (section 15) and Delete All Scan Data (section 16) --
 * both require an explicit two-step confirmation, never an immediate delete
 * on first tap. */
@Composable
fun ConfirmDialog(
    title: String,
    message: String,
    confirmLabel: String,
    onConfirm: () -> Unit,
    onDismiss: () -> Unit,
    confirmColor: Color = MaterialTheme.colorScheme.error,
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(title, style = MaterialTheme.typography.titleLarge) },
        text = { Text(message, style = MaterialTheme.typography.bodyMedium) },
        confirmButton = {
            TextButton(onClick = onConfirm) {
                Text(confirmLabel, color = confirmColor, style = MaterialTheme.typography.titleSmall)
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) { Text("Cancel", style = MaterialTheme.typography.titleSmall) }
        },
    )
}
