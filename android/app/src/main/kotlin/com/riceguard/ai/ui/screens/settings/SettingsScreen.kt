package com.riceguard.ai.ui.screens.settings

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.riceguard.ai.data.repository.ConnectionState
import com.riceguard.ai.di.AppContainer
import com.riceguard.ai.di.rememberViewModel
import com.riceguard.ai.ui.components.ConfirmDialog
import com.riceguard.ai.ui.components.ConnectionStatusChip
import com.riceguard.ai.ui.components.PrimaryButton
import com.riceguard.ai.ui.components.RiceGuardCard
import com.riceguard.ai.ui.components.ScreenTopBar

@Composable
fun SettingsScreen(container: AppContainer, onBack: () -> Unit) {
    val viewModel = rememberViewModel {
        SettingsViewModel(container.settingsRepository, container.connectionRepository, container.scanRepository)
    }
    val uiState by viewModel.uiState.collectAsState()
    val scanCount by viewModel.scanCount.collectAsState()
    var showDeleteAllConfirm by remember { mutableStateOf(false) }

    Scaffold(topBar = { ScreenTopBar(title = "Settings", onBack = onBack) }) { padding ->
        LazyColumn(
            modifier = Modifier.fillMaxSize().padding(padding),
            contentPadding = PaddingValues(20.dp),
        ) {
            item { SectionLabel("Connection") }
            item {
                RiceGuardCard(modifier = Modifier.fillMaxWidth()) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text("Status", style = MaterialTheme.typography.bodyMedium)
                        ConnectionStatusChip(state = uiState.connectionState)
                    }

                    HorizontalDivider(modifier = Modifier.padding(vertical = 16.dp))

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text("Automatic Discovery", style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.Medium)
                            Text(
                                "Finds the RiceGuard server on your Wi-Fi automatically. No IP address needed.",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                        Switch(checked = uiState.autoDiscoveryEnabled, onCheckedChange = viewModel::onAutoDiscoveryToggled)
                    }

                    if (!uiState.autoDiscoveryEnabled) {
                        HorizontalDivider(modifier = Modifier.padding(vertical = 16.dp))

                        Text(
                            "ADVANCED — MANUAL SERVER ADDRESS",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )

                        OutlinedTextField(
                            value = uiState.host,
                            onValueChange = viewModel::onHostChanged,
                            label = { Text("Server IP or hostname") },
                            placeholder = { Text("192.168.1.10") },
                            singleLine = true,
                            modifier = Modifier.fillMaxWidth().padding(top = 12.dp),
                        )
                        OutlinedTextField(
                            value = uiState.port,
                            onValueChange = viewModel::onPortChanged,
                            label = { Text("Port") },
                            placeholder = { Text("8000") },
                            singleLine = true,
                            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                            modifier = Modifier.fillMaxWidth().padding(top = 12.dp),
                        )

                        if (uiState.savedConfirmation && uiState.connectionState == ConnectionState.CONNECTED) {
                            Text(
                                "Saved — server reachable.",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.primary,
                                modifier = Modifier.padding(top = 10.dp),
                            )
                        } else if (uiState.savedConfirmation && uiState.connectionState == ConnectionState.OFFLINE) {
                            Text(
                                uiState.connectionErrorDetail ?: "Saved, but the server could not be reached.",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.error,
                                modifier = Modifier.padding(top = 10.dp),
                            )
                        }

                        PrimaryButton(
                            text = "Test Connection",
                            onClick = viewModel::saveAndTestConnection,
                            loading = uiState.testingConnection,
                            modifier = Modifier.fillMaxWidth().padding(top = 16.dp),
                        )
                    }
                }
            }

            item { SectionLabel("Privacy & Data") }
            item {
                RiceGuardCard(modifier = Modifier.fillMaxWidth()) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text("Save Scan History", style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.Medium)
                            Text(
                                "When off, results are shown but not stored on this device.",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                        Switch(checked = uiState.saveHistoryEnabled, onCheckedChange = viewModel::onSaveHistoryToggled)
                    }

                    HorizontalDivider(modifier = Modifier.padding(vertical = 16.dp))

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text("Stored scans", style = MaterialTheme.typography.bodyLarge)
                        Text("$scanCount", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                    }

                    PrimaryButton(
                        text = "Delete All Scan Data",
                        onClick = { showDeleteAllConfirm = true },
                        enabled = scanCount > 0,
                        containerColor = MaterialTheme.colorScheme.error,
                        modifier = Modifier.fillMaxWidth().padding(top = 16.dp),
                    )
                }
            }

            item { SectionLabel("About") }
            item {
                RiceGuardCard(modifier = Modifier.fillMaxWidth()) {
                    Text("RiceGuard", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                    Text(
                        "Version 1.1.0",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(top = 4.dp),
                    )
                    Text(
                        "Detection and verification run on your local PC over Wi-Fi. Treatment guidance is " +
                            "general agronomic information and is not a substitute for professional advice.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(top = 10.dp),
                    )
                    Text(
                        "Developed by: Parth Samarth",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(top = 10.dp),
                    )
                }
            }
        }
    }

    if (showDeleteAllConfirm) {
        ConfirmDialog(
            title = "Delete all scan data?",
            message = "This permanently removes all saved scan images, explanations, results, recommendations, and history from this device.",
            confirmLabel = "Delete Everything",
            onConfirm = {
                showDeleteAllConfirm = false
                viewModel.deleteAllHistory()
            },
            onDismiss = { showDeleteAllConfirm = false },
        )
    }
}

@Composable
private fun SectionLabel(text: String) {
    Text(
        text.uppercase(),
        style = MaterialTheme.typography.labelSmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
        modifier = Modifier.padding(top = 20.dp, bottom = 10.dp),
    )
}
