package com.riceguard.ai.ui.screens.home

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ChevronRight
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import com.riceguard.ai.R
import com.riceguard.ai.data.local.ScanRecord
import com.riceguard.ai.di.AppContainer
import com.riceguard.ai.di.rememberViewModel
import com.riceguard.ai.ui.components.ConnectionPanel
import com.riceguard.ai.ui.components.ConnectionStatusChip
import com.riceguard.ai.ui.components.PrimaryButton
import com.riceguard.ai.ui.components.RiceGuardBottomNav
import com.riceguard.ai.ui.components.RiceGuardCard
import com.riceguard.ai.ui.components.RiceGuardTab
import com.riceguard.ai.ui.components.SecondaryButton
import com.riceguard.ai.ui.components.rememberGalleryImagePicker
import java.io.File
import java.text.DateFormat
import java.util.Date

@Composable
fun HomeScreen(
    container: AppContainer,
    onScanClick: () -> Unit,
    onImagePicked: (File) -> Unit,
    onHistoryClick: () -> Unit,
    onSettingsClick: () -> Unit,
    onOpenScan: (String) -> Unit,
) {
    val viewModel = rememberViewModel { HomeViewModel(container.connectionRepository, container.scanRepository) }
    val connectionState by viewModel.connectionState.collectAsState()
    val connectionDetails by viewModel.connectionDetails.collectAsState()
    val connectionErrorDetail by viewModel.connectionErrorDetail.collectAsState()
    val recentScans by viewModel.recentScans.collectAsState()
    val pickImage = rememberGalleryImagePicker(container.imageStorage, onPicked = onImagePicked)
    var showConnectionPanel by remember { mutableStateOf(false) }

    Scaffold(
        bottomBar = {
            RiceGuardBottomNav(
                current = RiceGuardTab.HOME,
                onHome = {},
                onScan = onScanClick,
                onHistory = onHistoryClick,
                onSettings = onSettingsClick,
            )
        },
    ) { padding ->
        LazyColumn(
            modifier = Modifier.fillMaxSize().padding(padding),
            contentPadding = PaddingValues(20.dp),
            verticalArrangement = Arrangement.spacedBy(20.dp),
        ) {
            item {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column {
                        Text("Welcome back", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Text("RiceGuard", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.SemiBold)
                    }
                    ConnectionStatusChip(
                        state = connectionState,
                        modifier = Modifier.clickable { showConnectionPanel = true },
                    )
                }
            }

            item {
                RiceGuardCard(containerColor = MaterialTheme.colorScheme.primaryContainer) {
                    Text(
                        "Scan a rice leaf",
                        style = MaterialTheme.typography.titleLarge,
                        color = MaterialTheme.colorScheme.onPrimaryContainer,
                    )
                    Text(
                        "Capture a clear photo and get an instant diagnosis with treatment guidance.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onPrimaryContainer,
                        modifier = Modifier.padding(top = 6.dp, bottom = 18.dp),
                    )
                    PrimaryButton(
                        text = "Scan Rice Leaf",
                        onClick = onScanClick,
                        modifier = Modifier.fillMaxWidth(),
                        containerColor = MaterialTheme.colorScheme.primary,
                    )
                    SecondaryButton(
                        text = "Upload from Gallery",
                        onClick = pickImage,
                        modifier = Modifier.fillMaxWidth().padding(top = 10.dp),
                    )
                }
            }

            item {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text("Recent scans", style = MaterialTheme.typography.titleMedium)
                    if (recentScans.isNotEmpty()) {
                        IconButton(onClick = onHistoryClick) {
                            Icon(Icons.Filled.ChevronRight, contentDescription = "View all history")
                        }
                    }
                }
            }

            if (recentScans.isEmpty()) {
                item {
                    RiceGuardCard {
                        Text(
                            "No scans yet. Your analyzed rice leaves will appear here.",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
            } else {
                items(recentScans, key = { it.id }) { record ->
                    RecentScanRow(
                        record = record,
                        onClick = { onOpenScan(record.id) },
                        modifier = Modifier.animateItem(),
                    )
                }
            }
        }
    }

    if (showConnectionPanel) {
        ConnectionPanel(
            state = connectionState,
            details = connectionDetails,
            errorDetail = connectionErrorDetail,
            onRediscover = viewModel::rediscover,
            onDismiss = { showConnectionPanel = false },
        )
    }
}

@Composable
private fun RecentScanRow(record: ScanRecord, onClick: () -> Unit, modifier: Modifier = Modifier) {
    RiceGuardCard(
        modifier = modifier.fillMaxWidth().clickable(onClick = onClick),
        contentPadding = PaddingValues(12.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
            AsyncImage(
                model = File(record.originalImagePath),
                contentDescription = record.disease,
                modifier = Modifier.size(56.dp).clip(RoundedCornerShape(12.dp)),
                placeholder = painterResource(id = R.drawable.ic_launcher_foreground),
                error = painterResource(id = R.drawable.ic_launcher_foreground),
            )
            Spacer(Modifier.size(14.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(record.disease ?: "Unresolved", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
                Text(
                    DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.SHORT).format(Date(record.timestampMillis)),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            record.finalConfidence?.let {
                Text(
                    "${(it * 100).toInt()}%",
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.primary,
                    fontWeight = FontWeight.SemiBold,
                )
            }
            Icon(Icons.Filled.ChevronRight, contentDescription = null, tint = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}
