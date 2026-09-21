package com.riceguard.ai.ui.screens.history

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ChevronRight
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import com.riceguard.ai.R
import com.riceguard.ai.data.local.ScanRecord
import com.riceguard.ai.data.repository.ScanRepository
import com.riceguard.ai.di.rememberViewModel
import com.riceguard.ai.ui.components.EmptyState
import com.riceguard.ai.ui.components.PrimaryButton
import com.riceguard.ai.ui.components.RiceGuardBottomNav
import com.riceguard.ai.ui.components.RiceGuardCard
import com.riceguard.ai.ui.components.RiceGuardTab
import com.riceguard.ai.ui.components.ScreenTopBar
import com.riceguard.ai.ui.components.StatusChip
import com.riceguard.ai.ui.theme.riceGuardStatus
import java.io.File
import java.text.DateFormat
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

private val SECTION_DATE_FORMAT = SimpleDateFormat("EEEE, d MMMM yyyy", Locale.getDefault())
private val TODAY_FORMAT = SimpleDateFormat("yyyyMMdd", Locale.getDefault())

@Composable
fun HistoryScreen(
    scanRepository: ScanRepository,
    onOpenScan: (String) -> Unit,
    onBack: () -> Unit,
    onScanClick: () -> Unit,
) {
    val viewModel = rememberViewModel { HistoryViewModel(scanRepository) }
    val records by viewModel.history.collectAsState()
    val searchQuery by viewModel.searchQuery.collectAsState()
    val filter by viewModel.filter.collectAsState()
    val diseaseFilter by viewModel.diseaseFilter.collectAsState()
    val availableDiseases by viewModel.availableDiseases.collectAsState()

    Scaffold(
        topBar = { ScreenTopBar(title = "History", onBack = onBack) },
        bottomBar = {
            RiceGuardBottomNav(
                current = RiceGuardTab.HISTORY,
                onHome = onBack,
                onScan = onScanClick,
                onHistory = {},
                onSettings = onBack,
            )
        },
    ) { padding ->
        Column(modifier = Modifier.fillMaxSize().padding(padding)) {
            Column(modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 12.dp)) {
                OutlinedTextField(
                    value = searchQuery,
                    onValueChange = viewModel::onSearchQueryChanged,
                    placeholder = { Text("Search by Scan ID or disease") },
                    leadingIcon = { Icon(Icons.Filled.Search, contentDescription = null) },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
                Spacer(Modifier.size(10.dp))
                LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    items(HistoryFilter.entries) { option ->
                        FilterChip(
                            selected = filter == option,
                            onClick = { viewModel.onFilterChanged(option) },
                            label = { Text(option.label) },
                            colors = FilterChipDefaults.filterChipColors(
                                selectedContainerColor = MaterialTheme.colorScheme.primaryContainer,
                            ),
                        )
                    }
                    if (availableDiseases.isNotEmpty()) {
                        items(availableDiseases) { disease ->
                            FilterChip(
                                selected = diseaseFilter == disease,
                                onClick = { viewModel.onDiseaseFilterChanged(if (diseaseFilter == disease) null else disease) },
                                label = { Text(disease) },
                                colors = FilterChipDefaults.filterChipColors(
                                    selectedContainerColor = MaterialTheme.colorScheme.primaryContainer,
                                ),
                            )
                        }
                    }
                }
            }

            if (records.isEmpty()) {
                val noFiltersApplied = searchQuery.isBlank() && filter == HistoryFilter.ALL && diseaseFilter == null
                EmptyState(
                    icon = Icons.Filled.History,
                    title = if (noFiltersApplied) "No scans yet" else "No matching scans",
                    message = if (noFiltersApplied) "Your analyzed rice leaves will appear here." else "Try a different search term or filter.",
                    action = if (noFiltersApplied) {
                        { PrimaryButton(text = "Scan a Leaf", onClick = onScanClick, modifier = Modifier.padding(top = 8.dp)) }
                    } else {
                        null
                    },
                )
                return@Scaffold
            }

            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(start = 20.dp, end = 20.dp, bottom = 20.dp),
            ) {
                val grouped = records.groupBy { TODAY_FORMAT.format(Date(it.timestampMillis)) }
                grouped.forEach { (dayKey, dayRecords) ->
                    item(key = "header_$dayKey") {
                        Text(
                            dateSectionLabel(dayRecords.first().timestampMillis),
                            style = MaterialTheme.typography.labelLarge,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            fontWeight = FontWeight.SemiBold,
                            modifier = Modifier.padding(top = 16.dp, bottom = 8.dp),
                        )
                    }
                    items(dayRecords, key = { it.id }) { record ->
                        HistoryRow(
                            record = record,
                            onClick = { onOpenScan(record.id) },
                            modifier = Modifier.animateItem(),
                        )
                        Spacer(Modifier.size(12.dp))
                    }
                }
            }
        }
    }
}

private fun dateSectionLabel(timestampMillis: Long): String {
    val today = TODAY_FORMAT.format(Date())
    val yesterday = TODAY_FORMAT.format(Date(System.currentTimeMillis() - 86_400_000L))
    val key = TODAY_FORMAT.format(Date(timestampMillis))
    return when (key) {
        today -> "Today"
        yesterday -> "Yesterday"
        else -> SECTION_DATE_FORMAT.format(Date(timestampMillis))
    }
}

@Composable
private fun HistoryRow(record: ScanRecord, onClick: () -> Unit, modifier: Modifier = Modifier) {
    RiceGuardCard(
        modifier = modifier.fillMaxWidth().clickable(onClick = onClick),
        contentPadding = PaddingValues(12.dp),
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.fillMaxWidth(),
        ) {
            AsyncImage(
                model = File(record.originalImagePath),
                contentDescription = record.disease,
                modifier = Modifier.size(64.dp).clip(RoundedCornerShape(14.dp)),
                placeholder = painterResource(id = R.drawable.ic_launcher_foreground),
                error = painterResource(id = R.drawable.ic_launcher_foreground),
            )
            Spacer(Modifier.size(14.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(record.disease ?: "Unresolved", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
                Text(
                    "${record.id} • " + DateFormat.getTimeInstance(DateFormat.SHORT).format(Date(record.timestampMillis)),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 2.dp, bottom = 6.dp),
                )
                StatusRowChip(status = record.status)
            }
            record.finalConfidence?.let {
                Text(
                    "${(it * 100).toInt()}%",
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.primary,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.padding(end = 4.dp),
                )
            }
            Icon(Icons.Filled.ChevronRight, contentDescription = null, tint = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
private fun StatusRowChip(status: String) {
    val statusColors = riceGuardStatus
    val (label, color, container) = when (status) {
        "ok" -> Triple("Confident", statusColors.good, statusColors.goodContainer)
        "low_confidence" -> Triple("Low confidence", statusColors.warning, statusColors.warningContainer)
        "model_disagreement" -> Triple("Needs review", statusColors.caution, statusColors.cautionContainer)
        "not_recognized" -> Triple("Not a rice leaf", statusColors.caution, statusColors.cautionContainer)
        else -> Triple("Error", statusColors.critical, statusColors.criticalContainer)
    }
    StatusChip(label = label, color = color, containerColor = container, showDot = false)
}
