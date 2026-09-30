package com.riceguard.ai.ui.screens.history

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.riceguard.ai.data.local.ScanRecord
import com.riceguard.ai.data.repository.ScanRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn

enum class HistoryFilter(val label: String) { ALL("All"), HEALTHY("Healthy"), DISEASED("Diseased") }

class HistoryViewModel(scanRepository: ScanRepository) : ViewModel() {

    private val allHistory: StateFlow<List<ScanRecord>> = scanRepository.observeHistory()
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())

    private val _searchQuery = MutableStateFlow("")
    val searchQuery: StateFlow<String> = _searchQuery

    private val _filter = MutableStateFlow(HistoryFilter.ALL)
    val filter: StateFlow<HistoryFilter> = _filter

    private val _diseaseFilter = MutableStateFlow<String?>(null)
    val diseaseFilter: StateFlow<String?> = _diseaseFilter

    /** Every disease name actually present in history, for the per-disease
     * filter row -- never a hardcoded class list, so it can never drift from
     * what the server has actually returned. */
    val availableDiseases: StateFlow<List<String>> = allHistory
        .map { records -> records.mapNotNull { it.disease }.distinct().sorted() }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())

    val history: StateFlow<List<ScanRecord>> = combine(
        allHistory, _searchQuery, _filter, _diseaseFilter,
    ) { records, query, statusFilter, disease ->
        records
            .filter { record ->
                when (statusFilter) {
                    HistoryFilter.ALL -> true
                    HistoryFilter.HEALTHY -> record.disease?.equals("Healthy", ignoreCase = true) == true
                    HistoryFilter.DISEASED -> record.disease != null && !record.disease.equals("Healthy", ignoreCase = true)
                }
            }
            .filter { disease == null || it.disease == disease }
            .filter { record ->
                query.isBlank() ||
                    record.id.contains(query, ignoreCase = true) ||
                    record.disease?.contains(query, ignoreCase = true) == true
            }
    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())

    fun onSearchQueryChanged(query: String) {
        _searchQuery.value = query
    }

    fun onFilterChanged(filter: HistoryFilter) {
        _filter.value = filter
    }

    fun onDiseaseFilterChanged(disease: String?) {
        _diseaseFilter.value = disease
    }
}
