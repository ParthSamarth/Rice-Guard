package com.riceguard.ai.ui.screens.historydetail

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.riceguard.ai.data.local.ScanRecord
import com.riceguard.ai.data.repository.ScanRepository
import com.riceguard.ai.domain.model.Recommendation
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

data class HistoryDetailUiState(
    val record: ScanRecord? = null,
    val recommendation: Recommendation? = null,
    val loading: Boolean = true,
    val deleted: Boolean = false,
)

class HistoryDetailViewModel(private val scanRepository: ScanRepository, private val scanId: String) : ViewModel() {

    private val _uiState = MutableStateFlow(HistoryDetailUiState())
    val uiState: StateFlow<HistoryDetailUiState> = _uiState

    init {
        viewModelScope.launch {
            val record = scanRepository.getScanDetail(scanId)
            _uiState.value = HistoryDetailUiState(
                record = record,
                recommendation = scanRepository.recommendationFromJson(record?.recommendationJson),
                loading = false,
            )
        }
    }

    /** project brief section 15: removes the Room row, the original image,
     * detection image, Grad-CAM image, and any other cached files for this
     * scan -- ScanRepository.deleteScan deletes the whole per-scan folder in
     * one operation, so nothing can be partially deleted. */
    fun deleteScan() {
        viewModelScope.launch {
            scanRepository.deleteScan(scanId)
            _uiState.value = _uiState.value.copy(deleted = true)
        }
    }

    fun saveNotes(notes: String) {
        val trimmed = notes.trim().ifBlank { null }
        viewModelScope.launch {
            scanRepository.updateNotes(scanId, trimmed)
            _uiState.value = _uiState.value.copy(record = _uiState.value.record?.copy(notes = trimmed))
        }
    }
}
