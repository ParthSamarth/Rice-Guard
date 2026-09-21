package com.riceguard.ai.ui.screens.settings

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.riceguard.ai.data.repository.ConnectionRepository
import com.riceguard.ai.data.repository.ConnectionState
import com.riceguard.ai.data.repository.ScanRepository
import com.riceguard.ai.data.settings.SettingsRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

data class SettingsUiState(
    val host: String = "",
    val port: String = "8000",
    val saveHistoryEnabled: Boolean = true,
    val autoDiscoveryEnabled: Boolean = true,
    val connectionState: ConnectionState = ConnectionState.UNKNOWN,
    val connectionErrorDetail: String? = null,
    val testingConnection: Boolean = false,
    val savedConfirmation: Boolean = false,
)

class SettingsViewModel(
    private val settingsRepository: SettingsRepository,
    private val connectionRepository: ConnectionRepository,
    private val scanRepository: ScanRepository,
) : ViewModel() {

    private val _uiState = MutableStateFlow(SettingsUiState())
    val uiState: StateFlow<SettingsUiState> = _uiState

    val scanCount: StateFlow<Int> = scanRepository.observeHistoryCount()
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), 0)

    init {
        val current = settingsRepository.currentServerConfig.value
        _uiState.value = _uiState.value.copy(host = current.host, port = current.port.toString())
        viewModelScope.launch {
            settingsRepository.saveHistoryFlow.collect { enabled ->
                _uiState.value = _uiState.value.copy(saveHistoryEnabled = enabled)
            }
        }
        viewModelScope.launch {
            settingsRepository.autoDiscoveryEnabledFlow.collect { enabled ->
                _uiState.value = _uiState.value.copy(autoDiscoveryEnabled = enabled)
            }
        }
        viewModelScope.launch {
            connectionRepository.state.collect { state ->
                _uiState.value = _uiState.value.copy(connectionState = state, testingConnection = false)
            }
        }
        viewModelScope.launch {
            connectionRepository.lastErrorDetail.collect { detail ->
                _uiState.value = _uiState.value.copy(connectionErrorDetail = detail)
            }
        }
    }

    fun onHostChanged(host: String) {
        _uiState.value = _uiState.value.copy(host = host, savedConfirmation = false)
    }

    fun onPortChanged(port: String) {
        if (port.all { it.isDigit() } && port.length <= 5) {
            _uiState.value = _uiState.value.copy(port = port, savedConfirmation = false)
        }
    }

    fun onSaveHistoryToggled(enabled: Boolean) {
        viewModelScope.launch { settingsRepository.setSaveHistoryEnabled(enabled) }
    }

    fun onAutoDiscoveryToggled(enabled: Boolean) {
        viewModelScope.launch { settingsRepository.setAutoDiscoveryEnabled(enabled) }
    }

    fun saveAndTestConnection() {
        val port = _uiState.value.port.toIntOrNull() ?: return
        viewModelScope.launch {
            settingsRepository.setServerConfig(_uiState.value.host, port)
            _uiState.value = _uiState.value.copy(testingConnection = true, savedConfirmation = true)
            connectionRepository.checkNow()
        }
    }

    fun deleteAllHistory() {
        viewModelScope.launch { scanRepository.deleteAllHistory() }
    }
}
