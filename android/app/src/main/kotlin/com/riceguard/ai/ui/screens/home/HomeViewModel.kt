package com.riceguard.ai.ui.screens.home

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.riceguard.ai.data.local.ScanRecord
import com.riceguard.ai.data.repository.ConnectionDetails
import com.riceguard.ai.data.repository.ConnectionRepository
import com.riceguard.ai.data.repository.ConnectionState
import com.riceguard.ai.data.repository.ScanRepository
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

class HomeViewModel(
    private val connectionRepository: ConnectionRepository,
    scanRepository: ScanRepository,
) : ViewModel() {

    val connectionState: StateFlow<ConnectionState> = connectionRepository.state
    val connectionDetails: StateFlow<ConnectionDetails> = connectionRepository.details
    val connectionErrorDetail: StateFlow<String?> = connectionRepository.lastErrorDetail

    val recentScans: StateFlow<List<ScanRecord>> = scanRepository.observeHistory()
        .map { it.take(3) }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())

    init {
        checkConnection()
    }

    fun checkConnection() {
        viewModelScope.launch { connectionRepository.checkNow() }
    }

    fun rediscover() {
        connectionRepository.rediscover()
    }
}
