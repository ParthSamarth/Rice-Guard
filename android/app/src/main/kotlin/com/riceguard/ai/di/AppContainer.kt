package com.riceguard.ai.di

import android.content.Context
import com.riceguard.ai.data.discovery.ServiceDiscoveryRepository
import com.riceguard.ai.data.local.ImageStorage
import com.riceguard.ai.data.local.RiceGuardDatabase
import com.riceguard.ai.data.network.NetworkModule
import com.riceguard.ai.data.repository.ConnectionRepository
import com.riceguard.ai.data.repository.ScanRepository
import com.riceguard.ai.data.settings.SettingsRepository
import kotlinx.coroutines.CoroutineScope

/**
 * Plain manual dependency container (no Hilt/Dagger) -- kept deliberately
 * simple; fewer moving annotation-processor parts means fewer ways a build
 * can fail for reasons unrelated to app logic. One instance lives on
 * RiceGuardApplication.
 */
class AppContainer(context: Context, appScope: CoroutineScope) {

    val settingsRepository = SettingsRepository(context).also { it.observeAndMirror(appScope) }

    private val networkClients = NetworkModule.create(settingsRepository)

    private val discoveryRepository = ServiceDiscoveryRepository(context)

    val connectionRepository = ConnectionRepository(
        apiService = networkClients.apiService,
        settingsRepository = settingsRepository,
        discoveryRepository = discoveryRepository,
        discoveryProbeClient = networkClients.discoveryProbeClient,
    ).also { it.start(appScope) }

    private val database = RiceGuardDatabase.getInstance(context)
    val imageStorage = ImageStorage(context)

    val scanRepository = ScanRepository(
        apiService = networkClients.apiService,
        httpClient = networkClients.httpClient,
        settingsRepository = settingsRepository,
        scanDao = database.scanDao(),
        imageStorage = imageStorage,
    )
}
