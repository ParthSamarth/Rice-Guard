package com.riceguard.ai.data.repository

import com.google.gson.Gson
import com.riceguard.ai.data.discovery.DiscoveredServer
import com.riceguard.ai.data.discovery.ServiceDiscoveryRepository
import com.riceguard.ai.data.network.ApiService
import com.riceguard.ai.data.network.dto.HealthResponseDto
import com.riceguard.ai.data.settings.LastKnownServer
import com.riceguard.ai.data.settings.SettingsRepository
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.drop
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.IOException

/** FOUND and SEARCHING are new (auto-discovery); the rest predate it and
 * keep their existing meaning for manual mode. mDNS finding a service only
 * ever reaches FOUND -- CONNECTED is reserved for an actual verified
 * /health 200 from something that identifies itself as a RiceGuard server,
 * matching the "don't declare Connected just because mDNS found something"
 * requirement. */
enum class ConnectionState { UNKNOWN, SEARCHING, FOUND, CHECKING, CONNECTED, OFFLINE, NOT_CONFIGURED }

data class ConnectionDetails(
    val serverName: String? = null,
    val host: String? = null,
    val port: Int? = null,
    val lastSuccessfulConnection: LastKnownServer? = null,
    val autoDiscoveryEnabled: Boolean = true,
)

/**
 * Project brief section 18: the app must never fail silently on a network
 * problem -- every screen that needs the server first asks this repository,
 * which owns one shared, observable ConnectionState rather than each screen
 * independently guessing at reachability.
 *
 * Also the single place that decides whether the active server came from
 * automatic mDNS/NSD discovery or the user's manual Settings entry -- every
 * other screen (ScanRepository included) just reads [state] /
 * [SettingsRepository.currentServerConfig] and doesn't care which.
 */
class ConnectionRepository(
    private val apiService: ApiService,
    private val settingsRepository: SettingsRepository,
    private val discoveryRepository: ServiceDiscoveryRepository,
    private val discoveryProbeClient: OkHttpClient,
) {
    private val gson = Gson()

    private val _state = MutableStateFlow(ConnectionState.UNKNOWN)
    val state: StateFlow<ConnectionState> = _state

    private val _modelsReady = MutableStateFlow(false)
    val modelsReady: StateFlow<Boolean> = _modelsReady

    /** Why the last check landed on OFFLINE/SEARCHING, e.g. "Connection
     * refused" vs. "Connection timed out" vs. "no server found yet" -- null
     * once CONNECTED again. See [describeConnectionFailure]. */
    private val _lastErrorDetail = MutableStateFlow<String?>(null)
    val lastErrorDetail: StateFlow<String?> = _lastErrorDetail

    private val _details = MutableStateFlow(ConnectionDetails())
    val details: StateFlow<ConnectionDetails> = _details

    private var orchestrationJob: Job? = null
    private var searchStartedAtMillis: Long = 0L

    /** Called once from AppContainer with the app-level scope. Owns the
     * whole lifetime of auto-discovery: switches discovery on/off with the
     * user's preference, and for as long as it's on, keeps state in sync
     * with every server NSD finds/loses and every Wi-Fi network change --
     * this is what makes reconnect automatic (project brief section 4)
     * instead of something the user has to trigger by hand. */
    fun start(appScope: CoroutineScope) {
        appScope.launch {
            settingsRepository.autoDiscoveryEnabledFlow.distinctUntilChanged().collect { enabled ->
                _details.value = _details.value.copy(autoDiscoveryEnabled = enabled)
                orchestrationJob?.cancel()
                if (enabled) {
                    searchStartedAtMillis = System.currentTimeMillis()
                    _state.value = ConnectionState.SEARCHING
                    discoveryRepository.startDiscovery()
                    orchestrationJob = appScope.launch { runAutoDiscoveryLoop() }
                } else {
                    discoveryRepository.stopDiscovery()
                    checkNow()
                }
            }
        }
        appScope.launch {
            settingsRepository.lastKnownServerFlow.collect { last ->
                _details.value = _details.value.copy(lastSuccessfulConnection = last)
            }
        }
    }

    private suspend fun runAutoDiscoveryLoop(): Unit = coroutineScope {
        launch {
            discoveryRepository.discoveredServer.collect { server -> onDiscoveryUpdate(server) }
        }
        launch {
            // Skip the initial value -- discovery was just (re)started fresh
            // above, no need to restart it again on the very first tick.
            discoveryRepository.networkGeneration.drop(1).collect {
                searchStartedAtMillis = System.currentTimeMillis()
                _state.value = ConnectionState.SEARCHING
                discoveryRepository.restartDiscovery()
            }
        }
    }

    private suspend fun onDiscoveryUpdate(server: DiscoveredServer?) {
        if (server == null) {
            _modelsReady.value = false
            _details.value = _details.value.copy(serverName = null, host = null, port = null)
            if (_state.value == ConnectionState.CONNECTED || _state.value == ConnectionState.CHECKING) {
                _lastErrorDetail.value = "RiceGuard server unavailable"
                _state.value = ConnectionState.OFFLINE
            } else {
                _lastErrorDetail.value = wrongNetworkHintOrNull()
                _state.value = ConnectionState.SEARCHING
            }
            return
        }

        _state.value = ConnectionState.FOUND
        _lastErrorDetail.value = null
        _details.value = _details.value.copy(serverName = server.name, host = server.hosts.firstOrNull(), port = server.port)

        _state.value = ConnectionState.CHECKING
        verify(server)
    }

    /** A machine with virtual adapters (WSL, Hyper-V, Docker, a VPN client)
     * can advertise several candidate addresses for itself, and in testing
     * Android's own address ordering wasn't reliable enough to trust the
     * first one -- so this tries each candidate's /health in turn and
     * commits to whichever one actually answers, rather than gambling on a
     * single address. Discovery is the sole writer of the active base URL
     * while auto mode is on (see SettingsRepository docs); a candidate is
     * only written there once it's about to be probed. */
    private suspend fun verify(server: DiscoveredServer) {
        var lastFailureDetail: String? = null
        for (host in server.hosts) {
            val outcome = probeHealth(server, host)
            if (outcome == null) {
                // Success -- discovery is the sole writer of the active base
                // URL while auto mode is on (see SettingsRepository docs).
                // Only committed here, once a candidate is actually proven
                // reachable, so a losing candidate never briefly becomes the
                // URL real scan/predict traffic would be sent to.
                settingsRepository.setActiveDiscoveredConfig(host, server.port)
                return
            }
            lastFailureDetail = outcome
        }
        _modelsReady.value = false
        _lastErrorDetail.value = lastFailureDetail
            ?: "Found \"${server.name}\", but it didn't answer like a RiceGuard server."
        _state.value = ConnectionState.OFFLINE
    }

    /** Probes ONE candidate directly (bypassing ApiService/the shared
     * interceptor -- see NetworkModule's discoveryProbeClient docs) so
     * candidates can be tried in address order without racing the app-wide
     * "current server" value. Returns null on success (state already set to
     * CONNECTED), or a human-readable failure detail to try the next
     * candidate with. */
    private suspend fun probeHealth(server: DiscoveredServer, host: String): String? =
        withContext(Dispatchers.IO) {
            try {
                val request = Request.Builder().url("http://$host:${server.port}/health").build()
                discoveryProbeClient.newCall(request).execute().use { response ->
                    val bodyString = response.body?.string()
                    val body = bodyString?.let { runCatching { gson.fromJson(it, HealthResponseDto::class.java) }.getOrNull() }
                    if (response.isSuccessful && body != null && body.service.isNotBlank()) {
                        _modelsReady.value = body.modelsReady
                        _lastErrorDetail.value = null
                        _details.value = _details.value.copy(host = host)
                        settingsRepository.rememberLastKnownServer(
                            name = server.name,
                            host = host,
                            port = server.port,
                            timestampMillis = System.currentTimeMillis(),
                        )
                        _state.value = ConnectionState.CONNECTED
                        null
                    } else {
                        "Found \"${server.name}\", but it didn't answer like a RiceGuard server."
                    }
                }
            } catch (e: IOException) {
                describeConnectionFailure(e)
            } catch (e: Exception) {
                "Unexpected error: ${e.message ?: e::class.simpleName}"
            }
        }

    /** Distinguishes "still searching, first time ever" from "searching,
     * but this worked before on some network" without reading the Wi-Fi
     * SSID (which would need a location permission this feature doesn't
     * otherwise need) -- a time-based heuristic, not a real network-identity
     * check: if we've been searching a while AND a server has worked before
     * in this install, the most likely explanation is the wrong network. */
    private fun wrongNetworkHintOrNull(): String? {
        if (_details.value.lastSuccessfulConnection == null) return null
        val elapsed = System.currentTimeMillis() - searchStartedAtMillis
        return if (elapsed > 8_000) {
            "No RiceGuard server found on this network. Check that your phone and PC are on the same Wi-Fi."
        } else {
            null
        }
    }

    /** "Retry / Rediscover" from the connection panel (project brief
     * section 5). No-op in manual mode -- there [checkNow] already covers it. */
    fun rediscover() {
        if (!_details.value.autoDiscoveryEnabled) return
        searchStartedAtMillis = System.currentTimeMillis()
        _state.value = ConnectionState.SEARCHING
        _lastErrorDetail.value = null
        discoveryRepository.restartDiscovery()
    }

    /** On-demand check: in auto mode this is "Rediscover"; in manual mode
     * (auto-discovery off) it's the original one-shot /health probe against
     * whatever the user typed into Settings. */
    suspend fun checkNow(): ConnectionState {
        if (_details.value.autoDiscoveryEnabled) {
            rediscover()
            return _state.value
        }
        if (settingsRepository.currentServerConfig.value.host.isBlank()) {
            _state.value = ConnectionState.NOT_CONFIGURED
            _modelsReady.value = false
            _lastErrorDetail.value = null
            return _state.value
        }
        _state.value = ConnectionState.CHECKING
        _state.value = try {
            val response = apiService.health()
            if (response.isSuccessful && response.body() != null) {
                _modelsReady.value = response.body()!!.modelsReady
                _lastErrorDetail.value = null
                ConnectionState.CONNECTED
            } else {
                _modelsReady.value = false
                _lastErrorDetail.value = "Server responded with an error (HTTP ${response.code()})."
                ConnectionState.OFFLINE
            }
        } catch (e: IOException) {
            _modelsReady.value = false
            _lastErrorDetail.value = describeConnectionFailure(e)
            ConnectionState.OFFLINE
        } catch (e: Exception) {
            _modelsReady.value = false
            _lastErrorDetail.value = "Unexpected error: ${e.message ?: e::class.simpleName}"
            ConnectionState.OFFLINE
        }
        return _state.value
    }
}
