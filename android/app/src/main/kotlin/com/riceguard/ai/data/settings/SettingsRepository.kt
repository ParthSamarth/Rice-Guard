package com.riceguard.ai.data.settings

import android.content.Context
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.longPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.firstOrNull
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

private val Context.dataStore by preferencesDataStore(name = "riceguard_settings")

data class ServerConfig(val host: String, val port: Int) {
    val baseUrl: String get() = "http://$host:$port/"
}

/** What discovery last managed to reach, purely for display in the
 * connection panel ("Last successful connection") -- NEVER read as the
 * source of truth for the active base URL. The LAN IP a service was found
 * at can change on the next DHCP lease, so every reconnect re-verifies via
 * discovery + /health rather than trusting this. */
data class LastKnownServer(val name: String, val host: String, val port: Int, val timestampMillis: Long)

/**
 * Project brief section 19: the PC's IP is user-configurable, never
 * hard-coded. This repository is DataStore-backed (survives app restarts)
 * but also mirrors the current values into plain StateFlows
 * (currentServerConfig / currentSaveHistory) so
 * data/network/DynamicBaseUrlInterceptor -- a synchronous OkHttp interceptor
 * -- can read "the current setting" without suspending. The mirror is kept
 * live by [observeAndMirror], started once from RiceGuardApplication.
 *
 * Auto-discovery integration: [currentServerConfig] is the ONE value the
 * interceptor reads, regardless of whether it got there via the user typing
 * an IP in Settings ([setServerConfig], persisted) or via mDNS/NSD discovery
 * finding and health-verifying a server ([setActiveDiscoveredConfig],
 * in-memory only -- see ConnectionRepository). When [autoDiscoveryEnabled]
 * is on, discovery is the sole writer of the active config and the manual
 * host/port fields are inert (Settings screen disables them) so the two
 * writers never race.
 */
class SettingsRepository(private val context: Context) {

    private object Keys {
        val SERVER_HOST = stringPreferencesKey("server_host")
        val SERVER_PORT = intPreferencesKey("server_port")
        val SAVE_HISTORY = booleanPreferencesKey("save_history_enabled")
        val AUTO_DISCOVERY = booleanPreferencesKey("auto_discovery_enabled")
        val LAST_SERVER_NAME = stringPreferencesKey("last_server_name")
        val LAST_SERVER_HOST = stringPreferencesKey("last_server_host")
        val LAST_SERVER_PORT = intPreferencesKey("last_server_port")
        val LAST_SERVER_TIME = longPreferencesKey("last_server_time")
    }

    companion object {
        const val DEFAULT_PORT = 8000
        const val DEFAULT_HOST = "" // intentionally blank -- forces the user through Settings once (project brief section 19)
    }

    val serverConfigFlow: kotlinx.coroutines.flow.Flow<ServerConfig> = context.dataStore.data.map { prefs ->
        ServerConfig(
            host = prefs[Keys.SERVER_HOST] ?: DEFAULT_HOST,
            port = prefs[Keys.SERVER_PORT] ?: DEFAULT_PORT,
        )
    }

    val saveHistoryFlow: kotlinx.coroutines.flow.Flow<Boolean> = context.dataStore.data.map { prefs ->
        prefs[Keys.SAVE_HISTORY] ?: true // project brief section 26: default ON
    }

    val autoDiscoveryEnabledFlow: kotlinx.coroutines.flow.Flow<Boolean> = context.dataStore.data.map { prefs ->
        prefs[Keys.AUTO_DISCOVERY] ?: true // default ON -- the whole point is the user never has to think about it
    }

    val lastKnownServerFlow: kotlinx.coroutines.flow.Flow<LastKnownServer?> = context.dataStore.data.map { prefs ->
        val name = prefs[Keys.LAST_SERVER_NAME] ?: return@map null
        val host = prefs[Keys.LAST_SERVER_HOST] ?: return@map null
        val port = prefs[Keys.LAST_SERVER_PORT] ?: return@map null
        val time = prefs[Keys.LAST_SERVER_TIME] ?: return@map null
        LastKnownServer(name, host, port, time)
    }

    private val _currentServerConfig = MutableStateFlow(ServerConfig(DEFAULT_HOST, DEFAULT_PORT))
    val currentServerConfig: StateFlow<ServerConfig> = _currentServerConfig

    private val _currentSaveHistory = MutableStateFlow(true)
    val currentSaveHistory: StateFlow<Boolean> = _currentSaveHistory

    private val _currentAutoDiscoveryEnabled = MutableStateFlow(true)
    val currentAutoDiscoveryEnabled: StateFlow<Boolean> = _currentAutoDiscoveryEnabled

    /** Call once from Application.onCreate with the app-level CoroutineScope. */
    fun observeAndMirror(scope: CoroutineScope) {
        scope.launch {
            serverConfigFlow.collect { config ->
                // Manual config from DataStore only wins while auto-discovery
                // is off -- otherwise discovery's in-memory value (set via
                // setActiveDiscoveredConfig) stays authoritative, and this
                // emission (e.g. the one-time initial read) is ignored.
                if (!_currentAutoDiscoveryEnabled.value) _currentServerConfig.value = config
            }
        }
        scope.launch { saveHistoryFlow.collect { _currentSaveHistory.value = it } }
        scope.launch { autoDiscoveryEnabledFlow.collect { _currentAutoDiscoveryEnabled.value = it } }
    }

    suspend fun setServerConfig(host: String, port: Int) {
        context.dataStore.edit { prefs ->
            prefs[Keys.SERVER_HOST] = host.trim()
            prefs[Keys.SERVER_PORT] = port
        }
    }

    /** Discovery's write path -- updates only the in-memory active config
     * the interceptor reads. Never touches DataStore/the persisted manual
     * host+port, so switching auto-discovery off always falls back to
     * whatever the user last typed, untouched. */
    fun setActiveDiscoveredConfig(host: String, port: Int) {
        _currentServerConfig.value = ServerConfig(host, port)
    }

    suspend fun setSaveHistoryEnabled(enabled: Boolean) {
        context.dataStore.edit { prefs -> prefs[Keys.SAVE_HISTORY] = enabled }
    }

    suspend fun setAutoDiscoveryEnabled(enabled: Boolean) {
        context.dataStore.edit { prefs -> prefs[Keys.AUTO_DISCOVERY] = enabled }
        _currentAutoDiscoveryEnabled.value = enabled
        if (!enabled) {
            // Falling back to manual immediately -- re-read whatever's
            // persisted rather than waiting for the next DataStore emission.
            _currentServerConfig.value =
                serverConfigFlow.firstOrNull() ?: ServerConfig(DEFAULT_HOST, DEFAULT_PORT)
        }
    }

    suspend fun rememberLastKnownServer(name: String, host: String, port: Int, timestampMillis: Long) {
        context.dataStore.edit { prefs ->
            prefs[Keys.LAST_SERVER_NAME] = name
            prefs[Keys.LAST_SERVER_HOST] = host
            prefs[Keys.LAST_SERVER_PORT] = port
            prefs[Keys.LAST_SERVER_TIME] = timestampMillis
        }
    }
}
