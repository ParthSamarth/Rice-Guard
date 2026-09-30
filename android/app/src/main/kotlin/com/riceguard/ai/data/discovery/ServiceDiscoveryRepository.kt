package com.riceguard.ai.data.discovery

import android.content.Context
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.net.NetworkRequest
import android.net.nsd.NsdManager
import android.net.nsd.NsdServiceInfo
import android.net.wifi.WifiManager
import android.os.Build
import android.util.Log
import androidx.annotation.RequiresApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.util.concurrent.Executor
import java.util.concurrent.Executors

private const val TAG = "RiceGuardDiscovery"
const val RICEGUARD_SERVICE_TYPE = "_riceguard._tcp"

/** [hosts] can have more than one candidate address -- a machine with
 * virtual adapters (WSL, Hyper-V, Docker, a VPN client) advertises all of
 * its non-loopback addresses since it can't always tell which one the phone
 * can actually route to, and Android's own address ordering isn't reliable
 * either (observed in testing: NsdServiceInfo.hostAddresses does not
 * consistently keep the server's preferred-first ordering). The caller
 * (ConnectionRepository) is expected to try each until one actually answers
 * /health, rather than trusting the first entry. */
data class DiscoveredServer(val name: String, val hosts: List<String>, val port: Int)

/**
 * Wraps Android's NsdManager to find the RiceGuard server on the local
 * network via mDNS/DNS-SD (_riceguard._tcp), matching the advertisement
 * server/discovery.py registers. No IP address is ever hard-coded or typed
 * by the user in the automatic path.
 *
 * Discovery is push-based (mDNS multicast) -- once started it just waits for
 * onServiceFound/onServiceLost callbacks, so it never polls or "hammers"
 * the network (project brief requirement).
 *
 * Resolution: Android 14+ (API 34) offers registerServiceInfoCallback, which
 * keeps tracking a resolved service's current address for as long as it's
 * registered (survives the PC's IP changing via DHCP without the caller
 * re-resolving). That API-34-only surface is isolated in [ModernResolver]
 * (a separate @RequiresApi-annotated class, never referenced from a field of
 * that exact type here) so this class stays safely loadable on minSdk 26.
 * Below API 34, each onServiceFound triggers one classic one-shot
 * resolveService() call instead; a fresh IP after a DHCP change is only
 * picked up on the next mDNS re-announcement, which is an acceptable
 * trade-off for the older-OS fallback path.
 */
class ServiceDiscoveryRepository(context: Context) {
    private val appContext = context.applicationContext
    private val nsdManager = appContext.getSystemService(Context.NSD_SERVICE) as NsdManager
    private val connectivityManager =
        appContext.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
    private val wifiManager =
        appContext.applicationContext.getSystemService(Context.WIFI_SERVICE) as? WifiManager
    private val callbackExecutor: Executor = Executors.newSingleThreadExecutor()

    private var multicastLock: WifiManager.MulticastLock? = null
    private var discoveryListener: NsdManager.DiscoveryListener? = null
    private var modernResolver: Any? = null // ModernResolver, only touched when SDK_INT >= 34
    private var networkCallback: ConnectivityManager.NetworkCallback? = null
    private var discoveryActive = false

    private val _discoveredServer = MutableStateFlow<DiscoveredServer?>(null)
    val discoveredServer: StateFlow<DiscoveredServer?> = _discoveredServer

    /** Bumped every time the active Wi-Fi network changes (new network or
     * lost) -- ConnectionRepository restarts discovery+verification on each
     * change rather than trusting a previously resolved address. */
    private val _networkGeneration = MutableStateFlow(0)
    val networkGeneration: StateFlow<Int> = _networkGeneration

    fun startDiscovery() {
        if (discoveryActive) return
        discoveryActive = true
        acquireMulticastLock()
        registerNetworkCallback()

        val listener = object : NsdManager.DiscoveryListener {
            override fun onDiscoveryStarted(serviceType: String) {
                Log.i(TAG, "Discovery started for $serviceType")
            }

            override fun onServiceFound(serviceInfo: NsdServiceInfo) {
                Log.i(TAG, "Service found: ${serviceInfo.serviceName}")
                resolve(serviceInfo)
            }

            override fun onServiceLost(serviceInfo: NsdServiceInfo) {
                Log.i(TAG, "Service lost: ${serviceInfo.serviceName}")
                if (_discoveredServer.value?.name == serviceInfo.serviceName) {
                    _discoveredServer.value = null
                }
            }

            override fun onDiscoveryStopped(serviceType: String) {
                Log.i(TAG, "Discovery stopped for $serviceType")
            }

            override fun onStartDiscoveryFailed(serviceType: String, errorCode: Int) {
                Log.w(TAG, "Start discovery failed for $serviceType: $errorCode")
                discoveryActive = false
            }

            override fun onStopDiscoveryFailed(serviceType: String, errorCode: Int) {
                Log.w(TAG, "Stop discovery failed for $serviceType: $errorCode")
                try {
                    nsdManager.stopServiceDiscovery(this)
                } catch (e: Exception) {
                    Log.w(TAG, "retry stopServiceDiscovery threw", e)
                }
            }
        }
        discoveryListener = listener
        try {
            nsdManager.discoverServices(RICEGUARD_SERVICE_TYPE, NsdManager.PROTOCOL_DNS_SD, listener)
        } catch (e: Exception) {
            Log.w(TAG, "discoverServices threw", e)
            discoveryActive = false
        }
    }

    fun stopDiscovery() {
        if (!discoveryActive) return
        discoveryActive = false
        discoveryListener?.let {
            try {
                nsdManager.stopServiceDiscovery(it)
            } catch (e: Exception) {
                Log.w(TAG, "stopServiceDiscovery threw", e)
            }
        }
        discoveryListener = null
        stopModernResolver()
        unregisterNetworkCallback()
        releaseMulticastLock()
        _discoveredServer.value = null
    }

    /** Explicit user-triggered "Rediscover", and also used internally on
     * Wi-Fi network changes. */
    fun restartDiscovery() {
        stopDiscovery()
        startDiscovery()
    }

    private fun resolve(serviceInfo: NsdServiceInfo) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            resolveModern(serviceInfo)
        } else {
            resolveLegacy(serviceInfo)
        }
    }

    private fun resolveLegacy(serviceInfo: NsdServiceInfo) {
        val listener = object : NsdManager.ResolveListener {
            override fun onResolveFailed(info: NsdServiceInfo, errorCode: Int) {
                Log.w(TAG, "Resolve failed for ${info.serviceName}: $errorCode")
            }

            override fun onServiceResolved(info: NsdServiceInfo) {
                @Suppress("DEPRECATION") // NsdServiceInfo.host: single-address API, superseded by
                // hostAddresses on API 34+ (used in ModernResolver) -- this is the
                // deliberate < API 34 fallback path, not a stray unmigrated call.
                val host = info.host?.hostAddress ?: return
                _discoveredServer.value = DiscoveredServer(info.serviceName, listOf(host), info.port)
            }
        }
        try {
            @Suppress("DEPRECATION") // resolveService(NsdServiceInfo, ResolveListener): the
            // pre-API-34 resolve API; superseded by registerServiceInfoCallback on
            // newer platforms (see ModernResolver), kept here as the compatible
            // fallback for API < 34 as required.
            nsdManager.resolveService(serviceInfo, listener)
        } catch (e: Exception) {
            Log.w(TAG, "resolveService threw", e)
        }
    }

    private fun resolveModern(serviceInfo: NsdServiceInfo) {
        val resolver = getOrCreateModernResolver()
        resolver.resolve(serviceInfo)
    }

    @RequiresApi(Build.VERSION_CODES.UPSIDE_DOWN_CAKE)
    private fun getOrCreateModernResolver(): ModernResolver {
        (modernResolver as? ModernResolver)?.let { return it }
        val resolver = ModernResolver(
            nsdManager = nsdManager,
            executor = callbackExecutor,
            onResolved = { server -> _discoveredServer.value = server },
            onLost = { name -> if (_discoveredServer.value?.name == name) _discoveredServer.value = null },
            onUnsupported = { info -> resolveLegacy(info) },
        )
        modernResolver = resolver
        return resolver
    }

    private fun stopModernResolver() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            (modernResolver as? ModernResolver)?.stop()
        }
        modernResolver = null
    }

    private fun acquireMulticastLock() {
        // Several OEM Wi-Fi stacks (this app is verified on a Xiaomi/HyperOS
        // device) silently drop inbound multicast packets to save power
        // unless something explicitly asks for them -- without this, mDNS
        // replies from the server may simply never reach the app.
        try {
            val lock = wifiManager?.createMulticastLock("riceguard-mdns")
            lock?.setReferenceCounted(true)
            lock?.acquire()
            multicastLock = lock
        } catch (e: Exception) {
            Log.w(TAG, "MulticastLock acquire failed", e)
        }
    }

    private fun releaseMulticastLock() {
        try {
            multicastLock?.let { if (it.isHeld) it.release() }
        } catch (e: Exception) {
            Log.w(TAG, "MulticastLock release failed", e)
        }
        multicastLock = null
    }

    private fun registerNetworkCallback() {
        val request = NetworkRequest.Builder()
            .addTransportType(NetworkCapabilities.TRANSPORT_WIFI)
            .addCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
            .build()
        // registerNetworkCallback() synchronously replays onAvailable() for
        // any network that already satisfies the request at the moment of
        // registration (documented ConnectivityManager behavior) -- that is
        // NOT a real network change, it's just reporting the network
        // startDiscovery() was already about to search on. Without this
        // guard, that replay bumps networkGeneration, which
        // ConnectionRepository.runAutoDiscoveryLoop reacts to by calling
        // restartDiscovery() -- which re-registers this same callback, which
        // replays onAvailable() again, forever: an infinite restart loop
        // that never gives discoverServices() enough time to receive an
        // actual mDNS reply. Confirmed live on real Wi-Fi (not reproduced
        // over USB tethering, where this callback's network never matched
        // TRANSPORT_WIFI in the first place) -- logcat showed
        // "Wi-Fi network available -- rediscovering" firing every ~3-5ms.
        var firstCallback = true
        val callback = object : ConnectivityManager.NetworkCallback() {
            override fun onAvailable(network: Network) {
                if (firstCallback) {
                    firstCallback = false
                    Log.i(TAG, "Wi-Fi network available (already active at registration -- not a change)")
                    return
                }
                Log.i(TAG, "Wi-Fi network available -- rediscovering")
                _networkGeneration.value = _networkGeneration.value + 1
            }

            override fun onLost(network: Network) {
                firstCallback = false
                Log.i(TAG, "Wi-Fi network lost")
                _discoveredServer.value = null
                _networkGeneration.value = _networkGeneration.value + 1
            }
        }
        networkCallback = callback
        try {
            connectivityManager.registerNetworkCallback(request, callback)
        } catch (e: Exception) {
            Log.w(TAG, "registerNetworkCallback threw", e)
        }
    }

    private fun unregisterNetworkCallback() {
        networkCallback?.let {
            try {
                connectivityManager.unregisterNetworkCallback(it)
            } catch (e: Exception) {
                Log.w(TAG, "unregisterNetworkCallback threw", e)
            }
        }
        networkCallback = null
    }
}

/** API-34-only NSD resolution, isolated from [ServiceDiscoveryRepository] so
 * that class never holds a field of this exact type (kept as `Any?` there)
 * and this class is only ever instantiated behind an explicit SDK_INT
 * check -- avoids any class-verification risk on the minSdk-26 fallback
 * path, per Android's documented pattern for version-gated API surfaces. */
@RequiresApi(Build.VERSION_CODES.UPSIDE_DOWN_CAKE)
private class ModernResolver(
    private val nsdManager: NsdManager,
    private val executor: Executor,
    private val onResolved: (DiscoveredServer) -> Unit,
    private val onLost: (serviceName: String) -> Unit,
    private val onUnsupported: (NsdServiceInfo) -> Unit,
) {
    private var callback: NsdManager.ServiceInfoCallback? = null
    private var trackedName: String? = null

    fun resolve(serviceInfo: NsdServiceInfo) {
        stop()
        trackedName = serviceInfo.serviceName
        val cb = object : NsdManager.ServiceInfoCallback {
            override fun onServiceInfoCallbackRegistrationFailed(errorCode: Int) {
                Log.w(TAG, "ServiceInfoCallback registration failed ($errorCode) -- falling back")
                onUnsupported(serviceInfo)
            }

            override fun onServiceUpdated(updated: NsdServiceInfo) {
                val hosts = updated.hostAddresses.mapNotNull { it.hostAddress }
                if (hosts.isEmpty()) return
                onResolved(DiscoveredServer(updated.serviceName, hosts, updated.port))
            }

            override fun onServiceLost() {
                trackedName?.let(onLost)
            }

            override fun onServiceInfoCallbackUnregistered() {}
        }
        callback = cb
        try {
            nsdManager.registerServiceInfoCallback(serviceInfo, executor, cb)
        } catch (e: Exception) {
            Log.w(TAG, "registerServiceInfoCallback threw", e)
            onUnsupported(serviceInfo)
        }
    }

    fun stop() {
        callback?.let {
            try {
                nsdManager.unregisterServiceInfoCallback(it)
            } catch (e: Exception) {
                Log.w(TAG, "unregisterServiceInfoCallback threw", e)
            }
        }
        callback = null
        trackedName = null
    }
}
