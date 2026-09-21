package com.riceguard.ai.data.network

import com.riceguard.ai.data.settings.SettingsRepository
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

/** Holds the one shared OkHttpClient (reused for raw result-image downloads
 * by ScanRepository, not just the Retrofit-generated ApiService, so both
 * paths share the same dynamic-base-URL/timeout/logging configuration
 * instead of duplicating it), plus a second, separate short-timeout client
 * used only to probe auto-discovery candidate addresses (see
 * [NetworkModule.DISCOVERY_PROBE_CONNECT_TIMEOUT_SEC] on why it's separate). */
class NetworkClients(val apiService: ApiService, val httpClient: OkHttpClient, val discoveryProbeClient: OkHttpClient)

object NetworkModule {

    /** A GPU inference call can legitimately take several seconds (and
     * longer if another request is queued ahead of it -- server/
     * pipeline_service.py serializes GPU work), so the read timeout is
     * generous rather than tuned for a typical REST call. */
    private const val CONNECT_TIMEOUT_SEC = 8L
    private const val READ_TIMEOUT_SEC = 90L
    private const val WRITE_TIMEOUT_SEC = 60L

    /** A machine with virtual adapters (WSL, Hyper-V, Docker, a VPN client)
     * can advertise several candidate addresses for itself; ConnectionRepository
     * tries each one's /health in turn until one answers. At the main
     * client's 8s connect timeout, a handful of dead candidates can add up
     * to a slow first connect -- a local-LAN health probe should fail (or
     * succeed) in a couple of seconds, so this dedicated client uses a much
     * shorter timeout. It's also intentionally its own OkHttpClient with NO
     * DynamicBaseUrlInterceptor: probing needs to hit a SPECIFIC candidate
     * host per call, not whatever SettingsRepository currently considers
     * "active" (which the interceptor would otherwise force every request
     * onto, making concurrent/ordered candidate probing impossible). */
    private const val DISCOVERY_PROBE_CONNECT_TIMEOUT_SEC = 2L
    private const val DISCOVERY_PROBE_READ_TIMEOUT_SEC = 3L

    fun create(settingsRepository: SettingsRepository): NetworkClients {
        val logging = HttpLoggingInterceptor().apply {
            level = HttpLoggingInterceptor.Level.BASIC // never logs the image body itself
        }

        val client = OkHttpClient.Builder()
            .addInterceptor(DynamicBaseUrlInterceptor(settingsRepository))
            .addInterceptor(logging)
            .connectTimeout(CONNECT_TIMEOUT_SEC, TimeUnit.SECONDS)
            .readTimeout(READ_TIMEOUT_SEC, TimeUnit.SECONDS)
            .writeTimeout(WRITE_TIMEOUT_SEC, TimeUnit.SECONDS)
            .build()

        val discoveryProbeClient = OkHttpClient.Builder()
            .connectTimeout(DISCOVERY_PROBE_CONNECT_TIMEOUT_SEC, TimeUnit.SECONDS)
            .readTimeout(DISCOVERY_PROBE_READ_TIMEOUT_SEC, TimeUnit.SECONDS)
            .build()

        // baseUrl here is a placeholder Retrofit requires syntactically valid
        // at build time -- DynamicBaseUrlInterceptor overwrites host/port/
        // scheme on every real request, so this value is never actually used.
        val retrofit = Retrofit.Builder()
            .baseUrl("http://localhost:8000/")
            .client(client)
            .addConverterFactory(GsonConverterFactory.create())
            .build()

        return NetworkClients(retrofit.create(ApiService::class.java), client, discoveryProbeClient)
    }
}
