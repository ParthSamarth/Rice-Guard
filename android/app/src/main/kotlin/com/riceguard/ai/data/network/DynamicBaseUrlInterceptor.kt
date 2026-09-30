package com.riceguard.ai.data.network

import com.riceguard.ai.data.settings.SettingsRepository
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import okhttp3.Interceptor
import okhttp3.Response

/**
 * Retrofit's base URL is fixed at Retrofit.Builder() time, but this app's
 * server host/port is user-editable at any moment from the Settings screen
 * (project brief section 19). Rather than rebuild the whole Retrofit/OkHttp
 * stack on every settings change, one OkHttpClient is built once and this
 * interceptor rewrites each outgoing request's host/port from
 * [SettingsRepository.currentServerConfig] (a StateFlow mirror kept live by
 * SettingsRepository.observeAndMirror -- interceptors are synchronous, so
 * this reads its already-current in-memory value, not suspending DataStore).
 */
class DynamicBaseUrlInterceptor(private val settingsRepository: SettingsRepository) : Interceptor {
    override fun intercept(chain: Interceptor.Chain): Response {
        val original = chain.request()
        val config = settingsRepository.currentServerConfig.value

        if (config.host.isBlank()) {
            // No server configured yet -- let the request fail naturally
            // (ConnectException/UnknownHost) rather than silently substitute
            // a guess; the repository layer maps this to AppError.NoConnection.
            return chain.proceed(original)
        }

        val newUrl = original.url.newBuilder()
            .scheme("http")
            .host(config.host)
            .port(config.port)
            .build()

        return chain.proceed(original.newBuilder().url(newUrl).build())
    }
}

/** Used only by ConnectionChecker for a one-off "does this host:port even
 * parse" validation before saving a new server config from the Settings
 * screen (project brief section 19's "Test Connection"). */
fun isValidHostPort(host: String, port: Int): Boolean {
    if (host.isBlank() || port !in 1..65535) return false
    return "http://$host:$port/".toHttpUrlOrNull() != null
}
