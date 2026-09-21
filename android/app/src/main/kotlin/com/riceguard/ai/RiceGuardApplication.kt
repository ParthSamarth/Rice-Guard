package com.riceguard.ai

import android.app.Application
import com.riceguard.ai.di.AppContainer
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel

class RiceGuardApplication : Application() {

    private val appScope = CoroutineScope(SupervisorJob())
    lateinit var container: AppContainer
        private set

    override fun onCreate() {
        super.onCreate()
        container = AppContainer(this, appScope)
    }

    override fun onTerminate() {
        super.onTerminate()
        appScope.cancel()
    }
}
