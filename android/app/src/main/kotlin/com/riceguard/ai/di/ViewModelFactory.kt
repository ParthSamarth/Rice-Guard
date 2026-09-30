package com.riceguard.ai.di

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.compose.runtime.Composable

/** Generic factory bridging manual DI (AppContainer) to Compose's
 * viewModel(). Used as `viewModel { MyViewModel(container.xRepository) }`
 * via [rememberViewModel] below -- avoids a Hilt/Dagger dependency for a
 * project that must be authored without a local build to verify against. */
class SimpleViewModelFactory<T : ViewModel>(private val create: () -> T) : ViewModelProvider.Factory {
    @Suppress("UNCHECKED_CAST")
    override fun <U : ViewModel> create(modelClass: Class<U>): U = create() as U
}

@Composable
inline fun <reified T : ViewModel> rememberViewModel(noinline create: () -> T): T =
    viewModel(factory = SimpleViewModelFactory(create))
