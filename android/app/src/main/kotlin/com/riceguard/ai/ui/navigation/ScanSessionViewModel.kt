package com.riceguard.ai.ui.navigation

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.riceguard.ai.data.repository.ScanRepository
import com.riceguard.ai.domain.model.AppError
import com.riceguard.ai.domain.model.Outcome
import com.riceguard.ai.domain.model.ScanResult
import com.riceguard.ai.domain.model.ScanSource
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import java.io.File

sealed interface ScanUploadState {
    data object Idle : ScanUploadState
    data object Uploading : ScanUploadState
    data class Success(val result: ScanResult) : ScanUploadState
    data class Failed(val error: AppError) : ScanUploadState
}

/**
 * One instance shared across Camera -> PhotoConfirmation -> Processing ->
 * Result -> Explanation -> Recommendation -> (LowConfidence |
 * ModelDisagreement), scoped to the nav graph (see RiceGuardNavGraph) rather
 * than per-screen -- there is exactly one scan in flight at a time, so this
 * avoids re-fetching or serializing the result at every step.
 */
class ScanSessionViewModel(private val scanRepository: ScanRepository) : ViewModel() {

    private val _capturedPhoto = MutableStateFlow<File?>(null)
    val capturedPhoto: StateFlow<File?> = _capturedPhoto

    private val _photoSource = MutableStateFlow(ScanSource.CAMERA)
    val photoSource: StateFlow<ScanSource> = _photoSource

    private val _uploadState = MutableStateFlow<ScanUploadState>(ScanUploadState.Idle)
    val uploadState: StateFlow<ScanUploadState> = _uploadState

    fun onPhotoCaptured(file: File, source: ScanSource) {
        _capturedPhoto.value = file
        _photoSource.value = source
        _uploadState.value = ScanUploadState.Idle
    }

    /** project brief section 8: Retake discards the captured image locally
     * -- no network call has happened yet at this point, so there is
     * nothing server-side to undo. */
    fun retake() {
        _capturedPhoto.value?.delete()
        _capturedPhoto.value = null
        _uploadState.value = ScanUploadState.Idle
    }

    fun submitPhoto() {
        val photo = _capturedPhoto.value ?: return
        if (_uploadState.value is ScanUploadState.Uploading) return
        _uploadState.value = ScanUploadState.Uploading
        viewModelScope.launch {
            when (val outcome = scanRepository.submitScan(photo, _photoSource.value)) {
                is Outcome.Success -> _uploadState.value = ScanUploadState.Success(outcome.data)
                is Outcome.Failure -> _uploadState.value = ScanUploadState.Failed(outcome.error)
            }
        }
    }

    fun retryUpload() {
        if (_capturedPhoto.value != null) submitPhoto()
    }

    /** Called when the user leaves the result flow back to Home, or starts a
     * brand-new scan -- clears in-memory state (the staged capture file
     * itself was already either persisted by ScanRepository or is now
     * orphaned cache; cache cleanup is opportunistic, not safety-critical). */
    fun startNewScan() {
        _capturedPhoto.value = null
        _uploadState.value = ScanUploadState.Idle
    }
}
