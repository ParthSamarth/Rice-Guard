package com.riceguard.ai.domain.model

/** Every user-facing failure mode from project brief section 21, typed so
 * each screen can render its own specific copy instead of one generic
 * "error" bucket -- and so a raw exception/stack trace can never leak into
 * a Composable by accident (there is no "just show exception.message" path
 * here; every AppError already carries pre-written, friendly text). */
sealed interface AppError {
    data class NoConnection(val reason: String) : AppError
    data object ServerUnavailable : AppError
    data class InvalidImage(val reason: String) : AppError
    data class UploadFailed(val reason: String) : AppError
    data class ServerProcessingFailed(val requestId: String?, val reason: String) : AppError
    data class Unknown(val requestId: String?, val reason: String) : AppError
}

sealed interface Outcome<out T> {
    data class Success<T>(val data: T) : Outcome<T>
    data class Failure(val error: AppError) : Outcome<Nothing>
}

inline fun <T> Outcome<T>.onSuccess(block: (T) -> Unit): Outcome<T> {
    if (this is Outcome.Success) block(data)
    return this
}

inline fun <T> Outcome<T>.onFailure(block: (AppError) -> Unit): Outcome<T> {
    if (this is Outcome.Failure) block(error)
    return this
}
