package com.riceguard.ai.data.repository

import com.google.gson.Gson
import com.google.gson.JsonParseException
import com.riceguard.ai.data.local.ImageStorage
import com.riceguard.ai.data.local.ScanDao
import com.riceguard.ai.data.local.ScanRecord
import com.riceguard.ai.data.network.ApiService
import com.riceguard.ai.data.network.dto.ErrorResponseDto
import com.riceguard.ai.data.network.dto.RecommendationDto
import com.riceguard.ai.data.network.toDomain
import com.riceguard.ai.data.settings.SettingsRepository
import com.riceguard.ai.domain.model.AppError
import com.riceguard.ai.domain.model.Outcome
import com.riceguard.ai.domain.model.Recommendation
import com.riceguard.ai.domain.model.ScanResult
import com.riceguard.ai.domain.model.ScanSource
import com.riceguard.ai.domain.model.ScanStatus
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.asRequestBody
import java.io.File
import java.io.IOException

class ScanRepository(
    private val apiService: ApiService,
    private val httpClient: OkHttpClient,
    private val settingsRepository: SettingsRepository,
    private val scanDao: ScanDao,
    private val imageStorage: ImageStorage,
) {
    private val gson = Gson()
    private val backgroundScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    /** Uploads [photoFile] (already confirmed by the user -- project brief
     * section 8: nothing is sent before "Use Photo") and returns the parsed
     * result with fully-resolved (absolute) image URLs, ready to display
     * immediately. If "Save Scan History" is on, persistence to Room +
     * private storage happens in the background afterward (see
     * [persistIfEnabled]) so the Result screen doesn't wait on image
     * downloads to render. */
    suspend fun submitScan(photoFile: File, source: ScanSource): Outcome<ScanResult> {
        val part = MultipartBody.Part.createFormData(
            "image", photoFile.name, photoFile.asRequestBody("image/jpeg".toMediaType()),
        )

        val response = try {
            apiService.predict(part)
        } catch (e: IOException) {
            return Outcome.Failure(AppError.NoConnection(describeConnectionFailure(e)))
        } catch (e: JsonParseException) {
            return Outcome.Failure(AppError.Unknown(null, "The server sent a response this app could not understand."))
        } catch (e: Exception) {
            return Outcome.Failure(AppError.Unknown(null, e.message ?: "Unexpected error"))
        }

        if (!response.isSuccessful) {
            // errorBody() can only be read ONCE -- read it into a local
            // variable before parsing, rather than calling .string() twice
            // (the second call would see an already-consumed/closed body).
            val parsedError = parseErrorBody(response.errorBody()?.string())
            val requestId = parsedError?.requestId
            val message = parsedError?.message ?: "The server could not complete the analysis."
            val error = when (response.code()) {
                400 -> AppError.InvalidImage(message)
                503 -> AppError.ServerUnavailable
                500 -> AppError.ServerProcessingFailed(requestId, message)
                else -> AppError.Unknown(requestId, message)
            }
            return Outcome.Failure(error)
        }

        val body = response.body() ?: return Outcome.Failure(AppError.Unknown(null, "Empty server response."))
        val baseUrl = settingsRepository.currentServerConfig.value.baseUrl
        val result = body.toDomain().let { r ->
            r.copy(
                gradcamUrl = r.gradcamUrl?.let { resolveUrl(baseUrl, it) },
                detectionUrl = r.detectionUrl?.let { resolveUrl(baseUrl, it) },
            )
        }

        if (settingsRepository.currentSaveHistory.value) {
            backgroundScope.launch { persist(result, photoFile, source) }
        }

        return Outcome.Success(result)
    }

    private fun parseErrorBody(raw: String?): ErrorResponseDto? =
        try { if (raw.isNullOrBlank()) null else gson.fromJson(raw, ErrorResponseDto::class.java) }
        catch (_: Exception) { null }

    private fun resolveUrl(baseUrl: String, relative: String): String =
        if (relative.startsWith("http")) relative else baseUrl.trimEnd('/') + relative

    private suspend fun persist(result: ScanResult, photoFile: File, source: ScanSource) {
        try {
            val scanId = result.requestId
            photoFile.inputStream().use { imageStorage.saveOriginal(scanId, it) }

            val detectionPath = result.detectionUrl?.let { downloadTo(it, scanId, "detection.jpg") }
            val gradcamPath = result.gradcamUrl?.let { downloadTo(it, scanId, "gradcam.png") }

            val record = ScanRecord(
                id = scanId,
                timestampMillis = System.currentTimeMillis(),
                disease = result.finalDisease,
                finalConfidence = result.finalConfidence,
                status = when (result.status) {
                    ScanStatus.OK -> "ok"
                    ScanStatus.LOW_CONFIDENCE -> "low_confidence"
                    ScanStatus.MODEL_DISAGREEMENT -> "model_disagreement"
                    ScanStatus.NOT_RECOGNIZED -> "not_recognized"
                    ScanStatus.ERROR -> "error"
                },
                yoloPrediction = result.yoloPrediction,
                yoloConfidence = result.yoloConfidence,
                cnnPrediction = result.cnnPrediction,
                cnnConfidence = result.cnnConfidence,
                agreement = result.agreement,
                recommendationJson = result.recommendation?.let { gson.toJson(it.toDto()) },
                originalImagePath = imageStorage.originalPhotoFile(scanId).absolutePath,
                detectionImagePath = detectionPath?.absolutePath,
                gradcamImagePath = gradcamPath?.absolutePath,
                source = source.wireValue,
                pipelineProcessingTimeSec = result.pipelineProcessingTimeSec,
            )
            scanDao.insert(record)
        } catch (_: Exception) {
            // Best-effort: a failed background save must never crash the app
            // or corrupt the already-shown result -- the user already saw
            // their answer; losing the history entry is the only consequence,
            // and it fails silently ONLY here (never for the live result).
        }
    }

    private fun downloadTo(url: String, scanId: String, filename: String): File? {
        val request = Request.Builder().url(url).build()
        httpClient.newCall(request).execute().use { resp ->
            if (!resp.isSuccessful) return null
            val bytes = resp.body?.bytes() ?: return null
            return imageStorage.saveBytes(scanId, filename, bytes)
        }
    }

    fun observeHistory(): Flow<List<ScanRecord>> = scanDao.observeAll()

    fun observeHistoryCount(): Flow<Int> = scanDao.observeCount()

    suspend fun getScanDetail(id: String): ScanRecord? = scanDao.getById(id)

    suspend fun updateNotes(id: String, notes: String?) = scanDao.updateNotes(id, notes)

    fun recommendationFromJson(json: String?): Recommendation? =
        if (json.isNullOrBlank()) null
        else try { gson.fromJson(json, RecommendationDto::class.java).toDomain() } catch (_: Exception) { null }

    suspend fun deleteScan(id: String) {
        scanDao.deleteById(id)
        imageStorage.deleteScan(id)
    }

    suspend fun deleteAllHistory() {
        scanDao.deleteAll()
        imageStorage.deleteAll()
    }
}

private fun Recommendation.toDto() = RecommendationDto(
    available = available,
    requested = null,
    reason = unavailableReason,
    disease = disease,
    causalAgent = causalAgent,
    description = description,
    symptoms = symptoms,
    management = management,
    treatment = treatment,
    prevention = prevention,
    requiresExpertValidation = requiresExpertValidation,
    disclaimer = disclaimer,
)
