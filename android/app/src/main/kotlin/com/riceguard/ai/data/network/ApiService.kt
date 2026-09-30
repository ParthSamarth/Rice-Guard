package com.riceguard.ai.data.network

import com.riceguard.ai.data.network.dto.HealthResponseDto
import com.riceguard.ai.data.network.dto.PredictResponseDto
import okhttp3.MultipartBody
import retrofit2.Response
import retrofit2.http.GET
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part

/** Mirrors server/main.py's routes exactly. Uses Response<T> (not a bare
 * suspend return) everywhere so the repository layer can distinguish
 * "server answered with an error" (400/500/503, a structured ErrorResponseDto
 * body -- see ApiResultMapper) from "never reached the server at all"
 * (IOException, handled by the repository's try/catch), which the UI needs
 * to tell apart (project brief section 21: "network unavailable" vs.
 * "server processing failure" are different screens). */
interface ApiService {

    @GET("health")
    suspend fun health(): Response<HealthResponseDto>

    @Multipart
    @POST("predict")
    suspend fun predict(@Part image: MultipartBody.Part): Response<PredictResponseDto>
}
