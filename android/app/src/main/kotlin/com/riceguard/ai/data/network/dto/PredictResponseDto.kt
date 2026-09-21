package com.riceguard.ai.data.network.dto

import com.google.gson.annotations.SerializedName

/** Mirrors server/schemas.py exactly -- one field-for-field copy, on
 * purpose, so a schema drift between server and app is a compile-time (or at
 * worst a silent-null-field) issue to notice quickly, not a guessing game. */

data class HealthResponseDto(
    val status: String,
    val service: String,
    @SerializedName("models_ready") val modelsReady: Boolean,
    val yolo: Boolean,
    val resnet50: Boolean,
    val device: String?,
    @SerializedName("queue_depth") val queueDepth: Int,
)

data class DetectionDto(
    @SerializedName("class") val className: String,
    @SerializedName("class_id") val classId: Int,
    val confidence: Float,
    @SerializedName("bbox_xyxy") val bboxXyxy: List<Float>,
)

data class RegionDto(
    @SerializedName("region_index") val regionIndex: Int?,
    @SerializedName("yolo_class") val yoloClass: String?,
    @SerializedName("yolo_confidence") val yoloConfidence: Float?,
    @SerializedName("bbox_xyxy") val bboxXyxy: List<Float>?,
    @SerializedName("cnn_class") val cnnClass: String?,
    @SerializedName("cnn_confidence") val cnnConfidence: Float?,
    @SerializedName("cnn_probabilities") val cnnProbabilities: Map<String, Float>?,
    val agreement: Boolean?,
    @SerializedName("gradcam_url") val gradcamUrl: String?,
)

data class AgreementDetailDto(
    val status: String,
    @SerializedName("yolo_prediction") val yoloPrediction: String?,
    @SerializedName("cnn_prediction") val cnnPrediction: String?,
    @SerializedName("yolo_confidence") val yoloConfidence: Float?,
    @SerializedName("cnn_confidence") val cnnConfidence: Float?,
)

data class RecommendationDto(
    val available: Boolean,
    val requested: String?,
    val reason: String?,
    val disease: String?,
    @SerializedName("causal_agent") val causalAgent: String?,
    val description: String?,
    val symptoms: List<String>?,
    val management: List<String>?,
    val treatment: List<String>?,
    val prevention: List<String>?,
    @SerializedName("requires_expert_validation") val requiresExpertValidation: Boolean?,
    val disclaimer: String?,
)

data class PredictResponseDto(
    @SerializedName("request_id") val requestId: String,
    val status: String,
    @SerializedName("decision_path") val decisionPath: String,
    @SerializedName("final_disease") val finalDisease: String?,
    @SerializedName("final_confidence") val finalConfidence: Float?,
    @SerializedName("yolo_prediction") val yoloPrediction: String?,
    @SerializedName("yolo_confidence") val yoloConfidence: Float?,
    @SerializedName("cnn_prediction") val cnnPrediction: String?,
    @SerializedName("cnn_confidence") val cnnConfidence: Float?,
    val agreement: Boolean,
    @SerializedName("agreement_detail") val agreementDetail: AgreementDetailDto?,
    val detections: List<DetectionDto>,
    val regions: List<RegionDto>,
    @SerializedName("gradcam_url") val gradcamUrl: String?,
    @SerializedName("detection_url") val detectionUrl: String?,
    val recommendation: RecommendationDto?,
    @SerializedName("thresholds_used") val thresholdsUsed: Map<String, Float>,
    @SerializedName("pipeline_processing_time_sec") val pipelineProcessingTimeSec: Float,
    @SerializedName("server_processing_time_sec") val serverProcessingTimeSec: Float,
)

data class ErrorResponseDto(
    @SerializedName("request_id") val requestId: String?,
    val status: String,
    val message: String,
)
