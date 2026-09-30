package com.riceguard.ai.domain.model

/** Mirrors server/schemas.py:PredictResponse -- one place the UI reads a
 * completed scan's result from, independent of the Retrofit DTO shape. */
data class ScanResult(
    val requestId: String,
    val status: ScanStatus,
    val decisionPath: DecisionPath,
    val finalDisease: String?,
    val finalConfidence: Float?,
    val yoloPrediction: String?,
    val yoloConfidence: Float?,
    val cnnPrediction: String?,
    val cnnConfidence: Float?,
    val agreement: Boolean,
    val detections: List<Detection>,
    val gradcamUrl: String?,
    val detectionUrl: String?,
    val recommendation: Recommendation?,
    val pipelineProcessingTimeSec: Float?,
    val cnnProbabilities: Map<String, Float>?,
)

enum class ScanStatus { OK, LOW_CONFIDENCE, MODEL_DISAGREEMENT, NOT_RECOGNIZED, ERROR;

    companion object {
        fun fromWire(value: String): ScanStatus = when (value) {
            "ok" -> OK
            "low_confidence" -> LOW_CONFIDENCE
            "model_disagreement" -> MODEL_DISAGREEMENT
            "not_recognized" -> NOT_RECOGNIZED
            else -> ERROR
        }
    }
}

enum class DecisionPath { YOLO_CNN_VERIFICATION, CNN_ONLY_FULL_IMAGE;

    companion object {
        fun fromWire(value: String): DecisionPath = when (value) {
            "yolo_cnn_verification" -> YOLO_CNN_VERIFICATION
            else -> CNN_ONLY_FULL_IMAGE
        }
    }
}

data class Detection(
    val className: String,
    val confidence: Float,
    val bboxXyxy: List<Float>,
)

data class Recommendation(
    val available: Boolean,
    val disease: String?,
    val causalAgent: String?,
    val description: String?,
    val symptoms: List<String>,
    val management: List<String>,
    val treatment: List<String>,
    val prevention: List<String>,
    val requiresExpertValidation: Boolean,
    val disclaimer: String?,
    val unavailableReason: String? = null,
)
