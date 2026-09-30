package com.riceguard.ai.data.network

import com.riceguard.ai.data.network.dto.DetectionDto
import com.riceguard.ai.data.network.dto.PredictResponseDto
import com.riceguard.ai.data.network.dto.RecommendationDto
import com.riceguard.ai.domain.model.DecisionPath
import com.riceguard.ai.domain.model.Detection
import com.riceguard.ai.domain.model.Recommendation
import com.riceguard.ai.domain.model.ScanResult
import com.riceguard.ai.domain.model.ScanStatus

fun PredictResponseDto.toDomain(): ScanResult = ScanResult(
    requestId = requestId,
    status = ScanStatus.fromWire(status),
    decisionPath = DecisionPath.fromWire(decisionPath),
    finalDisease = finalDisease,
    finalConfidence = finalConfidence,
    yoloPrediction = yoloPrediction,
    yoloConfidence = yoloConfidence,
    cnnPrediction = cnnPrediction,
    cnnConfidence = cnnConfidence,
    agreement = agreement,
    detections = detections.map { it.toDomain() },
    gradcamUrl = gradcamUrl,
    detectionUrl = detectionUrl,
    recommendation = recommendation?.toDomain(),
    pipelineProcessingTimeSec = pipelineProcessingTimeSec,
    cnnProbabilities = regions.firstOrNull()?.cnnProbabilities,
)

fun DetectionDto.toDomain(): Detection = Detection(
    className = className,
    confidence = confidence,
    bboxXyxy = bboxXyxy,
)

fun RecommendationDto.toDomain(): Recommendation = Recommendation(
    available = available,
    disease = disease,
    causalAgent = causalAgent,
    description = description,
    symptoms = symptoms.orEmpty(),
    management = management.orEmpty(),
    treatment = treatment.orEmpty(),
    prevention = prevention.orEmpty(),
    requiresExpertValidation = requiresExpertValidation ?: true,
    disclaimer = disclaimer,
    unavailableReason = reason,
)
