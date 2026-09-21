"""
server/schemas.py

Pydantic response models for the RiceGuard AI API. These are a presentation
layer over pipeline.inference_pipeline.RiceDiseasePipeline.run()'s own return
dict (see that module's run() docstring for the authoritative schema) --
nothing here computes a metric or a prediction; every field is copied
straight from the pipeline's own output, with local filesystem paths
translated into fetchable /results/{request_id}/{filename} URLs (never a raw
PC path -- see server/security.py:path_to_result_url).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str = "RiceGuard AI"
    models_ready: bool
    yolo: bool
    resnet50: bool
    device: Optional[str] = None
    queue_depth: int = Field(0, description="Requests currently waiting for the single GPU worker slot.")


class DetectionOut(BaseModel):
    class_: str = Field(alias="class")
    class_id: int
    confidence: float
    bbox_xyxy: list[float]

    class Config:
        populate_by_name = True


class RegionOut(BaseModel):
    region_index: Optional[int]
    yolo_class: Optional[str]
    yolo_confidence: Optional[float]
    bbox_xyxy: Optional[list[float]]
    cnn_class: Optional[str]
    cnn_confidence: Optional[float]
    cnn_probabilities: Optional[dict]
    agreement: Optional[bool]
    gradcam_url: Optional[str]


class AgreementDetail(BaseModel):
    status: str
    yolo_prediction: Optional[str]
    cnn_prediction: Optional[str]
    yolo_confidence: Optional[float]
    cnn_confidence: Optional[float]


class RecommendationOut(BaseModel):
    available: bool
    requested: Optional[str] = None
    reason: Optional[str] = None
    disease: Optional[str] = None
    causal_agent: Optional[str] = None
    description: Optional[str] = None
    symptoms: Optional[list[str]] = None
    management: Optional[list[str]] = None
    treatment: Optional[list[str]] = None
    prevention: Optional[list[str]] = None
    requires_expert_validation: Optional[bool] = None
    disclaimer: Optional[str] = None


class PredictResponse(BaseModel):
    request_id: str
    status: str  # "ok" | "model_disagreement" | "low_confidence"
    decision_path: str  # "yolo_cnn_verification" | "cnn_only_full_image"

    final_disease: Optional[str]
    final_confidence: Optional[float]

    yolo_prediction: Optional[str]
    yolo_confidence: Optional[float]
    cnn_prediction: Optional[str]
    cnn_confidence: Optional[float]

    agreement: bool
    agreement_detail: Optional[AgreementDetail]

    detections: list[DetectionOut]
    regions: list[RegionOut]

    gradcam_url: Optional[str]
    detection_url: Optional[str]

    recommendation: Optional[RecommendationOut]

    thresholds_used: dict
    pipeline_processing_time_sec: float
    server_processing_time_sec: float


class ErrorResponse(BaseModel):
    request_id: Optional[str] = None
    status: str = "error"
    message: str
