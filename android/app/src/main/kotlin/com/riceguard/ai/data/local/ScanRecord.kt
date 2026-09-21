package com.riceguard.ai.data.local

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * Local scan history (project brief section 13). Images are NEVER stored as
 * blobs in this table -- only paths into this app's private storage
 * (context.filesDir/scans/<id>/...), per the brief's explicit instruction.
 * `recommendationJson` exists because the recommendation knowledge base
 * lives only on the PC (recommendation/knowledge_base.json) -- there is no
 * offline way to re-derive it from just a disease name on-device, so the
 * full recommendation returned at scan time is persisted verbatim, pre-
 * serialized to a JSON string by the repository layer (Gson, the same
 * library already used for the Retrofit responses), so History Detail works
 * with the PC completely disconnected. A plain String needs no Room
 * TypeConverter, unlike a nested object would.
 */
@Entity(tableName = "scan_records")
data class ScanRecord(
    @PrimaryKey val id: String, // the server's request_id, e.g. RG-20260819-0001 -- already unique
    val timestampMillis: Long,
    val disease: String?,
    val finalConfidence: Float?,
    val status: String, // "ok" | "low_confidence" | "model_disagreement"
    val yoloPrediction: String?,
    val yoloConfidence: Float?,
    val cnnPrediction: String?,
    val cnnConfidence: Float?,
    val agreement: Boolean,
    val recommendationJson: String?,
    val originalImagePath: String,
    val detectionImagePath: String?,
    val gradcamImagePath: String?,
    val source: String = "camera", // ScanSource.wireValue: "camera" | "gallery"
    val notes: String? = null, // user-editable, set from Scan Detail
    val pipelineProcessingTimeSec: Float? = null, // inference duration, from the server's own response
    val serverVersion: String? = null, // not currently returned by the server; kept nullable for when it is
)
