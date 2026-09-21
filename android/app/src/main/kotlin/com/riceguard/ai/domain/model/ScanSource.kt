package com.riceguard.ai.domain.model

/** Where a scan's photo came from -- shown in History/Scan Detail so a
 * gallery-uploaded scan is never confused with a fresh camera capture. */
enum class ScanSource(val wireValue: String, val label: String) {
    CAMERA("camera", "Camera"),
    GALLERY("gallery", "Gallery");

    companion object {
        fun fromWire(value: String): ScanSource = entries.firstOrNull { it.wireValue == value } ?: CAMERA
    }
}
