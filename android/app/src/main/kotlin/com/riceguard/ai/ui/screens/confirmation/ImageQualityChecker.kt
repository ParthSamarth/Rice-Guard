package com.riceguard.ai.ui.screens.confirmation

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.File
import kotlin.math.abs

data class ImageQualityResult(val isAcceptable: Boolean, val warnings: List<String>)

/**
 * project brief section 9: "basic image-quality validation... NOT disease
 * classification." Deliberately simple, dependency-free heuristics -- no ML
 * model, no leaf-detection (that would itself be a small object-detection
 * problem, out of scope for "is this photo usable"). Runs on a downsampled
 * copy for speed/memory, never the full-resolution original.
 */
object ImageQualityChecker {

    private const val MIN_DIMENSION = 300
    private const val DARK_THRESHOLD = 35     // mean luma 0-255
    private const val BRIGHT_THRESHOLD = 235
    private const val BLUR_GRADIENT_THRESHOLD = 8.0 // mean absolute gradient; lower = blurrier

    fun check(file: File): ImageQualityResult {
        val warnings = mutableListOf<String>()

        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.absolutePath, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
            return ImageQualityResult(isAcceptable = false, warnings = listOf("This image could not be read."))
        }
        if (bounds.outWidth < MIN_DIMENSION || bounds.outHeight < MIN_DIMENSION) {
            warnings += "Resolution is quite low."
        }

        val sample = decodeDownsampled(file, targetSize = 200) ?: return ImageQualityResult(
            isAcceptable = warnings.isEmpty(), warnings = warnings,
        )

        val luma = meanLuma(sample)
        when {
            luma < DARK_THRESHOLD -> warnings += "The photo looks very dark."
            luma > BRIGHT_THRESHOLD -> warnings += "The photo looks overexposed."
        }

        val gradient = meanGradient(sample)
        if (gradient < BLUR_GRADIENT_THRESHOLD) {
            warnings += "The photo may be blurry."
        }

        sample.recycle()

        // Hard-unusable only for a genuinely unreadable file (checked above);
        // everything else is a soft warning the user can override (section 9:
        // "do not force the user to retake unless the image is technically
        // unusable").
        return ImageQualityResult(isAcceptable = true, warnings = warnings)
    }

    private fun decodeDownsampled(file: File, targetSize: Int): Bitmap? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.absolutePath, bounds)
        var sampleSize = 1
        while (bounds.outWidth / (sampleSize * 2) >= targetSize && bounds.outHeight / (sampleSize * 2) >= targetSize) {
            sampleSize *= 2
        }
        val opts = BitmapFactory.Options().apply { inSampleSize = sampleSize }
        return BitmapFactory.decodeFile(file.absolutePath, opts)
    }

    private fun meanLuma(bitmap: Bitmap): Double {
        var sum = 0L
        var count = 0
        val stepX = maxOf(1, bitmap.width / 64)
        val stepY = maxOf(1, bitmap.height / 64)
        var y = 0
        while (y < bitmap.height) {
            var x = 0
            while (x < bitmap.width) {
                val pixel = bitmap.getPixel(x, y)
                val r = (pixel shr 16) and 0xFF
                val g = (pixel shr 8) and 0xFF
                val b = pixel and 0xFF
                sum += (0.299 * r + 0.587 * g + 0.114 * b).toLong()
                count++
                x += stepX
            }
            y += stepY
        }
        return if (count > 0) sum.toDouble() / count else 128.0
    }

    /** Coarse blur proxy: mean absolute difference between horizontally
     * adjacent sampled pixels' luma. A sharp, detailed photo has many strong
     * local transitions; a blurry one is smoothed out and scores low. This
     * is a heuristic, not an edge-detection algorithm -- it is only meant to
     * catch obviously/extremely blurred captures. */
    private fun meanGradient(bitmap: Bitmap): Double {
        var sum = 0.0
        var count = 0
        val step = maxOf(1, bitmap.width / 80)
        var y = 0
        while (y < bitmap.height) {
            var x = 0
            while (x < bitmap.width - step) {
                val p1 = bitmap.getPixel(x, y)
                val p2 = bitmap.getPixel(x + step, y)
                val l1 = ((p1 shr 16 and 0xFF) + (p1 shr 8 and 0xFF) + (p1 and 0xFF)) / 3.0
                val l2 = ((p2 shr 16 and 0xFF) + (p2 shr 8 and 0xFF) + (p2 and 0xFF)) / 3.0
                sum += abs(l1 - l2)
                count++
                x += step
            }
            y += maxOf(1, bitmap.height / 80)
        }
        return if (count > 0) sum / count else 100.0
    }
}
