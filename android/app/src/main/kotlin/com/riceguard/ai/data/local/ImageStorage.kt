package com.riceguard.ai.data.local

import android.content.Context
import java.io.File
import java.io.InputStream
import java.util.UUID

/**
 * All scan-related images live under context.filesDir/scans/<scanId>/ --
 * private app storage, never MediaStore/shared storage (project brief
 * section 13: "store images in private app storage"). One folder per scan
 * makes "delete this scan" (section 15) and "delete everything" (section 16)
 * both a single recursive directory delete, with no risk of leaving an
 * orphaned file behind.
 */
class ImageStorage(private val context: Context) {

    private fun scansRoot(): File = File(context.filesDir, "scans").apply { mkdirs() }

    fun scanDir(scanId: String): File = File(scansRoot(), scanId).apply { mkdirs() }

    /** A fresh capture doesn't have a server-assigned request_id yet (that
     * only exists after a successful upload), so it's staged in the cache
     * dir under a random name until the user presses "Use Photo" and the
     * upload succeeds -- ScanRepository then copies it into scanDir(id). */
    fun newCaptureStagingFile(cacheDir: File): File =
        File(cacheDir, "capture_${UUID.randomUUID()}.jpg")

    fun originalPhotoFile(scanId: String): File = File(scanDir(scanId), "original.jpg")

    fun savedFileFor(scanId: String, name: String): File = File(scanDir(scanId), name)

    /** Copies a just-captured/picked photo (already on-device, e.g. from
     * CameraX's own capture output or a content:// picker Uri stream) into
     * this scan's private folder. */
    fun saveOriginal(scanId: String, input: InputStream): File {
        val dest = originalPhotoFile(scanId)
        input.use { src -> dest.outputStream().use { out -> src.copyTo(out) } }
        return dest
    }

    fun saveBytes(scanId: String, filename: String, bytes: ByteArray): File {
        val dest = savedFileFor(scanId, filename)
        dest.writeBytes(bytes)
        return dest
    }

    /** project brief section 15: deleting one scan must remove its original
     * image, detection image, and Grad-CAM image together -- this call
     * removes the whole per-scan folder in one operation so nothing can be
     * partially deleted. */
    fun deleteScan(scanId: String) {
        scanDir(scanId).deleteRecursively()
    }

    /** project brief section 16: Delete All Scan Data. */
    fun deleteAll() {
        scansRoot().deleteRecursively()
    }

    /** Any scan folder not backed by a Room row (e.g. "Save History" was OFF
     * for that scan, or a previous run crashed mid-save) -- called from the
     * repository after a fresh read of current DB ids, never on a guess. */
    fun deleteOrphans(keepScanIds: Set<String>) {
        val root = scansRoot()
        root.listFiles()?.forEach { dir ->
            if (dir.isDirectory && dir.name !in keepScanIds) {
                dir.deleteRecursively()
            }
        }
    }
}
