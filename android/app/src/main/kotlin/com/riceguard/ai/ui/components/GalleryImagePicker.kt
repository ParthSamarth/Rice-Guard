package com.riceguard.ai.ui.components

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.LocalContext
import com.riceguard.ai.data.local.ImageStorage
import java.io.File

/** Android's modern Photo Picker -- no READ_MEDIA_IMAGES/READ_EXTERNAL_STORAGE
 * permission needed on any supported API level, unlike an ACTION_GET_CONTENT
 * or a manual MediaStore query would require. The chosen image is copied into
 * the same private-cache staging file CameraX's own capture uses, so a
 * gallery pick and a camera capture feed the exact same confirm/upload
 * pipeline from this point on. Returns a launch function to call from a
 * button's onClick. */
@Composable
fun rememberGalleryImagePicker(imageStorage: ImageStorage, onPicked: (File) -> Unit): () -> Unit {
    val context = LocalContext.current
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        val staging = imageStorage.newCaptureStagingFile(context.cacheDir)
        context.contentResolver.openInputStream(uri)?.use { input ->
            staging.outputStream().use { output -> input.copyTo(output) }
        }
        onPicked(staging)
    }
    return { launcher.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) }
}
