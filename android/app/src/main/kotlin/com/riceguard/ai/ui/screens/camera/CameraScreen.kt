package com.riceguard.ai.ui.screens.camera

import android.Manifest
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.WindowInsetsSides
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.only
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.FlashOff
import androidx.compose.material.icons.filled.FlashOn
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.PermissionStatus
import com.google.accompanist.permissions.rememberPermissionState
import com.riceguard.ai.data.local.ImageStorage
import com.riceguard.ai.data.repository.ConnectionRepository
import com.riceguard.ai.data.repository.ConnectionState
import com.riceguard.ai.domain.model.ScanSource
import com.riceguard.ai.ui.components.ConnectionStatusChip
import com.riceguard.ai.ui.components.PrimaryButton
import com.riceguard.ai.ui.components.RiceGuardHapticStyle
import com.riceguard.ai.ui.components.rememberGalleryImagePicker
import com.riceguard.ai.ui.components.rememberHapticPerformer
import kotlinx.coroutines.launch
import java.io.File

@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun CameraScreen(
    imageStorage: ImageStorage,
    connectionRepository: ConnectionRepository,
    onPhotoCaptured: (File, ScanSource) -> Unit,
    onClose: () -> Unit,
) {
    val cameraPermission = rememberPermissionState(Manifest.permission.CAMERA)

    LaunchedEffect(Unit) {
        if (cameraPermission.status != PermissionStatus.Granted) {
            cameraPermission.launchPermissionRequest()
        }
    }

    Box(modifier = Modifier.fillMaxSize().background(Color.Black)) {
        when (cameraPermission.status) {
            is PermissionStatus.Granted -> CameraContent(
                imageStorage = imageStorage,
                connectionRepository = connectionRepository,
                onPhotoCaptured = { file -> onPhotoCaptured(file, ScanSource.CAMERA) },
                onImagePicked = { file -> onPhotoCaptured(file, ScanSource.GALLERY) },
                onClose = onClose,
            )
            is PermissionStatus.Denied -> CameraPermissionDenied(onClose)
        }
    }
}

@Composable
private fun CameraPermissionDenied(onClose: () -> Unit) {
    Column(
        modifier = Modifier.fillMaxSize().padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text("Unable to open camera.", style = MaterialTheme.typography.titleLarge, color = Color.White)
        Text(
            "RiceGuard needs camera access to scan a rice leaf. Enable it in your device's app settings.",
            style = MaterialTheme.typography.bodyMedium,
            color = Color.White.copy(alpha = 0.8f),
            modifier = Modifier.padding(top = 8.dp),
        )
        PrimaryButton(text = "Close", onClick = onClose, modifier = Modifier.fillMaxWidth().padding(top = 24.dp))
    }
}

@Composable
private fun CameraContent(
    imageStorage: ImageStorage,
    connectionRepository: ConnectionRepository,
    onPhotoCaptured: (File) -> Unit,
    onImagePicked: (File) -> Unit,
    onClose: () -> Unit,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val scope = rememberCoroutineScope()
    val performHaptic = rememberHapticPerformer()
    val pickImage = rememberGalleryImagePicker(imageStorage, onPicked = onImagePicked)

    val previewView = remember { PreviewView(context) }
    val imageCapture = remember { ImageCapture.Builder().build() }
    var isCapturing by remember { mutableStateOf(false) }
    var captureError by remember { mutableStateOf<String?>(null) }
    val connectionState by connectionRepository.state.collectAsState()

    var camera by remember { mutableStateOf<Camera?>(null) }
    var zoomRatio by remember { mutableFloatStateOf(1f) }
    var minZoomRatio by remember { mutableFloatStateOf(1f) }
    var maxZoomRatio by remember { mutableFloatStateOf(1f) }
    var torchOn by remember { mutableStateOf(false) }
    var hasFlash by remember { mutableStateOf(false) }

    DisposableEffect(lifecycleOwner) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
            try {
                cameraProvider.unbindAll()
                val boundCamera = cameraProvider.bindToLifecycle(
                    lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture,
                )
                camera = boundCamera
                hasFlash = boundCamera.cameraInfo.hasFlashUnit()
                val initialZoomState = boundCamera.cameraInfo.zoomState.value
                minZoomRatio = initialZoomState?.minZoomRatio ?: 1f
                maxZoomRatio = initialZoomState?.maxZoomRatio ?: 1f
                zoomRatio = initialZoomState?.zoomRatio ?: 1f
            } catch (_: Exception) {
                captureError = "Unable to open camera."
            }
        }, ContextCompat.getMainExecutor(context))

        onDispose {
            ProcessCameraProvider.getInstance(context).get().unbindAll()
        }
    }

    AndroidView(
        factory = { previewView },
        modifier = Modifier
            .fillMaxSize()
            .pointerInput(Unit) {
                detectTransformGestures { _, _, zoomChange, _ ->
                    val cam = camera ?: return@detectTransformGestures
                    zoomRatio = (zoomRatio * zoomChange).coerceIn(minZoomRatio, maxZoomRatio)
                    cam.cameraControl.setZoomRatio(zoomRatio)
                }
            },
    )

    // Framing guide -- subtle, not cluttered (project brief section 10): a
    // single translucent rounded outline, no extra icons/labels inside it.
    Box(modifier = Modifier.fillMaxSize().padding(48.dp), contentAlignment = Alignment.Center) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(280.dp)
                .clip(RoundedCornerShape(24.dp))
                .border(2.dp, Color.White.copy(alpha = 0.7f), RoundedCornerShape(24.dp)),
        )
    }

    Column(modifier = Modifier.fillMaxSize()) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .windowInsetsPadding(WindowInsets.safeDrawing.only(WindowInsetsSides.Top + WindowInsetsSides.Horizontal))
                .padding(16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                IconButton(onClick = onClose) {
                    Icon(Icons.Filled.Close, contentDescription = "Close", tint = Color.White)
                }
                IconButton(
                    onClick = {
                        val cam = camera ?: return@IconButton
                        performHaptic(RiceGuardHapticStyle.LIGHT)
                        val newTorchOn = !torchOn
                        cam.cameraControl.enableTorch(newTorchOn)
                        torchOn = newTorchOn
                    },
                    enabled = hasFlash,
                ) {
                    Icon(
                        imageVector = if (torchOn) Icons.Filled.FlashOn else Icons.Filled.FlashOff,
                        contentDescription = if (torchOn) "Turn off flashlight" else "Turn on flashlight",
                        tint = if (hasFlash) Color.White else Color.White.copy(alpha = 0.35f),
                    )
                }
                IconButton(onClick = pickImage) {
                    Icon(Icons.Filled.PhotoLibrary, contentDescription = "Upload from gallery", tint = Color.White)
                }
            }
            ConnectionStatusChip(state = connectionState)
        }

        Spacer(modifier = Modifier.weight(1f))

        Column(
            modifier = Modifier
                .fillMaxWidth()
                .windowInsetsPadding(WindowInsets.safeDrawing.only(WindowInsetsSides.Bottom))
                .padding(bottom = 32.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(
                "%.1fx".format(zoomRatio),
                style = MaterialTheme.typography.labelMedium,
                color = Color.White,
                modifier = Modifier
                    .clip(RoundedCornerShape(100))
                    .background(Color.Black.copy(alpha = 0.45f))
                    .padding(horizontal = 16.dp, vertical = 8.dp),
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                "Keep the leaf visible • Good lighting • Avoid blur • Hold steady",
                style = MaterialTheme.typography.labelMedium,
                color = Color.White,
                modifier = Modifier
                    .clip(RoundedCornerShape(100))
                    .background(Color.Black.copy(alpha = 0.45f))
                    .padding(horizontal = 16.dp, vertical = 8.dp),
            )
            Spacer(modifier = Modifier.height(20.dp))
            captureError?.let {
                Text(it, color = MaterialTheme.colorScheme.errorContainer, style = MaterialTheme.typography.bodySmall)
                Spacer(modifier = Modifier.height(8.dp))
            }
            ShutterButton(
                isCapturing = isCapturing,
                onClick = {
                    if (isCapturing) return@ShutterButton
                    performHaptic(RiceGuardHapticStyle.LIGHT)
                    isCapturing = true
                    captureError = null
                    val outFile = imageStorage.newCaptureStagingFile(context.cacheDir)
                    val outputOptions = ImageCapture.OutputFileOptions.Builder(outFile).build()
                    imageCapture.takePicture(
                        outputOptions,
                        ContextCompat.getMainExecutor(context),
                        object : ImageCapture.OnImageSavedCallback {
                            override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                                isCapturing = false
                                onPhotoCaptured(outFile)
                            }

                            override fun onError(exc: ImageCaptureException) {
                                isCapturing = false
                                captureError = "This image could not be read."
                            }
                        },
                    )
                },
            )
        }
    }
}

@Composable
private fun ShutterButton(isCapturing: Boolean, onClick: () -> Unit) {
    val interactionSource = remember { MutableInteractionSource() }
    val pressed by interactionSource.collectIsPressedAsState()
    val scale by animateFloatAsState(
        targetValue = if (pressed) 0.94f else 1f,
        animationSpec = spring(dampingRatio = Spring.DampingRatioNoBouncy, stiffness = Spring.StiffnessMedium),
        label = "shutterPressScale",
    )

    Box(
        modifier = Modifier
            .size(76.dp)
            .clip(CircleShape)
            .background(Color.White.copy(alpha = 0.25f)),
        contentAlignment = Alignment.Center,
    ) {
        if (isCapturing) {
            CircularProgressIndicator(color = Color.White, modifier = Modifier.size(36.dp), strokeWidth = 3.dp)
        } else {
            IconButton(
                onClick = onClick,
                interactionSource = interactionSource,
                modifier = Modifier.size(64.dp).scale(scale).clip(CircleShape).background(Color.White),
            ) {
                Icon(Icons.Filled.CameraAlt, contentDescription = "Capture", tint = Color.Black)
            }
        }
    }
}
