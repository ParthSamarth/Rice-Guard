package com.riceguard.ai.ui.navigation

import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.runtime.Composable
import androidx.compose.ui.unit.IntOffset
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.riceguard.ai.di.AppContainer
import com.riceguard.ai.di.rememberViewModel
import com.riceguard.ai.domain.model.ScanSource
import com.riceguard.ai.domain.model.ScanStatus
import com.riceguard.ai.ui.screens.camera.CameraScreen
import com.riceguard.ai.ui.screens.confirmation.PhotoConfirmationScreen
import com.riceguard.ai.ui.screens.explanation.ExplanationScreen
import com.riceguard.ai.ui.screens.history.HistoryScreen
import com.riceguard.ai.ui.screens.historydetail.HistoryDetailScreen
import com.riceguard.ai.ui.screens.home.HomeScreen
import com.riceguard.ai.ui.screens.processing.ProcessingScreen
import com.riceguard.ai.ui.screens.recommendation.RecommendationScreen
import com.riceguard.ai.ui.screens.result.ResultScreen
import com.riceguard.ai.ui.screens.settings.SettingsScreen
import com.riceguard.ai.ui.screens.splash.SplashScreen
import com.riceguard.ai.ui.screens.uncertainty.LowConfidenceScreen
import com.riceguard.ai.ui.screens.uncertainty.ModelDisagreementScreen
import com.riceguard.ai.ui.screens.uncertainty.NotRecognizedScreen

@Composable
fun RiceGuardNavGraph(container: AppContainer) {
    val navController = rememberNavController()
    // Shared for the whole capture -> result -> explanation -> recommendation
    // flow (see ScanSessionViewModel's own doc comment for why this is
    // deliberately NOT per-destination).
    val scanSession = rememberViewModel { ScanSessionViewModel(container.scanRepository) }

    // One consistent transition policy for every destination (project brief
    // animation pass, "one motion language"): forward navigation slides the
    // new screen in from the right while the old one drifts slightly left
    // (a soft parallax, not a full-width slide) with a fade; back
    // navigation mirrors it. Short + tween-based on purpose -- a full-screen
    // replace reads as "the wrong place" if it springs/bounces.
    val transitionSpec = tween<IntOffset>(280, easing = FastOutSlowInEasing)
    val fadeSpecIn = tween<Float>(220, easing = FastOutSlowInEasing)
    val fadeSpecOut = tween<Float>(200, easing = FastOutSlowInEasing)

    NavHost(
        navController = navController,
        startDestination = Screen.Splash.route,
        enterTransition = { slideInHorizontally(transitionSpec) { it / 4 } + fadeIn(fadeSpecIn) },
        exitTransition = { slideOutHorizontally(transitionSpec) { -it / 6 } + fadeOut(fadeSpecOut) },
        popEnterTransition = { slideInHorizontally(transitionSpec) { -it / 6 } + fadeIn(fadeSpecIn) },
        popExitTransition = { slideOutHorizontally(transitionSpec) { it / 4 } + fadeOut(fadeSpecOut) },
    ) {

        composable(Screen.Splash.route) {
            SplashScreen(
                onFinished = {
                    navController.navigate(Screen.Home.route) {
                        popUpTo(Screen.Splash.route) { inclusive = true }
                    }
                },
            )
        }

        composable(Screen.Home.route) {
            HomeScreen(
                container = container,
                onScanClick = {
                    scanSession.startNewScan()
                    navController.navigate(Screen.Camera.route)
                },
                onImagePicked = { file ->
                    scanSession.startNewScan()
                    scanSession.onPhotoCaptured(file, ScanSource.GALLERY)
                    navController.navigate(Screen.PhotoConfirmation.route)
                },
                onHistoryClick = { navController.navigate(Screen.History.route) },
                onSettingsClick = { navController.navigate(Screen.Settings.route) },
                onOpenScan = { id -> navController.navigate(Screen.HistoryDetail.createRoute(id)) },
            )
        }

        composable(Screen.Camera.route) {
            CameraScreen(
                imageStorage = container.imageStorage,
                connectionRepository = container.connectionRepository,
                onPhotoCaptured = { file, source ->
                    scanSession.onPhotoCaptured(file, source)
                    navController.navigate(Screen.PhotoConfirmation.route)
                },
                onClose = { navController.popBackStack(Screen.Home.route, inclusive = false) },
            )
        }

        composable(Screen.PhotoConfirmation.route) {
            PhotoConfirmationScreen(
                scanSession = scanSession,
                onRetake = {
                    scanSession.retake()
                    navController.popBackStack()
                },
                onUsePhoto = {
                    navController.navigate(Screen.Processing.route)
                },
            )
        }

        composable(Screen.Processing.route) {
            ProcessingScreen(
                scanSession = scanSession,
                onDone = { status ->
                    val target = when (status) {
                        ScanStatus.LOW_CONFIDENCE -> Screen.LowConfidence.route
                        ScanStatus.MODEL_DISAGREEMENT -> Screen.ModelDisagreement.route
                        ScanStatus.NOT_RECOGNIZED -> Screen.NotRecognized.route
                        else -> Screen.Result.route
                    }
                    navController.navigate(target) { popUpTo(Screen.Camera.route) { inclusive = true } }
                },
                onRetryFailed = { navController.popBackStack(Screen.PhotoConfirmation.route, inclusive = false) },
                onGiveUp = { navController.popBackStack(Screen.Home.route, inclusive = false) },
            )
        }

        composable(Screen.Result.route) {
            ResultScreen(
                scanSession = scanSession,
                onViewExplanation = { navController.navigate(Screen.Explanation.route) },
                onViewRecommendation = { navController.navigate(Screen.Recommendation.route) },
                onDone = { navController.popBackStack(Screen.Home.route, inclusive = false) },
            )
        }

        composable(Screen.Explanation.route) {
            ExplanationScreen(
                scanSession = scanSession,
                onBack = { navController.popBackStack() },
            )
        }

        composable(Screen.Recommendation.route) {
            RecommendationScreen(
                scanSession = scanSession,
                onBack = { navController.popBackStack() },
            )
        }

        composable(Screen.LowConfidence.route) {
            LowConfidenceScreen(
                scanSession = scanSession,
                onRetake = {
                    scanSession.startNewScan()
                    navController.navigate(Screen.Camera.route) { popUpTo(Screen.Home.route) }
                },
                onDone = { navController.popBackStack(Screen.Home.route, inclusive = false) },
            )
        }

        composable(Screen.NotRecognized.route) {
            NotRecognizedScreen(
                onRetake = {
                    scanSession.startNewScan()
                    navController.navigate(Screen.Camera.route) { popUpTo(Screen.Home.route) }
                },
                onDone = { navController.popBackStack(Screen.Home.route, inclusive = false) },
            )
        }

        composable(Screen.ModelDisagreement.route) {
            ModelDisagreementScreen(
                scanSession = scanSession,
                onRetake = {
                    scanSession.startNewScan()
                    navController.navigate(Screen.Camera.route) { popUpTo(Screen.Home.route) }
                },
                onDone = { navController.popBackStack(Screen.Home.route, inclusive = false) },
            )
        }

        composable(Screen.History.route) {
            HistoryScreen(
                scanRepository = container.scanRepository,
                onOpenScan = { id -> navController.navigate(Screen.HistoryDetail.createRoute(id)) },
                onBack = { navController.popBackStack() },
                onScanClick = {
                    scanSession.startNewScan()
                    navController.navigate(Screen.Camera.route)
                },
            )
        }

        composable(Screen.HistoryDetail.route) { backStackEntry ->
            val scanId = backStackEntry.arguments?.getString(Screen.HistoryDetail.ARG_SCAN_ID).orEmpty()
            HistoryDetailScreen(
                scanId = scanId,
                scanRepository = container.scanRepository,
                onBack = { navController.popBackStack() },
                onDeleted = { navController.popBackStack() },
            )
        }

        composable(Screen.Settings.route) {
            SettingsScreen(
                container = container,
                onBack = { navController.popBackStack() },
            )
        }
    }
}
