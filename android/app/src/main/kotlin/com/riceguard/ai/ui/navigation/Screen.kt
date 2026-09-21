package com.riceguard.ai.ui.navigation

/** All 13 required screens (project brief section 12). Result/Explanation/
 * Recommendation/LowConfidence/ModelDisagreement all read from the shared
 * ScanSessionViewModel (see RiceGuardNavGraph) rather than taking nav args --
 * there is exactly one "current scan" in flight at a time by construction of
 * the flow, so a graph-scoped ViewModel is simpler and safer than
 * serializing a whole ScanResult through a route string. Only HistoryDetail
 * navigates to an arbitrary (not-current) scan, so it alone takes an id arg.
 */
sealed class Screen(val route: String) {
    data object Splash : Screen("splash")
    data object Home : Screen("home")
    data object Camera : Screen("camera")
    data object PhotoConfirmation : Screen("photo_confirmation")
    data object Processing : Screen("processing")
    data object Result : Screen("result")
    data object Explanation : Screen("explanation")
    data object Recommendation : Screen("recommendation")
    data object LowConfidence : Screen("low_confidence")
    data object ModelDisagreement : Screen("model_disagreement")
    data object NotRecognized : Screen("not_recognized")
    data object History : Screen("history")
    data object Settings : Screen("settings")

    data object HistoryDetail : Screen("history_detail/{scanId}") {
        fun createRoute(scanId: String) = "history_detail/$scanId"
        const val ARG_SCAN_ID = "scanId"
    }
}
