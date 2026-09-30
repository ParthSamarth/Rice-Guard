package com.riceguard.ai.ui.components

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable

enum class RiceGuardTab { HOME, SCAN, HISTORY, SETTINGS }

/** project brief section 12, screen 2's suggested nav: Home / Scan / History
 * / Settings. "Scan" is a one-shot action (opens the camera flow), not a
 * screen you stay on -- so it is never the "selected" tab, it just launches
 * the flow when tapped, same as a prominent shortcut placed in the bar. */
@Composable
fun RiceGuardBottomNav(
    current: RiceGuardTab,
    onHome: () -> Unit,
    onScan: () -> Unit,
    onHistory: () -> Unit,
    onSettings: () -> Unit,
) {
    NavigationBar {
        NavigationBarItem(
            selected = current == RiceGuardTab.HOME,
            onClick = onHome,
            icon = { Icon(Icons.Filled.Home, contentDescription = "Home") },
            label = { Text("Home") },
        )
        NavigationBarItem(
            selected = false,
            onClick = onScan,
            icon = { Icon(Icons.Filled.CameraAlt, contentDescription = "Scan") },
            label = { Text("Scan") },
        )
        NavigationBarItem(
            selected = current == RiceGuardTab.HISTORY,
            onClick = onHistory,
            icon = { Icon(Icons.Filled.History, contentDescription = "History") },
            label = { Text("History") },
        )
        NavigationBarItem(
            selected = current == RiceGuardTab.SETTINGS,
            onClick = onSettings,
            icon = { Icon(Icons.Filled.Settings, contentDescription = "Settings") },
            label = { Text("Settings") },
        )
    }
}
