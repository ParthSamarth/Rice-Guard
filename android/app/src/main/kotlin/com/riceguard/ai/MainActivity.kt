package com.riceguard.ai

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.material3.MaterialTheme
import com.riceguard.ai.ui.navigation.RiceGuardNavGraph
import com.riceguard.ai.ui.theme.RiceGuardTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        val container = (application as RiceGuardApplication).container
        setContent {
            RiceGuardApp(container)
        }
    }
}

@Composable
private fun RiceGuardApp(container: com.riceguard.ai.di.AppContainer) {
    RiceGuardTheme {
        Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
            RiceGuardNavGraph(container = container)
        }
    }
}
