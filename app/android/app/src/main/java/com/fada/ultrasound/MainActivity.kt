package com.fada.ultrasound

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import com.fada.ultrasound.ui.FADANavHost
import com.fada.ultrasound.ui.theme.FADATheme

/**
 * Main entry point for the FADA Android app.
 *
 * DISCLAIMER: This is a research prototype for educational purposes only.
 * NOT intended for clinical use or medical diagnosis.
 */
class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        setContent {
            FADATheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    FADANavHost()
                }
            }
        }
    }
}
