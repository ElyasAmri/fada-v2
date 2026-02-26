package com.fada.ultrasound.ui

import android.graphics.Bitmap
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.fada.ultrasound.viewmodel.InferenceViewModel

/**
 * Navigation routes for the app.
 */
sealed class Screen(val route: String) {
    data object Main : Screen("main")
    data object Camera : Screen("camera")
    data object Results : Screen("results")
}

/**
 * Main navigation host for the app.
 */
@Composable
fun FADANavHost() {
    val navController = rememberNavController()
    val viewModel: InferenceViewModel = viewModel()

    // Shared captured image state
    var capturedBitmap: Bitmap? by remember { mutableStateOf(null) }

    NavHost(
        navController = navController,
        startDestination = Screen.Main.route
    ) {
        composable(Screen.Main.route) {
            MainScreen(
                viewModel = viewModel,
                onNavigateToCamera = {
                    navController.navigate(Screen.Camera.route)
                },
                onNavigateToResults = {
                    navController.navigate(Screen.Results.route)
                }
            )
        }

        composable(Screen.Camera.route) {
            CameraScreen(
                onImageCaptured = { bitmap ->
                    capturedBitmap = bitmap
                    viewModel.setCapturedImage(bitmap)
                    navController.popBackStack()
                },
                onNavigateBack = {
                    navController.popBackStack()
                }
            )
        }

        composable(Screen.Results.route) {
            ResultsScreen(
                viewModel = viewModel,
                onNavigateBack = {
                    navController.popBackStack()
                },
                onNewImage = {
                    viewModel.clearSelection()
                    navController.popBackStack()
                }
            )
        }
    }
}
