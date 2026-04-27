package com.fada.ultrasound.ui

import androidx.compose.animation.AnimatedContentTransitionScope
import androidx.compose.animation.core.tween
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.fada.ultrasound.viewmodel.InferenceViewModel

/**
 * Navigation routes for the app.
 */
sealed class Screen(val route: String) {
    data object Chat : Screen("chat")
    data object Conversations : Screen("conversations")
    data object Models : Screen("models")
    data object Settings : Screen("settings")
    data object Camera : Screen("camera")
    data object Results : Screen("results")
}

/**
 * Main navigation host for the app.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun FADANavHost() {
    val navController = rememberNavController()
    val viewModel: InferenceViewModel = viewModel()
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = navBackStackEntry?.destination?.route
    val showGlobalTopBar = currentRoute != Screen.Camera.route && currentRoute != Screen.Results.route
    val showSettingsAction = currentRoute != Screen.Settings.route && currentRoute != Screen.Models.route

    Scaffold(
        topBar = {
            if (showGlobalTopBar) {
                TopAppBar(
                    title = { Text("FADA") },
                    navigationIcon = {
                        if (currentRoute != null && currentRoute != Screen.Chat.route) {
                            IconButton(onClick = { navController.popBackStack() }) {
                                Icon(
                                    imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                                    contentDescription = "Back"
                                )
                            }
                        }
                    },
                    actions = {
                        if (showSettingsAction) {
                            IconButton(
                                onClick = {
                                    navController.navigate(Screen.Settings.route) {
                                        launchSingleTop = true
                                    }
                                }
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Settings,
                                    contentDescription = "Settings"
                                )
                            }
                        }
                    }
                )
            }
        }
    ) { paddingValues ->
        NavHost(
            navController = navController,
            startDestination = Screen.Chat.route,
            modifier = Modifier.padding(paddingValues),
            enterTransition = {
                slideIntoContainer(
                    AnimatedContentTransitionScope.SlideDirection.Left,
                    animationSpec = tween(220)
                )
            },
            exitTransition = {
                slideOutOfContainer(
                    AnimatedContentTransitionScope.SlideDirection.Left,
                    animationSpec = tween(220)
                )
            },
            popEnterTransition = {
                slideIntoContainer(
                    AnimatedContentTransitionScope.SlideDirection.Right,
                    animationSpec = tween(220)
                )
            },
            popExitTransition = {
                slideOutOfContainer(
                    AnimatedContentTransitionScope.SlideDirection.Right,
                    animationSpec = tween(220)
                )
            }
        ) {
            composable(Screen.Chat.route) {
                ChatScreen(
                    viewModel = viewModel,
                    onNavigateToCamera = {
                        navController.navigate(Screen.Camera.route)
                    },
                    onNavigateToConversations = {
                        navController.navigate(Screen.Conversations.route) {
                            launchSingleTop = true
                        }
                    }
                )
            }

            composable(Screen.Conversations.route) {
                ConversationsScreen(
                    viewModel = viewModel,
                    onOpenConversation = {
                        navController.popBackStack(Screen.Chat.route, inclusive = false)
                    }
                )
            }

            composable(Screen.Models.route) {
                ModelsScreen(viewModel = viewModel)
            }

            composable(Screen.Settings.route) {
                SettingsScreen(
                    viewModel = viewModel,
                    onNavigateToModels = {
                        navController.navigate(Screen.Models.route) {
                            launchSingleTop = true
                        }
                    }
                )
            }

            composable(Screen.Camera.route) {
                CameraScreen(
                    onImageCaptured = { bitmap ->
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
}
