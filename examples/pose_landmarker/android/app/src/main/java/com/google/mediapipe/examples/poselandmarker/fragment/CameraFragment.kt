/*
 * Copyright 2023 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.mediapipe.examples.poselandmarker.fragment

import android.annotation.SuppressLint
import android.content.res.Configuration
import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.graphics.Point
import android.media.MediaPlayer
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.util.Base64
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.camera.core.AspectRatio
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.lifecycle.lifecycleScope
import androidx.navigation.Navigation
import app.rive.runtime.kotlin.core.Direction
import app.rive.runtime.kotlin.core.Loop
import com.google.mediapipe.examples.poselandmarker.MainViewModel
import com.google.mediapipe.examples.poselandmarker.PoseLandmarkerHelper
import com.google.mediapipe.examples.poselandmarker.R
import com.google.mediapipe.examples.poselandmarker.databinding.FragmentCameraBinding
import com.google.mediapipe.tasks.components.containers.Landmark
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import io.github.sceneview.collision.Vector3
import io.github.sceneview.math.Position
import io.github.sceneview.node.CylinderNode
import kotlinx.coroutines.*
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.math.sqrt

fun isPointInsideTriangle(point: Point, triangle: List<Point>): Boolean {
    if (triangle.size != 3) throw IllegalArgumentException("삼각형은 반드시 세 개의 꼭짓점을 가져야 합니다.")

    val (a, b, c) = triangle

    // 삼각형의 전체 면적
    val areaABC = calculateTriangleArea(a, b, c)

    // 점과 삼각형 꼭짓점들로 이루어진 세 개의 삼각형 면적
    val areaPAB = calculateTriangleArea(point, a, b)
    val areaPBC = calculateTriangleArea(point, b, c)
    val areaPCA = calculateTriangleArea(point, c, a)

    // 점이 삼각형 안에 있다면 세 면적의 합이 삼각형의 전체 면적과 같아야 함
    return abs(areaABC - (areaPAB + areaPBC + areaPCA)) < 1e-6
}

fun calculateTriangleArea(p1: Point, p2: Point, p3: Point): Float {
    return abs((p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)) / 2f)
}

fun isPointInsideTrapezoid(
    point: Point,
    trapezoid: List<Point> // 네 개의 꼭짓점 (시계 방향 또는 반시계 방향)
): Boolean {
    if (trapezoid.size != 4) throw IllegalArgumentException("사다리꼴은 반드시 네 개의 꼭짓점을 가져야 합니다.")

    val (a, b, c, d) = trapezoid

    // 사다리꼴을 두 개의 삼각형으로 나눔
    val triangle1 = listOf(a, b, c)
    val triangle2 = listOf(b, c, d)

    // 점이 두 삼각형 중 하나라도 포함되면 사다리꼴 안에 있음

//    println ("triangle1 : ${isPointInsideTriangle(point, triangle1)}, ${isPointInsideTriangle(point, triangle2)}")
    return isPointInsideTriangle(point, triangle1) || isPointInsideTriangle(point, triangle2)
}

class CameraFragment : Fragment(), PoseLandmarkerHelper.LandmarkerListener {

    companion object {
        private const val TAG = "Pose Landmarker"
    }

    private var _fragmentCameraBinding: FragmentCameraBinding? = null

    private val fragmentCameraBinding
        get() = _fragmentCameraBinding!!

    private lateinit var poseLandmarkerHelper: PoseLandmarkerHelper
    private val viewModel: MainViewModel by activityViewModels()
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var cameraFacing = CameraSelector.LENS_FACING_BACK

    /** Blocking ML operations are performed using this executor */
    private lateinit var backgroundExecutor: ExecutorService
    private var imageSendJob: Job? = null
    private var lastImageSent = false

//    private lateinit var renderer: CubeRenderer

    private var previousLandmarks: List<NormalizedLandmark>? = null
//    private lateinit var smoothedLandmarks: MutableList<NormalizedLandmark>
    private lateinit var smoothedLandmarks: MutableList<NormalizedLandmark>

    // LPF 필터 계수 (0.0 ~ 1.0, 클수록 현재 데이터에 민감)
    private val lpfAlpha = 0.7f
    private val movementThreshold = 0.35f
    private lateinit var mediaPlayer: MediaPlayer

//    private val landmarkNodes: ArrayList<CylinderNode> = ArrayList<CylinderNode>()

//    private lateinit var openGLView: OpenGLView
    private val checkLandmark = arrayOf<Int>(7, 8, 11, 13, 15, 16, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 31, 32)
    private val trapezoid = listOf(
        Point(210 + 40, 0 + 40),
        Point(430 - 40, 0 + 40),
        Point(100 + 40, 480 - 40),
        Point(540 - 40, 480 - 40)
    )

    // 저역통과 필터(LPF) 적용
    private fun applyLPF(currentLandmarks: List<NormalizedLandmark>) {
        for (i in currentLandmarks.indices) {
            val current = currentLandmarks[i]
            val smoothed = smoothedLandmarks[i]


            // x, y, z에 각각 LPF 적용
            smoothedLandmarks[i] = NormalizedLandmark.create(
                lpfAlpha * current.x() + (1 - lpfAlpha) * smoothed.x(),
                lpfAlpha * current.y() + (1 - lpfAlpha) * smoothed.y(),
                lpfAlpha * current.z() + (1 - lpfAlpha) * smoothed.z(),
                current.visibility(), // visibility는 그대로 유지,
                current.presence()
            )
        }
    }

    // 갑작스러운 움직임 감지 함수
    private fun detectSuddenMovement(filteredLandmarks: List<NormalizedLandmark>) {
        previousLandmarks?.let { prevLandmarks ->
            for (i in filteredLandmarks.indices) {
                val dx = filteredLandmarks[i].x() - prevLandmarks[i].x()
                val dy = filteredLandmarks[i].y() - prevLandmarks[i].y()
                val dz = filteredLandmarks[i].z() - prevLandmarks[i].z()

                // 변화량 계산
                val distance = sqrt((dx * dx + dy * dy + dz * dz).toDouble()).toFloat()
                if (distance > movementThreshold) {
                    // 갑작스러운 움직임 감지 시
                    activity?.runOnUiThread {
                        // Toast 메시지 출력
                        Toast.makeText(requireContext(),"위험합니다. 천천히 움직여 주세요.", Toast.LENGTH_SHORT).show()
                        if (!mediaPlayer.isPlaying) {
                            mediaPlayer.start()
                        }
                    }
                    break
                }
            }
        }
        // 현재 랜드마크를 이전 랜드마크로 업데이트
        previousLandmarks = filteredLandmarks.toList()
    }

    override fun onResume() {
        super.onResume()
        // Make sure that all permissions are still present, since the
        // user could have removed them while the app was in paused state.
        if (!PermissionsFragment.hasPermissions(requireContext())) {
            Navigation.findNavController(
                requireActivity(), R.id.fragment_container
            ).navigate(R.id.action_camera_to_permissions)
        }

        // Start the PoseLandmarkerHelper again when users come back
        // to the foreground.
        backgroundExecutor.execute {
            if(this::poseLandmarkerHelper.isInitialized) {
                if (poseLandmarkerHelper.isClose()) {
                    poseLandmarkerHelper.setupPoseLandmarker()
                }
            }
        }

        //"State Machine"
        _fragmentCameraBinding?.riveAnimationView?.play(
            "State Machine",
            Loop.AUTO,
            Direction.AUTO,
            isStateMachine = true
        )

        _fragmentCameraBinding?.riveAnimationView?.setNumberState("State Machine", "brow", 50f)
    }

    override fun onPause() {
        super.onPause()
        if(this::poseLandmarkerHelper.isInitialized) {
            viewModel.setMinPoseDetectionConfidence(poseLandmarkerHelper.minPoseDetectionConfidence)
            viewModel.setMinPoseTrackingConfidence(poseLandmarkerHelper.minPoseTrackingConfidence)
            viewModel.setMinPosePresenceConfidence(poseLandmarkerHelper.minPosePresenceConfidence)
            viewModel.setDelegate(poseLandmarkerHelper.currentDelegate)

            // Close the PoseLandmarkerHelper and release resources
            backgroundExecutor.execute { poseLandmarkerHelper.clearPoseLandmarker() }
        }
    }

    override fun onDestroyView() {
        _fragmentCameraBinding = null
        super.onDestroyView()

        // Shut down our background executor
        backgroundExecutor.shutdown()
        backgroundExecutor.awaitTermination(
            Long.MAX_VALUE, TimeUnit.NANOSECONDS
        )
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _fragmentCameraBinding =
            FragmentCameraBinding.inflate(inflater, container, false)

        mediaPlayer = MediaPlayer.create(requireContext(), R.raw.slowmove)

        val point = Point(346, 378)
        println("point : $point")
        println("trapezoid : $trapezoid")
        isPointInsideTrapezoid(point, trapezoid)

        return fragmentCameraBinding.root
    }

    @SuppressLint("MissingPermission")
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // Initialize our background executor
        backgroundExecutor = Executors.newSingleThreadExecutor()

        // Wait for the views to be properly laid out
        fragmentCameraBinding.viewFinder.post {
            // Set up the camera and its use cases
            setUpCamera()
        }

//        fragmentCameraBinding.glSurfaceView.setEGLContextClientVersion(2)
//        fragmentCameraBinding.glSurfaceView.setRenderer(renderer)
//        fragmentCameraBinding.glSurfaceView.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY // 연속 렌더링

        poseLandmarkerHelper = PoseLandmarkerHelper(
            context = requireContext(),
            runningMode = RunningMode.LIVE_STREAM,
            minPoseDetectionConfidence = viewModel.currentMinPoseDetectionConfidence,
            minPoseTrackingConfidence = viewModel.currentMinPoseTrackingConfidence,
            minPosePresenceConfidence = viewModel.currentMinPosePresenceConfidence,
            currentDelegate = viewModel.currentDelegate,
            poseLandmarkerHelperListener = this
        )

        imageSendJob = CoroutineScope(Dispatchers.IO).launch {
            while (isActive) {
                if (viewModel.bedStatus.value == true) {
                    // Convert preview image to Base64
                    val base64Image = getBitmapFromPreview()?.let { convertPreviewImageToBase64(it) }
                    if (base64Image != null) {
                        callLambdaFunction(base64Image)
                    }
                    delay(5000) // 5초마다 전송
                    lastImageSent = false
                } else if (!lastImageSent) {
                    // Convert and send once when status is false
                    val base64Image = getBitmapFromPreview()?.let { convertPreviewImageToBase64(it) }
                    if (base64Image != null) {
                        callLambdaFunction(base64Image)
                        lastImageSent = true
                    }
                }
                delay(1000) // 상태를 지속적으로 확인
            }
        }

        initBottomSheetControls()
    }

    suspend fun callLambdaFunction(base64Image: String) {
        val url = URL("https://7gb9b8se68.execute-api.ap-northeast-2.amazonaws.com/prod/upload")
        val jsonBody = JSONObject().apply {
            put("duid", "smartbed1")
            put("fileContent", base64Image)
        }

        withContext(Dispatchers.IO) {
            try {
                val connection = url.openConnection() as HttpURLConnection
                connection.requestMethod = "POST"
                connection.setRequestProperty("Content-Type", "application/json")
                connection.doOutput = true

                connection.outputStream.use { os ->
                    val input = jsonBody.toString().toByteArray()
                    os.write(input, 0, input.size)
                }

                val responseCode = connection.responseCode
                println("Response Code: $responseCode")
            } catch (e: Exception) {
                println("Error calling Lambda: $e")
            }
        }
    }

    private fun initBottomSheetControls() {

    }

    // Update the values displayed in the bottom sheet. Reset Poselandmarker
    // helper.
    private fun updateControlsUi() {
        if(this::poseLandmarkerHelper.isInitialized) {

            // Needs to be cleared instead of reinitialized because the GPU
            // delegate needs to be initialized on the thread using it when applicable
            backgroundExecutor.execute {
                poseLandmarkerHelper.clearPoseLandmarker()
                poseLandmarkerHelper.setupPoseLandmarker()
            }
            fragmentCameraBinding.overlay.clear()
        }
    }

    // Initialize CameraX, and prepare to bind the camera use cases
    private fun setUpCamera() {
        val cameraProviderFuture =
            ProcessCameraProvider.getInstance(requireContext())
        cameraProviderFuture.addListener(
            {
                // CameraProvider
                cameraProvider = cameraProviderFuture.get()

                // Build and bind the camera use cases
                bindCameraUseCases()
            }, ContextCompat.getMainExecutor(requireContext())
        )
    }

    // Declare and bind preview, capture and analysis use cases
    @SuppressLint("UnsafeOptInUsageError")
    private fun bindCameraUseCases() {

        // CameraProvider
        val cameraProvider = cameraProvider
            ?: throw IllegalStateException("Camera initialization failed.")

        val cameraSelector =
            CameraSelector.Builder().requireLensFacing(cameraFacing).build()

        // Preview. Only using the 4:3 ratio because this is the closest to our models
        preview = Preview.Builder().setTargetAspectRatio(AspectRatio.RATIO_16_9)
            .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
            .build()

        // ImageAnalysis. Using RGBA 8888 to match how our models work
        imageAnalyzer =
            ImageAnalysis.Builder().setTargetAspectRatio(AspectRatio.RATIO_16_9)
                .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                // The analyzer can then be assigned to the instance
                .also {
                    it.setAnalyzer(backgroundExecutor) { image ->
                        detectPose(image)
                    }
                }

        // Must unbind the use-cases before rebinding them
        cameraProvider.unbindAll()

        try {
            // A variable number of use-cases can be passed here -
            // camera provides access to CameraControl & CameraInfo
            camera = cameraProvider.bindToLifecycle(
                this, cameraSelector, preview, imageAnalyzer
            )

            // Attach the viewfinder's surface provider to preview use case
            preview?.setSurfaceProvider(fragmentCameraBinding.viewFinder.surfaceProvider)
        } catch (exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

    private fun detectPose(imageProxy: ImageProxy) {
        if(this::poseLandmarkerHelper.isInitialized) {
            poseLandmarkerHelper.detectLiveStream(
                imageProxy = imageProxy,
                isFrontCamera = cameraFacing == CameraSelector.LENS_FACING_FRONT
            )
        }
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        imageAnalyzer?.targetRotation =
            fragmentCameraBinding.viewFinder.display.rotation
    }

    // Update UI after pose have been detected. Extracts original
    // image height/width to scale and place the landmarks properly through
    // OverlayView
    override fun onResults(
        resultBundle: PoseLandmarkerHelper.ResultBundle
    ) {
        activity?.runOnUiThread {
            if (_fragmentCameraBinding != null) {

                if (resultBundle.results.size > 0 && resultBundle.results.first().landmarks().size > 0) {
                    if (!::smoothedLandmarks.isInitialized) {
                        // 초기화 시 현재 랜드마크를 그대로 사용
                        smoothedLandmarks = resultBundle.results.first().landmarks().first().toMutableList()
                    }
                    applyLPF(resultBundle.results.first().landmarks().first())
                    detectSuddenMovement(smoothedLandmarks)

//                    println("코 : ${resultBundle.results.first().landmarks().first()[0]} ${smoothedLandmarks[0]}")
                    fragmentCameraBinding.overlay.resultsLandmark = smoothedLandmarks
//                    println("코 X : ${smoothedLandmarks[0].x() * 640f}, 코 Y : ${smoothedLandmarks[0].y() * 480f} ")

                    // 끼임 방지
                    checkLandmark.forEach {
                        val point = Point(
                            (smoothedLandmarks[it].x() * 640f).toInt(),
                            (smoothedLandmarks[it].y() * 480f).toInt())
                        val chk = isPointInsideTrapezoid(
                            point, trapezoid)

                        if (!chk) println("$it : ${smoothedLandmarks[it]} $point")
                    }
                }

                // Pass necessary information to OverlayView for drawing on the canvas
                fragmentCameraBinding.overlay.setResults(
                    resultBundle.results.first(),
                    resultBundle.inputImageHeight,
                    resultBundle.inputImageWidth,
                    RunningMode.LIVE_STREAM
                )

//                println ("resultBundle.inputImageHeight : ${resultBundle.inputImageHeight}, resultBundle.inputImageWidth : ${resultBundle.inputImageWidth}")

                // Force a redraw
//                fragmentCameraBinding.glSurfaceView.requestRender()
//                fragmentCameraBinding.glSurfaceView.invalidate()
                fragmentCameraBinding.overlay.invalidate()
            }
        }
    }

//    private fun displayLandmarks(landmarks: List<NormalizedLandmark>) {
//        // 이전 노드 제거
//        for (node in landmarkNodes) {
//            fragmentCameraBinding.sceneView.addChildNode(node)
////            fragmentCameraBinding.sceneView.getScene().removeChild(node)
//        }
//        landmarkNodes.clear()
//
//        // 새로운 랜드마크 노드 추가
//        for (landmark in landmarks) {
//            val position = Vector3(landmark.x(), landmark.y(), landmark.z())
//
//            fragmentCameraBinding.sceneView.
//            val landmarkNode: CylinderNode =
//                CylinderNode(sceneView.getTransformationSystem())
//            landmarkNode.setParent(sceneView.getScene())
//            landmarkNode.setRenderable(landmarkRenderable)
//            landmarkNode.setLocalPosition(position)
//
//            landmarkNodes.add(landmarkNode)
//        }
//    }

    override fun onError(error: String, errorCode: Int) {
        activity?.runOnUiThread {
            Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show()
        }
    }

//    fun getBitmapFromPreview(): Bitmap? {
//        return fragmentCameraBinding.viewFinder.bitmap
//    }

    suspend fun getBitmapFromPreview(): Bitmap? = withContext(Dispatchers.Main) {
        fragmentCameraBinding.viewFinder.bitmap
    }

    fun convertPreviewImageToBase64(bitmap: Bitmap): String {
        val byteArrayOutputStream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 10, byteArrayOutputStream)
        val byteArray = byteArrayOutputStream.toByteArray()
        return Base64.encodeToString(byteArray, Base64.DEFAULT)
    }
}
