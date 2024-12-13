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
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Base64
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.annotation.RequiresApi
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
import com.google.mlkit.vision.pose.PoseLandmark
import com.shashank.sony.fancytoastlib.FancyToast
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
    private val landmarks = arrayOf(
        "brow", "chin", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
        "sternum", "rib", "left_knee", "right_knee", "left_ilium", "right_ilium",
        "left_elbow", "right_elbow", "left_calf", "right_calf", "left_anklebone",
        "right_anklebone", "left_in_anklebone", "right_in_anklebone", "occipital",
        "left_scapula", "right_scapula", "left_ancon", "right_ancon", "sacrum",
        "left_heel", "right_heel", "left_hip", "right_hip"
    )
    val damageMap = mutableMapOf<String, Int>()

//    fun updateDamage(zValues: FloatArray, damageMap: MutableMap<String, Int>) {
//        for ((index, z) in zValues.withIndex()) {
//            val landmark = landmarks[index]
//            if (z >= 0) {
//                // z값이 양수면 데미지 누적
//                damageMap[landmark] = (damageMap[landmark]?.plus(10) ?: 10).coerceAtMost(255)
//            } else {
//                // z값이 음수면 데미지 차감 (최소 0 이하로 내려가지 않게)
//                damageMap[landmark] = (damageMap[landmark]?.minus(5) ?: 0).coerceAtLeast(0)
//            }
//        }
//    }

    private val checkLandmark = arrayOf<Int>(7, 8, 11, 13, 15, 16, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 31, 32)
    private val trapezoid = listOf(
        Point(210 + 40, 0 + 40),
        Point(430 - 40, 0 + 40),
        Point(100 + 40, 480 - 40),
        Point(540 - 40, 480 - 40)
    )

    private val handler = Handler(Looper.getMainLooper())
    private var lastResultTime = System.currentTimeMillis()
    private val damageReductionInterval = 5 * 1000L
    private val stepup = 2;
    private val stepdown = 1;

    private fun reduceDamageOverTime() {
        handler.postDelayed(object : Runnable {
            override fun run() {
                println("reduceDamageOverTime run")
                val currentTime = System.currentTimeMillis()
                if (currentTime - lastResultTime >= damageReductionInterval) {
                    // 데미지 감소 로직
                    for (key in damageMap.keys) {
                        damageMap[key] = (damageMap[key]?.minus(200) ?: 0).coerceAtLeast(0)
                        val value = damageMap[key] ?: 0
                        _fragmentCameraBinding?.riveAnimationView?.setNumberState("State Machine", key, value.toFloat() / 10f)
                    }
                    println("데미지 감소 적용: $damageMap")
                }
                handler.postDelayed(this, damageReductionInterval)
            }
        }, damageReductionInterval)
    }

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

//                        Toast.makeText(requireContext(),"위험합니다. 천천히 움직여 주세요.", Toast.LENGTH_SHORT).show()
                        FancyToast.makeText(
                            requireContext(),
                            "위험합니다. 천천히 움직여 주세요.",
                            FancyToast.LENGTH_SHORT,
                            FancyToast.WARNING,
                            false).show()

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

//        _fragmentCameraBinding?.riveAnimationView?.setNumberState("State Machine", "brow", 50f)
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
        landmarks.forEach { damageMap[it] = 0 }
        reduceDamageOverTime()

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

    fun normalizeZ(z: Float, minZ: Float, maxZ: Float): Float {
        return ((maxZ - z) / (maxZ - minZ)) * 100f
    }

    // Update UI after pose have been detected. Extracts original
    // image height/width to scale and place the landmarks properly through
    // OverlayView
    override fun onResults(
        resultBundle: PoseLandmarkerHelper.ResultBundle
    ) {
        if (_fragmentCameraBinding != null) {
            if (resultBundle.results.size > 0 && resultBundle.results.first().landmarks().size > 0) {
                lastResultTime = System.currentTimeMillis()

                if (!::smoothedLandmarks.isInitialized) {
                    // 초기화 시 현재 랜드마크를 그대로 사용
                    smoothedLandmarks = resultBundle.results.first().landmarks().first().toMutableList()
                }
                applyLPF(resultBundle.results.first().landmarks().first())
                detectSuddenMovement(smoothedLandmarks)

                fragmentCameraBinding.overlay.resultsLandmark = smoothedLandmarks

                // 끼임 방지
//                checkLandmark.forEach {
//                    val point = Point(
//                        (smoothedLandmarks[it].x() * 640f).toInt(),
//                        (smoothedLandmarks[it].y() * 480f).toInt())
//                    val chk = isPointInsideTrapezoid(
//                        point, trapezoid)
//
//                    if (!chk) println("끼임 : $it : ${smoothedLandmarks[it]} $point")
//                }

                val marks = resultBundle.results.first().landmarks().first().toMutableList()
                val nose = marks[0]
                val leftHip = marks[23]
                val rightHip = marks[24]
                
                val baseline = (leftHip.y() + rightHip.y()) / 2f
                if (nose.y() > baseline) {
                    println("정상")
                } else {
                    println("역방향")
                }
                

                // 눈썹
                val leftEyebrow = marks[1]
                val rightEyebrow = marks[4]
                val foreheadZ = (leftEyebrow.z() + rightEyebrow.z()) / 2
                if (foreheadZ >= 0) {
                    damageMap["brow"] = (damageMap["brow"]?.plus(stepup) ?: stepup).coerceAtMost(1000)
                } else {
                    damageMap["brow"] = (damageMap["brow"]?.minus(stepdown) ?: 0).coerceAtLeast(0)
                }

                val brow = damageMap["brow"]?.toFloat()  ?: 0f
                _fragmentCameraBinding?.riveAnimationView?.setNumberState("State Machine", "brow", brow / 10f)

                // 턱
                val leftEar = marks[7]
                val rightEar = marks[8]
                val chinZ = (nose.z() + leftEar.z() + rightEar.z()) / 3
                if (chinZ >= 0) {
                    damageMap["chin"] = (damageMap["chin"]?.plus(stepup) ?: stepup).coerceAtMost(1000)
                } else {
                    damageMap["chin"] = (damageMap["chin"]?.minus(stepdown) ?: 0).coerceAtLeast(0)
                }

                val chin = damageMap["chin"]?.toFloat()  ?: 0f
                _fragmentCameraBinding?.riveAnimationView?.setNumberState("State Machine", "chin", chin / 10f)

                // 왼쪽 귀
                if (leftEar.z() >= 0) {
                    damageMap["left_ear"] = (damageMap["left_ear"]?.plus(stepup) ?: stepup).coerceAtMost(1000)
                } else {
                    damageMap["left_ear"] = (damageMap["left_ear"]?.minus(stepdown) ?: 0).coerceAtLeast(0)
                }
                val left_ear = damageMap["left_ear"]?.toFloat()  ?: 0f
                _fragmentCameraBinding?.riveAnimationView?.setNumberState("State Machine", "left_ear", left_ear / 10f)

                // 오른쪽 귀
                if (rightEar.z() >= 0) {
                    damageMap["right_ear"] = (damageMap["right_ear"]?.plus(stepup) ?: stepup).coerceAtMost(1000)
                } else {
                    damageMap["right_ear"] = (damageMap["right_ear"]?.minus(stepdown) ?: 0).coerceAtLeast(0)
                }
                val right_ear = damageMap["right_ear"]?.toFloat()  ?: 0f
                _fragmentCameraBinding?.riveAnimationView?.setNumberState("State Machine", "right_ear", right_ear / 10f)

//                11 - left shoulder
//                12 - right shoulder

                // 왼쪽 어깨
                val leftshoulder = marks[11]
                if (leftshoulder.z() >= 0) {
                    damageMap["left_shoulder"] = (damageMap["left_shoulder"]?.plus(stepup) ?: stepup).coerceAtMost(1000)
                } else {
                    damageMap["left_shoulder"] = (damageMap["left_shoulder"]?.minus(stepdown) ?: 0).coerceAtLeast(0)
                }

                val left_shoulder = damageMap["left_shoulder"]?.toFloat()  ?: 0f
                _fragmentCameraBinding?.riveAnimationView?.setNumberState("State Machine", "left_shoulder", left_shoulder / 10f)

                // 오른쪽 어깨
                val rightshoulder = marks[12]
                if (rightshoulder.z() >= 0) {
                    damageMap["right_shoulder"] = (damageMap["right_shoulder"]?.plus(stepup) ?: stepup).coerceAtMost(1000)
                } else {
                    damageMap["right_shoulder"] = (damageMap["right_shoulder"]?.minus(stepdown) ?: 0).coerceAtLeast(0)
                }

                val right_shoulder = damageMap["right_shoulder"]?.toFloat()  ?: 0f
                _fragmentCameraBinding?.riveAnimationView?.setNumberState("State Machine", "right_shoulder", right_shoulder / 10f)

                // 흉골

                val shoulderCenterZ = (leftshoulder.z() + rightshoulder.z()) / 2
                val hipCenterZ = (leftHip.z() + rightHip.z()) / 2
                val sternumZ = (shoulderCenterZ + hipCenterZ) / 2
                if (sternumZ >= 0) {
                    damageMap["sternum"] = (damageMap["sternum"]?.plus(stepup) ?: stepup).coerceAtMost(1000)
                } else {
                    damageMap["sternum"] = (damageMap["sternum"]?.minus(stepdown) ?: 0).coerceAtLeast(0)
                }
                val sternum = damageMap["sternum"]?.toFloat()  ?: 0f
                _fragmentCameraBinding?.riveAnimationView?.setNumberState("State Machine", "sternum", sternum / 10f)

                // 갈비뼈
                val ribZ = shoulderCenterZ + (hipCenterZ - shoulderCenterZ) * 0.33f
                if (ribZ >= 0) {
                    damageMap["rib"] = (damageMap["rib"]?.plus(stepup) ?: stepup).coerceAtMost(1000)
                } else {
                    damageMap["rib"] = (damageMap["rib"]?.minus(stepdown) ?: 0).coerceAtLeast(0)
                }
                val rib = damageMap["rib"]?.toFloat()  ?: 0f
                _fragmentCameraBinding?.riveAnimationView?.setNumberState("State Machine", "rib", rib / 10f)

                // 왼쪽 무릎
                val leftknee = marks[25]
                if (leftknee.z() >= 0) {
                    damageMap["left_knee"] = (damageMap["left_knee"]?.plus(stepup) ?: stepup).coerceAtMost(1000)
                } else {
                    damageMap["left_knee"] = (damageMap["left_knee"]?.minus(stepdown) ?: 0).coerceAtLeast(0)
                }
                val left_knee = damageMap["left_knee"]?.toFloat()  ?: 0f
                _fragmentCameraBinding?.riveAnimationView?.setNumberState("State Machine", "left_knee", left_knee / 10f)

                // 오른쪽 무릎
                val rightknee = marks[26]
                if (rightknee.z() >= 0) {
                    damageMap["right_knee"] = (damageMap["right_knee"]?.plus(stepup) ?: stepup).coerceAtMost(1000)
                } else {
                    damageMap["right_knee"] = (damageMap["right_knee"]?.minus(stepdown) ?: 0).coerceAtLeast(0)
                }
                val right_knee = damageMap["right_knee"]?.toFloat()  ?: 0f
                _fragmentCameraBinding?.riveAnimationView?.setNumberState("State Machine", "right_knee", right_knee / 10f)

                // 왼쪽 장골
                if (leftHip.z() >= 0) {
                    damageMap["left_ilium"] = (damageMap["left_ilium"]?.plus(stepup) ?: stepup).coerceAtMost(1000)
                } else {
                    damageMap["left_ilium"] = (damageMap["left_ilium"]?.minus(stepdown) ?: 0).coerceAtLeast(0)
                }
                val left_ilium = damageMap["left_ilium"]?.toFloat()  ?: 0f
                _fragmentCameraBinding?.riveAnimationView?.setNumberState("State Machine", "left_ilium", left_ilium / 10f)

                // 오른쪽 장골
                if (rightHip.z() >= 0) {
                    damageMap["right_ilium"] = (damageMap["right_ilium"]?.plus(stepup) ?: stepup).coerceAtMost(1000)
                } else {
                    damageMap["right_ilium"] = (damageMap["right_ilium"]?.minus(stepdown) ?: 0).coerceAtLeast(0)
                }
                val right_ilium = damageMap["right_ilium"]?.toFloat()  ?: 0f
                _fragmentCameraBinding?.riveAnimationView?.setNumberState("State Machine", "right_ilium", right_ilium / 10f)

                val leftelbow = marks[13]
                val rightelbow = marks[14]

                // 왼쪽 팔꿈치
                if (leftelbow.z() >= 0 && rightelbow.z() <= 0) {
                    damageMap["left_elbow"] = (damageMap["left_elbow"]?.plus(stepup) ?: stepup).coerceAtMost(1000)
                } else {
                    damageMap["left_elbow"] = (damageMap["left_elbow"]?.plus(stepup) ?: stepup).coerceAtMost(1000)
                }
                val left_elbow = damageMap["left_elbow"]?.toFloat()  ?: 0f
                _fragmentCameraBinding?.riveAnimationView?.setNumberState("State Machine", "left_elbow", left_elbow / 10f)

                // 오른쪽 팔꿈치
                if (rightelbow.z() >= 0 && leftelbow.z() <= 0) {
                    damageMap["right_elbow"] = (damageMap["right_elbow"]?.plus(stepup) ?: stepup).coerceAtMost(1000)
                } else {
                    damageMap["right_elbow"] = (damageMap["right_elbow"]?.plus(stepup) ?: stepup).coerceAtMost(1000)
                }
                val right_elbow = damageMap["right_elbow"]?.toFloat()  ?: 0f
                _fragmentCameraBinding?.riveAnimationView?.setNumberState("State Machine", "right_elbow", right_elbow / 10f)

                // 왼쪽 종아리
                // 왼쪽 바깥쪽 발목뼈
                if (leftknee.z() >= 0 && rightknee.z() <= 0) {
                    damageMap["left_calf"] = (damageMap["left_calf"]?.plus(stepup) ?: stepup).coerceAtMost(1000)
                    damageMap["left_anklebone"] = (damageMap["left_anklebone"]?.plus(stepup) ?: stepup).coerceAtMost(1000)
                } else {
                    damageMap["left_calf"] = (damageMap["left_calf"]?.plus(stepup) ?: stepup).coerceAtMost(1000)
                    damageMap["left_anklebone"] = (damageMap["left_anklebone"]?.plus(stepup) ?: stepup).coerceAtMost(1000)
                }
                val left_calf = damageMap["left_calf"]?.toFloat()  ?: 0f
                val left_anklebone = damageMap["left_anklebone"]?.toFloat()  ?: 0f
                _fragmentCameraBinding?.riveAnimationView?.setNumberState("State Machine", "left_calf", left_calf / 10f)
                _fragmentCameraBinding?.riveAnimationView?.setNumberState("State Machine", "left_anklebone", left_anklebone / 10f)

                // 오른쪽 종아리
                // 오른쪽 바깥쪽 발목뼈
                if (rightknee.z() >= 0 && leftknee.z() <= 0) {
                    damageMap["right_calf"] = (damageMap["right_calf"]?.plus(stepup) ?: stepup).coerceAtMost(1000)
                    damageMap["right_anklebone"] = (damageMap["right_anklebone"]?.plus(stepup) ?: stepup).coerceAtMost(1000)
                } else {
                    damageMap["right_calf"] = (damageMap["right_calf"]?.plus(stepup) ?: stepup).coerceAtMost(1000)
                    damageMap["right_anklebone"] = (damageMap["right_anklebone"]?.plus(stepup) ?: stepup).coerceAtMost(1000)
                }
                val right_calf = damageMap["right_calf"]?.toFloat()  ?: 0f
                val right_anklebone = damageMap["right_anklebone"]?.toFloat()  ?: 0f
                _fragmentCameraBinding?.riveAnimationView?.setNumberState("State Machine", "right_calf", right_calf / 10f)
                _fragmentCameraBinding?.riveAnimationView?.setNumberState("State Machine", "right_anklebone", right_anklebone / 10f)


                val leftankle = marks[27]
                val rightankle = marks[28]
                //왼쪽 안쪽 발목뼈
                if (leftankle.z() >= 0 && leftHip.z() <= 0) {
                    damageMap["left_in_anklebone"] = (damageMap["left_in_anklebone"]?.plus(stepup) ?: stepup).coerceAtMost(1000)
                } else {
                    damageMap["left_in_anklebone"] = (damageMap["left_in_anklebone"]?.minus(stepdown) ?: 0).coerceAtLeast(0)
                }
                val left_in_anklebone = damageMap["left_in_anklebone"]?.toFloat()  ?: 0f
                _fragmentCameraBinding?.riveAnimationView?.setNumberState("State Machine", "left_in_anklebone", left_in_anklebone / 10f)

                //오른쪽 안쪽 발목뼈
                if (rightankle.z() >= 0 && rightHip.z() <= 0) {
                    damageMap["right_in_anklebone"] = (damageMap["right_in_anklebone"]?.plus(stepup) ?: stepup).coerceAtMost(1000)
                } else {
                    damageMap["right_in_anklebone"] = (damageMap["right_in_anklebone"]?.minus(stepdown) ?: 0).coerceAtLeast(0)
                }
                val right_in_anklebone = damageMap["right_in_anklebone"]?.toFloat()  ?: 0f
                _fragmentCameraBinding?.riveAnimationView?.setNumberState("State Machine", "right_in_anklebone", right_in_anklebone / 10f)

                // 후두골
                val earCenterZ = (leftEar.z() + rightEar.z()) / 2
                val occipitalZ = earCenterZ - (nose.z() - earCenterZ)
                if (occipitalZ >= 0) {
                    damageMap["occipital"] = (damageMap["occipital"]?.plus(stepup) ?: stepup).coerceAtMost(1000)
                } else {
                    damageMap["occipital"] = (damageMap["occipital"]?.minus(stepdown) ?: 0).coerceAtLeast(0)
                }
                val occipital = damageMap["occipital"]?.toFloat()  ?: 0f
                _fragmentCameraBinding?.riveAnimationView?.setNumberState("State Machine", "occipital", occipital / 10f)

                //왼쪽 견갑골
                if (leftshoulder.z() >= 0 && rightshoulder.z() >= 0) {
                    damageMap["left_scapula"] = (damageMap["left_scapula"]?.plus(stepup) ?: stepup).coerceAtMost(1000)
                } else {
                    damageMap["left_scapula"] = (damageMap["left_scapula"]?.minus(stepdown) ?: 0).coerceAtLeast(0)
                }
                val left_scapula = damageMap["left_scapula"]?.toFloat()  ?: 0f
                _fragmentCameraBinding?.riveAnimationView?.setNumberState("State Machine", "left_scapula", left_scapula / 10f)

                // 오른쪽 견갑골
                if (rightshoulder.z() >= 0 && leftshoulder.z() >= 0) {
                    damageMap["right_scapula"] = (damageMap["right_scapula"]?.plus(stepup) ?: stepup).coerceAtMost(1000)
                } else {
                    damageMap["right_scapula"] = (damageMap["right_scapula"]?.minus(stepdown) ?: 0).coerceAtLeast(0)
                }
                val right_scapula = damageMap["right_scapula"]?.toFloat()  ?: 0f
                _fragmentCameraBinding?.riveAnimationView?.setNumberState("State Machine", "right_scapula", right_scapula / 10f)

                // 왼쪽 앙콘
                // 오른쪽 앙콘
                if (leftelbow.z() >= 0 && rightelbow.z() >= 0) {
                    damageMap["left_ancon"] = (damageMap["left_ancon"]?.plus(stepup) ?: stepup).coerceAtMost(1000)
                    damageMap["right_ancon"] = (damageMap["right_ancon"]?.plus(stepup) ?: stepup).coerceAtMost(1000)
                } else {
                    damageMap["left_ancon"] = (damageMap["left_ancon"]?.minus(stepdown) ?: 0).coerceAtLeast(0)
                    damageMap["right_ancon"] = (damageMap["right_ancon"]?.minus(stepdown) ?: 0).coerceAtLeast(0)
                }
                val left_ancon = damageMap["left_ancon"]?.toFloat()  ?: 0f
                val right_ancon = damageMap["right_ancon"]?.toFloat()  ?: 0f

                _fragmentCameraBinding?.riveAnimationView?.setNumberState("State Machine", "left_ancon", left_ancon / 10f)
                _fragmentCameraBinding?.riveAnimationView?.setNumberState("State Machine", "right_ancon", right_ancon / 10f)

                // 천골
                // 왼쪽 힙
                // 오른쪽 힙
                if (hipCenterZ >= 0) {
                    damageMap["sacrum"] = (damageMap["sacrum"]?.plus(stepup) ?: stepup).coerceAtMost(1000)
                    if (leftHip.z() >= 0) {
                        damageMap["left_hip"] = (damageMap["left_hip"]?.plus(stepup) ?: stepup).coerceAtMost(1000)
                        if (rightHip.z() <= 0) {
                            damageMap["right_hip"] = (damageMap["right_hip"]?.minus(stepdown) ?: 0).coerceAtLeast(0)
                        } else {
                            damageMap["right_hip"] = (damageMap["right_hip"]?.plus(stepup) ?: stepup).coerceAtMost(1000)
                        }
                    } else {
                        damageMap["left_hip"] = (damageMap["left_hip"]?.minus(stepdown) ?: 0).coerceAtLeast(0)
                        if (rightHip.z() >= 0) {
                            damageMap["right_hip"] = (damageMap["right_hip"]?.plus(stepup) ?: stepup).coerceAtMost(1000)
                        } else {
                            damageMap["right_hip"] = (damageMap["right_hip"]?.minus(stepdown) ?: 0).coerceAtLeast(0)
                        }
                    }
                } else {
                    damageMap["sacrum"] = (damageMap["sacrum"]?.minus(stepdown) ?: 0).coerceAtLeast(0)
                    damageMap["left_hip"] = (damageMap["left_hip"]?.minus(stepdown) ?: 0).coerceAtLeast(0)
                    damageMap["right_hip"] = (damageMap["right_hip"]?.minus(stepdown) ?: 0).coerceAtLeast(0)
                }
                val sacrum = damageMap["sacrum"]?.toFloat()  ?: 0f
                val left_hip = damageMap["left_hip"]?.toFloat()  ?: 0f
                val right_hip = damageMap["right_hip"]?.toFloat()  ?: 0f
                _fragmentCameraBinding?.riveAnimationView?.setNumberState("State Machine", "sacrum", sacrum / 10f)
                _fragmentCameraBinding?.riveAnimationView?.setNumberState("State Machine", "left_hip", left_hip / 10f)
                _fragmentCameraBinding?.riveAnimationView?.setNumberState("State Machine", "right_hip", right_hip / 10f)


                val leftheel = marks[29]
                val rightheel = marks[30]
                // 왼쪽 발꿈치
                if (leftheel.z() >= 0) {
                    damageMap["left_heel"] = (damageMap["left_heel"]?.plus(stepup) ?: stepup).coerceAtMost(1000)
                } else {
                    damageMap["left_heel"] = (damageMap["left_heel"]?.minus(stepdown) ?: 0).coerceAtLeast(0)
                }
                val left_heel = damageMap["left_heel"]?.toFloat()  ?: 0f
                _fragmentCameraBinding?.riveAnimationView?.setNumberState("State Machine", "left_heel", left_heel / 10f)

                // 오른쪽 발꿈치
                if (rightheel.z() >= 0) {
                    damageMap["right_heel"] = (damageMap["right_heel"]?.plus(stepup) ?: stepup).coerceAtMost(1000)
                } else {
                    damageMap["right_heel"] = (damageMap["right_heel"]?.minus(stepdown) ?: 0).coerceAtLeast(0)
                }
                val right_heel = damageMap["right_heel"]?.toFloat()  ?: 0f
                _fragmentCameraBinding?.riveAnimationView?.setNumberState("State Machine", "right_heel", right_heel / 10f)



                // 히트맵
// 코 (인덱스 0), 왼쪽 엉덩이 (인덱스 23), 왼쪽 어깨 (인덱스 11)의 z좌표
//                val noseZ = worldLandmarks[0].z()
//                val leftHipZ = worldLandmarks[23].z()
//                val leftShoulderZ = worldLandmarks[11].z()

//                println ("noseZ : $noseZ, leftHipZ: $leftHipZ, leftShoulderZ: $leftShoulderZ")

//// 상체와 하체의 z좌표 차이 계산
//                val bodyZDiff = abs(noseZ - leftHipZ)
//                val upperBodyZDiff = abs(leftShoulderZ - leftHipZ)
//
//// 임계값 설정
//                val zThreshold = 0.1f
//
//                if (bodyZDiff < zThreshold) {
//                    println("누워있는 상태입니다.")
//                } else if (upperBodyZDiff > zThreshold) {
//                    println("앉아있는 상태입니다.")
//                } else {
//                    println("기타 자세입니다.")
//                }
//                _fragmentCameraBinding?.riveAnimationView?.setNumberState("State Machine", "brow", foreheadDistance)
            }
        }

        if (_fragmentCameraBinding != null) {
            activity?.runOnUiThread {
                // Pass necessary information to OverlayView for drawing on the canvas
                fragmentCameraBinding.overlay.setResults(
                    resultBundle.results.first(),
                    resultBundle.inputImageHeight,
                    resultBundle.inputImageWidth,
                    RunningMode.LIVE_STREAM
                )

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
