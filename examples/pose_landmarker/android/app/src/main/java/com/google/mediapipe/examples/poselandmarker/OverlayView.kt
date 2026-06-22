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
package com.google.mediapipe.examples.poselandmarker

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RadialGradient
import android.graphics.Shader
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import androidx.lifecycle.ViewModelProvider
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

class OverlayView(context: Context?, attrs: AttributeSet?) :
    View(context, attrs) {

    private var results: PoseLandmarkerResult? = null
    private var pointPaint = Paint()
    private var linePaint = Paint()

    private var pointLPFPaint = Paint()
    private var lineLPFPaint = Paint()

    private var scaleFactor: Float = 1f
    private var scaleFactorResult: Float = 1f
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1

    var resultsLandmark: List<NormalizedLandmark>? = null

    val viewModel: MainViewModel by lazy {
        ViewModelProvider(MainActivity.instance)[MainViewModel::class.java]
    }

    init {
        initPaints()
    }

    fun clear() {
        results = null
        pointPaint.reset()
        linePaint.reset()
        invalidate()
        initPaints()
    }

    private fun initPaints() {
//        linePaint.color =
//            ContextCompat.getColor(context!!, R.color.mp_color_primary)
        linePaint.color =  Color.CYAN
        linePaint.strokeWidth = LANDMARK_STROKE_WIDTH
        linePaint.style = Paint.Style.STROKE

        pointPaint.color = Color.YELLOW
        pointPaint.strokeWidth = LANDMARK_STROKE_WIDTH
        pointPaint.style = Paint.Style.FILL

        pointLPFPaint.color = Color.RED

        pointLPFPaint.strokeWidth = LANDMARK_STROKE_WIDTH
        pointLPFPaint.style = Paint.Style.FILL

        lineLPFPaint.color =  Color.DKGRAY
        lineLPFPaint.strokeWidth = LANDMARK_STROKE_WIDTH
        lineLPFPaint.style = Paint.Style.STROKE
    }

    private val rectPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 5f
    }

    private val rectRedPaint = Paint().apply {
        color = Color.argb(0.25f, 1f, 0f, 0f)
        style = Paint.Style.STROKE
        strokeWidth = 3f
    }

//    private var cropLeft = 0
//    private var cropTop = 0
//    private var cropRight = 0
//    private var cropBottom = 0

    private var topLeftX = 0f
    private var topLeftY = 0f
    private var topRightX = 0f
    private var topRightY = 0f
    private var bottomLeftX = 0f
    private var bottomLeftY = 0f
    private var bottomRightX = 0f
    private var bottomRightY = 0f

    private var topredLeftX = 0f
    private var topredLeftY = 0f
    private var topredRightX = 0f
    private var topredRightY = 0f
    private var bottomredLeftX = 0f
    private var bottomredLeftY = 0f
    private var bottomredRightX = 0f
    private var bottomredRightY = 0f

    private fun createTrapezoidPath(): android.graphics.Path {
        val path = android.graphics.Path()
        path.moveTo(topLeftX, topLeftY)       // Move to top-left corner
        path.lineTo(topRightX, topRightY)     // Draw line to top-right corner
        path.lineTo(bottomRightX, bottomRightY) // Draw line to bottom-right corner
        path.lineTo(bottomLeftX, bottomLeftY)   // Draw line to bottom-left corner
        path.close()  // Close the path
        return path
    }

    private fun createRedzonePath(): android.graphics.Path {
        val path = android.graphics.Path()
//        path.moveTo(topLeftX + 66, topLeftY + 60)       // Move to top-left corner
//        path.lineTo(topRightX - 66, topRightY + 60)     // Draw line to top-right corner
//        path.lineTo(bottomRightX - 100, bottomRightY - 100) // Draw line to bottom-right corner
//        path.lineTo(bottomLeftX + 100, bottomLeftY - 100)   // Draw line to bottom-left corner

        path.moveTo(topredLeftX, topredLeftY)       // Move to top-left corner
        path.lineTo(topredRightX, topredRightY)     // Draw line to top-right corner
        path.lineTo(bottomredRightX, bottomredRightY) // Draw line to bottom-right corner
        path.lineTo(bottomredLeftX, bottomredLeftY)   // Draw line to bottom-left corner
        path.close()  // Close the path
        return path
    }

    val minZ = -0.3f  // 카메라에 매우 가까울 때
    val maxZ = 0.3f   // 카메라로부터 멀 때

    // 한 랜드마크의 z값이 있을 때
    fun getAlphaFromZ(z: Float): Int {
        // z값을 0~1 범위로 정규화: (z - minZ) / (maxZ - minZ)
        val normalized = (z - minZ) / (maxZ - minZ)
        // normalized를 0~1 사이 값으로 클램핑
        val clamped = normalized.coerceIn(0f, 1f)
        // 알파값은 255에서 시작해서 z가 멀어질수록 투명해진다고 가정
        // 가까울수록 불투명 -> z가 minZ일 때 알파=255, maxZ일 때 알파=0
        val alpha = (clamped * 255).toInt()
        return alpha
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        val paintBlue = Paint().apply {
            color = Color.BLUE
            style = Paint.Style.FILL
        }
        val paintRed = Paint().apply {
            color = Color.RED
            style = Paint.Style.FILL
        }

        val radialGradient = RadialGradient(
            2.5f,
            2.5f,
            2f,
            intArrayOf(Color.TRANSPARENT, Color.RED),
            null,
            Shader.TileMode.MIRROR
        )

        val gradientPaint = Paint().apply {
            isAntiAlias = true
            shader = radialGradient
        }


//        canvas.drawRect(0F, 0F, 100F, 200F, paintBlue)
        results?.let { poseLandmarkerResult ->
//            poseLandmarkerResult.worldLandmarks()
            for(landmark in poseLandmarkerResult.landmarks()) {
                PoseLandmarker.POSE_LANDMARKS.forEach {
                    canvas.drawLine(
                        (132 + (poseLandmarkerResult.landmarks().get(0).get(it!!.start()).x() * imageWidth)) * scaleFactorResult,
                        poseLandmarkerResult.landmarks().get(0).get(it.start()).y() * imageHeight * scaleFactorResult,
                        (132 + (poseLandmarkerResult.landmarks().get(0).get(it.end()).x() * imageWidth)) * scaleFactorResult,
                        poseLandmarkerResult.landmarks().get(0).get(it.end()).y() * imageHeight * scaleFactorResult,
                        linePaint)
                }

                PoseLandmarker.POSE_LANDMARKS.forEach {

                    resultsLandmark?.let { rr ->
                        canvas.drawLine(
                            (132 + (rr.get(it!!.start()).x() * imageWidth)) * scaleFactorResult,
                            rr.get(it.start()).y() * imageHeight * scaleFactorResult,
                            (132 + (rr.get(it.end()).x() * imageWidth)) * scaleFactorResult,
                            rr.get(it.end()).y() * imageHeight * scaleFactorResult,
                            lineLPFPaint)
                    }
                }
                resultsLandmark?.let {
//                    var idx = 0
                    for(n in it) {
//                        println("it.indexOf(n) :  ${idx++}   =========== z: ${n.z()}")
//                        pointLPFPaint.alpha = getAlphaFromZ(n.z())
                        canvas.drawPoint(
                            (132 + (n.x() * imageWidth)) * scaleFactorResult,
                            n.y() * imageHeight * scaleFactorResult,
                            pointLPFPaint
                        )
                    }
                }

//                val tmp = ArrayList<FloatArray>()
//                val nose = poseLandmarkerResult.landmarks().get(0).get(0)
//                if (nose.z() < 0) {
//                    val leftEye = poseLandmarkerResult.landmarks().get(0).get(2)
//                    val rightEye = poseLandmarkerResult.landmarks().get(0).get(5)
//
//                    val headCenter = if (leftEye != null && rightEye != null && nose != null) {
//                        val x = (leftEye.x() + rightEye.x() + nose.x()) / 3
//                        val y = (leftEye.y() + rightEye.y() + nose.y()) / 3
//                        val z = (leftEye.z() + rightEye.z() + nose.z()) / 3
//                        floatArrayOf(x, y, z)
//                    } else {
//                        null
//                    }
//
//                    val headDirection = if (headCenter != null) {
//                        floatArrayOf(
//                            headCenter[0] - nose.x(),
//                            headCenter[1] - nose.y(),
//                            headCenter[2] - nose.z()
//                        )
//                    } else {
//                        null
//                    }
//
//                    val occiputPosition = if (headDirection != null) {
//                        // 머리의 깊이 값을 추정하여 스케일링 (예: 머리 깊이의 절반)
//                        val headDepth = 0.1f // 머리의 깊이 (미터 단위, 필요시 조정)
//                        val scale = headDepth / sqrt(
//                            headDirection[0] * headDirection[0] +
//                                    headDirection[1] * headDirection[1] +
//                                    headDirection[2] * headDirection[2]
//                        )
//
//                        floatArrayOf(
//                            nose.x() + headDirection[0] * scale,
//                            nose.y() + headDirection[1] * scale
//                        )
//                    } else {
//                        null
//                    }
//
//                    if (occiputPosition != null) {
//                        tmp.add(occiputPosition)
//
//                        canvas.drawCircle(
//                            occiputPosition[0] * 100,
//                            occiputPosition[1] * 200,
//                            5f,
//                            gradientPaint)
//                    }
//
//                }
//
//                val leftS = poseLandmarkerResult.landmarks().get(0).get(11)
//                if (leftS.z() > 0) {
//                    tmp.add(floatArrayOf(leftS.x(), leftS.y()))
//                    canvas.drawCircle(
//                        leftS.x() * 100,
//                        leftS.y() * 200,
//                        5f,
//                        gradientPaint)
//                }
//                val rightS = poseLandmarkerResult.landmarks().get(0).get(12)
//                if (rightS.z() > 0) {
//                    tmp.add(floatArrayOf(rightS.x(), rightS.y()))
//                    canvas.drawCircle(
//                        rightS.x() * 100,
//                        rightS.y() * 200,
//                        5f,
//                        gradientPaint)
//                }
//                val leftH= poseLandmarkerResult.landmarks().get(0).get(23)
//                if (leftH.z() > 0) {
//                    tmp.add(floatArrayOf(leftH.x(), leftH.y()))
//                    canvas.drawCircle(
//                        leftH.x() * 100,
//                        leftH.y() * 200,
//                        5f,
//                        gradientPaint)
//                }
//                val rightH = poseLandmarkerResult.landmarks().get(0).get(24)
//                if (rightH.z() > 0) {
//                    tmp.add(floatArrayOf(rightH.x(), rightH.y()))
//                    canvas.drawCircle(
//                        rightH.x() * 100,
//                        rightH.y() * 200,
//                        5f,
//                        gradientPaint)
//                }
//                val leftHl= poseLandmarkerResult.landmarks().get(0).get(29)
//                if (leftHl.z() > 0) {
//                    tmp.add(floatArrayOf(leftHl.x(), leftHl.y()))
//                    canvas.drawCircle(
//                        leftHl.x() * 100,
//                        leftHl.y() * 200,
//                        5f,
//                        gradientPaint)
//                }
//                val rightHl = poseLandmarkerResult.landmarks().get(0).get(30)
//                if (rightHl.z() > 0) {
//                    tmp.add(floatArrayOf(rightHl.x(), rightHl.y()))
//                    canvas.drawCircle(
//                        rightHl.x() * 100,
//                        rightHl.y() * 200,
//                        5f,
//                        gradientPaint)
//                }
//                val leftk= poseLandmarkerResult.landmarks().get(0).get(25)
//                if (leftk.z() > 0) {
//                    tmp.add(floatArrayOf(leftk.x(), leftk.y()))
//                    canvas.drawCircle(
//                        leftk.x() * 100,
//                        leftk.y() * 200,
//                        5f,
//                        gradientPaint)
//                }
//                val rightk = poseLandmarkerResult.landmarks().get(0).get(26)
//                if (rightk.z() > 0) {
//                    tmp.add(floatArrayOf(rightk.x(), rightk.y()))
//                    canvas.drawCircle(
//                        rightk.x() * 100,
//                        rightk.y() * 200,
//                        5f,
//                        gradientPaint)
//                }
//                val lefte= poseLandmarkerResult.landmarks().get(0).get(13)
//                if (lefte.z() > 0) {
//                    tmp.add(floatArrayOf(lefte.x(), lefte.y()))
//                    canvas.drawCircle(
//                        lefte.x() * 100,
//                        lefte.y() * 200,
//                        5f,
//                        gradientPaint)
//                }
//                val righte = poseLandmarkerResult.landmarks().get(0).get(14)
//                if (righte.z() > 0) {
//                    tmp.add(floatArrayOf(righte.x(), righte.y()))
//                    canvas.drawCircle(
//                        righte.x() * 100,
//                        righte.y() * 200,
//                        5f,
//                        gradientPaint)
//                }
//
//                // 골반 중심 좌표 계산
//                val pelvisCenter = if (leftH != null && rightH != null) {
//                    val x = (leftH.x() + rightH.x()) / 2
//                    val y = (leftH.y() + rightH.y()) / 2
//                    val z = (leftH.z() + rightH.z()) / 2
//                    floatArrayOf(x, y, z)
//                } else {
//                    null
//                }
//
//                // 어깨 중심 좌표 계산
//                val shoulderCenter = if (leftS != null && rightS != null) {
//                    val x = (leftS.x() + rightS.x()) / 2
//                    val y = (leftS.y() + rightS.y()) / 2
//                    val z = (leftS.z() + rightS.z()) / 2
//                    floatArrayOf(x, y, z)
//                } else {
//                    null
//                }
//
//                // 척추 방향 벡터 계산 (어깨 중심 -> 골반 중심)
//                val spineDirection = if (pelvisCenter != null && shoulderCenter != null) {
//                    floatArrayOf(
//                        pelvisCenter[0] - shoulderCenter[0],
//                        pelvisCenter[1] - shoulderCenter[1],
//                        pelvisCenter[2] - shoulderCenter[2]
//                    )
//                } else {
//                    null
//                }
//
//                // 꼬리뼈 위치 계산
//                val tailbonePosition = if (spineDirection != null && pelvisCenter != null) {
//                    // 척추 방향 벡터를 따라 아래로 연장 (스케일링)
//                    val spineLength = sqrt(
//                        spineDirection[0] * spineDirection[0] +
//                                spineDirection[1] * spineDirection[1] +
//                                spineDirection[2] * spineDirection[2]
//                    )
//                    val scale = 0.02f // 골반에서 꼬리뼈까지의 상대적인 거리 (필요에 따라 조정)
//
//                    // 단위 벡터로 만들기
//                    val unitSpineDirection = floatArrayOf(
//                        spineDirection[0] / spineLength,
//                        spineDirection[1] / spineLength,
//                        spineDirection[2] / spineLength
//                    )
//
//                    // 꼬리뼈 위치 계산
//                    floatArrayOf(
//                        pelvisCenter[0] + unitSpineDirection[0] * scale,
//                        pelvisCenter[1] + unitSpineDirection[1] * scale,
//                        pelvisCenter[2] + unitSpineDirection[2] * scale
//                    )
//                } else {
//                    null
//                }
//
//                if (tailbonePosition != null) {
//                    tmp.add(floatArrayOf(tailbonePosition[0], tailbonePosition[1]))
//                    canvas.drawCircle(
//                        tailbonePosition[0] * 100,
//                        tailbonePosition[1] * 200,
//                        5f,
//                        gradientPaint)
//                }
//
//                // 2D로 표시
//                drawPressurePoints(canvas, tmp)
            }
        }
//        canvas.drawRect(cropLeft.toFloat(), cropTop.toFloat(), cropRight.toFloat(), cropBottom.toFloat(), rectPaint)
        val trapezoidPath = createTrapezoidPath()
        canvas.drawPath(trapezoidPath, rectPaint)

        val redzonePath = createRedzonePath()
        canvas.drawPath(redzonePath, rectRedPaint)
    }

    fun drawPressurePoints(canvas: Canvas, points: List<FloatArray>) {
        val paint = Paint().apply {
            color = Color.RED
            style = Paint.Style.FILL
            alpha = 60
        }

        points.forEach { point ->
            canvas.drawCircle(
                (132 + (point[0] * imageWidth)) * scaleFactorResult,
                point[1] * imageHeight * scaleFactorResult,
                40f,
                paint)
        }
    }

    private var lastStateChangeTime: Long = 0L
    fun setResults(
        poseLandmarkerResults: PoseLandmarkerResult,
        imageHeight: Int,
        imageWidth: Int,
        runningMode: RunningMode = RunningMode.IMAGE
    ) {
        results = poseLandmarkerResults

        val currentTime = System.currentTimeMillis()
        if (results!!.landmarks().size > 0) {
            if (viewModel.rawbedStatus.value != true) {
                viewModel.rawbedStatus.value = true
                lastStateChangeTime = currentTime
            }
        } else {
            if (viewModel.rawbedStatus.value != false) {
                viewModel.rawbedStatus.value = false
                lastStateChangeTime = currentTime
            }
        }

        if (currentTime - lastStateChangeTime >= 2000) {
            if (viewModel.bedStatus.value != viewModel.rawbedStatus.value)
                viewModel.bedStatus.value = viewModel.rawbedStatus.value
        }

        this.imageHeight = imageHeight
        this.imageWidth = imageWidth

//        println("imageWidth : $imageWidth, imageHeight : $imageHeight")

        scaleFactor = when (runningMode) {
            RunningMode.IMAGE,
            RunningMode.VIDEO -> {
                min(width * 1f / imageWidth, height * 1f / imageHeight)
            }
            RunningMode.LIVE_STREAM -> {
                // PreviewView is in FILL_START mode. So we need to scale up the
                // landmarks to match with the size that the captured images will be
                // displayed.
                max(width * 1f / imageWidth, height * 1f / imageHeight)
            }
        }
        scaleFactorResult = min(width * 1f / imageWidth, height * 1f / imageHeight)

        val cropWidth = 160
        val cropWidthBottom = 320
        val cropHeight = 340

        topLeftX = ((imageWidth - cropWidth) / 2 * scaleFactor)
        topLeftY = 0f
        topRightX = topLeftX + (cropWidth * scaleFactor).toInt()
        topRightY = 0f
        bottomLeftX = ((imageWidth - cropWidthBottom) / 2 * scaleFactor)
        bottomLeftY = topLeftY + (cropHeight * scaleFactor).toInt()
        bottomRightX = bottomLeftX + (cropWidthBottom * scaleFactor).toInt()
        bottomRightY = bottomLeftY


        val cropredWidth = 130
        val cropredWidthBottom = 260
        val cropredHeight = 290
        topredLeftX = ((imageWidth - cropredWidth) / 2 * scaleFactor)
        topredLeftY = 20f  * scaleFactor
        topredRightX = topredLeftX + (cropredWidth * scaleFactor).toInt()
        topredRightY = topredLeftY
        bottomredLeftX = ((imageWidth - cropredWidthBottom) / 2 * scaleFactor)
        bottomredLeftY = topredLeftY + (cropredHeight * scaleFactor).toInt()
        bottomredRightX = bottomredLeftX + (cropredWidthBottom * scaleFactor).toInt()
        bottomredRightY = bottomredLeftY

//        topredLeftX = 250 * scaleFactor
//        topredLeftY = 40 * scaleFactor
//        topredRightX = 390 * scaleFactor
//        topredRightY = topredLeftY
//        bottomredLeftX = 100 * scaleFactor
//        bottomredLeftY = 440 * scaleFactor
//        bottomredRightX = 500 * scaleFactor
//        bottomredRightY = bottomredLeftY

//        invalidate()
    }

    companion object {
        private const val LANDMARK_STROKE_WIDTH = 6F
    }
}