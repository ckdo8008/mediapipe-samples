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
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import androidx.lifecycle.ViewModelProvider
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import kotlin.math.max
import kotlin.math.min

class OverlayView(context: Context?, attrs: AttributeSet?) :
    View(context, attrs) {

    private var results: PoseLandmarkerResult? = null
    private var pointPaint = Paint()
    private var linePaint = Paint()

    private var scaleFactor: Float = 1f
    private var scaleFactorResult: Float = 1f
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1

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
    }

    private val rectPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 5f
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

    private fun createTrapezoidPath(): android.graphics.Path {
        val path = android.graphics.Path()
        path.moveTo(topLeftX, topLeftY)       // Move to top-left corner
        path.lineTo(topRightX, topRightY)     // Draw line to top-right corner
        path.lineTo(bottomRightX, bottomRightY) // Draw line to bottom-right corner
        path.lineTo(bottomLeftX, bottomLeftY)   // Draw line to bottom-left corner
        path.close()  // Close the path
        return path
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)
        results?.let { poseLandmarkerResult ->
            for(landmark in poseLandmarkerResult.landmarks()) {
                // Draw the green rectangle

                for(normalizedLandmark in landmark) {
                    canvas.drawPoint(
                        (132 + (normalizedLandmark.x() * imageWidth)) * scaleFactorResult,
                        normalizedLandmark.y() * imageHeight * scaleFactorResult,
                        pointPaint
                    )
                }

                PoseLandmarker.POSE_LANDMARKS.forEach {
                    canvas.drawLine(
                        (132 + (poseLandmarkerResult.landmarks().get(0).get(it!!.start()).x() * imageWidth)) * scaleFactorResult,
                        poseLandmarkerResult.landmarks().get(0).get(it.start()).y() * imageHeight * scaleFactorResult,
                        (132 + (poseLandmarkerResult.landmarks().get(0).get(it.end()).x() * imageWidth)) * scaleFactorResult,
                        poseLandmarkerResult.landmarks().get(0).get(it.end()).y() * imageHeight * scaleFactorResult,
                        linePaint)
                }
            }
        }
//        canvas.drawRect(cropLeft.toFloat(), cropTop.toFloat(), cropRight.toFloat(), cropBottom.toFloat(), rectPaint)
        val trapezoidPath = createTrapezoidPath()
        canvas.drawPath(trapezoidPath, rectPaint)
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
        val cropHeight = 270

        topLeftX = ((imageWidth - cropWidth) / 2 * scaleFactor)
        topLeftY = 70 * scaleFactor

        topRightX = topLeftX + (cropWidth * scaleFactor).toInt()
        topRightY = topLeftY

        bottomLeftX = ((imageWidth - cropWidthBottom) / 2 * scaleFactor)
        bottomLeftY = topLeftY + (cropHeight * scaleFactor).toInt()

        bottomRightX = bottomLeftX + (cropWidthBottom * scaleFactor).toInt()
        bottomRightY = bottomLeftY

        invalidate()
    }

    companion object {
        private const val LANDMARK_STROKE_WIDTH = 6F
    }
}