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

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import androidx.activity.viewModels
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleObserver
import androidx.lifecycle.OnLifecycleEvent
import androidx.lifecycle.ProcessLifecycleOwner
import androidx.navigation.fragment.NavHostFragment
import androidx.navigation.ui.setupWithNavController
import app.rive.runtime.kotlin.core.Rive
import com.google.mediapipe.examples.poselandmarker.databinding.ActivityMainBinding
import kotlinx.coroutines.CoroutineScope

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import org.opencv.android.OpenCVLoader
import java.io.BufferedReader
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL
import java.nio.charset.StandardCharsets

import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter


class MainActivity : AppCompatActivity(), LifecycleObserver {
    private lateinit var activityMainBinding: ActivityMainBinding
    private val viewModel : MainViewModel by viewModels()

    private val robotSN = "01940078-604d-776b-bc09-c13991cacf21"
    private val userId = "user000"
    private var token: String? = null
    private var refreshToken: String? = null
    private var lastTokenDate: Long = System.currentTimeMillis()
    private var accessTokenExpireDate: Long = 0
    private val utcStart = OffsetDateTime.now(ZoneOffset.UTC)
    private var otolithiasis: Boolean = false
    private var serviceRequest: Boolean = false

    companion object {
        lateinit var instance: MainActivity
            private set
    }

    private suspend fun putPosture(): Boolean {
        val urlString = "https://www.careplace.co.kr/robotApi/robot/posture"
        val url = URL(urlString)
//        println (utcStart.toString())
        val formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ssX")
        val data = JSONObject().apply {
            put("userId", userId)
            put("activityStartTime", utcStart.format(formatter))
//            put("activityEndTime", "")
//            put("postureSettingTime", OffsetDateTime.now(ZoneOffset.UTC).format(formatter))
            put("currentPosture", "")
            put("otolithiasis", otolithiasis)
            put("serviceRequest", serviceRequest)
        }

        val requestBody = JSONObject().apply {
            put("robotId", robotSN)
            put("robotType", "posture")
            put("status", "active")
            put("data", data)
        }
//        val connection = url.openConnection() as HttpURLConnection

        return withContext(Dispatchers.IO) {
            try {
                val connection = url.openConnection() as HttpURLConnection
                connection.requestMethod = "PUT"
                connection.setRequestProperty("Content-Type", "application/json")
                token?.let {
                    connection.setRequestProperty("Authorization", "Bearer $it")
                }

                connection.doOutput = true

                connection.outputStream.use { os ->
                    val input = requestBody.toString().toByteArray()
                    println(requestBody.toString())
                    os.write(input, 0, input.size)
                }

                val responseCode = connection.responseCode
                if (responseCode >= 200 && responseCode < 400) {
                    val allText: String = connection.inputStream.bufferedReader().use(BufferedReader::readText)
                    println("putBackPosture 응답 본문:")
                    val jsonObject = JSONObject(allText)
                    println(jsonObject)
                    true
                }
                else {
                    if (responseCode == 401) {
                        refreshToken()
                    }

//                    token = null
//                    refreshToken = null
                    val allText: String = connection.inputStream.bufferedReader().use(BufferedReader::readText)
                    println("putBackPosture 응답 본문:")
                    println(allText)
                    false
                }
            } catch (e: Exception) {
                println("Error calling Lambda: $e")
                refreshToken()
                false
            }
        }
    }

    private suspend fun putBackPosture(): Boolean {
        val urlString = "https://www.careplace.co.kr/robotApi/robot/posture"
        val url = URL(urlString)
        println (utcStart.toString())
        val formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ssX")
        val data = JSONObject().apply {
            put("userId", userId)
            put("activityStartTime", utcStart.format(formatter))
            put("activityEndTime", OffsetDateTime.now(ZoneOffset.UTC).format(formatter))
            put("postureSettingTime", OffsetDateTime.now(ZoneOffset.UTC).format(formatter))
            put("currentPosture", "")
            put("otolithiasis", otolithiasis)
            put("serviceRequest", serviceRequest)
        }

        val requestBody = JSONObject().apply {
            put("robotId", robotSN)
            put("robotType", "posture")
            put("status", "active")
            put("data", data)
        }
//        val connection = url.openConnection() as HttpURLConnection

        return withContext(Dispatchers.IO) {
            try {
                val connection = url.openConnection() as HttpURLConnection
                connection.requestMethod = "PUT"
                connection.setRequestProperty("Content-Type", "application/json")
                token?.let {
                    connection.setRequestProperty("Authorization", "Bearer $it")
                }

                connection.doOutput = true

                connection.outputStream.use { os ->
                    val input = requestBody.toString().toByteArray()
                    println(requestBody.toString())
                    os.write(input, 0, input.size)
                }

                val responseCode = connection.responseCode
                if (responseCode >= 200 && responseCode < 400) {
                    val allText: String = connection.inputStream.bufferedReader().use(BufferedReader::readText)
                    println("putBackPosture 응답 본문:")
                    val jsonObject = JSONObject(allText)
                    println(jsonObject)
                    true
                }
                else {
//                    token = null
//                    refreshToken = null
                    val allText: String = connection.inputStream.bufferedReader().use(BufferedReader::readText)
                    println("putBackPosture 응답 본문:")
                    println(allText)
                    refreshToken()
                    false
                }
            } catch (e: Exception) {
                println("Error calling Lambda: $e")
                refreshToken()
                false
            }
        }
    }

    private suspend fun putExitPosture(): Boolean {
        val urlString = "https://www.careplace.co.kr/robotApi/robot/posture"
        val url = URL(urlString)
        println (utcStart.toString())
        val formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ssX")
        val data = JSONObject().apply {
            put("userId", userId)
            put("activityStartTime", utcStart.format(formatter))
            put("activityEndTime", OffsetDateTime.now(ZoneOffset.UTC).format(formatter))
            put("postureSettingTime", OffsetDateTime.now(ZoneOffset.UTC).format(formatter))
            put("currentPosture", "")
            put("otolithiasis", otolithiasis)
            put("serviceRequest", serviceRequest)
        }

        val requestBody = JSONObject().apply {
            put("robotId", robotSN)
            put("robotType", "posture")
            put("status", "off")
            put("data", data)
        }
//        val connection = url.openConnection() as HttpURLConnection

        return withContext(Dispatchers.IO) {
            try {
                val connection = url.openConnection() as HttpURLConnection
                connection.requestMethod = "PUT"
                connection.setRequestProperty("Content-Type", "application/json")
                token?.let {
                    connection.setRequestProperty("Authorization", "Bearer $it")
                }

                connection.doOutput = true

                connection.outputStream.use { os ->
                    val input = requestBody.toString().toByteArray()
                    println(requestBody.toString())
                    os.write(input, 0, input.size)
                }

                val responseCode = connection.responseCode
                if (responseCode >= 200 && responseCode < 400) {
                    val allText: String = connection.inputStream.bufferedReader().use(BufferedReader::readText)
                    println("putExitPosture 응답 본문:")
//                    println(allText)
                    val jsonObject = JSONObject(allText)
                    println(jsonObject)
//                    refreshToken = jsonObject.getString("refreshToken")
//                    token = jsonObject.getString("accessToken")
//
//                    lastTokenDate = System.currentTimeMillis()
//                    println("accessTokenExpireDate : ${jsonObject.getLong("accessTokenExpireDate")}")
//                    accessTokenExpireDate = jsonObject.getLong("accessTokenExpireDate")
                    true
                }
                else {
//                    token = null
//                    refreshToken = null
                    val allText: String = connection.inputStream.bufferedReader().use(BufferedReader::readText)
                    println("putExitPosture 응답 본문:")
                    println(allText)
                    refreshToken()
                    false
                }
            } catch (e: Exception) {
                println("Error calling Lambda: $e")
                refreshToken()
                false
            }
        }
    }

    private suspend fun refreshToken() {
        val urlString = "https://www.careplace.co.kr/robotApi/auth/refreshToken"
        val url = URL(urlString)
        val requestBody = JSONObject().apply {
            put("username", robotSN)
            put("refreshToken", refreshToken)
        }
//        val connection = url.openConnection() as HttpURLConnection

        withContext(Dispatchers.IO) {
            try {
                val connection = url.openConnection() as HttpURLConnection
                connection.requestMethod = "POST"
                connection.setRequestProperty("Content-Type", "application/json")
                token?.let {
                    connection.setRequestProperty("Authorization", "Bearer $it")
                }

                connection.doOutput = true

                connection.outputStream.use { os ->
                    val input = requestBody.toString().toByteArray()
                    os.write(input, 0, input.size)
                }

                val responseCode = connection.responseCode
                if (responseCode >= 200 && responseCode < 400) {
                    val allText: String = connection.inputStream.bufferedReader().use(BufferedReader::readText)
                    println("응답 본문:")
                    println(allText)
                    val jsonObject = JSONObject(allText)
//                    val resultObject = jsonObject.getJSONObject("result")
//                    val itemsObject = resultObject.getJSONObject("items")
//                    itemsObject.getString("status")
//                    jsonObject.getString("refreshToken")
                    println(jsonObject)
                    refreshToken = jsonObject.getString("refreshToken")
                    token = jsonObject.getString("accessToken")

                    lastTokenDate = System.currentTimeMillis()
                    println("accessTokenExpireDate : ${jsonObject.getLong("accessTokenExpireDate")}")
                    accessTokenExpireDate = jsonObject.getLong("accessTokenExpireDate")
                }
                else {
                    token = null
                    refreshToken = null
                }
            } catch (e: Exception) {
                println("Error calling Lambda: $e")
            }
        }
    }

    private suspend fun getToken(){
        val urlString = "https://www.careplace.co.kr/robotApi/auth/getToken"
        val url = URL(urlString)
        val requestBody = JSONObject().apply {
            put("robotId", robotSN)
        }
//        val connection = url.openConnection() as HttpURLConnection

        withContext(Dispatchers.IO) {
            try {
                val connection = url.openConnection() as HttpURLConnection
                connection.requestMethod = "POST"
                connection.setRequestProperty("Content-Type", "application/json")
                connection.doOutput = true

                connection.outputStream.use { os ->
                    val input = requestBody.toString().toByteArray()
                    os.write(input, 0, input.size)
                }

                val responseCode = connection.responseCode
                if (responseCode >= 200 && responseCode < 400){
                    val allText: String = connection.inputStream.bufferedReader().use(BufferedReader::readText)
                    println("응답 본문:")
                    println(allText)
                    val jsonObject = JSONObject(allText)
//                    val resultObject = jsonObject.getJSONObject("result")
//                    val itemsObject = resultObject.getJSONObject("items")
//                    itemsObject.getString("status")
                    refreshToken = jsonObject.getString("refreshToken")
                    token = jsonObject.getString("accessToken")
                    lastTokenDate = System.currentTimeMillis()
                    println("accessTokenExpireDate : ${jsonObject.getLong("accessTokenExpireDate")}")
                    accessTokenExpireDate = jsonObject.getLong("accessTokenExpireDate")
                }
                else {
                    token = null
                    refreshToken = null
                }
            } catch (e: Exception) {
                println("Error calling Lambda: $e")
                null
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        ProcessLifecycleOwner.get().lifecycle.addObserver(this)

        OpenCVLoader.initLocal()
        Rive.init(this)

        instance = this
        activityMainBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(activityMainBinding.root)

        viewModel.bedStatus.observe(this) { state ->
            if (state) {
                activityMainBinding.status.text = "자리에있음"
                otolithiasis = false
            }
            else {
                activityMainBinding.status.text = "자리비움"
                otolithiasis = true
            }

            CoroutineScope(Dispatchers.IO).launch {
                callLambdaFunction(activityMainBinding.status.text as String)
                if (!putPosture()) putPosture()
            }
        }

        viewModel.callStatus.observe(this) { state ->
            if (state) {
                if (serviceRequest != true) {
                    serviceRequest = true
                    CoroutineScope(Dispatchers.IO).launch {
                        if (!putPosture()) putPosture()
                    }
                }
            } else {
                if (serviceRequest != false) {
                    serviceRequest = false
                    CoroutineScope(Dispatchers.IO).launch {
                        if (!putPosture()) putPosture()
                    }
                }
            }
        }

        CoroutineScope(Dispatchers.IO).launch {
            getToken()
//            println("token : $token")
//            refreshToken()
        }

        
        viewModel.bedStatus.value = true
    }

    private suspend fun getStatus(): String? {
        val url = URL("https://7gb9b8se68.execute-api.ap-northeast-2.amazonaws.com/prod/status") // API Gateway URL
        val jsonBody = JSONObject().apply {
            put("duid", "smartbed1")
        }

        return withContext(Dispatchers.IO) {
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
                if (responseCode >= 200 && responseCode < 400){
                    val allText: String = connection.inputStream.bufferedReader().use(BufferedReader::readText)
                    val jsonObject = JSONObject(allText)
                    val resultObject = jsonObject.getJSONObject("result")
                    val itemsObject = resultObject.getJSONObject("items")
                    itemsObject.getString("status")
                }
                else {
                    null
                }
            } catch (e: Exception) {
                println("Error calling Lambda: $e")
                null
            }
        }
    }

    private suspend fun callLambdaFunction(status: String) {
        val url = URL("https://7gb9b8se68.execute-api.ap-northeast-2.amazonaws.com/prod/update") // API Gateway URL
        val jsonBody = JSONObject().apply {
            put("duid", "smartbed1")
            put("status", status)
        }

        withContext(Dispatchers.IO) {
            var isNotCall = false
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
//                println("Response Code: $responseCode")
            } catch (e: Exception) {
//                println("Error calling Lambda: $e")
                isNotCall = true
            }

            if (!isNotCall) {
                if (getStatus() != status) {
//                    println("Fail data : ${getStatus()}")
                    callLambdaFunction(status)
                }
            }
        }
    }

    override fun onBackPressed() {
        finish()
    }

    override fun onDestroy() {
        super.onDestroy()

        Log.d("MainActivity", "onDestroy: Activity is being destroyed")

        CoroutineScope(Dispatchers.IO).launch {
            if (!putExitPosture()) putExitPosture()
        }
    }

    @OnLifecycleEvent(Lifecycle.Event.ON_STOP)
    fun onAppBackgrounded() {
        println("앱이 백그라운드로 이동했습니다. 데이터를 저장하세요.")
        CoroutineScope(Dispatchers.IO).launch {
            if (!putExitPosture()) putExitPosture()
        }
    }

    @OnLifecycleEvent(Lifecycle.Event.ON_START)
    fun onAppForegrounded() {
        println("앱이 포그라운드로 복귀했습니다.")
        CoroutineScope(Dispatchers.IO).launch {
            if (!putBackPosture()) putBackPosture()
        }
    }
}