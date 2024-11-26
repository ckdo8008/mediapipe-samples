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
import androidx.activity.viewModels
import androidx.navigation.fragment.NavHostFragment
import androidx.navigation.ui.setupWithNavController
import com.google.mediapipe.examples.poselandmarker.databinding.ActivityMainBinding
import kotlinx.coroutines.CoroutineScope

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import org.opencv.android.OpenCVLoader
import java.io.BufferedReader
import java.net.HttpURLConnection
import java.net.URL


class MainActivity : AppCompatActivity() {
    private lateinit var activityMainBinding: ActivityMainBinding
    private val viewModel : MainViewModel by viewModels()

    companion object {
        lateinit var instance: MainActivity
            private set
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        OpenCVLoader.initLocal()

        instance = this
        activityMainBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(activityMainBinding.root)

        viewModel.bedStatus.observe(this) { state ->
            if (state)
                activityMainBinding.status.text = "자리에있음"
            else
                activityMainBinding.status.text = "자리비움"

            CoroutineScope(Dispatchers.IO).launch {
                callLambdaFunction(activityMainBinding.status.text as String)
            }
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

            if (getStatus() != status) {
                println("Fail data : ${getStatus()}")
                callLambdaFunction(status)
            }
        }
    }

    override fun onBackPressed() {
        finish()
    }
}