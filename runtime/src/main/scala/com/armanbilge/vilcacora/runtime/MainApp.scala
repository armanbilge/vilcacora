/*
 * Copyright 2023 Arman Bilge
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.armanbilge.vilcacora.runtime

import cats.effect.{IO, IOApp}
import com.armanbilge.vilcacora.ir._
import vilcacora.onnx.Translator
import vilcacora.onnx.proto.ModelProto
import java.nio.file.{Files, Paths}
import scala.util.Random

object MainApp extends IOApp.Simple {

  // Helper to load the ONNX model from a file in the project's root directory.
  private def loadModelFromFile(path: String): IO[ModelProto] = IO {
    println(s"Loading model from: $path")
    val bytes = Files.readAllBytes(Paths.get(path))
    ModelProto.parseFrom(bytes)
  }

  def run: IO[Unit] = {
    // 1. Load the ONNX model and translate it to our internal representation (IR).
    // IO.fromEither will raise an error and fail the IO if translation fails.
    val modelIRIO: IO[ModelIR] = for {
      modelProto <- loadModelFromFile("static_svm.onnx")
      _ <- IO.println("Model loaded. Translating to ModelIR...")
      modelIR <- IO.fromEither(
        Translator.translate(modelProto).left.map { errorMsg =>
          new RuntimeException(s"Translation failed: $errorMsg")
        },
      )
      _ <- IO.println("Translation successful.")
    } yield modelIR

    modelIRIO
      .flatMap { modelIR =>
        // 2. Define the input data for the model's primary input tensor.
        // The shape and name must match the model's graph input.
        val inputData: Map[String, Array[Float]] = Map(
          "features_float32x30" -> Array.fill(30)(Random.nextFloat()),
        )

        IO.println("\n--- Running Interpreter ---") >>
          // 3. Execute the full model graph via the interpreter.
          Interpreter.execute(modelIR, inputData).flatMap { outputMap =>
            IO.println("Inference complete!") >>
              IO {
                // 4. Print the final results from the specified graph outputs.
                println("\n--- Inference Outputs ---")
                outputMap.foreach { case (name, array) =>
                  val contentString = array match {
                    case arr: Array[Int] => arr.mkString("Array(", ", ", ")")
                    case arr: Array[Float] => arr.mkString("Array(", ", ", ")")
                    case arr: Array[Double] => arr.mkString("Array(", ", ", ")")
                    case _ => "Unknown array type"
                  }
                  println(s"Output tensor '$name': $contentString")
                }
              }
          }
      }
      .handleErrorWith {
        // This will catch errors from translation or execution.
        err =>
          IO.println(s"An error occurred: ${err.getMessage}\n${err.getStackTrace.mkString("\n")}")
      }
  }
}
