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



package com.armanbilge.runtime

import scala.scalanative.unsafe._
import scala.scalanative.unsigned._


@link("mlpack")
@link("openblas")
@extern object NativeOnnxOps {

  def onnx_add_matrix_double(
      A_data: Ptr[CDouble], rowsA: CSize, colsA: CSize,
      B_data: Ptr[CDouble], rowsB: CSize, colsB: CSize,
      C_data: Ptr[CDouble]): Unit = extern

  def onnx_mul_matrix_double(
      A_data: Ptr[CDouble], rowsA: CSize, colsA: CSize,
      B_data: Ptr[CDouble], rowsB: CSize, colsB: CSize,
      D_data: Ptr[CDouble]): Unit = extern

  def onnx_cast_double_to_float(
      input_data: Ptr[CDouble], rows: CSize, cols: CSize,
      output_data: Ptr[CFloat]): Unit = extern

  def onnx_cast_float_to_double(
      input_data: Ptr[CFloat], rows: CSize, cols: CSize,
      output_data: Ptr[CDouble]): Unit = extern
}


object Main {
  def main(args: Array[String]): Unit = {
    Zone { implicit z =>
      println("Calling C++ mlpack/Armadillo for ONNX operations from Scala Native...")

      def printMatrix(name: String, data: Array[Double], rows: Int, cols: Int): Unit = {
        println(s"$name ($rows x $cols):")
        for (r <- 0 until rows) {
          print("  [")
          for (c <- 0 until cols) {
            print(f"${data(r * cols + c)}%.2f")
            if (c < cols - 1) print(", ")
          }
          println("]")
        }
      }

      def printFloatMatrix(name: String, data: Array[Float], rows: Int, cols: Int): Unit = {
        println(s"$name ($rows x $cols):")
        for (r <- 0 until rows) {
          print("  [")
          for (c <- 0 until cols) {
            print(f"${data(r * cols + c)}%.2f")
            if (c < cols - 1) print(", ")
          }
          println("]")
        }
      }

      val rows = 2
      val cols = 2
      val size = rows * cols

      val matrixA = Array[CDouble](1.0, 2.0, 3.0, 4.0)
      val matrixB = Array[CDouble](5.0, 6.0, 7.0, 8.0)

      val ptrA = alloc[CDouble](size)
      val ptrB = alloc[CDouble](size)

      for (i <- 0 until size) {
        !(ptrA + i) = matrixA(i)
        !(ptrB + i) = matrixB(i)
      }

      printMatrix("Matrix A", matrixA.map(_.toDouble), rows, cols)
      printMatrix("Matrix B", matrixB.map(_.toDouble), rows, cols)

      println("\n--- Performing ONNX Add ---")
      val resultAddPtr = alloc[CDouble](size)
      NativeOnnxOps.onnx_add_matrix_double(
        ptrA, rows.toCSize, cols.toCSize,
        ptrB, rows.toCSize, cols.toCSize,
        resultAddPtr
      )
      val resultAddArray = Array.ofDim[Double](size)
      for (i <- 0 until size) resultAddArray(i) = !(resultAddPtr + i)
      printMatrix("Result of A + B", resultAddArray, rows, cols)

      println("\n--- Performing ONNX Mul ---")
      val resultMulPtr = alloc[CDouble](size)
      NativeOnnxOps.onnx_mul_matrix_double(
        ptrA, rows.toCSize, cols.toCSize,
        ptrB, rows.toCSize, cols.toCSize,
        resultMulPtr
      )
      val resultMulArray = Array.ofDim[Double](size)
      for (i <- 0 until size) resultMulArray(i) = !(resultMulPtr + i)
      printMatrix("Result of A * B", resultMulArray, rows, cols)

      println("\n--- Performing ONNX Cast (Double to Float) ---")
      val inputCastDouble = Array[CDouble](1.1, 2.2, 3.3, 4.4)
      val inputCastDoublePtr = alloc[CDouble](size)
      for (i <- 0 until size) !(inputCastDoublePtr + i) = inputCastDouble(i)

      val resultCastFloatPtr = alloc[CFloat](size)
      NativeOnnxOps.onnx_cast_double_to_float(
        inputCastDoublePtr, rows.toCSize, cols.toCSize,
        resultCastFloatPtr
      )
      val resultCastFloatArray = Array.ofDim[Float](size)
      for (i <- 0 until size) resultCastFloatArray(i) = !(resultCastFloatPtr + i)
      printMatrix("Input Double Matrix", inputCastDouble.map(_.toDouble), rows, cols)
      printFloatMatrix("Result Float Matrix", resultCastFloatArray, rows, cols)

      println("\n--- Performing ONNX Cast (Float to Double) ---")
      val inputCastFloat = Array[CFloat](10.1f, 20.2f, 30.3f, 40.4f)
      val inputCastFloatPtr = alloc[CFloat](size)
      for (i <- 0 until size) !(inputCastFloatPtr + i) = inputCastFloat(i)

      val resultCastDoublePtr = alloc[CDouble](size)
      NativeOnnxOps.onnx_cast_float_to_double(
        inputCastFloatPtr, rows.toCSize, cols.toCSize,
        resultCastDoublePtr
      )
      val resultCastDoubleArray = Array.ofDim[Double](size)
      for (i <- 0 until size) resultCastDoubleArray(i) = !(resultCastDoublePtr + i)
      printFloatMatrix("Input Float Matrix", inputCastFloat.map(_.toFloat), rows, cols)
      printMatrix("Result Double Matrix", resultCastDoubleArray, rows, cols)
    }
  }
}

