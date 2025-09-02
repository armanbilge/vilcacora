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

import scala.scalanative.unsafe._

/** CBLAS constants - defined outside the extern object */
object BLASConstants {
  // CBLAS layout options
  final val CblasRowMajor: CInt = 101
  final val CblasColMajor: CInt = 102

  // CBLAS transpose options
  final val CblasNoTrans: CInt = 111
  final val CblasTrans: CInt = 112
  final val CblasConjTrans: CInt = 113
}

/** Scala Native bindings for OpenBLAS CBLAS functions */
@link("openblas")
@extern
object BLAS {
  // Double precision matrix multiplication
  def cblas_dgemm(
      layout: CInt,
      transA: CInt,
      transB: CInt,
      M: CInt,
      N: CInt,
      K: CInt,
      alpha: CDouble,
      A: Ptr[CDouble],
      lda: CInt,
      B: Ptr[CDouble],
      ldb: CInt,
      beta: CDouble,
      C: Ptr[CDouble],
      ldc: CInt,
  ): Unit = extern

  // Single precision matrix multiplication
  def cblas_sgemm(
      layout: CInt,
      transA: CInt,
      transB: CInt,
      M: CInt,
      N: CInt,
      K: CInt,
      alpha: CFloat,
      A: Ptr[CFloat],
      lda: CInt,
      B: Ptr[CFloat],
      ldb: CInt,
      beta: CFloat,
      C: Ptr[CFloat],
      ldc: CInt,
  ): Unit = extern
}
