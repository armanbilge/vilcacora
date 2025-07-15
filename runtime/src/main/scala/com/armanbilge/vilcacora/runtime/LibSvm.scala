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

/** A safe, idiomatic Scala API for the LIBSVM C library. It provides type aliases, constants, and
  * convenient accessors for the C structs, while isolating the raw C function bindings.
  */
object LibSvm {

  // --- Type Aliases for LIBSVM C Structs ---
  // This is a concise way to define the memory layout of C structs in modern Scala Native.
  type svm_node = CStruct2[CInt, CDouble]
  type svm_parameter =
    CStruct5[CInt, CInt, CInt, CDouble, CDouble] // Only fields needed for prediction
  type svm_model = CStruct8[
    Ptr[svm_parameter],
    CInt, // nr_class
    CInt, // l
    Ptr[Ptr[svm_node]], // SV
    Ptr[Ptr[CDouble]], // sv_coef
    Ptr[CDouble], // rho
    Ptr[CInt], // label
    Ptr[CInt], // nSV
  ]

  // --- LIBSVM Constants ---
  // These are regular Scala vals, as they are part of our Scala API, not external C variables.
  val LINEAR: CInt = 0
  val POLY: CInt = 1
  val RBF: CInt = 2
  val SIGMOID: CInt = 3

  // --- Extension Methods for convenient, type-safe struct field access ---

  implicit class SvmNodeOps(val ptr: Ptr[svm_node]) extends AnyVal {
    def index: CInt = ptr._1; def index_=(v: CInt): Unit = ptr._1 = v
    def value: CDouble = ptr._2; def value_=(v: CDouble): Unit = ptr._2 = v
  }

  implicit class SvmParameterOps(val ptr: Ptr[svm_parameter]) extends AnyVal {
    def svm_type: CInt = ptr._1; def svm_type_=(v: CInt): Unit = ptr._1 = v
    def kernel_type: CInt = ptr._2; def kernel_type_=(v: CInt): Unit = ptr._2 = v
    def degree: CInt = ptr._3; def degree_=(v: CInt): Unit = ptr._3 = v
    def gamma: CDouble = ptr._4; def gamma_=(v: CDouble): Unit = ptr._4 = v
    def coef0: CDouble = ptr._5; def coef0_=(v: CDouble): Unit = ptr._5 = v
  }

  implicit class SvmModelOps(val ptr: Ptr[svm_model]) extends AnyVal {
    def param: Ptr[svm_parameter] = ptr._1; def param_=(v: Ptr[svm_parameter]): Unit = ptr._1 = v
    def nr_class: CInt = ptr._2; def nr_class_=(v: CInt): Unit = ptr._2 = v
    def l: CInt = ptr._3; def l_=(v: CInt): Unit = ptr._3 = v
    def SV: Ptr[Ptr[svm_node]] = ptr._4; def SV_=(v: Ptr[Ptr[svm_node]]): Unit = ptr._4 = v
    def sv_coef: Ptr[Ptr[CDouble]] = ptr._5; def sv_coef_=(v: Ptr[Ptr[CDouble]]): Unit = ptr._5 = v
    def rho: Ptr[CDouble] = ptr._6; def rho_=(v: Ptr[CDouble]): Unit = ptr._6 = v
    def label: Ptr[CInt] = ptr._7; def label_=(v: Ptr[CInt]): Unit = ptr._7 = v
    def nSV: Ptr[CInt] = ptr._8; def nSV_=(v: Ptr[CInt]): Unit = ptr._8 = v
  }

  // --- Public API function that delegates to the C binding ---

  /** Performs prediction and returns decision values (scores). */
  def svm_predict_values(
      model: Ptr[svm_model],
      x: Ptr[svm_node],
      dec_values: Ptr[CDouble],
  ): CDouble = extern_functions.svm_predict_values(model, x, dec_values)

  // --- Private, raw C bindings ---
  // This object is marked @extern and only contains the raw function stubs.
  @link("svm")
  @extern
  private object extern_functions {
    def svm_predict_values(
        model: Ptr[svm_model],
        x: Ptr[svm_node],
        dec_values: Ptr[CDouble],
    ): CDouble = extern
  }
}
