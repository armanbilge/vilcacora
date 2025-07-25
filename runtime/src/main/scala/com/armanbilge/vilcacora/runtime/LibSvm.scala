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

/** A safe, idiomatic Scala API for the svm_wrapper.cpp
  */

// Scala Native bindings for the C wrapper functions
@link("svm")
@extern
object LibSVM {
  // Model creation
  def create_svm_param(
      svm_type: CInt,
      kernel_type: CInt,
      degree: CInt,
      gamma: CDouble,
      coef0: CDouble,
  ): Ptr[Byte] = extern

  def create_svm_model(
      param: Ptr[Byte],
      nr_class: CInt,
      l: CInt,
      support_vectors: Ptr[CDouble],
      num_features: CInt,
      coefficients: Ptr[CDouble],
      rho: Ptr[CDouble],
      class_labels: Ptr[CInt],
      n_sv_per_class: Ptr[CInt],
  ): Ptr[Byte] = extern

  // Prediction
  def svm_predict_with_scores(
      model: Ptr[Byte],
      features: Ptr[CDouble],
      num_features: CInt,
      class_scores: Ptr[CDouble],
  ): CInt = extern

  // Debug function
  def debug_model_info(model: Ptr[Byte]): Unit = extern

  // Use LibSVM's native cleanup function
  def svm_free_and_destroy_model(model_ptr_ptr: Ptr[Ptr[Byte]]): Unit = extern
}
