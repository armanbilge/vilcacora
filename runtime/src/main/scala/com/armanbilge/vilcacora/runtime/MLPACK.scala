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

/** Scala Native bindings for MLPack C++ wrapper functions
  *
  * All operations are single-inference with no batching or threading.
  */
@linkCppRuntime
@extern
object MLPack {

  /** Direct convolution - no struct overhead, writes to pre-allocated output */
  def F_perform_convolution_direct(
      // Parameters passed directly
      num_output_maps: CSize,
      kernel_height: CSize,
      kernel_width: CSize,
      stride_height: CSize,
      stride_width: CSize,
      auto_pad: CInt,
      use_bias: CInt,

      // Data pointers
      input_ptr: Ptr[CFloat],
      input_height: CSize,
      input_width: CSize,
      input_channels: CSize,
      kernel_ptr: Ptr[CFloat],
      bias_ptr: Ptr[CFloat],

      // Pre-allocated output + dimension outputs
      output_ptr: Ptr[CFloat],
      output_height: Ptr[CSize],
      output_width: Ptr[CSize],
      output_channels: Ptr[CSize],
  ): Unit = extern
  def perform_convolution_direct(
      // Parameters passed directly
      num_output_maps: CSize,
      kernel_height: CSize,
      kernel_width: CSize,
      stride_height: CSize,
      stride_width: CSize,
      auto_pad: CInt,
      use_bias: CInt,
      // Data pointers
      input_ptr: Ptr[Double],
      input_height: CSize,
      input_width: CSize,
      input_channels: CSize,
      kernel_ptr: Ptr[Double],
      bias_ptr: Ptr[Double],

      // Pre-allocated output + dimension outputs
      output_ptr: Ptr[Double],
      output_height: Ptr[CSize],
      output_width: Ptr[CSize],
      output_channels: Ptr[CSize],
  ): Unit = extern

  def F_perform_maxpooling_direct(
      kernel_height: CSize,
      kernel_width: CSize,
      stride_height: CSize,
      stride_width: CSize,
      input_ptr: Ptr[CFloat],
      input_height: CSize,
      input_width: CSize,
      input_channels: CSize,
      output_ptr: Ptr[CFloat],
      output_height: Ptr[CSize],
      output_width: Ptr[CSize],
      output_channels: Ptr[CSize],
  ): Unit = extern
  def perform_maxpooling_direct(
      kernel_height: CSize,
      kernel_width: CSize,
      stride_height: CSize,
      stride_width: CSize,
      input_ptr: Ptr[Double],
      input_height: CSize,
      input_width: CSize,
      input_channels: CSize,
      output_ptr: Ptr[Double],
      output_height: Ptr[CSize],
      output_width: Ptr[CSize],
      output_channels: Ptr[CSize],
  ): Unit = extern
  def F_perform_softmax_direct(
      input_ptr: Ptr[CFloat],
      input_size: CSize,
      output_ptr: Ptr[CFloat], // Same size as input
  ): Unit = extern
  def perform_softmax_direct(
      input_ptr: Ptr[Double],
      input_size: CSize,
      output_ptr: Ptr[Double], // Same size as input
  ): Unit = extern
}
