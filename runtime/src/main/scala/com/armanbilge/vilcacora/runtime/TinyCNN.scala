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

/** Scala Native bindings for tiny-cnn C++ wrapper functions */

@linkCppRuntime
@extern
object TinyCNN {

  // Single inference convolution - NO BATCHING
  def conv2d_single_inference(
      input_data: Ptr[CFloat], // Single input tensor
      weights: Ptr[CFloat], // Convolution weights
      bias: Ptr[CFloat], // Bias values
      output: Ptr[CFloat], // Output buffer
      // Input dimensions (single sample)
      input_height: CInt,
      input_width: CInt,
      input_channels: CInt,
      // Convolution parameters
      kernel_height: CInt,
      kernel_width: CInt,
      output_channels: CInt,
      stride_h: CInt,
      stride_w: CInt,
      pad_h: CInt,
      pad_w: CInt,
  ): CInt = extern

  // Single inference max pooling - NO BATCHING
  def maxpool2d_single_inference(
      input_data: Ptr[CFloat], // Single input tensor
      output: Ptr[CFloat], // Output buffer
      // Input dimensions (single sample)
      input_height: CInt,
      input_width: CInt,
      channels: CInt,
      // Pooling parameters
      kernel_height: CInt,
      kernel_width: CInt,
      stride_h: CInt,
      stride_w: CInt,
      pad_h: CInt,
      pad_w: CInt,
  ): CInt = extern

  // Utility functions
  def calculate_output_size(
      input_size: CInt,
      kernel_size: CInt,
      stride: CInt,
      padding: CInt,
  ): CInt = extern
}
