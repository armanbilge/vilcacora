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
  * MLPack uses column-major memory layout (Armadillo) and returns flattened outputs. All operations
  * are single-inference with no batching or threading.
  */
@linkCppRuntime
@extern
object MLPack {

  /** Result structure returned by MLPack operations */
  type OpResult = CStruct4[
    Ptr[CDouble], // data pointer to flattened output
    CSize, // height of output tensor
    CSize, // width of output tensor
    CSize, // channels of output tensor
  ]
  type FOpResult = CStruct4[
    Ptr[CFloat], // data pointer to flattened output
    CSize, // height of output tensor
    CSize, // width of output tensor
    CSize, // channels of output tensor
  ]

  /** Convolution parameters structure */
  type ConvolutionParameters = CStruct6[
    CSize, // num_output_maps
    CSize, // kernel_height
    CSize, // kernel_width
    CSize, // stride_height
    CSize, // stride_width
    CInt, // auto_pad (0=VALID, 1=SAME)
  ]

  /** Max pooling parameters structure */
  type MaxPoolingParameters = CStruct4[
    CSize, // kernel_height
    CSize, // kernel_width
    CSize, // stride_height
    CSize, // stride_width
  ]

  /** Create convolution parameters */
  def create_convolution_parameters(
      num_output_maps: CSize,
      kernel_height: CSize,
      kernel_width: CSize,
      stride_height: CSize,
      stride_width: CSize,
      auto_pad: CInt,
  ): Ptr[ConvolutionParameters] = extern

  /** Create max pooling parameters */
  def create_maxpooling_parameters(
      kernel_height: CSize,
      kernel_width: CSize,
      stride_height: CSize,
      stride_width: CSize,
  ): Ptr[MaxPoolingParameters] = extern

  /** Perform convolution operation for float 32
    *
    * @param params
    *   Convolution parameters
    * @param input_ptr
    *   Input data (row-major, will be converted internally)
    * @param input_height
    *   Height of input tensor
    * @param input_width
    *   Width of input tensor
    * @param input_channels
    *   Number of input channels
    * @param kernel_ptr
    *   Kernel weights (OIHW format, will be converted to column-major)
    * @param bias_ptr
    *   Bias data (can be null)
    * @param use_bias
    *   Whether to use bias
    * @return
    *   FOpResult with flattened output data and dimensions
    */
  def F_perform_convolution(
      params: Ptr[ConvolutionParameters],
      input_ptr: Ptr[CFloat],
      input_height: CSize,
      input_width: CSize,
      input_channels: CSize,
      kernel_ptr: Ptr[CFloat],
      bias_ptr: Ptr[CFloat],
      use_bias: CInt,
  ): Ptr[FOpResult] = extern

  /** Perform convolution operation for float 64
    *
    * @param params
    *   Convolution parameters
    * @param input_ptr
    *   Input data (row-major, will be converted internally)
    * @param input_height
    *   Height of input tensor
    * @param input_width
    *   Width of input tensor
    * @param input_channels
    *   Number of input channels
    * @param kernel_ptr
    *   Kernel weights (OIHW format, will be converted to column-major)
    * @param bias_ptr
    *   Bias data (can be null)
    * @param use_bias
    *   Whether to use bias
    * @return
    *   OpResult with flattened output data and dimensions
    */
  def perform_convolution(
      params: Ptr[ConvolutionParameters],
      input_ptr: Ptr[CDouble],
      input_height: CSize,
      input_width: CSize,
      input_channels: CSize,
      kernel_ptr: Ptr[CDouble],
      bias_ptr: Ptr[CDouble],
      use_bias: CInt,
  ): Ptr[OpResult] = extern

  /** Perform max pooling operation for float 32
    *
    * @param params
    *   Max pooling parameters
    * @param input_ptr
    *   Input data (row-major, will be converted internally)
    * @param input_height
    *   Height of input tensor
    * @param input_width
    *   Width of input tensor
    * @param input_channels
    *   Number of input channels
    * @return
    *   FOpResult with flattened output data and dimensions
    */
  def F_perform_maxpooling(
      params: Ptr[MaxPoolingParameters],
      input_ptr: Ptr[CFloat],
      input_height: CSize,
      input_width: CSize,
      input_channels: CSize,
  ): Ptr[FOpResult] = extern

  /** Perform max pooling operation for float 64
    *
    * @param params
    *   Max pooling parameters
    * @param input_ptr
    *   Input data (row-major, will be converted internally)
    * @param input_height
    *   Height of input tensor
    * @param input_width
    *   Width of input tensor
    * @param input_channels
    *   Number of input channels
    * @return
    *   OpResult with flattened output data and dimensions
    */
  def perform_maxpooling(
      params: Ptr[MaxPoolingParameters],
      input_ptr: Ptr[CDouble],
      input_height: CSize,
      input_width: CSize,
      input_channels: CSize,
  ): Ptr[OpResult] = extern

  /** Free OpResult memory allocated by MLPack */
  def free_op_result(result: Ptr[OpResult]): Unit = extern

  /** Free FOpResult memory allocated by MLPack */
  def free_Fop_result(result: Ptr[FOpResult]): Unit = extern

  /** Free convolution parameters */
  def free_convolution_parameters(params: Ptr[ConvolutionParameters]): Unit = extern

  /** Free max pooling parameters */
  def free_maxpooling_parameters(params: Ptr[MaxPoolingParameters]): Unit = extern

  /** Calculate output size for convolution/pooling operations */
  def calculate_output_size(
      input_size: CSize,
      kernel_size: CSize,
      stride: CSize,
      pad_before: CSize,
      pad_after: CSize,
  ): CSize = extern

  /** Helper function to convert row-major input to column-major for MLPack
    *
    * @param input_ptr
    *   Row-major input data
    * @param height
    *   Height of tensor
    * @param width
    *   Width of tensor
    * @param channels
    *   Number of channels
    * @return
    *   Pointer to column-major converted data (caller must free)
    */
  def convert_row_to_column_major(
      input_ptr: Ptr[CDouble],
      height: CSize,
      width: CSize,
      channels: CSize,
  ): Ptr[CDouble] = extern

  /** Helper function to convert row-major input to column-major for MLPack (float 32)
    *
    * @param input_ptr
    *   Row-major input data
    * @param height
    *   Height of tensor
    * @param width
    *   Width of tensor
    * @param channels
    *   Number of channels
    * @return
    *   Pointer to column-major converted data (caller must free)
    */
  def F_convert_row_to_column_major(
      input_ptr: Ptr[CFloat],
      height: CSize,
      width: CSize,
      channels: CSize,
  ): Ptr[CFloat] = extern

  /** Helper function to convert column-major output back to row-major
    *
    * @param output_ptr
    *   Column-major output data from MLPack
    * @param height
    *   Height of tensor
    * @param width
    *   Width of tensor
    * @param channels
    *   Number of channels
    * @return
    *   Pointer to row-major converted data (caller must free)
    */
  def convert_column_to_row_major(
      output_ptr: Ptr[CDouble],
      height: CSize,
      width: CSize,
      channels: CSize,
  ): Ptr[CDouble] = extern

  /** Helper function to convert column-major output back to row-major (float 32)
    *
    * @param output_ptr
    *   Column-major output data from MLPack
    * @param height
    *   Height of tensor
    * @param width
    *   Width of tensor
    * @param channels
    *   Number of channels
    * @return
    *   Pointer to row-major converted data (caller must free)
    */

  def F_convert_column_to_row_major(
      output_ptr: Ptr[CFloat],
      height: CSize,
      width: CSize,
      channels: CSize,
  ): Ptr[CFloat] = extern

  /** Helper function to convert OIHW kernel weights to MLPack column-major format
    *
    * @param kernel_ptr
    *   OIHW format kernel weights
    * @param output_channels
    *   Number of output channels
    * @param input_channels
    *   Number of input channels
    * @param kernel_height
    *   Height of kernel
    * @param kernel_width
    *   Width of kernel
    * @return
    *   Pointer to converted weights (caller must free)
    */
  def convert_oihw_to_mlpack_format(
      kernel_ptr: Ptr[CDouble],
      output_channels: CSize,
      input_channels: CSize,
      kernel_height: CSize,
      kernel_width: CSize,
  ): Ptr[CDouble] = extern

  /** Helper function to convert OIHW kernel weights to MLPack column-major format for float 32
    *
    * @param kernel_ptr
    *   OIHW format kernel weights
    * @param output_channels
    *   Number of output channels
    * @param input_channels
    *   Number of input channels
    * @param kernel_height
    *   Height of kernel
    * @param kernel_width
    *   Width of kernel
    * @return
    *   Pointer to converted weights (caller must free)
    */
  def F_convert_oihw_to_mlpack_format(
      kernel_ptr: Ptr[CFloat],
      output_channels: CSize,
      input_channels: CSize,
      kernel_height: CSize,
      kernel_width: CSize,
  ): Ptr[CFloat] = extern

  /** Perform softmax operation for float 64
    *
    * @param input_ptr
    *   Input data
    * @param input_height
    *   Height of input tensor
    * @param input_width
    *   Width of input tensor
    * @param input_channels
    *   Number of input channels
    * @return
    *   OpResult with softmax output data and same dimensions
    */
  def perform_softmax(
      input_ptr: Ptr[CDouble],
      input_height: CSize,
      input_width: CSize,
      input_channels: CSize,
  ): Ptr[OpResult] = extern

  /** Perform softmax operation for float 32
    *
    * @param input_ptr
    *   Input data
    * @param input_height
    *   Height of input tensor
    * @param input_width
    *   Width of input tensor
    * @param input_channels
    *   Number of input channels
    * @return
    *   FOpResult with softmax output data and same dimensions
    */
  def F_perform_softmax(
      input_ptr: Ptr[CFloat],
      input_height: CSize,
      input_width: CSize,
      input_channels: CSize,
  ): Ptr[FOpResult] = extern

  /** Free memory allocated by conversion functions */
  def free_converted_memory(ptr: Ptr[CDouble]): Unit = extern

  def F_free_converted_memory(ptr: Ptr[CFloat]): Unit = extern // float32
}
