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

package com.armanbilge.vilcacora.ir

/** Represents the data type of a tensor's elements and its size in bytes.
  */
sealed abstract class DataType { def sizeInBytes: Int }
object DataType {
  case object Float32 extends DataType { override def sizeInBytes: Int = 4 }
  case object Float64 extends DataType { override def sizeInBytes: Int = 8 } // For ONNX DOUBLE
  case object Float16 extends DataType { override def sizeInBytes: Int = 2 }
  case object BFloat16 extends DataType { override def sizeInBytes: Int = 2 }

  case object Int64 extends DataType { override def sizeInBytes: Int = 8 }
  case object Int32 extends DataType { override def sizeInBytes: Int = 4 }
  case object Int16 extends DataType { override def sizeInBytes: Int = 2 }
  case object Int8 extends DataType { override def sizeInBytes: Int = 1 }

  case object UInt64 extends DataType { override def sizeInBytes: Int = 8 }
  case object UInt32 extends DataType { override def sizeInBytes: Int = 4 }
  case object UInt16 extends DataType { override def sizeInBytes: Int = 2 }
  case object UInt8 extends DataType { override def sizeInBytes: Int = 1 }

  case object Bool extends DataType { override def sizeInBytes: Int = 1 }
  // Add other supported types  from ONNX
}

/** Describes a single, named memory buffer (a tensor).
  */
final case class Allocation(
    name: String,
    dataType: DataType,
    shape: List[Int],
    initialData: Option[Array[Byte]] = None,
)

/** representation of the `kernel_type` attribute for SVM nodes.
  */
sealed trait SVMKernel
object SVMKernel {
  case object LINEAR extends SVMKernel
  case object POLY extends SVMKernel
  case object RBF extends SVMKernel
  case object SIGMOID extends SVMKernel

  def fromString(s: String): Either[String, SVMKernel] = s.toUpperCase match {
    case "LINEAR" => Right(LINEAR)
    case "POLY" => Right(POLY)
    case "RBF" => Right(RBF)
    case "SIGMOID" => Right(SIGMOID)
    case other => Left(s"Unsupported SVM Kernel: $other")
  }
}

/** Type-safe representation of the `post_transform` attribute for SVM nodes. */
sealed trait PostTransform
object PostTransform {
  case object NONE extends PostTransform
  case object SOFTMAX extends PostTransform
  case object LOGISTIC extends PostTransform
  case object SOFTMAX_ZERO extends PostTransform
  case object PROBIT extends PostTransform

  def fromString(s: String): Either[String, PostTransform] = s.toUpperCase match {
    case "NONE" => Right(NONE)
    case "SOFTMAX" => Right(SOFTMAX)
    case "LOGISTIC" => Right(LOGISTIC)
    case "SOFTMAX_ZERO" => Right(SOFTMAX_ZERO)
    case "PROBIT" => Right(PROBIT)
    case other => Left(s"Unsupported PostTransform: $other")
  }
}

/** An ADT representing a single operation in the computation graph.
  */
sealed abstract class Operation {
  def inputs: Seq[String]
  def outputs: Seq[String]
}

object Operation {
  final case class MatMul(inputA: String, inputB: String, output: String) extends Operation {
    override def inputs: Seq[String] = Seq(inputA, inputB)
    override def outputs: Seq[String] = Seq(output)
  }

  final case class Add(inputA: String, inputB: String, output: String) extends Operation {
    override def inputs: Seq[String] = Seq(inputA, inputB)
    override def outputs: Seq[String] = Seq(output)
  }

  /** Represents an `SVMClassifier` operation. All configuration attributes are stored as fields in
    * the case class.
    */
  final case class SVMClassifier(
      input: String,
      outputLabel: String,
      outputScores: String,
      // --- Attributes ---
      classLabels: Seq[Long],
      coefficients: Seq[Float],
      kernelType: SVMKernel,
      kernelParams: Seq[Float],
      postTransform: PostTransform,
      rho: Seq[Float],
      supportVectors: Seq[Float],
      vectorsPerClass: Seq[Long],
  ) extends Operation {
    override def inputs: Seq[String] = Seq(input)
    override def outputs: Seq[String] = Seq(outputLabel, outputScores)
  }

  /** Represents an `SVMRegressor` operation. All configuration attributes are stored as fields in
    * the case class.
    */
  final case class SVMRegressor(
      input: String,
      output: String,
      // --- Attributes ---
      coefficients: Seq[Float],
      kernelParams: Seq[Float],
      kernelType: SVMKernel,
      nSupports: Long,
      oneClass: Boolean,
      postTransform: PostTransform,
      rho: Seq[Float],
      supportVectors: Seq[Float],
  ) extends Operation {
    override def inputs: Seq[String] = Seq(input)
    override def outputs: Seq[String] = Seq(output)
  }
  // Add more operations here...
}

/** The top-level container for the entire parsed and translated model.
  */
case class ModelIR(
    name: String,
    operations: Seq[Operation],
    allocations: Map[String, Allocation],
    graphInputs: Seq[String],
    graphOutputs: Seq[String],
)
