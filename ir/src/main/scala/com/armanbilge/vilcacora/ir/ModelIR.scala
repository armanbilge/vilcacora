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
sealed abstract class SVMKernel
object SVMKernel {
  case object Linear extends SVMKernel
  case object Poly extends SVMKernel
  case object Rbf extends SVMKernel
  case object Sigmoid extends SVMKernel

  def fromString(s: String): Either[String, SVMKernel] = s.toUpperCase match {
    case "LINEAR" => Right(Linear)
    case "POLY" => Right(Poly)
    case "RBF" => Right(Rbf)
    case "SIGMOID" => Right(Sigmoid)
    case other => Left(s"Unsupported SVM Kernel: $other")
  }
}

/** Type-safe representation of the `post_transform` attribute for SVM nodes. */
sealed abstract class PostTransform
object PostTransform {
  case object None extends PostTransform
  case object Softmax extends PostTransform
  case object Logistic extends PostTransform
  case object Softmax_Zero extends PostTransform
  case object Probit extends PostTransform

  def fromString(s: String): Either[String, PostTransform] = s.toUpperCase match {
    case "NONE" => Right(None)
    case "SOFTMAX" => Right(Softmax)
    case "LOGISTIC" => Right(Logistic)
    case "SOFTMAX_ZERO" => Right(Softmax_Zero)
    case "PROBIT" => Right(Probit)
    case other => Left(s"Unsupported PostTransform: $other")
  }
}

/** An ADT representing a single operation in the computation graph.
  */
sealed abstract class Operation {
  def inputs: List[String]
  def outputs: List[String]
}

object Operation {
  final case class MatMul(inputA: String, inputB: String, output: String) extends Operation {
    override def inputs: List[String] = List(inputA, inputB)
    override def outputs: List[String] = List(output)
  }

  final case class Add(inputA: String, inputB: String, output: String) extends Operation {
    override def inputs: List[String] = List(inputA, inputB)
    override def outputs: List[String] = List(output)
  }

  // Operation for element wise multiplication
  final case class Mul(inputA: String, inputB: String, output: String) extends Operation {
    override def inputs: List[String] = List(inputA, inputB)
    override def outputs: List[String] = List(output)
  }

  // Operation for changing a tensor's data type
  final case class Cast(input: String, output: String, to: DataType) extends Operation {
    override def inputs: List[String] = List(input)
    override def outputs: List[String] = List(output)
  }

  /** Represents an `SVMClassifier` operation. All configuration attributes are stored as fields in
    * the case class.
    */
  final case class SVMClassifier(
      input: String,
      outputLabel: String,
      outputScores: String,
      // --- Attributes ---
      classLabels: List[Long],
      coefficients: Array[Double],
      kernelType: SVMKernel,
      kernelParams: List[Double],
      postTransform: PostTransform,
      rho: List[Double],
      supportVectors: Array[Double],
      vectorsPerClass: List[Long],
  ) extends Operation {
    override def inputs: List[String] = List(input)
    override def outputs: List[String] = List(outputLabel, outputScores)
  }

  /** Represents an `SVMRegressor` operation. All configuration attributes are stored as fields in
    * the case class.
    */
  final case class SVMRegressor(
      input: String,
      output: String,
      // --- Attributes ---
      coefficients: Array[Double],
      kernelParams: List[Double],
      kernelType: SVMKernel,
      nSupports: Long,
      oneClass: Boolean,
      postTransform: PostTransform,
      rho: List[Double],
      supportVectors: Array[Double],
  ) extends Operation {
    override def inputs: List[String] = List(input)
    override def outputs: List[String] = List(output)
  }
  // Add more operations here...
}

/** The top-level container for the entire parsed and translated model.
  */
case class ModelIR(
    name: String,
    operations: List[Operation],
    allocations: Map[String, Allocation],
    graphInputs: List[String],
    graphOutputs: List[String],
)
