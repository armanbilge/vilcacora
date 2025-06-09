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
sealed trait DataType { def sizeInBytes: Int }
object DataType {
  case object Float32 extends DataType { override def sizeInBytes: Int = 4 }
  case object Int64 extends DataType { override def sizeInBytes: Int = 8 }
  case object Int32 extends DataType { override def sizeInBytes: Int = 4 }
  case object UInt8 extends DataType { override def sizeInBytes: Int = 1 }
  // Add other supported types  from ONNX
}

/** Describes a single, named memory buffer (a tensor).
  */
case class Allocation(
    name: String,
    dataType: DataType,
    shape: Seq[Int],
    initialData: Option[Array[Byte]] = None,
)

/** An ADT representing a single operation in the computation graph.
  */
sealed trait Operation {
  def inputs: Seq[String]
  def outputs: Seq[String]
}

object Operation {
  case class MatMul(inputA: String, inputB: String, output: String) extends Operation {
    override def inputs: Seq[String] = Seq(inputA, inputB)
    override def outputs: Seq[String] = Seq(output)
  }

  case class Add(inputA: String, inputB: String, output: String) extends Operation {
    override def inputs: Seq[String] = Seq(inputA, inputB)
    override def outputs: Seq[String] = Seq(output)
  }
  // Add other operations here...
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
