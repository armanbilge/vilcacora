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

import cats.effect.{IO, Resource}
import cats.syntax.all._
import com.armanbilge.vilcacora.ir._
import com.armanbilge.vilcacora.runtime.LibSvm._
import scala.scalanative.unsafe._
import scala.scalanative.libc.stdlib
import scala.scalanative.libc.string.memcpy
import scala.scalanative.unsigned._
import scala.collection.mutable.ListBuffer

/** The core execution engine for a translated `ModelIR`. It manages native memory and executes
  * model operations within a Cats Effect IO context.
  */
object Interpreter {

  /** A type alias mapping a tensor's name to its pointer in native memory. */
  type MemoryMap = Map[String, Ptr[Byte]]

  /** Executes a complete ModelIR graph.
    *
    * The process is separated into two stages:
    *   1. A synchronous validation of the model to fail-fast on unsupported operations. 2. An
    *      asynchronous, resource-safe execution of the graph operations within an IO context.
    *
    * @param model
    *   The intermediate representation of the model to execute.
    * @param inputs
    *   A map of input tensor names to their corresponding Scala arrays.
    * @return
    *   An IO containing a map of output tensor names to their resulting Scala arrays.
    */
  def execute(model: ModelIR, inputs: Map[String, Array[_]]): IO[Map[String, Array[_]]] = {
    validateModel(model)

    val outputArrays = createOutputArrays(model)
    memoryResource(model, inputs, outputArrays)
      .use { memoryMap =>
        model.operations.traverse_ { op =>
          executeOperation(op, memoryMap, model) <* IO.cede
        }
      }
      .as(outputArrays)
  }

  /** Synchronously validates the model definition, throwing a `NotImplementedError` if any
    * operation or data type cast is not supported. This ensures the interpreter fails before any
    * memory is allocated or side effects are scheduled.
    */
  private def validateModel(model: ModelIR): Unit =
    model.operations.foreach {
      case _: Operation.SVMClassifier | _: Operation.Add | _: Operation.Mul =>
        () // Supported
      case op: Operation.Cast =>
        val from = model.allocations(op.input).dataType
        val to = model.allocations(op.output).dataType
        (from, to) match {
          case (f, t) if f == t => ()
          case (DataType.Float64, DataType.Float32) => ()
          case (DataType.Float32, DataType.Float64) => ()
          case (from, to) =>
            throw new NotImplementedError(s"Cast from $from to $to is not implemented.")
        }
      case other =>
        throw new NotImplementedError(s"Operation not implemented: ${other.getClass.getSimpleName}")
    }

  /** A `Resource` that manages all memory for the graph execution.
    *   - Input and output tensors get direct pointers to the memory of their Scala arrays
    *     (zero-copy).
    *   - Intermediate and constant tensors are allocated in native memory using `malloc`. The
    *     `Resource` guarantees that all `malloc`'d memory is freed after execution.
    */
  private def memoryResource(
      model: ModelIR,
      inputs: Map[String, Array[_]],
      outputs: Map[String, Array[_]],
  ): Resource[IO, MemoryMap] = {
    val acquire = IO {
      val mallocedPtrs = ListBuffer.empty[Ptr[Byte]]
      val memoryMap = model.allocations.map { case (name, allocation) =>
        val ptr: Ptr[Byte] =
          if (inputs.contains(name)) {
            inputs(name).at(0).asInstanceOf[Ptr[Byte]]
          } else if (outputs.contains(name)) {
            outputs(name).at(0).asInstanceOf[Ptr[Byte]]
          } else {
            val totalBytes = (allocation.shape.product * allocation.dataType.sizeInBytes).toUSize
            val p = stdlib.malloc(totalBytes)
            if (p == null) throw new OutOfMemoryError(s"Failed to allocate tensor '$name'")

            allocation.initialData.foreach(data =>
              memcpy(p, data.at(0).asInstanceOf[Ptr[Byte]], data.length.toUSize),
            )
            mallocedPtrs += p
            p
          }
        name -> ptr
      }
      (memoryMap.toMap, mallocedPtrs.toList)
    }

    Resource
      .make(acquire) { case (_, ptrsToFree) =>
        IO(ptrsToFree.foreach(stdlib.free))
      }
      .map(_._1)
  }

  /** Dispatches a single operation to its corresponding handler function. */
  private def executeOperation(op: Operation, memory: MemoryMap, model: ModelIR): IO[Unit] =
    op match {
      case op: Operation.SVMClassifier => handleSvmClassifier(op, memory, model)
      case op: Operation.Add => handleAdd(op, memory, model)
      case op: Operation.Mul => handleMul(op, memory, model)
      case op: Operation.Cast => handleCast(op, memory, model)
      case other =>
        // This case is unreachable due to the pre-validation step.
        // It remains as a safeguard against internal logic errors.
        throw new NotImplementedError(s"Operation not implemented: ${other.getClass.getSimpleName}")
    }

  /** Handles element-wise addition for float tensors. */
  private def handleAdd(op: Operation.Add, memory: MemoryMap, model: ModelIR): IO[Unit] = IO {
    val count = model.allocations(op.outputs.head).shape.product
    val inputA = memory(op.inputs(0)).asInstanceOf[Ptr[CFloat]]
    val inputB = memory(op.inputs(1)).asInstanceOf[Ptr[CFloat]]
    val output = memory(op.outputs.head).asInstanceOf[Ptr[CFloat]]

    var i = 0
    while (i < count) {
      !(output + i) = !(inputA + i) + !(inputB + i)
      i += 1
    }
  }

  /** Handles element-wise multiplication for float tensors. */
  private def handleMul(op: Operation.Mul, memory: MemoryMap, model: ModelIR): IO[Unit] = IO {
    val count = model.allocations(op.outputs.head).shape.product
    val inputA = memory(op.inputs(0)).asInstanceOf[Ptr[CFloat]]
    val inputB = memory(op.inputs(1)).asInstanceOf[Ptr[CFloat]]
    val output = memory(op.outputs.head).asInstanceOf[Ptr[CFloat]]

    var i = 0
    while (i < count) {
      !(output + i) = !(inputA + i) * !(inputB + i)
      i += 1
    }
  }

  /** Handles casting between supported data types (Float32 <-> Float64). */
  private def handleCast(op: Operation.Cast, memory: MemoryMap, model: ModelIR): IO[Unit] = IO {
    val inputAlloc = model.allocations(op.input)
    val outputAlloc = model.allocations(op.output)
    val count = inputAlloc.shape.product
    val inputPtr = memory(op.input)
    val outputPtr = memory(op.output)

    (inputAlloc.dataType, outputAlloc.dataType) match {
      case (from, to) if from == to =>
        ()

      case (DataType.Float64, DataType.Float32) =>
        val in = inputPtr.asInstanceOf[Ptr[CDouble]]
        val out = outputPtr.asInstanceOf[Ptr[CFloat]]
        var i = 0
        while (i < count) {
          !(out + i) = (!(in + i)).toFloat
          i += 1
        }

      case (DataType.Float32, DataType.Float64) =>
        val in = inputPtr.asInstanceOf[Ptr[CFloat]]
        val out = outputPtr.asInstanceOf[Ptr[CDouble]]
        var i = 0
        while (i < count) {
          !(out + i) = (!(in + i)).toDouble
          i += 1
        }

      case (from, to) =>
        // Unreachable due to pre-validation.
        throw new IllegalStateException(s"Unvalidated cast from $from to $to encountered.")
    }
  }

  /** Handles the SVMClassifier operation by constructing a native LibSvm model, performing the
    * prediction, and writing the results to the output tensors.
    */
  private def handleSvmClassifier(
      op: Operation.SVMClassifier,
      memory: MemoryMap,
      model: ModelIR,
  ): IO[Unit] = {
    val numFeatures = model.allocations(op.input).shape.last

    val svmResources = for {
      modelPtr <- buildSvmModelResource(op, numFeatures)
      // Allocate native memory for the SVM input vector.
      svmInputNode <- malloc[svm_node](sizeof[svm_node] * (numFeatures + 1).toUSize)
    } yield (modelPtr, svmInputNode)

    svmResources.use { case (modelPtr, svmInputNode) =>
      IO {
        val inputPtr = memory(op.input).asInstanceOf[Ptr[CDouble]]

        // Populate the svm_node array with feature data from the input tensor.
        var i = 0
        while (i < numFeatures) {
          val node = svmInputNode + i
          node.index = i + 1 // LibSvm is 1-based.
          node.value = !(inputPtr + i)
          i += 1
        }
        (svmInputNode + numFeatures).index = -1 // Add the terminator node.

        // Predict and copy results back to the output tensors.
        val nrClass = op.classLabels.size
        val decValuesCount = nrClass * (nrClass - 1) / 2
        val decisionValuesPtr = stackalloc[CDouble](decValuesCount.toUInt)
        val predictedLabel = svm_predict_values(modelPtr, svmInputNode, decisionValuesPtr)

        !memory(op.outputLabel).asInstanceOf[Ptr[CInt]] = predictedLabel.toInt
        memcpy(
          memory(op.outputScores).asInstanceOf[Ptr[CDouble]],
          decisionValuesPtr.asInstanceOf[Ptr[Byte]],
          decValuesCount.toUSize * sizeof[CDouble],
        )
        ()
      }
    }
  }

  /** A resource-safe helper for `stdlib.malloc`. Throws `OutOfMemoryError` on allocation failure.
    */
  private def malloc[T](size: CSize): Resource[IO, Ptr[T]] =
    Resource
      .make(IO {
        val p = stdlib.malloc(size)
        if (p == null) throw new OutOfMemoryError(s"Failed to allocate $size bytes")
        p
      })(ptr => IO(stdlib.free(ptr)))
      .map(_.asInstanceOf[Ptr[T]])

  /** Constructs a native `svm_model` struct from the model's IR definition. All native memory
    * required for the struct (for params, coefficients, vectors, etc.) is allocated and managed
    * within a single `Resource` scope to ensure it is all safely deallocated after use.
    */
  private def buildSvmModelResource(
      op: Operation.SVMClassifier,
      numFeatures: Int,
  ): Resource[IO, Ptr[svm_model]] = {
    val nrClass = op.classLabels.size
    val numSupportVectors = op.vectorsPerClass.sum.toInt

    // A single resource that acquires all necessary native memory for the SVM model struct.
    val allAllocs = (
      malloc[svm_model](sizeof[svm_model]),
      malloc[svm_parameter](sizeof[svm_parameter]),
      malloc[CInt](sizeof[CInt] * nrClass.toUSize),
      malloc[CDouble](sizeof[CDouble] * op.rho.size.toUSize),
      malloc[CInt](sizeof[CInt] * nrClass.toUSize),
      malloc[Ptr[svm_node]](sizeof[Ptr[svm_node]] * numSupportVectors.toUSize),
      malloc[Ptr[CDouble]](sizeof[Ptr[CDouble]] * (nrClass - 1).toUSize),
      (0 until numSupportVectors).toList
        .traverse(_ => malloc[svm_node](sizeof[svm_node] * (numFeatures + 1).toUSize)),
      (0 until nrClass - 1).toList
        .traverse(_ => malloc[CDouble](sizeof[CDouble] * numSupportVectors.toUSize)),
    ).tupled

    allAllocs.map { case (model, param, label, rho, nSV, svs, svCoefs, svRows, coefRows) =>
      // Populate svm_parameter
      param.svm_type = 0 // SVM_TYPE_C_SVC
      param.kernel_type = op.kernelType match {
        case SVMKernel.Linear => LINEAR
        case SVMKernel.Poly => POLY
        case SVMKernel.Rbf => RBF
        case SVMKernel.Sigmoid => SIGMOID
      }
      param.gamma = op.kernelParams.headOption.getOrElse(0.0)
      param.coef0 = op.kernelParams.drop(1).headOption.getOrElse(0.0)
      param.degree = op.kernelParams.drop(2).headOption.map(_.toInt).getOrElse(3)

      // Populate arrays with model data
      var i = 0
      while (i < op.classLabels.length) { !(label + i) = op.classLabels(i).toInt; i += 1 }
      i = 0
      while (i < op.rho.length) { !(rho + i) = op.rho(i); i += 1 }
      i = 0
      while (i < op.vectorsPerClass.length) { !(nSV + i) = op.vectorsPerClass(i).toInt; i += 1 }

      // Populate the support vector nodes
      i = 0
      while (i < svRows.length) {
        val svRowPtr = svRows(i)
        !(svs + i) = svRowPtr
        var j = 0
        while (j < numFeatures) {
          val node = svRowPtr + j
          node.index = j + 1
          node.value = op.supportVectors(i * numFeatures + j)
          j += 1
        }
        (svRowPtr + numFeatures).index = -1 // Terminator node
        i += 1
      }

      // Populate the support vector coefficients
      i = 0
      while (i < coefRows.length) {
        val coefRowPtr = coefRows(i)
        !(svCoefs + i) = coefRowPtr
        var j = 0
        while (j < numSupportVectors) {
          val index = i * numSupportVectors + j
          !(coefRowPtr + j) = op.coefficients(index)
          j += 1
        }
        i += 1
      }

      // Link all the populated memory to the main svm_model struct
      model.param = param
      model.nr_class = nrClass
      model.l = numSupportVectors
      model.label = label
      model.rho = rho
      model.nSV = nSV
      model.SV = svs
      model.sv_coef = svCoefs
      model
    }
  }

  /** Creates empty Scala arrays for each graph output. These arrays will be pointed to by the
    * `memoryResource` and written to directly from native code.
    */
  private def createOutputArrays(model: ModelIR): Map[String, Array[_]] =
    model.graphOutputs.map { name =>
      val allocation = model.allocations(name)
      val size = allocation.shape.product
      val array: Array[_] = allocation.dataType match {
        case DataType.Float32 => new Array[Float](size)
        case DataType.Int32 => new Array[Int](size)
        case DataType.Float64 => new Array[Double](size)
        case DataType.Int64 => new Array[Long](size)
        case other => throw new Exception(s"Unsupported output data type: $other")
      }
      name -> array
    }.toMap
}
