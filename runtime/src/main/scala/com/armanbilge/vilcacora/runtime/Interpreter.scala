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
import com.armanbilge.vilcacora.runtime.LibSVM
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
  def execute(
      model: ModelIR,
      inputs: Map[String, Array[_]],
  ): Resource[IO, IO[Map[String, Array[_]]]] = {
    validateModel(model)
    val outputArrays: Map[String, Array[_]] = createOutputArrays(model)

    memoryResource(model, inputs, outputArrays).flatMap { memoryMap =>
      val opResources: List[Resource[IO, IO[Unit]]] =
        model.operations.map(op => executeOperation(op, memoryMap, model))

      val combined: Resource[IO, List[IO[Unit]]] = opResources.sequence

      combined.map { opIOs =>
        val runOps: IO[Unit] = opIOs.traverse_(_ *> IO.cede)
        runOps.as(outputArrays)
      }
    }
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
  private def executeOperation(
      op: Operation,
      memory: MemoryMap,
      model: ModelIR,
  ): Resource[IO, IO[Unit]] =
    op match {
      case op: Operation.SVMClassifier => handleSvmClassifier(op, memory, model)
      case op: Operation.Add => Resource.pure(handleAdd(op, memory, model))
      case op: Operation.Mul => Resource.pure(handleMul(op, memory, model))
      case op: Operation.Cast => Resource.pure(handleCast(op, memory, model))
      case other =>
        // This case is unreachable due to the pre-validation step.
        // It remains as a safeguard against internal logic errors.
        throw new NotImplementedError(s"Operation not implemented: ${other.getClass.getSimpleName}")
    }

  /** Handles element-wise addition for both Float32 and Float64 tensors. */
  private def handleAdd(op: Operation.Add, memory: MemoryMap, model: ModelIR): IO[Unit] = IO {
    val count = model.allocations(op.outputs.head).shape.product
    val inputAAlloc = model.allocations(op.inputs(0))
    val inputBAlloc = model.allocations(op.inputs(1))
    val outputAlloc = model.allocations(op.outputs.head)

    // Verify all tensors have the same data type
    require(
      inputAAlloc.dataType == inputBAlloc.dataType &&
        inputBAlloc.dataType == outputAlloc.dataType,
      s"Add operation requires all tensors to have the same data type. " +
        s"Got: ${inputAAlloc.dataType}, ${inputBAlloc.dataType}, ${outputAlloc.dataType}",
    )

    inputAAlloc.dataType match {
      case DataType.Float32 =>
        val inputA = memory(op.inputs(0)).asInstanceOf[Ptr[CFloat]]
        val inputB = memory(op.inputs(1)).asInstanceOf[Ptr[CFloat]]
        val output = memory(op.outputs.head).asInstanceOf[Ptr[CFloat]]

        var i = 0
        while (i < count) {
          !(output + i) = !(inputA + i) + !(inputB + i)
          i += 1
        }

      case DataType.Float64 =>
        val inputA = memory(op.inputs(0)).asInstanceOf[Ptr[CDouble]]
        val inputB = memory(op.inputs(1)).asInstanceOf[Ptr[CDouble]]
        val output = memory(op.outputs.head).asInstanceOf[Ptr[CDouble]]

        var i = 0
        while (i < count) {
          !(output + i) = !(inputA + i) + !(inputB + i)
          i += 1
        }

      case DataType.Int32 =>
        val inputA = memory(op.inputs(0)).asInstanceOf[Ptr[CInt]]
        val inputB = memory(op.inputs(1)).asInstanceOf[Ptr[CInt]]
        val output = memory(op.outputs.head).asInstanceOf[Ptr[CInt]]

        var i = 0
        while (i < count) {
          !(output + i) = !(inputA + i) + !(inputB + i)
          i += 1
        }

      case DataType.Int64 =>
        val inputA = memory(op.inputs(0)).asInstanceOf[Ptr[CLongLong]]
        val inputB = memory(op.inputs(1)).asInstanceOf[Ptr[CLongLong]]
        val output = memory(op.outputs.head).asInstanceOf[Ptr[CLongLong]]

        var i = 0
        while (i < count) {
          !(output + i) = !(inputA + i) + !(inputB + i)
          i += 1
        }

      case unsupported =>
        throw new NotImplementedError(s"Add operation not implemented for data type: $unsupported")
    }
  }

  /** Handles element-wise multiplication for both Float32 and Float64 tensors. */
  private def handleMul(op: Operation.Mul, memory: MemoryMap, model: ModelIR): IO[Unit] = IO {
    val count = model.allocations(op.outputs.head).shape.product
    val inputAAlloc = model.allocations(op.inputs(0))

    inputAAlloc.dataType match {
      case DataType.Float32 =>
        val inputA = memory(op.inputs(0)).asInstanceOf[Ptr[CFloat]]
        val inputB = memory(op.inputs(1)).asInstanceOf[Ptr[CFloat]]
        val output = memory(op.outputs.head).asInstanceOf[Ptr[CFloat]]

        var i = 0
        while (i < count) {
          !(output + i) = !(inputA + i) * !(inputB + i)
          i += 1
        }

      case DataType.Float64 =>
        val inputA = memory(op.inputs(0)).asInstanceOf[Ptr[CDouble]]
        val inputB = memory(op.inputs(1)).asInstanceOf[Ptr[CDouble]]
        val output = memory(op.outputs.head).asInstanceOf[Ptr[CDouble]]

        var i = 0
        while (i < count) {
          !(output + i) = !(inputA + i) * !(inputB + i)
          i += 1
        }

      case DataType.Int32 =>
        val inputA = memory(op.inputs(0)).asInstanceOf[Ptr[CInt]]
        val inputB = memory(op.inputs(1)).asInstanceOf[Ptr[CInt]]
        val output = memory(op.outputs.head).asInstanceOf[Ptr[CInt]]

        var i = 0
        while (i < count) {
          !(output + i) = !(inputA + i) * !(inputB + i)
          i += 1
        }

      case DataType.Int64 =>
        val inputA = memory(op.inputs(0)).asInstanceOf[Ptr[CLongLong]]
        val inputB = memory(op.inputs(1)).asInstanceOf[Ptr[CLongLong]]
        val output = memory(op.outputs.head).asInstanceOf[Ptr[CLongLong]]

        var i = 0
        while (i < count) {
          !(output + i) = !(inputA + i) * !(inputB + i)
          i += 1
        }

      case unsupported =>
        throw new NotImplementedError(s"Mul operation not implemented for data type: $unsupported")
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
  /** Handles the SVMClassifier operation using the C++ wrapper for robust LibSVM integration. This
    * approach eliminates struct layout issues and provides ONNX-compliant per-class scores.
    */
  private def handleSvmClassifier(
      op: Operation.SVMClassifier,
      memory: MemoryMap,
      model: ModelIR,
  ): Resource[IO, IO[Unit]] = {
    val numFeatures = model.allocations(op.input).shape.last

    // Create SVM model using C++ wrapper with proper resource management
    createSvmModelResource(op, numFeatures).map { svmModel =>
      IO {
        // Get input and output pointers from memory map
        val inputPtr = memory(op.input).asInstanceOf[Ptr[CDouble]]
        val scoresPtr = memory(op.outputScores).asInstanceOf[Ptr[CDouble]]
        val labelPtr = memory(op.outputLabel).asInstanceOf[Ptr[CInt]]

        // Single function call - all complexity handled in C++
        val predictedLabel = LibSVM.svm_predict_with_scores(
          svmModel,
          inputPtr,
          numFeatures,
          scoresPtr,
        )

        // Write back the predicted label
        !labelPtr = predictedLabel

        ()
      }
    }
  }

  /** Creates an SVM model using the C++ wrapper functions with proper resource management. All
    * memory allocation and model construction is handled in C for maximum reliability.
    */
  private def createSvmModelResource(
      op: Operation.SVMClassifier,
      numFeatures: Int,
  ): Resource[IO, Ptr[Byte]] = {
    val nrClass = op.classLabels.size
    val numSupportVectors = op.vectorsPerClass.sum.toInt

    for {
      // Create SVM parameter using C++ wrapper
      param <- createSvmParameterResource(op)

      // Create managed arrays for model data
      supportVectorsPtr <- createManagedDoubleArray(op.supportVectors)
      coefficientsPtr <- createManagedDoubleArray(op.coefficients)
      rhoPtr <- createManagedDoubleArray(op.rho.toArray)
      classLabelsPtr <- createManagedIntArray(op.classLabels.map(_.toInt).toArray)
      vectorsPerClassPtr <- createManagedIntArray(op.vectorsPerClass.map(_.toInt).toArray)

      // Create the SVM model using C++ wrapper with LibSVM's native cleanup
      svmModel <- Resource.make(IO {
        LibSVM.create_svm_model(
          param,
          nrClass,
          numSupportVectors,
          supportVectorsPtr,
          numFeatures,
          coefficientsPtr,
          rhoPtr,
          classLabelsPtr,
          vectorsPerClassPtr,
        )
      })(model =>
        IO {
          // Use LibSVM's native cleanup function
          val modelPtrPtr = stdlib.malloc(sizeof[Ptr[Byte]]).asInstanceOf[Ptr[Ptr[Byte]]]
          !modelPtrPtr = model
          LibSVM.svm_free_and_destroy_model(modelPtrPtr)
          stdlib.free(modelPtrPtr.asInstanceOf[Ptr[Byte]])
        },
      )

    } yield svmModel
  }

  /** Creates SVM parameter using C++ wrapper with proper resource management. */
  private def createSvmParameterResource(op: Operation.SVMClassifier): Resource[IO, Ptr[Byte]] = {
    val kernelType = op.kernelType match {
      case SVMKernel.Linear => 0
      case SVMKernel.Poly => 1
      case SVMKernel.Rbf => 2
      case SVMKernel.Sigmoid => 3
    }

    val gamma = op.kernelParams.headOption.getOrElse(0.0)
    val coef0 = op.kernelParams.drop(1).headOption.getOrElse(0.0)
    val degree = op.kernelParams.drop(2).headOption.map(_.toInt).getOrElse(3)

    Resource.make(IO {
      LibSVM.create_svm_param(
        svm_type = 0, // C_SVC
        kernel_type = kernelType,
        degree = degree,
        gamma = gamma,
        coef0 = coef0,
      )
    })(param => IO(stdlib.free(param)))
  }

  /** Helper to create managed double array for C++ wrapper calls. */
  private def createManagedDoubleArray(values: Array[Double]): Resource[IO, Ptr[CDouble]] =
    Resource.make(IO {
      val ptr = stdlib
        .malloc(sizeof[CDouble] * values.length.toUSize)
        .asInstanceOf[Ptr[CDouble]]
      if (ptr == null) throw new OutOfMemoryError(s"Failed to allocate ${values.length} doubles")

      for (i <- values.indices)
        ptr(i) = values(i)
      ptr
    })(ptr => IO(stdlib.free(ptr.asInstanceOf[Ptr[Byte]])))

  /** Helper to create managed int array for C++ wrapper calls. */
  private def createManagedIntArray(values: Array[Int]): Resource[IO, Ptr[CInt]] =
    Resource.make(IO {
      val ptr = stdlib
        .malloc(sizeof[CInt] * values.length.toUSize)
        .asInstanceOf[Ptr[CInt]]
      if (ptr == null) throw new OutOfMemoryError(s"Failed to allocate ${values.length} ints")

      for (i <- values.indices)
        ptr(i) = values(i)
      ptr
    })(ptr => IO(stdlib.free(ptr.asInstanceOf[Ptr[Byte]])))

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
