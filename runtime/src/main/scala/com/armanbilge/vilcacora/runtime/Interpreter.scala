/*
 * Copyright 2023 Arman Bilge
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
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

/** The core execution engine for a translated `ModelIR`. It manages memory and executes operations
  * within a Cats Effect IO context.
  */
object Interpreter {

  // A type alias for clarity: maps tensor names to their C memory pointers.
  type MemoryMap = Map[String, Ptr[Byte]]

  /** Executes a complete ModelIR graph from inputs to outputs. */
  def execute(model: ModelIR, inputs: Map[String, Array[_]]): IO[Map[String, Array[_]]] = {
    // Pre-allocate Scala arrays that will hold the final results.
    // The execution engine will write directly into the memory of these arrays.
    val outputArrays = createOutputArrays(model)

    // `Resource.use` guarantees that natively-allocated memory for intermediate
    // tensors is safely freed, while inputs and outputs use zero-copy pointers.
    memoryResource(model, inputs, outputArrays).use { memoryMap =>
      // The core execution loop. It executes each operation and then yields control
      // to the Cats Effect runtime to maintain application responsiveness.
      model.operations.traverse_ { op =>
        executeOperation(op, memoryMap, model) <* IO.cede
      }
    }.as(outputArrays) // Return the populated output arrays.
  }

  /** A `Resource` that manages memory for the graph execution.
    *
    * - **Inputs/Outputs**: Obtains direct pointers to the provided Scala arrays (zero-copy).
    * - **Intermediates/Constants**: Allocates native memory via `malloc` and ensures it's freed.
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
            // For inputs, get a direct pointer to the Scala array's memory.
            inputs(name).at(0).asInstanceOf[Ptr[Byte]]
          } else if (outputs.contains(name)) {
            // For outputs, get a direct pointer to the pre-allocated Scala array's memory.
            outputs(name).at(0).asInstanceOf[Ptr[Byte]]
          } else {
            // For intermediates and constants, allocate native memory.
            val totalBytes = (allocation.shape.product * allocation.dataType.sizeInBytes).toUSize
            val p = stdlib.malloc(totalBytes)
            if (p == null) throw new OutOfMemoryError(s"Failed to allocate tensor '$name'")

            // Copy any initial data (e.g., for constants).
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

    Resource.make(acquire) { case (_, ptrsToFree) =>
      // The release action only frees the memory we explicitly `malloc`'d.
      // Memory for input/output arrays is managed by the Scala GC.
      IO(ptrsToFree.foreach(stdlib.free))
    }.map(_._1)
  }

  /** Executes a single operation by dispatching to the appropriate handler. */
  private def executeOperation(op: Operation, memory: MemoryMap, model: ModelIR): IO[Unit] =
    op match {
      case op: Operation.SVMClassifier => handleSvmClassifier(op, memory, model)
      case op: Operation.Add => handleElementWise(op, memory, model)(_ + _)
      case op: Operation.Mul => handleElementWise(op, memory, model)(_ * _)
      case op: Operation.Cast => handleCast(op, memory, model)
      case other =>
        IO.raiseError(
          new NotImplementedError(s"Operation not implemented: ${other.getClass.getSimpleName}"),
        )
    }

  // --- Operation Handlers ---

  private def handleElementWise(op: Operation, memory: MemoryMap, model: ModelIR)(
      f: (Float, Float) => Float,
  ): IO[Unit] = IO {
    val outputAllocation = model.allocations(op.outputs.head)
    val count = outputAllocation.shape.product

    val inputA = memory(op.inputs(0)).asInstanceOf[Ptr[CFloat]]
    val inputB = memory(op.inputs(1)).asInstanceOf[Ptr[CFloat]]
    val output = memory(op.outputs(0)).asInstanceOf[Ptr[CFloat]]

    for (i <- 0 until count)
      !(output + i) = f(!(inputA + i), !(inputB + i))
  }

  private def handleCast(op: Operation.Cast, memory: MemoryMap, model: ModelIR): IO[Unit] = IO {
    val inputAlloc = model.allocations(op.input)
    val outputAlloc = model.allocations(op.output)

    val inputPtr = memory(op.input)
    val outputPtr = memory(op.output)
    val count = inputAlloc.shape.product

    (inputAlloc.dataType, outputAlloc.dataType) match {
      case (from, to) if from == to =>
        val _ = memcpy(outputPtr, inputPtr, (count * from.sizeInBytes).toUSize)
      case (DataType.Float64, DataType.Float32) =>
        val in = inputPtr.asInstanceOf[Ptr[CDouble]]
        val out = outputPtr.asInstanceOf[Ptr[CFloat]]
        for (i <- 0 until count)
          !(out + i) = (!(in + i)).toFloat
      case (DataType.Float32, DataType.Float64) =>
        val in = inputPtr.asInstanceOf[Ptr[CFloat]]
        val out = outputPtr.asInstanceOf[Ptr[CDouble]]
        for (i <- 0 until count)
          !(out + i) = (!(in + i)).toDouble
      // Add more cases for other data types
      case (from, to) =>
        throw new NotImplementedError(s"Cast from $from to $to is not implemented.")
    }
  }

  private def handleSvmClassifier(
      op: Operation.SVMClassifier,
      memory: MemoryMap,
      model: ModelIR,
  ): IO[Unit] = {
    // The number of features is the size of the last dimension of the input tensor.
    val numFeatures = model.allocations(op.input).shape.last

    val svmResources = for {
      modelPtr <- buildSvmModelResource(op, numFeatures)
      // Allocate space for features + 1 for the terminator node.
      svmInputNode <- malloc[svm_node](sizeof[svm_node] * (numFeatures + 1).toUSize)
    } yield (modelPtr, svmInputNode)

    svmResources.use { case (modelPtr, svmInputNode) =>
      // Using a for-comprehension here breaks the logic into clean, sequential IO steps.
      for {
        inputPtr <- IO.pure(memory(op.input).asInstanceOf[Ptr[CDouble]])

        // This entire block is now a single IO action that performs the C interop.
        predictionResult <- IO {
          // 1. Fill the svm_node struct with input data from the input tensor.
          for (i <- 0 until numFeatures) {
            val node = svmInputNode + i
            node.index = i + 1 // LIBSVM indices are 1-based.
            node.value = !(inputPtr + i)
          }
          (svmInputNode + numFeatures).index = -1 // Terminator node.

          // 2. Prepare for and call the prediction function.
          val nrClass = op.classLabels.size
          val decValuesCount = nrClass * (nrClass - 1) / 2
          val decisionValuesPtr = stackalloc[CDouble](decValuesCount.toUInt)
          val predictedLabel = svm_predict_values(modelPtr, svmInputNode, decisionValuesPtr)

          // 3. Return the results needed for the next step.
          (predictedLabel, decisionValuesPtr, decValuesCount)
        }

        // 4. Deconstruct the results and write them directly to the output tensor memory.
        (predictedLabel, decisionValuesPtr, decValuesCount) = predictionResult
        _ <- IO {
          val labelOutputPtr = memory(op.outputLabel).asInstanceOf[Ptr[CInt]]
          !labelOutputPtr = predictedLabel.toInt

          val scoresOutputPtr = memory(op.outputScores).asInstanceOf[Ptr[CDouble]]
          memcpy(
            scoresOutputPtr,
            decisionValuesPtr.asInstanceOf[Ptr[Byte]],
            decValuesCount.toUSize * sizeof[CDouble],
          )
        }
      } yield () // The for-comprehension must yield Unit.
    }
  }

  /** Helper to safely `malloc` memory within a `Resource` scope. */
  private def malloc[T](size: CSize): Resource[IO, Ptr[T]] =
    Resource.make(IO(stdlib.malloc(size)))(ptr => IO(stdlib.free(ptr))).map(_.asInstanceOf[Ptr[T]])

  /** Constructs a `Ptr[svm_model]` from our IR, wrapped in a `Resource` for guaranteed memory
    * safety.
    */
  private def buildSvmModelResource(
      op: Operation.SVMClassifier,
      numFeatures: Int,
  ): Resource[IO, Ptr[svm_model]] = {
    val nrClass = op.classLabels.size
    val numSupportVectors = op.vectorsPerClass.sum.toInt

    for {
      model <- malloc[svm_model](sizeof[svm_model])
      param <- malloc[svm_parameter](sizeof[svm_parameter])
      label <- malloc[CInt](sizeof[CInt] * nrClass.toUSize)
      rho <- malloc[CDouble](sizeof[CDouble] * op.rho.size.toUSize)
      nSV <- malloc[CInt](sizeof[CInt] * nrClass.toUSize)
      svs <- malloc[Ptr[svm_node]](sizeof[Ptr[svm_node]] * numSupportVectors.toUSize)
      svCoefs <- malloc[Ptr[CDouble]](sizeof[Ptr[CDouble]] * (nrClass - 1).toUSize)
      svRows <- (0 until numSupportVectors).toList.traverse(_ =>
        malloc[svm_node](sizeof[svm_node] * (numFeatures + 1).toUSize),
      )
      coefRows <- (0 until nrClass - 1).toList.traverse(_ =>
        malloc[CDouble](sizeof[CDouble] * numSupportVectors.toUSize),
      )
    } yield {
      param.svm_type = 0 // C_SVC
      param.kernel_type = op.kernelType match {
        case SVMKernel.Linear => LibSvm.LINEAR
        case SVMKernel.Poly => LibSvm.POLY
        case SVMKernel.Rbf => LibSvm.RBF
        case SVMKernel.Sigmoid => LibSvm.SIGMOID
      }
      param.gamma = op.kernelParams.headOption.getOrElse(0.0)
      param.coef0 = op.kernelParams.drop(1).headOption.getOrElse(0.0)
      param.degree = op.kernelParams.drop(2).headOption.map(_.toInt).getOrElse(3)

      op.classLabels.zipWithIndex.foreach { case (l, i) => !(label + i) = l.toInt }
      op.rho.zipWithIndex.foreach { case (r, i) => !(rho + i) = r }
      op.vectorsPerClass.zipWithIndex.foreach { case (n, i) => !(nSV + i) = n.toInt }

      svRows.zipWithIndex.foreach { case (svRowPtr, i) =>
        !(svs + i) = svRowPtr
        for (j <- 0 until numFeatures) {
          val node = svRowPtr + j
          node.index = j + 1
          node.value = op.supportVectors(i * numFeatures + j)
        }
        (svRowPtr + numFeatures).index = -1
      }

      coefRows.zipWithIndex.foreach { case (coefRowPtr, i) =>
        !(svCoefs + i) = coefRowPtr
        for (j <- 0 until numSupportVectors) {
          // ONNX coefficients are flat with shape [(nr_class-1), num_support_vectors].
          // `i` is the class-pair index, `j` is the support-vector index.
          // This formula correctly accesses the flat array in row-major order.
          val index = i * numSupportVectors + j
          !(coefRowPtr + j) = op.coefficients(index)
        }
      }

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

  /** Creates empty Scala arrays for each graph output, ready to be written to directly. */
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