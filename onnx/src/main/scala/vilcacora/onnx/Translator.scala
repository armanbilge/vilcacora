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

package vilcacora.onnx

import com.armanbilge.vilcacora.ir._
import vilcacora.onnx.proto._
import java.nio.ByteBuffer
import java.nio.ByteOrder
import cats.syntax.all._

/** Translates an ONNX `ModelProto` into a custom, type-safe `ModelIR`.
  *
  * The primary goals of this translator are:
  *   1. To convert the protobuf-based ONNX graph into a more easily consumable Scala ADT. 2. To
  *      resolve all memory requirements at compile-time by creating `Allocation` objects for every
  *      tensor (inputs, outputs, weights, and intermediate results).
  */
object Translator {

  /** The main entry point for translation.
    *
    * @param model
    *   The ONNX model loaded from a protobuf file.
    *
    * @return
    *   An `Either` containing the translated `ModelIR` on success, or an error message on failure.
    */
  def translate(model: ModelProto): Either[String, ModelIR] =
    for {
      graph <- model.graph.toRight("Model does not contain a graph")
      allocations <- buildAllocations(graph)

      // Translate all nodes into IR operations.

      operations <- graph.node.toList.traverse(translateNode)

      graphInputs = graph.input.map(_.name)
      graphOutputs = graph.output.map(_.name)

    } yield ModelIR(
      name = graph.name,
      operations = operations,
      allocations = allocations,
      graphInputs = graphInputs.toList,
      graphOutputs = graphOutputs.toList,
    )

  /** Validates that an ONNX node has the expected number of inputs and outputs. This prevents
    * runtime errors from unsafe access like `.head` or `(1)`.
    */
  private[onnx] def checkArity(
      node: NodeProto,
      expectedInputs: Int,
      expectedOutputs: Int,
  ): Either[String, Unit] =
    if (node.input.size == expectedInputs && node.output.size == expectedOutputs) {
      Right(())
    } else {
      Left(
        s"Node '${node.name}' (opType: ${node.opType}) expects $expectedInputs inputs and $expectedOutputs outputs, but got ${node.input.size} and ${node.output.size}",
      )
    }

  /** Gathers all tensor definitions from the graph and creates a map of named `Allocation` objects.
    * This includes inputs, outputs, constant initializers, and intermediate tensors.
    */
  private[onnx] def buildAllocations(
      graph: GraphProto,
  ): Either[String, Map[String, Allocation]] = {
    // Collect all tensor *declarations* (which define shape and type).
    val allValueProtos = graph.input ++ graph.valueInfo ++ graph.output

    // Create allocations for declared tensors (without initial data).
    val valueAllocations: Either[String, List[Allocation]] = allValueProtos
      .distinctBy(_.name)
      .toList
      .traverse(valueInfo => createAllocation(valueInfo, None))

    // Create allocations for constant tensors, which include initial data.
    val initializerAllocations: Either[String, List[Allocation]] =
      graph.initializer.toList.traverse(createAllocationFromInitializer)
    // This new section manually creates allocations for intermediate tensors
    // that are not explicitly declared in the ONNX graph's value_info.

    val manuallyCreatedAllocs = for {
      valAllocs <- valueAllocations
      initAllocs <- initializerAllocations
      // Create a temporary map of all known allocations so far for lookups.
      existingAllocs = (valAllocs ++ initAllocs).map(a => a.name -> a).toMap

      // Iterate through all nodes to find any that need special handling.
      newAllocs <- graph.node.toList.flatTraverse { node =>
        node.opType match {
          case "SVMClassifier" =>
            for {
              _ <- checkArity(node, 1, 2) // Ensure SVMClassifier has 2 outputs
              scoresOutputName = node.output(1)
              // Check if an allocation for the scores tensor already exists.
              allocations <-
                if (existingAllocs.contains(scoresOutputName)) {
                  // If it exists, we don't need to do anything.
                  Right(List.empty[Allocation])
                } else {
                  // If it doesn't exist, create it manually.
                  for {
                    // Get the input tensor's allocation to infer the batch size.
                    inputAlloc <- existingAllocs
                      .get(node.input.head)
                      .toRight(
                        s"SVM input '${node.input.head}' not found in allocations.",
                      )
                    batchSize <- inputAlloc.shape.headOption.toRight(
                      s"Input '${inputAlloc.name}' for SVM has no dimensions.",
                    )

                    // Get the number of classes from the node's attributes.
                    attributes = new OnnxAttributeHelper(node)
                    classLabels <- attributes.getInts("classlabels_ints")
                    numClasses = classLabels.size

                    // The ONNX spec defines the scores output as a float tensor.
                    // We default to Float32. Shape is [batch_size, num_classes].
                    scoresAlloc = Allocation(
                      name = scoresOutputName,
                      dataType = DataType.Float32,
                      shape = List(batchSize.toInt, numClasses),
                      initialData = None,
                    )
                  } yield List(scoresAlloc)
                }
            } yield allocations

          case _ =>
            // For all other operators, we assume their outputs are properly declared.
            Right(List.empty[Allocation])
        }
      }
    } yield newAllocs

    for {
      valAllocs <- valueAllocations
      initAllocs <- initializerAllocations
      manualAllocs <- manuallyCreatedAllocs
    } yield (valAllocs ++ initAllocs ++ manualAllocs).map(a => a.name -> a).toMap
  }

  /** Translates a single ONNX `NodeProto` into its corresponding IR `Operation`.
    */
  private[onnx] def translateNode(node: NodeProto): Either[String, Operation] = {
    val attributes = new OnnxAttributeHelper(node)
    node.opType match {
      // Group simple binary operators
      case "MatMul" | "Add" | "Mul" =>
        for {
          // The arity check ensures the .head and (1) accessors below are safe.
          _ <- checkArity(node, expectedInputs = 2, expectedOutputs = 1)
          op <- node.opType match {
            case "MatMul" =>
              Right(Operation.MatMul(node.input.head, node.input(1), node.output.head))
            case "Add" => Right(Operation.Add(node.input.head, node.input(1), node.output.head))
            case "Mul" => Right(Operation.Mul(node.input.head, node.input(1), node.output.head))
            case _ => Left("Internal error: Unreachable code in operator matching")
          }
        } yield op

      case "Cast" =>
        for {
          _ <- checkArity(node, expectedInputs = 1, expectedOutputs = 1)
          toValue <- attributes.getInt("to")
          dataType <- fromOnnxDataType(toValue.toInt)
        } yield Operation.Cast(node.input.head, node.output.head, dataType)

      case "SVMClassifier" =>
        for {
          _ <- checkArity(node, expectedInputs = 1, expectedOutputs = 2)
          classLabels <- attributes.getInts("classlabels_ints")
          coefficients <- attributes.getFloats("coefficients")
          kernelParams <- attributes.getFloats("kernel_params")
          kernelTypeStr <- attributes.getString("kernel_type")
          kernelType <- SVMKernel.fromString(kernelTypeStr)
          postTransformStr <- attributes.getString("post_transform")
          postTransform <- PostTransform.fromString(postTransformStr)
          rho <- attributes.getFloats("rho")
          supportVectors <- attributes.getFloats("support_vectors")
          vectorsPerClass <- attributes.getInts("vectors_per_class")
        } yield Operation.SVMClassifier(
          input = node.input.head,
          outputLabel = node.output.head,
          outputScores = node.output(1),
          classLabels = classLabels.toList,
          coefficients = coefficients.map(_.toDouble).toArray,
          kernelType = kernelType,
          kernelParams = kernelParams.map(_.toDouble).toList,
          postTransform = postTransform,
          rho = rho.map(_.toDouble).toList,
          supportVectors = supportVectors.map(_.toDouble).toArray,
          vectorsPerClass = vectorsPerClass.toList,
        )
      case unsupported => Left(s"Unsupported operation type: $unsupported")
    }
  }

  /** Creates an `Allocation` from a tensor declaration (`ValueInfoProto`). This is used for tensors
    * whose memory must be allocated but whose initial value is not known.
    */
  private[onnx] def createAllocation(
      valueInfo: ValueInfoProto,
      initialData: Option[Array[Byte]],
  ): Either[String, Allocation] =
    for {
      name <- Option(valueInfo.name).filter(_.nonEmpty).toRight("ValueInfo is missing a name")
      typeProto <- valueInfo.`type`.toRight(s"ValueInfo '$name' is missing a type")
      tensorType <- typeProto.value match {
        case TypeProto.Value.TensorType(t) => Right(t)
        case _ =>
          Left(
            s"ValueInfo '$name' is not a tensor type, but ${typeProto.value.getClass.getSimpleName}",
          )
      }
      dataType <- fromOnnxDataType(tensorType.elemType)
      shapeProto <- tensorType.shape.toRight(s"Tensor '$name' has no shape")
      shape <- parseShape(shapeProto)
    } yield Allocation(name, dataType, shape, initialData)

  /** Creates an `Allocation` from a constant tensor (`TensorProto`). This is used for weights and
    * biases, and includes extracting the raw byte data.
    */
  private[onnx] def createAllocationFromInitializer(
      tensor: TensorProto,
  ): Either[String, Allocation] =
    for {
      name <- Option(tensor.name).filter(_.nonEmpty).toRight("Initializer is missing a name")
      dataType <- fromOnnxDataType(tensor.dataType)
      shape = tensor.dims.map(_.toInt).toList
      data <- extractBytes(tensor, dataType)
    } yield Allocation(name, dataType, shape, Some(data))

  /** Converts an ONNX `TensorShapeProto` into a `List[Int]`.
    */
  private[onnx] def parseShape(shapeProto: TensorShapeProto): Either[String, List[Int]] =
    // `traverse` will attempt to convert each dimension. If any dimension fails
    // (returns a Left), the entire operation will fail and return that Left.
    shapeProto.dim.toList.traverse { dim =>
      dim.value match {
        // The success case: the dimension has a fixed integer value.
        case TensorShapeProto.Dimension.Value.DimValue(value) =>
          Right(value.toInt)

        // The failure case: the dimension is a named parameter (e.g., 'N' or 'batch_size').
        case TensorShapeProto.Dimension.Value.DimParam(name) =>
          Left(
            s"Model uses a dynamic dimension parameter ('$name'). This translator requires static shapes and does not handle dynamic inputs.",
          )

        // The failure case: the dimension is unspecified.
        case TensorShapeProto.Dimension.Value.Empty =>
          Left(
            "Model has an unknown dimension. This translator requires static shapes.",
          )
      }
    }

  /** Maps an ONNX integer data type code to the corresponding IR `DataType`. */
  private[onnx] def fromOnnxDataType(onnxType: Int): Either[String, DataType] =
    onnxType match {
      case 1 => Right(DataType.Float32)
      case 11 => Right(DataType.Float64)
      case 10 => Right(DataType.Float16)
      case 16 => Right(DataType.BFloat16)
      case 6 => Right(DataType.Int32)
      case 7 => Right(DataType.Int64)
      case 5 => Right(DataType.Int16)
      case 3 => Right(DataType.Int8)
      case 12 => Right(DataType.UInt32)
      case 13 => Right(DataType.UInt64)
      case 4 => Right(DataType.UInt16)
      case 2 => Right(DataType.UInt8)
      case 9 => Right(DataType.Bool)
      case _ => Left(s"Unsupported ONNX data type code: $onnxType")
    }

  /** Extracts the tensor's weight data into a raw `Array[Byte]`. It prioritizes the efficient
    * `rawData` field. If that's empty, it falls back to reconstructing the byte array from typed
    * data fields (e.g., `floatData`).
    */
  private[onnx] def extractBytes(
      tensor: TensorProto,
      dataType: DataType,
  ): Either[String, Array[Byte]] =
    // The `rawData` field is the preferred and most common way to store tensor data.
    if (tensor.rawData.nonEmpty) {
      Right(tensor.rawData.toByteArray())
    } else {
      // Fallback for models that use the repeated typed fields instead.
      val elementCount = tensor.dims.map(_.toInt).product
      val buffer = ByteBuffer.allocate(elementCount * dataType.sizeInBytes)
      // ONNX standard specifies little-endian byte order.
      buffer.order(ByteOrder.LITTLE_ENDIAN)

      tensor.dataType match {
        case 1 => tensor.floatData.foreach(buffer.putFloat)
        case 11 => tensor.doubleData.foreach(buffer.putDouble)
        case 6 => tensor.int32Data.foreach(buffer.putInt)
        case 7 => tensor.int64Data.foreach(buffer.putLong)
        // Note: Other types like int16 are typically stored in `rawData` or `int32Data`.
        case unsupportedType =>
          return Left(
            s"Extracting typed data for tensor '${tensor.name}' is not supported for type code $unsupportedType. The data should be in the 'raw_data' field.",
          )
      }
      Right(buffer.array())
    }

  /** A private helper class to simplify and safely access attributes from a `NodeProto`. This
    * encapsulates the boilerplate of finding an attribute by name and extracting its typed value.
    */
  private class OnnxAttributeHelper(node: NodeProto) {
    private val attributeMap: Map[String, AttributeProto] =
      node.attribute.map(attr => attr.name -> attr).toMap

    def getString(name: String): Either[String, String] =
      attributeMap
        .get(name)
        .toRight(s"Missing attribute '$name' in node '${node.name}'")
        .map(_.s.toStringUtf8())

    def getInt(name: String): Either[String, Long] =
      attributeMap.get(name).toRight(s"Missing attribute '$name' in node '${node.name}'").map(_.i)

    def getFloats(name: String): Either[String, Seq[Float]] =
      attributeMap
        .get(name)
        .toRight(s"Missing attribute '$name' in node '${node.name}'")
        .map(_.floats)

    def getInts(name: String): Either[String, Seq[Long]] =
      attributeMap
        .get(name)
        .toRight(s"Missing attribute '$name' in node '${node.name}'")
        .map(_.ints)
  }
}
