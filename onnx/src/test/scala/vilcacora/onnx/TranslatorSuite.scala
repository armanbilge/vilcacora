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

import com.armanbilge.vilcacora.ir.{DataType, Operation, PostTransform, SVMKernel, AutoPad}
import vilcacora.onnx.proto.{AttributeProto, ModelProto, NodeProto, TensorProto}
import com.google.protobuf.ByteString
import munit.FunSuite
import java.io.InputStream
import java.nio.{ByteBuffer, ByteOrder}

class TranslatorSuite extends FunSuite {

  // --- Unit tests for individual node translations ---

  test("translateNode should translate a MatMul node") {
    val node = NodeProto(opType = "MatMul", input = Seq("A", "B"), output = Seq("C"))
    assertEquals(Translator.translateNode(node), Right(Operation.MatMul("A", "B", "C")))
  }

  test("translateNode should translate an Add node") {
    val node = NodeProto(opType = "Add", input = Seq("X", "Y"), output = Seq("Z"))
    assertEquals(Translator.translateNode(node), Right(Operation.Add("X", "Y", "Z")))
  }

  test("translateNode should translate a Cast node") {
    val node = NodeProto(
      opType = "Cast",
      input = Seq("in"),
      output = Seq("out"),
      attribute = Seq(
        AttributeProto(name = "to", i = 7, `type` = AttributeProto.AttributeType.INT),
      ), // 7 = INT64
    )
    assertEquals(Translator.translateNode(node), Right(Operation.Cast("in", "out", DataType.Int64)))
  }

  // --- REWRITTEN TEST FOR SVMCLASSIFIER ---
  test(
    "translateNode should correctly translate an SVMClassifier node with floating point tolerance",
  ) {
    val node = NodeProto(
      opType = "SVMClassifier",
      input = Seq("features"),
      output = Seq("label", "scores"),
      attribute = Seq(
        AttributeProto(name = "classlabels_ints", ints = Seq(0L, 1L)),
        AttributeProto(name = "coefficients", floats = Seq(1.5f, -1.5f)),
        AttributeProto(name = "kernel_params", floats = Seq(0.5f, 0.1f)), // Note: 0.1f is inexact
        AttributeProto(name = "kernel_type", s = ByteString.copyFromUtf8("RBF")),
        AttributeProto(name = "post_transform", s = ByteString.copyFromUtf8("SOFTMAX")),
        AttributeProto(name = "rho", floats = Seq(0.9f)),
        AttributeProto(name = "support_vectors", floats = Seq(1.1f, 2.2f)),
        AttributeProto(name = "vectors_per_class", ints = Seq(1L, 1L)),
      ),
    )

    val result = Translator.translateNode(node)
    assert(result.isRight, "SVMClassifier translation failed unexpectedly")

    // Define a small tolerance for comparing floating point numbers
    val epsilon = 1e-6

    result.foreach {
      case op: Operation.SVMClassifier =>
        assertEquals(op.classLabels, List(0L, 1L))
        assertEquals(op.kernelType, SVMKernel.Rbf)
        assertEquals(op.postTransform, PostTransform.Softmax)

        // Use assertEquals with a tolerance (epsilon) for doubles
        assertEqualsDouble(op.kernelParams.head, 0.5, epsilon)
        assertEqualsDouble(op.kernelParams(1), 0.1, epsilon)
        assertEqualsDouble(op.rho.head, 0.9, epsilon)

        // Compare arrays of doubles element-by-element with tolerance
        op.coefficients.zip(Array(1.5, -1.5)).foreach { case (actual, expected) =>
          assertEqualsDouble(actual, expected, epsilon)
        }
        op.supportVectors.zip(Array(1.1, 2.2)).foreach { case (actual, expected) =>
          assertEqualsDouble(actual, expected, epsilon)
        }
      case other => fail(s"Expected SVMClassifier but got $other")
    }
  }
  test("translateNode should translate a Div node") {
    val node =
      NodeProto(opType = "Div", input = Seq("Numerator", "Denominator"), output = Seq("Quotient"))
    assertEquals(
      Translator.translateNode(node),
      Right(Operation.Div("Numerator", "Denominator", "Quotient")),
    )
  }

  test("translateNode should translate a Relu node") {
    val node = NodeProto(opType = "Relu", input = Seq("X"), output = Seq("Y"))
    assertEquals(Translator.translateNode(node), Right(Operation.Relu("X", "Y")))
  }

  test("translateNode should translate a Reshape node") {
    val node = NodeProto(opType = "Reshape", input = Seq("data", "shape"), output = Seq("reshaped"))
    // Test with the default 'allowzero' attribute
    assertEquals(
      Translator.translateNode(node),
      Right(Operation.Reshape("data", "shape", "reshaped", allowzero = false)),
    )
  }

  test("translateNode should translate a Constant node") {
    // Prepare the raw byte data for a Float32 tensor with value [1.0f, 2.0f]
    val byteBuffer = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
    byteBuffer.putFloat(1.0f)
    byteBuffer.putFloat(2.0f)
    val rawBytes = byteBuffer.array()

    // The constant data is stored as a TensorProto inside an AttributeProto
    val tensorProto = TensorProto(
      dataType = 1, // Float32
      dims = Seq(2),
      rawData = ByteString.copyFrom(rawBytes),
    )
    val attribute = AttributeProto(
      name = "value",
      t = Some(tensorProto),
      `type` = AttributeProto.AttributeType.TENSOR,
    )

    val node = NodeProto(
      opType = "Constant",
      input = Nil,
      output = Seq("const_out"),
      attribute = Seq(attribute),
    )

    val result = Translator.translateNode(node)
    assert(result.isRight, "Constant translation failed")
    result.foreach {
      case op: Operation.Constant =>
        assertEquals(op.output, "const_out")
        assertEquals(op.dataType, DataType.Float32)
        assertEquals(op.shape, List(2))
        assert(op.value.sameElements(rawBytes), "Raw byte data did not match")
      case other => fail(s"Expected Constant but got $other")
    }
  }

  test("translateNode should translate a Conv node") {
    val node = NodeProto(
      opType = "Conv",
      input = Seq("X", "W", "B"), // Input, Weights, Bias
      output = Seq("Y"),
      attribute = Seq(
        AttributeProto(name = "kernel_shape", ints = Seq(3L, 3L)),
        AttributeProto(name = "strides", ints = Seq(1L, 1L)),
        AttributeProto(name = "pads", ints = Seq(1L, 1L, 1L, 1L)),
        AttributeProto(name = "dilations", ints = Seq(1L, 1L)),
        AttributeProto(name = "group", i = 1L),
        AttributeProto(name = "auto_pad", s = ByteString.copyFromUtf8("NOTSET")),
      ),
    )

    val expected = Operation.Conv(
      input = "X",
      weight = "W",
      bias = Some("B"),
      output = "Y",
      autoPad = AutoPad.NotSet,
      dilations = List(1, 1),
      group = 1,
      kernelShape = List(3, 3),
      pads = List(1, 1, 1, 1),
      strides = List(1, 1),
    )

    assertEquals(Translator.translateNode(node), Right(expected))
  }

  test("translateNode should translate a MaxPool node") {
    val node = NodeProto(
      opType = "MaxPool",
      input = Seq("X"),
      output = Seq("Y"),
      attribute = Seq(
        AttributeProto(name = "kernel_shape", ints = Seq(2L, 2L)),
        AttributeProto(name = "strides", ints = Seq(2L, 2L)),
        AttributeProto(name = "pads", ints = Seq(0L, 0L, 0L, 0L)),
        AttributeProto(name = "ceil_mode", i = 0L),
        AttributeProto(name = "storage_order", i = 0L),
        AttributeProto(name = "dilations", ints = Seq(1L, 1L)),
        AttributeProto(name = "auto_pad", s = ByteString.copyFromUtf8("NOTSET")),
      ),
    )

    val expected = Operation.MaxPool(
      input = "X",
      output = "Y",
      autoPad = AutoPad.NotSet,
      ceilMode = false,
      dilations = List(1, 1),
      kernelShape = List(2, 2),
      pads = List(0, 0, 0, 0),
      storageOrder = 0,
      strides = List(2, 2),
    )

    assertEquals(Translator.translateNode(node), Right(expected))
  }

  // --- End-to-end tests using model files from resources ---

  // Helper now includes the leading slash, which is the correct way to specify an absolute path on the classpath
  def loadModelFromPath(modelPath: String): ModelProto = {
    val stream: InputStream = getClass.getResourceAsStream(modelPath)
    if (stream == null)
      throw new IllegalArgumentException(s"Resource not found: $modelPath")
    try ModelProto.parseFrom(stream)
    finally stream.close()
  }

  test("translate should successfully process a valid, fully static model") {
    // This test now depends on the file 'static_svm.onnx' being in 'onnx/src/test/resources'
    val modelProto = loadModelFromPath("/static_svm.onnx")
    val irResult = Translator.translate(modelProto)

    assert(irResult.isRight, "Translation failed unexpectedly for a valid model")
    irResult.foreach { modelIR =>
      assertEquals(modelIR.graphInputs, List("features_float32x30"))
      assertEquals(modelIR.operations.length, 5)
    }
  }

  test("translate should fail gracefully for a model with dynamic shapes") {
    // This test now depends on the file 'dynamic_svm.onnx' being in 'onnx/src/test/resources'
    val modelProto = loadModelFromPath("/dynamic_svm.onnx")
    val irResult = Translator.translate(modelProto)

    assert(irResult.isLeft, "Translation should have failed but it succeeded")
    irResult.left.foreach { error =>
      assert(error.contains("unknown dimension"))
    }
  }
}
