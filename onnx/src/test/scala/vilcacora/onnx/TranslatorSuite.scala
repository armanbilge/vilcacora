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

import com.armanbilge.vilcacora.ir.{DataType, Operation, PostTransform, SVMKernel}
import vilcacora.onnx.proto.{AttributeProto, ModelProto, NodeProto}
import com.google.protobuf.ByteString
import munit.FunSuite
import java.nio.file.{Files, Paths}

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
        AttributeProto(name = "to", i = 6, `type` = AttributeProto.AttributeType.INT),
      ), // 6 = INT64
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

  // --- End-to-end tests using model files from resources ---

  // Helper now includes the leading slash, which is the correct way to specify an absolute path on the classpath
  def loadModelFromPath(modelPath: String): ModelProto = {
    val bytes = Files.readAllBytes(Paths.get(modelPath))
    ModelProto.parseFrom(bytes)
  }

  test("translate should successfully process a valid, fully static model") {
    // This test now depends on the file 'static_svm.onnx' being in 'onnx/src/test/resources'
    val modelProto = loadModelFromPath("onnx/src/test/resources/static_svm.onnx")
    val irResult = Translator.translate(modelProto)

    assert(irResult.isRight, "Translation failed unexpectedly for a valid model")
    irResult.foreach { modelIR =>
      assertEquals(modelIR.graphInputs, List("features_float32x30"))
      assertEquals(modelIR.operations.length, 5)
    }
  }

  test("translate should fail gracefully for a model with dynamic shapes") {
    // This test now depends on the file 'dynamic_svm.onnx' being in 'onnx/src/test/resources'
    val modelProto = loadModelFromPath("onnx/src/test/resources/dynamic_svm.onnx")
    val irResult = Translator.translate(modelProto)

    assert(irResult.isLeft, "Translation should have failed but it succeeded")
    irResult.left.foreach { error =>
      assert(error.contains("unknown dimension"))
    }
  }
}
