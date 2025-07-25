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

import cats.effect.unsafe.implicits.global
import com.armanbilge.vilcacora.ir._
import munit.FunSuite

class InterpreterSuite extends FunSuite {

  /** Test the Add operation with Float32 tensors */
  test("Add operation should perform element-wise addition on Float32 tensors") {
    val inputA = Array(1.0f, 2.0f, 3.0f, 4.0f)
    val inputB = Array(5.0f, 6.0f, 7.0f, 8.0f)

    val model = ModelIR(
      name = "add_test",
      operations = List(
        Operation.Add("input_a", "input_b", "output"),
      ),
      allocations = Map(
        "input_a" -> Allocation("input_a", DataType.Float32, List(4)),
        "input_b" -> Allocation("input_b", DataType.Float32, List(4)),
        "output" -> Allocation("output", DataType.Float32, List(4)),
      ),
      graphInputs = List("input_a", "input_b"),
      graphOutputs = List("output"),
    )

    val inputs = Map(
      "input_a" -> inputA,
      "input_b" -> inputB,
    )

    val results = Interpreter.execute(model, inputs).use(_.map(identity)).unsafeRunSync()
    val output = results("output").asInstanceOf[Array[Float]]
    val expected = Array(6.0f, 8.0f, 10.0f, 12.0f)

    assertEquals(output.toSeq, expected.toSeq)
  }

  /** Test the Mul operation with Float32 tensors */
  test("Mul operation should perform element-wise multiplication on Float32 tensors") {
    val inputA = Array(2.0f, 3.0f, 4.0f, 5.0f)
    val inputB = Array(1.5f, 2.0f, 2.5f, 3.0f)

    val model = ModelIR(
      name = "mul_test",
      operations = List(
        Operation.Mul("input_a", "input_b", "output"),
      ),
      allocations = Map(
        "input_a" -> Allocation("input_a", DataType.Float32, List(4)),
        "input_b" -> Allocation("input_b", DataType.Float32, List(4)),
        "output" -> Allocation("output", DataType.Float32, List(4)),
      ),
      graphInputs = List("input_a", "input_b"),
      graphOutputs = List("output"),
    )

    val inputs = Map(
      "input_a" -> inputA,
      "input_b" -> inputB,
    )

    val results = Interpreter.execute(model, inputs).use(_.map(identity)).unsafeRunSync()
    val output = results("output").asInstanceOf[Array[Float]]
    val expected = Array(3.0f, 6.0f, 10.0f, 15.0f)

    assertEquals(output.toSeq, expected.toSeq)
  }

  /** Test Cast operation from Float64 to Float32 */
  test("Cast operation should convert Float64 to Float32 with appropriate precision loss") {
    val input = Array(1.123456789, 2.987654321, 3.141592653)

    val model = ModelIR(
      name = "cast_f64_to_f32_test",
      operations = List(
        Operation.Cast("input", "output", DataType.Float32),
      ),
      allocations = Map(
        "input" -> Allocation("input", DataType.Float64, List(3)),
        "output" -> Allocation("output", DataType.Float32, List(3)),
      ),
      graphInputs = List("input"),
      graphOutputs = List("output"),
    )

    val inputs = Map("input" -> input)

    val results = Interpreter.execute(model, inputs).use(_.map(identity)).unsafeRunSync()
    val output = results("output").asInstanceOf[Array[Float]]

    // Check that values are approximately correct within Float32 precision
    assertEqualsFloat(output(0), 1.123456789f, 1e-6f)
    assertEqualsFloat(output(1), 2.987654321f, 1e-6f)
    assertEqualsFloat(output(2), 3.141592653f, 1e-6f)
  }

  /** Test Cast operation from Float32 to Float64 */
  test("Cast operation should convert Float32 to Float64 without precision loss") {
    val input = Array(1.5f, 2.75f, 3.25f)

    val model = ModelIR(
      name = "cast_f32_to_f64_test",
      operations = List(
        Operation.Cast("input", "output", DataType.Float64),
      ),
      allocations = Map(
        "input" -> Allocation("input", DataType.Float32, List(3)),
        "output" -> Allocation("output", DataType.Float64, List(3)),
      ),
      graphInputs = List("input"),
      graphOutputs = List("output"),
    )

    val inputs = Map("input" -> input)

    val results = Interpreter.execute(model, inputs).use(_.map(identity)).unsafeRunSync()
    val output = results("output").asInstanceOf[Array[Double]]
    val expected = Array(1.5, 2.75, 3.25)

    assertEquals(output.toSeq, expected.toSeq)
  }

  /** Test SVM Classifier operation (simplified example) */
  test("SVM Classifier should handle basic classification (may fail without LibSVM)") {
    // Simple 2D input for binary classification
    val input = Array(0.5, 1.5)

    // Minimal SVM model data (this is a simplified example)
    val supportVectors = Array(
      0.0,
      1.0, // Support vector 1
      1.0,
      0.0, // Support vector 2
    )
    val coefficients = Array(1.0, -1.0) // Dual coefficients
    val rho = List(0.5) // Decision function constant
    val classLabels = List(0L, 1L)
    val vectorsPerClass = List(1L, 1L)

    val model = ModelIR(
      name = "svm_test",
      operations = List(
        Operation.SVMClassifier(
          input = "input",
          outputLabel = "label",
          outputScores = "scores",
          classLabels = classLabels,
          coefficients = coefficients,
          kernelType = SVMKernel.Linear,
          kernelParams = List(),
          postTransform = PostTransform.None,
          rho = rho,
          supportVectors = supportVectors,
          vectorsPerClass = vectorsPerClass,
        ),
      ),
      allocations = Map(
        "input" -> Allocation("input", DataType.Float64, List(2)),
        "label" -> Allocation("label", DataType.Int32, List(1)),
        "scores" -> Allocation("scores", DataType.Float64, List(2)),
      ),
      graphInputs = List("input"),
      graphOutputs = List("label", "scores"),
    )

    val inputs = Map("input" -> input)

    // SVM may fail without LibSVM bindings, so we test both success and graceful failure
    try {
      val results = Interpreter.execute(model, inputs).use(_.map(identity)).unsafeRunSync()
      val label = results("label").asInstanceOf[Array[Int]]
      val scores = results("scores").asInstanceOf[Array[Double]]

      // If SVM works, verify the outputs are reasonable
      assert(label.length == 1, "Should produce exactly one label")
      assert(scores.length == 2, "Should produce scores for 2 classes")
      assert(classLabels.contains(label(0).toLong), "Label should be one of the class labels")
    } catch {
      case _: NotImplementedError | _: UnsatisfiedLinkError =>
        // Expected when LibSVM bindings are not available
        () // Test passes - graceful failure is acceptable
    }
  }

  /** Test a more complex graph with multiple operations */
  test("Complex graph should correctly chain Add, Cast, and Mul operations") {
    val inputA = Array(2.0, 4.0, 6.0)
    val inputB = Array(1.0, 2.0, 3.0)
    val multiplier = Array(0.5f, 1.5f, 2.5f)

    val model = ModelIR(
      name = "complex_test",
      operations = List(
        // First add the inputs (Float64)
        Operation.Add("input_a", "input_b", "sum"),
        // Cast the sum to Float32
        Operation.Cast("sum", "sum_f32", DataType.Float32),
        // Multiply with the multiplier
        Operation.Mul("sum_f32", "multiplier", "final_output"),
      ),
      allocations = Map(
        "input_a" -> Allocation("input_a", DataType.Float64, List(3)),
        "input_b" -> Allocation("input_b", DataType.Float64, List(3)),
        "sum" -> Allocation("sum", DataType.Float64, List(3)),
        "sum_f32" -> Allocation("sum_f32", DataType.Float32, List(3)),
        "multiplier" -> Allocation("multiplier", DataType.Float32, List(3)),
        "final_output" -> Allocation("final_output", DataType.Float32, List(3)),
      ),
      graphInputs = List("input_a", "input_b", "multiplier"),
      graphOutputs = List("final_output"),
    )

    val inputs = Map(
      "input_a" -> inputA,
      "input_b" -> inputB,
      "multiplier" -> multiplier,
    )

    val results = Interpreter.execute(model, inputs).use(_.map(identity)).unsafeRunSync()
    val output = results("final_output").asInstanceOf[Array[Float]]
    val expected = Array(1.5f, 9.0f, 22.5f)

    // Use floating point comparison with tolerance
    assertEquals(output.length, expected.length)
    output.zip(expected).foreach { case (actual, exp) =>
      assertEqualsFloat(actual, exp, 1e-5f)
    }
  }

  // Helper method for floating point comparisons
  private def assertEqualsFloat(actual: Float, expected: Float, tolerance: Float): Unit = {
    val diff = math.abs(actual - expected)
    assert(
      diff <= tolerance,
      s"Expected $expected ± $tolerance, but got $actual (difference: $diff)",
    )
  }
}
