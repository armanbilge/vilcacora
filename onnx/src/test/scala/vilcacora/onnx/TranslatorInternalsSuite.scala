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
import com.google.protobuf.ByteString
import munit.FunSuite
import java.nio.{ByteBuffer, ByteOrder}

class TranslatorInternalsSuite extends FunSuite {

  // --- Data Type Mapping ---

  test("fromOnnxDataType should map known type codes") {
    assertEquals(Translator.fromOnnxDataType(1), Right(DataType.Float32))
    // ONNX code 6 is INT32, 7 is INT64
    assertEquals(Translator.fromOnnxDataType(6), Right(DataType.Int32))
    assertEquals(Translator.fromOnnxDataType(7), Right(DataType.Int64))
  }
  test("fromOnnxDataType should fail on unknown type codes") {
    assert(Translator.fromOnnxDataType(999).isLeft)
  }

  // --- Shape Parsing Logic ---

  test("parseShape should correctly parse a static shape") {
    val shapeProto = TensorShapeProto(
      dim = Seq(
        TensorShapeProto.Dimension(value = TensorShapeProto.Dimension.Value.DimValue(1)),
        TensorShapeProto.Dimension(value = TensorShapeProto.Dimension.Value.DimValue(30)),
      ),
    )
    assertEquals(Translator.parseShape(shapeProto), Right(List(1, 30)))
  }

  test("parseShape should fail on a dynamic dimension parameter") {
    val shapeProto = TensorShapeProto(
      dim =
        Seq(TensorShapeProto.Dimension(value = TensorShapeProto.Dimension.Value.DimParam("unk_0"))),
    )
    val result = Translator.parseShape(shapeProto)
    assert(result.isLeft, "Should fail for dynamic named dimension")
    result.left.foreach(err => assert(err.contains("dynamic dimension parameter")))
  }

  test("parseShape should fail on an empty (unknown) dimension") {
    val shapeProto =
      TensorShapeProto(dim =
        Seq(TensorShapeProto.Dimension(value = TensorShapeProto.Dimension.Value.Empty)),
      )
    val result = Translator.parseShape(shapeProto)
    assert(result.isLeft, "Should fail for un-inferred dimension")
    result.left.foreach(err => assert(err.contains("unknown dimension")))
  }

  // --- Allocation Creation ---

  test("createAllocation should build a valid Allocation from a ValueInfoProto") {
    val shape = TensorShapeProto(dim =
      Seq(TensorShapeProto.Dimension(TensorShapeProto.Dimension.Value.DimValue(10))),
    )
    val typeProto = TypeProto(value =
      TypeProto.Value.TensorType(TypeProto.Tensor(elemType = 1, shape = Some(shape))),
    )
    val valueInfo = ValueInfoProto(name = "test_tensor", `type` = Some(typeProto))

    val result = Translator.createAllocation(valueInfo, None)

    val expected = Allocation("test_tensor", DataType.Float32, List(10), None)
    assertEquals(result, Right(expected))
  }

  test("createAllocationFromInitializer should build an Allocation containing data") {
    val rawBytes = Array[Byte](1, 2, 3, 4)
    val tensorProto = TensorProto(
      name = "weights",
      dims = Seq(4),
      dataType = 3, // INT8
      rawData = ByteString.copyFrom(rawBytes),
    )

    val result = Translator.createAllocationFromInitializer(tensorProto)
    assert(result.isRight)
    result.foreach { alloc =>
      assertEquals(alloc.name, "weights")
      assertEquals(alloc.dataType, DataType.Int8)
      assert(alloc.initialData.isDefined)
      alloc.initialData.foreach(data => assert(data.sameElements(rawBytes)))
    }
  }

  // --- Byte Extraction Logic ---

  test("extractBytes should prioritize raw_data field") {
    val rawBytes = Array[Byte](0, 0, -128, 63) // 1.0f
    val tensor = TensorProto(dims = Seq(1), dataType = 1, rawData = ByteString.copyFrom(rawBytes))

    val result = Translator.extractBytes(tensor, DataType.Float32)
    assert(result.map(_.toSeq) == Right(rawBytes.toSeq))
  }

  test("extractBytes should fall back to typed data fields") {
    val tensor = TensorProto(dims = Seq(2), dataType = 1, floatData = Seq(1.0f, -1.0f))
    val expectedBytes =
      ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putFloat(1.0f).putFloat(-1.0f).array()

    val result = Translator.extractBytes(tensor, DataType.Float32)
    assert(result.map(_.toSeq) == Right(expectedBytes.toSeq))
  }
}
