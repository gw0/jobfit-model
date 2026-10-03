"""Unit tests for export.py's ONNX graph helpers, on a tiny graph. Skipped without onnx/onnxruntime."""

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
onnxruntime = pytest.importorskip("onnxruntime")

import export  # noqa: E402
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

WEIGHT = np.random.default_rng(0).normal(size=(4, 3)).astype(np.float32)


def _model():
    """x -> down_proj MatMul (weight) -> y; y @ y.T (activation by activation) -> z."""
    nodes = [helper.make_node("MatMul", ["x", "w"], ["y"], name="/layers.0/mlp/down_proj/MatMul"),
             helper.make_node("Transpose", ["y"], ["yt"], perm=[1, 0]),
             helper.make_node("MatMul", ["y", "yt"], ["z"], name="/layers.0/self_attn/MatMul")]
    graph = helper.make_graph(
        nodes, "tiny", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3]),
         helper.make_tensor_value_info("z", TensorProto.FLOAT, [2, 2])],
        [numpy_helper.from_array(WEIGHT, "w")])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)  # ORT 1.21's max


def test_matmul_names_splits_weight_from_activation_matmuls(tmp_path):
    path = tmp_path / "model.onnx"
    onnx.save(_model(), str(path))
    assert export.matmul_names(path) == (["/layers.0/mlp/down_proj/MatMul"], ["/layers.0/self_attn/MatMul"])


def test_dequantize_weights_keeps_outputs_close_and_stores_int8():
    model = _model()
    export.dequantize_weights(model, {"/layers.0/mlp/down_proj/MatMul"})
    onnx.checker.check_model(model)
    assert [i.data_type for i in model.graph.initializer if i.name.endswith("_int8")] == [TensorProto.INT8]
    assert "w" not in {i.name for i in model.graph.initializer}

    x = np.random.default_rng(1).normal(size=(2, 4)).astype(np.float32)
    y, z = onnxruntime.InferenceSession(model.SerializeToString()).run(None, {"x": x})
    np.testing.assert_allclose(y, x @ WEIGHT, atol=0.05)
    np.testing.assert_allclose(z, y @ y.T, atol=1e-4)
