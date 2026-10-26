#!/usr/bin/env python3
"""Build the Triton model repository: ONNX MiniLM with fused mean pooling and L2 normalization.

Usage: python scripts/prepare_triton_models.py [SEQ_LEN]   (default 64)
Writes triton_repo/model_repository/minilm_nobatch and minilm_batch (C5a and C5).
"""
import os
import shutil
import sys

import numpy as np
import onnx
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from onnx import TensorProto, helper, numpy_helper
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEQ_LEN = int(sys.argv[1]) if len(sys.argv) > 1 else 64
REPO = "triton_repo/model_repository"
OUT_NAME = "sentence_embedding"

model = onnx.load(hf_hub_download(MODEL_NAME, "onnx/model.onnx"))
opset = max((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), default=0)
assert opset >= 13, f"need opset >= 13, got {opset}"

graph = model.graph
input_names = [i.name for i in graph.input]
hidden_name = graph.output[0].name
hidden_dim = graph.output[0].type.tensor_type.shape.dim[-1].dim_value or 384
print("inputs:", input_names, "| hidden output:", hidden_name, "| dim:", hidden_dim, "| opset:", opset)
assert "attention_mask" in input_names


def add_init(name, arr):
    graph.initializer.append(numpy_helper.from_array(arr, name))


add_init("pool_axes_unsq", np.array([2], dtype=np.int64))
add_init("pool_axes_sum", np.array([1], dtype=np.int64))
add_init("pool_eps", np.array(1e-9, dtype=np.float32))

graph.node.extend([
    helper.make_node("Cast", ["attention_mask"], ["pool_mask_f"], to=TensorProto.FLOAT),
    helper.make_node("Unsqueeze", ["pool_mask_f", "pool_axes_unsq"], ["pool_mask_u"]),
    helper.make_node("Mul", [hidden_name, "pool_mask_u"], ["pool_masked"]),
    helper.make_node("ReduceSum", ["pool_masked", "pool_axes_sum"], ["pool_sum"], keepdims=0),
    helper.make_node("ReduceSum", ["pool_mask_f", "pool_axes_sum"], ["pool_count"], keepdims=1),
    helper.make_node("Clip", ["pool_count", "pool_eps"], ["pool_count_c"]),
    helper.make_node("Div", ["pool_sum", "pool_count_c"], ["pool_mean"]),
    helper.make_node("LpNormalization", ["pool_mean"], [OUT_NAME], axis=1, p=2),
])
del graph.output[:]
graph.output.append(helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, ["batch_size", hidden_dim]))
onnx.checker.check_model(model)

os.makedirs("triton_repo/build", exist_ok=True)
fused = "triton_repo/build/minilm_fused.onnx"
onnx.save(model, fused)

# Verify against the PyTorch reference before shipping it to Triton.
st = SentenceTransformer(MODEL_NAME, device="cpu")
queries = ["reset my password", "refund for double charge", "app crashes on login",
           "cannot export report to csv", "invoice shows wrong tax", "how do I delete my account"]
enc = st.tokenizer(queries, padding="max_length", truncation=True, max_length=SEQ_LEN, return_tensors="np")
feeds = {n: (np.asarray(enc[n]) if n in enc else np.zeros_like(enc["input_ids"])).astype(np.int64)
         for n in input_names}
got = ort.InferenceSession(fused, providers=["CPUExecutionProvider"]).run([OUT_NAME], feeds)[0]
ref = st.encode(queries, normalize_embeddings=True)
cos = (got * ref).sum(axis=1)
longest = max(len(st.tokenizer(q)["input_ids"]) for q in queries)
print(f"min cosine vs PyTorch: {cos.min():.6f} | longest sample query: {longest} tokens (pad length {SEQ_LEN})")
assert cos.min() > 0.9999, "fused model does not match PyTorch"


def config(name, batching):
    inputs = ",\n".join(
        f'  {{ name: "{n}" data_type: TYPE_INT64 dims: [ {SEQ_LEN} ] }}' for n in input_names)
    dyn = "dynamic_batching { max_queue_delay_microseconds: 0 }\n" if batching else ""
    return f'''name: "{name}"
backend: "onnxruntime"
max_batch_size: 32
input [
{inputs}
]
output [
  {{ name: "{OUT_NAME}" data_type: TYPE_FP32 dims: [ {hidden_dim} ] }}
]
instance_group [ {{ count: 4 kind: KIND_CPU }} ]
{dyn}parameters {{ key: "intra_op_thread_count" value: {{ string_value: "1" }} }}
parameters {{ key: "inter_op_thread_count" value: {{ string_value: "1" }} }}
'''


for name, batching in (("minilm_nobatch", False), ("minilm_batch", True)):
    version_dir = os.path.join(REPO, name, "1")
    os.makedirs(version_dir, exist_ok=True)
    shutil.copy(fused, os.path.join(version_dir, "model.onnx"))
    with open(os.path.join(REPO, name, "config.pbtxt"), "w") as f:
        f.write(config(name, batching))
print("wrote", REPO)
