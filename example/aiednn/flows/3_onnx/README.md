# Flow 3: ONNX

Two front ends onto the same partition graph API:

- **a) `aiegraph.dnn_onnx_graph.OnnxGraph`**: imports an ONNX model directly, one graph op per node; the standard
  opset maps onto aiednn ops, Q/DQ int8 included. No onnxruntime is needed: unsupported partitions run through
  `onnx.reference`.
  ```python
  og = OnnxGraph.load(build_graph_lib(dir), "model.onnx"); print(og.report()); y = og.run({"x": x})
  ```
- **b) The ONNX Runtime plugin execution provider** (`aiegraph.dnn_ort_ep`, onnxruntime >= 1.23):
  GetCapability = `get_partitions`, Compile = `partition_compile`, Compute = `compiled_partition_execute`.
  ORT's CPU EP runs what aiednn does not claim.
  ```python
  sess = dnn_ort_ep.session("model.onnx")      # builds + registers libaiednn_ort_ep.so once
  y = sess.run(None, {"x": x})
  ```

```bash
python example/aiednn/flows/3_onnx/run.py [--model resnet18 | --onnx <file.onnx>]   # --no-ep: only a)
```

```
4 partitions: 4 supported, 0 unsupported; 7 / 7 ONNX nodes inside supported partitions
a) OnnxGraph: max rel diff vs onnxruntime CPU 2.61e-07
b) ORT plugin EP: providers ['aiednn', 'CPUExecutionProvider'], ... max rel diff 2.61e-07
PASS: flow 3 ONNX
```

The EP build fetches ONNX Runtime's C headers once (or set `AIEDNN_ORT_INCLUDE`). For a board, an ONNX
graph is emitted as a C app with `aiegraph.dnn_graph_app.emit_graph_app` (its docstring has the ONNX
ResNet-18 example).
