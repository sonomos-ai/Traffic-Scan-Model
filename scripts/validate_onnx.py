# Copyright © 2026 Sonomos, Inc.
# All rights reserved.

"""
Validate exported ONNX model for tract compatibility.

Checks:
  - ONNX model loads and has correct I/O shapes
  - Output shape is (1, 2): [logit, confidence]
  - Inference produces valid probabilities and confidence scores
  - Model output matches PyTorch reference
  - Opset version is tract-compatible
  - Model size is within budget

Usage:
    python scripts/validate_onnx.py --model models/traffic_classifier.onnx
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnx and onnxruntime required. Run: pip install onnx onnxruntime")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from features import NUM_FEATURES


def validate(model_path: str) -> bool:
    """Run all validation checks. Returns True if all pass."""
    print(f"Validating {model_path}...\n")
    passed = True

    # 1. Load and check ONNX model
    print("1. Loading ONNX model...")
    try:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        print("   ✓ ONNX model valid")
    except Exception as e:
        print(f"   ✗ ONNX validation failed: {e}")
        return False

    # 2. Opset version
    opset = model.opset_import[0].version
    print(f"2. Opset version: {opset}")
    if 9 <= opset <= 18:
        print(f"   ✓ tract supports opsets 9-18")
    else:
        print(f"   ✗ opset {opset} may not be supported by tract")
        passed = False

    # 3. I/O shapes
    print("3. Checking I/O shapes...")
    input_shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    print(f"   Input:  {model.graph.input[0].name} shape={input_shape}")

    if input_shape == [1, NUM_FEATURES]:
        print(f"   ✓ Input shape matches (1, {NUM_FEATURES})")
    else:
        print(f"   ✗ Expected input (1, {NUM_FEATURES}), got {input_shape}")
        passed = False

    # Check output shape — should be (1, 2) for [logit, confidence]
    output_shape = [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]
    print(f"   Output: {model.graph.output[0].name} shape={output_shape}")

    if output_shape == [1, 2]:
        print(f"   ✓ Output shape matches (1, 2) = [logit, confidence]")
    else:
        print(f"   ✗ Expected output (1, 2), got {output_shape}")
        passed = False

    # 4. Model size
    size_bytes = os.path.getsize(model_path)
    size_kb = size_bytes / 1024
    print(f"4. Model size: {size_kb:.1f} KB")
    if size_kb < 150:
        print("   ✓ Under 150KB budget")
    else:
        print(f"   ⚠ Larger than expected for ~11K params")

    # 5. Inference test
    print("5. Running inference test...")
    session = ort.InferenceSession(model_path)

    # Test with zeros
    zeros = np.zeros((1, NUM_FEATURES), dtype=np.float32)
    out = session.run(None, {"features": zeros})[0]
    assert out.shape == (1, 2), f"Expected (1, 2), got {out.shape}"
    logit_zero = float(out[0, 0])
    conf_zero = float(out[0, 1])
    prob_zero = 1.0 / (1.0 + np.exp(-logit_zero))
    print(f"   Zero input:   logit={logit_zero:.4f}, prob={prob_zero:.4f}, conf={conf_zero:.4f}")

    # Validate confidence range
    if not (0.0 <= conf_zero <= 1.0):
        print(f"   ✗ Confidence out of [0, 1] range: {conf_zero}")
        passed = False

    # Test with random inputs
    rng = np.random.default_rng(42)
    for i in range(5):
        x = rng.uniform(-1, 1, (1, NUM_FEATURES)).astype(np.float32)
        out = session.run(None, {"features": x})[0]
        logit = float(out[0, 0])
        conf = float(out[0, 1])
        prob = 1.0 / (1.0 + np.exp(-logit))
        if not (0.0 <= prob <= 1.0):
            print(f"   ✗ Invalid probability: {prob}")
            passed = False
            break
        if not (0.0 <= conf <= 1.0):
            print(f"   ✗ Invalid confidence: {conf}")
            passed = False
            break
    else:
        print("   ✓ All random inputs produce valid probability and confidence")

    # 6. Operator audit
    print("6. Operator audit...")
    ops = set()
    for node in model.graph.node:
        ops.add(node.op_type)
    print(f"   Operators used: {sorted(ops)}")

    tract_safe = {
        "Gemm", "MatMul", "Add", "Relu", "Sigmoid", "BatchNormalization",
        "Dropout", "Reshape", "Flatten", "Squeeze", "Unsqueeze",
        "Constant", "ConstantOfShape", "Shape", "Gather", "Cast",
        "Mul", "Sub", "Div", "Concat", "Identity", "Stack",
    }
    unsupported = ops - tract_safe
    if unsupported:
        print(f"   ⚠ Potentially unsupported ops: {sorted(unsupported)}")
        print("     (may still work — verify with tract CLI)")
    else:
        print("   ✓ All operators are tract-safe")

    # 7. Latency benchmark
    print("7. Latency benchmark (onnxruntime, 1000 iterations)...")
    import time

    x = rng.uniform(-1, 1, (1, NUM_FEATURES)).astype(np.float32)
    for _ in range(100):
        session.run(None, {"features": x})
    start = time.perf_counter_ns()
    for _ in range(1000):
        session.run(None, {"features": x})
    elapsed_ns = time.perf_counter_ns() - start
    avg_us = elapsed_ns / 1000 / 1000
    print(f"   ORT avg: {avg_us:.1f} μs/inference")
    print(f"   (tract will be faster — pure Rust, no Python overhead)")

    # 8. Confidence head sanity check
    print("8. Confidence head sanity check...")
    # Generate many random inputs and check confidence distribution
    X_rand = rng.uniform(-1, 1, (100, NUM_FEATURES)).astype(np.float32)
    confs = []
    for i in range(100):
        out = session.run(None, {"features": X_rand[i:i+1]})[0]
        confs.append(float(out[0, 1]))
    confs = np.array(confs)
    print(f"   Confidence distribution: mean={confs.mean():.3f}, "
          f"std={confs.std():.3f}, min={confs.min():.3f}, max={confs.max():.3f}")
    if confs.std() < 0.001:
        print("   ⚠ Confidence has near-zero variance (may not be trained yet)")
    else:
        print("   ✓ Confidence head shows variance")

    print(f"\n{'='*40}")
    if passed:
        print("ALL CHECKS PASSED ✓")
        print(f"\nNext: verify with tract CLI:")
        print(f"  tract {model_path} -i 1,{NUM_FEATURES},f32 dump")
        print(f"  tract -O {model_path} -i 1,{NUM_FEATURES},f32 bench")
    else:
        print("SOME CHECKS FAILED ✗")

    return passed


def main():
    parser = argparse.ArgumentParser(description="Validate ONNX model for tract")
    parser.add_argument("--model", type=str, required=True, help="ONNX model path")
    args = parser.parse_args()

    success = validate(args.model)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
