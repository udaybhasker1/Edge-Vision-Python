#!/usr/bin/env python3
"""
build_engine.py — VisionEdge Week 1, Track A deliverable.

Takes an ONNX model (e.g. YOLOv10 exported via torch.onnx.export) and
produces a serialized TensorRT .engine file, at FP16 precision.

Pipeline (per the Week 1 roadmap):
  1. Load + validate the ONNX graph with onnx.checker.
  2. Sanity-check the ONNX model's outputs against onnxruntime as a
     numerical reference (catches broken exports before the expensive
     TensorRT build).
  3. Parse the ONNX graph into a TensorRT network.
  4. Build a serialized FP16 engine and write it to disk.

Usage:
    python build_engine.py --onnx yolov10.onnx --engine yolov10_fp16.engine
    python build_engine.py --onnx yolov10.onnx --engine yolov10_fp16.engine \
        --input-shape 1,3,640,640 --workspace-gb 4 --precision fp16

Requires (on a CUDA/TensorRT-capable machine):
    pip install onnx onnxruntime-gpu tensorrt numpy
    (tensorrt is normally installed via the NVIDIA TensorRT .deb/.whl,
    matched to your CUDA version — see the environment checklist in
    the project roadmap, section 3.3)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np # type: ignore


def parse_args():
    p = argparse.ArgumentParser(description="Validate ONNX and build a TensorRT engine.")
    p.add_argument("--onnx", required=True, type=str, help="Path to input .onnx file")
    p.add_argument("--engine", required=True, type=str, help="Path to output .engine file")
    p.add_argument(
        "--input-shape",
        type=str,
        default="1,3,640,640",
        help="Fixed NCHW input shape used for the engine build, comma-separated (default: 1,3,640,640)",
    )
    p.add_argument(
        "--precision",
        type=str,
        choices=["fp16", "fp32"],
        default="fp16",
        help="Engine precision. FP16 is the recommended starting point per the roadmap (default: fp16)",
    )
    p.add_argument(
        "--workspace-gb",
        type=float,
        default=2.0,
        help="TensorRT builder workspace size in GiB (default: 2.0)",
    )
    p.add_argument(
        "--skip-onnxruntime-check",
        action="store_true",
        help="Skip the onnxruntime numerical sanity check (not recommended)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Step 1-2: ONNX validation
# ---------------------------------------------------------------------------

def validate_onnx(onnx_path: Path, input_shape: tuple[int, ...], run_ort_check: bool) -> None:
    import onnx # type: ignore

    print(f"[1/4] Loading ONNX model: {onnx_path}")
    model = onnx.load(str(onnx_path))

    print("[1/4] Running onnx.checker.check_model(...)")
    onnx.checker.check_model(model)
    print("[1/4] ONNX graph is structurally valid.")

    if not run_ort_check:
        print("[2/4] Skipping onnxruntime sanity check (--skip-onnxruntime-check).")
        return

    try:
        import onnxruntime as ort # type: ignore
    except ImportError:
        print(
            "[2/4] WARNING: onnxruntime not installed — skipping numerical sanity check. "
            "Install with `pip install onnxruntime-gpu` to enable it.",
            file=sys.stderr,
        )
        return

    print("[2/4] Running a dummy inference pass through onnxruntime as a reference...")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)

    input_meta = session.get_inputs()[0]
    dummy_input = np.random.rand(*input_shape).astype(np.float32)

    start = time.perf_counter()
    outputs = session.run(None, {input_meta.name: dummy_input})
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"[2/4] onnxruntime forward pass OK in {elapsed_ms:.2f} ms.")
    for i, out in enumerate(outputs):
        print(f"       output[{i}] shape={out.shape} dtype={out.dtype}")


# ---------------------------------------------------------------------------
# Step 3-4: TensorRT engine build
# ---------------------------------------------------------------------------

def build_engine(onnx_path: Path, engine_path: Path, input_shape: tuple[int, ...],
                  precision: str, workspace_gb: float) -> None:
    import tensorrt as trt # type: ignore

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    print(f"[3/4] Parsing ONNX model into TensorRT network: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("ERROR: Failed to parse ONNX model.", file=sys.stderr)
            for i in range(parser.num_errors):
                print(f"  parser error {i}: {parser.get_error(i)}", file=sys.stderr)
            sys.exit(1)
    print(f"[3/4] Parsed OK. Network has {network.num_layers} layers, "
          f"{network.num_inputs} input(s), {network.num_outputs} output(s).")

    config = builder.create_builder_config()
    workspace_bytes = int(workspace_gb * (1 << 30))
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)

    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            print("WARNING: platform reports no fast FP16 support; build will proceed anyway.",
                  file=sys.stderr)
        config.set_flag(trt.BuilderFlag.FP16)
        print("[3/4] FP16 precision flag set.")
    else:
        print("[3/4] Using FP32 precision.")

    # Fix the input shape (min == opt == max) for a predictable, static-shape
    # engine build, per the Week 1 plan.
    input_tensor = network.get_input(0)
    profile = builder.create_optimization_profile()
    profile.set_shape(input_tensor.name, input_shape, input_shape, input_shape)
    config.add_optimization_profile(profile)
    print(f"[3/4] Fixed input shape for '{input_tensor.name}': {input_shape}")

    print("[4/4] Building serialized TensorRT engine (this can take a few minutes)...")
    t0 = time.perf_counter()
    serialized_engine = builder.build_serialized_network(network, config)
    build_time = time.perf_counter() - t0

    if serialized_engine is None:
        print("ERROR: Engine build failed — see TensorRT log output above.", file=sys.stderr)
        sys.exit(1)

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    size_mb = engine_path.stat().st_size / (1024 * 1024)
    print(f"[4/4] Engine build complete in {build_time:.1f}s. "
          f"Wrote {size_mb:.1f} MB to {engine_path}")


def main():
    args = parse_args()
    onnx_path = Path(args.onnx)
    engine_path = Path(args.engine)
    input_shape = tuple(int(x) for x in args.input_shape.split(","))

    if not onnx_path.exists():
        print(f"ERROR: ONNX file not found: {onnx_path}", file=sys.stderr)
        sys.exit(1)

    validate_onnx(onnx_path, input_shape, run_ort_check=not args.skip_onnxruntime_check)
    build_engine(onnx_path, engine_path, input_shape, args.precision, args.workspace_gb)

    print("\nDone. Next step: run benchmarks/benchmark_engine.py against "
          f"{engine_path} to confirm single-image FPS.")


if __name__ == "__main__":
    main()
