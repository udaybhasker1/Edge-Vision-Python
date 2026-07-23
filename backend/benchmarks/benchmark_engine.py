#!/usr/bin/env python3
"""
benchmark_engine.py — VisionEdge Week 1, Track A deliverable.

Loads a compiled TensorRT .engine file, runs repeated inference on a
single static image (or synthetic random tensor if no image is given),
and reports FPS. This is the "before" reference used again in the
Mid-Project performance audit (native PyTorch vs. TensorRT, >=3x gate).

Usage:
    python benchmark_engine.py --engine yolov10_fp16.engine
    python benchmark_engine.py --engine yolov10_fp16.engine --image sample.jpg \
        --input-shape 1,3,640,640 --iterations 500 --warmup 50
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np # type: ignore


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark FPS of a TensorRT engine on a single static image.")
    p.add_argument("--engine", required=True, type=str, help="Path to .engine file")
    p.add_argument("--image", type=str, default=None,
                   help="Path to a static image file. If omitted, a random tensor is used instead.")
    p.add_argument("--input-shape", type=str, default="1,3,640,640",
                   help="NCHW input shape matching the engine build (default: 1,3,640,640)")
    p.add_argument("--iterations", type=int, default=200, help="Timed inference iterations (default: 200)")
    p.add_argument("--warmup", type=int, default=20, help="Warmup iterations excluded from timing (default: 20)")
    return p.parse_args()


def load_image_as_tensor(image_path: str, input_shape: tuple[int, ...]) -> np.ndarray:
    """Loads an image from disk, resizes to the engine's expected HxW, normalizes to [0,1], NCHW float32."""
    import cv2 # type: ignore

    _, c, h, w = input_shape
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img = cv2.resize(img, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)  # -> NCHW
    if c != 3:
        raise ValueError(f"Expected 3 input channels, engine expects {c}")
    return np.ascontiguousarray(img)


def main():
    args = parse_args()
    engine_path = Path(args.engine)
    input_shape = tuple(int(x) for x in args.input_shape.split(","))

    if not engine_path.exists():
        print(f"ERROR: engine file not found: {engine_path}", file=sys.stderr)
        sys.exit(1)

    import tensorrt as trt # type: ignore
    import pycuda.autoinit  # type: ignore # noqa: F401  (initializes the CUDA context)
    import pycuda.driver as cuda # type: ignore

    logger = trt.Logger(trt.Logger.WARNING)

    print(f"Loading engine: {engine_path}")
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        print("ERROR: failed to deserialize engine.", file=sys.stderr)
        sys.exit(1)

    context = engine.create_execution_context()

    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    context.set_input_shape(input_name, input_shape)

    if args.image:
        print(f"Using static image: {args.image}")
        host_input = load_image_as_tensor(args.image, input_shape)
    else:
        print("No --image given; using a random synthetic tensor.")
        host_input = np.random.rand(*input_shape).astype(np.float32)

    output_shape = tuple(context.get_tensor_shape(output_name))
    host_output = np.empty(output_shape, dtype=np.float32)

    d_input = cuda.mem_alloc(host_input.nbytes)
    d_output = cuda.mem_alloc(host_output.nbytes)
    stream = cuda.Stream()

    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    def run_once():
        cuda.memcpy_htod_async(d_input, host_input, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_output, d_output, stream)
        stream.synchronize()

    print(f"Warming up ({args.warmup} iterations)...")
    for _ in range(args.warmup):
        run_once()

    print(f"Benchmarking ({args.iterations} iterations) on a single static image...")
    latencies_ms = []
    t_start = time.perf_counter()
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        run_once()
        latencies_ms.append((time.perf_counter() - t0) * 1000)
    total_elapsed = time.perf_counter() - t_start

    latencies_ms = np.array(latencies_ms)
    fps = args.iterations / total_elapsed

    print("\n--- Benchmark Results ---")
    print(f"Engine:            {engine_path}")
    print(f"Input shape:       {input_shape}")
    print(f"Iterations:        {args.iterations} (warmup: {args.warmup})")
    print(f"Mean latency:      {latencies_ms.mean():.3f} ms")
    print(f"P50 latency:       {np.percentile(latencies_ms, 50):.3f} ms")
    print(f"P99 latency:       {np.percentile(latencies_ms, 99):.3f} ms")
    print(f"Throughput (FPS):  {fps:.2f}")


if __name__ == "__main__":
    main()
