"""
trt_engine.py — VisionEdge Week 2, Track B support module.

A small reusable wrapper around a deserialized TensorRT engine, so the
Week 2 inference loop (and later, the multi-stream orchestrator in Week 4)
don't each re-implement the CUDA buffer/binding boilerplate that
benchmarks/benchmark_engine.py used in Week 1.

Requires: tensorrt, pycuda, numpy
"""

import numpy as np


class TRTEngine:
    """Loads a serialized .engine file and runs synchronous inference on a
    single input tensor, returning the output as a numpy array."""

    def __init__(self, engine_path: str, input_shape: tuple[int, ...]):
        import tensorrt as trt # type: ignore
        import pycuda.autoinit  # type: ignore # noqa: F401 — initializes the CUDA context
        import pycuda.driver as cuda # type: ignore

        self._cuda = cuda
        logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        self.context.set_input_shape(self.input_name, input_shape)

        self.input_shape = input_shape
        self.output_shape = tuple(self.context.get_tensor_shape(self.output_name))

        self._host_output = np.empty(self.output_shape, dtype=np.float32)
        input_nbytes = int(np.prod(input_shape)) * np.dtype(np.float32).itemsize
        self._d_input = cuda.mem_alloc(input_nbytes)
        self._d_output = cuda.mem_alloc(self._host_output.nbytes)
        self._stream = cuda.Stream()

        self.context.set_tensor_address(self.input_name, int(self._d_input))
        self.context.set_tensor_address(self.output_name, int(self._d_output))

    def infer(self, input_array: np.ndarray) -> np.ndarray:
        """Runs one synchronous inference pass. input_array must already be
        contiguous float32 in the engine's expected NCHW shape. This is the
        Week 2 "still CPU-side" path — Week 3 replaces the H2D copy below
        with a direct GPU-pointer bind."""
        assert input_array.shape == self.input_shape, (
            f"expected {self.input_shape}, got {input_array.shape}"
        )
        cuda = self._cuda
        cuda.memcpy_htod_async(self._d_input, np.ascontiguousarray(input_array), self._stream)
        self.context.execute_async_v3(stream_handle=self._stream.handle)
        cuda.memcpy_dtoh_async(self._host_output, self._d_output, self._stream)
        self._stream.synchronize()
        return self._host_output.copy()
