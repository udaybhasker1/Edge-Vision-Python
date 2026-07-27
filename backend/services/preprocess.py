"""
preprocess.py — VisionEdge Week 2 support module.

Converts a decoded RGB24 numpy frame (HxWx3, uint8) from decoder.py into
the NCHW float32 tensor TensorRT expects. This is still a CPU-side numpy
op for Week 2 — Week 3 replaces this with a GPU-resident CuPy equivalent
so the frame never has to leave VRAM.
"""

import cv2 # type: ignore
import numpy as np # type: ignore


def preprocess_frame(frame_rgb: np.ndarray, input_hw: tuple[int, int]) -> np.ndarray:
    """
    frame_rgb: HxWx3 uint8, RGB24 (as produced by decoder.DecodedFrame.array)
    input_hw: (height, width) expected by the TensorRT engine

    Returns: 1x3xHxW float32, values in [0, 1], contiguous.
    """
    h, w = input_hw
    resized = cv2.resize(frame_rgb, (w, h))
    normalized = resized.astype(np.float32) / 255.0
    chw = np.transpose(normalized, (2, 0, 1))
    return np.ascontiguousarray(np.expand_dims(chw, axis=0))
