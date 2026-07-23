# VisionEdge — Week 1 Deliverables (Track A + Track B)

## What's here

```
visionedge/
├── backend/
│   ├── engine_build/
│   │   └── build_engine.py      # Track A: ONNX validate + TensorRT engine compile
│   └── streaming/
│       ├── server.py            # Track B: aiortc server + HTTP signaling
│       └── static/index.html    # Raw JS smoke-test page (not the deliverable, just a quick check)
├── benchmarks/
│   └── benchmark_engine.py      # Track A: single-image FPS benchmark
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Track B: top-level React app
│   │   ├── main.jsx
│   │   ├── components/VideoPlayer.jsx
│   │   └── hooks/useWebRTC.js
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
└── media/                       # put a sample.mp4 here for the server to stream
```

## Track A — Model Compilation

```bash
# 1. Export your YOLOv10 checkpoint to ONNX first (not included here — that's
#    the ML engineer's existing torch.onnx.export step referenced in the roadmap).

# 2. Validate + compile:
python backend/engine_build/build_engine.py \
    --onnx yolov10.onnx \
    --engine yolov10_fp16.engine \
    --input-shape 1,3,640,640 \
    --precision fp16

# 3. Benchmark single-image FPS:
python benchmarks/benchmark_engine.py \
    --engine yolov10_fp16.engine \
    --input-shape 1,3,640,640 \
    --iterations 200
```

Requires a CUDA + TensorRT machine: `pip install onnx onnxruntime-gpu tensorrt pycuda numpy`.

**Exit criterion check:** `build_engine.py` exits 0 and writes a `.engine` file only
after both the ONNX-checker validation and the onnxruntime numerical sanity
check pass — so a successful run *is* "a working .engine file that runs
inference on a single frame," and `benchmark_engine.py` gives you the FPS
number to record.

## Track B — WebRTC Foundation

```bash
# Terminal 1 — backend
pip install aiortc aiohttp av
python backend/streaming/server.py --video media/sample.mp4 --port 8080

# Terminal 2 — frontend
cd frontend
npm install
npm run dev
# open http://localhost:5173
```

**Exit criterion check:** opening the React app at `localhost:5173` should
show `media/sample.mp4` playing live via WebRTC end to end, sourced from the
Python aiortc server. If you just want a fast smoke test without the React
build, `http://localhost:8080/` serves a minimal raw-JS page that does the
same offer/answer handshake.

## What I could not verify in this environment

This sandbox has no GPU, no CUDA/TensorRT, no internet access for
`pip`/`npm install`, so I could not execute or hardware-test either script
here. The code targets the real TensorRT 10.x Python API (`build_serialized_network`,
`execute_async_v3`, `set_tensor_address`) and the real aiortc API
(`RTCPeerConnection`, `MediaPlayer`, `MediaRelay`), matching what the roadmap's
section 3 stack specifies — but you should run both on your actual CUDA/TensorRT
box and confirm against the Week 1 exit criteria before treating this as
submission-final. Two things worth double-checking on first run:
- The output tensor name assumption in `benchmark_engine.py` (`engine.get_tensor_name(1)`)
  — if your YOLOv10 export has multiple output tensors, adjust accordingly.
- TensorRT API changes between minor versions (e.g. 8.x vs 10.x) — the script
  targets the 10.x API named in the roadmap's tech stack (section 3.1).
