# VisionEdge — AI-Powered Real-Time Video Analytics Platform

## Overview

VisionEdge is a real-time AI video analytics platform designed for intelligent surveillance and smart-city applications.

The system processes video streams through an AI-powered pipeline and delivers low-latency video output using WebRTC streaming technology.

The objective of VisionEdge is to build a scalable video processing system capable of handling multiple high-resolution camera streams by combining:

- Real-time video decoding
- AI-based object detection
- GPU-accelerated inference
- Low-latency WebRTC streaming

The project focuses on replacing traditional CPU-heavy video processing workflows with an optimized hardware-aware architecture.

---

# Problem Statement

Traditional computer vision video pipelines face several performance challenges:

- CPU-based video decoding creates processing bottlenecks
- Frequent CPU-GPU memory transfers increase latency
- Real-time AI inference becomes difficult for multiple high-resolution streams
- Traditional streaming methods introduce additional delay

VisionEdge addresses these challenges by designing a modular AI video pipeline:

---

# System Architecture

Current VisionEdge architecture:
             Video Source
                  |
                  |
                  ↓

             decoder.py
      (Extracts video frames)

                  |
                  |
                  ↓

         inference_loop.py
    (Controls AI processing pipeline)

                  |
                  |
                  ↓

         video_track.py
   (Connects processed frames to WebRTC)

                  |
                  |
                  ↓

             server.py
      (WebRTC signaling server)

                  |
                  |
                  ↓

          React Frontend
      (Live video dashboard)
      
---

# Key Features

## Real-Time Video Streaming

Implemented:

- WebRTC-based low latency streaming
- Browser-based video playback
- SDP offer/answer exchange
- Peer connection management
- Backend-to-frontend video transmission

---

## Modular AI Processing Pipeline

VisionEdge separates video streaming and AI processing into independent modules.

The pipeline consists of:

- Frame decoding
- Image preprocessing
- AI inference
- Frame processing
- WebRTC streaming

This modular approach allows future integration of multiple AI models and multiple camera streams.

---

## GPU Acceleration Ready

The architecture is designed to support NVIDIA GPU acceleration.

Planned optimizations:

- NVIDIA NVDEC hardware video decoding
- TensorRT optimized inference
- CUDA acceleration
- GPU memory optimization
- Zero-copy processing pipeline

---

# Technology Stack

## Backend

| Technology | Purpose |
|------------|---------|
| Python | Backend development |
| aiohttp | Web server and signaling |
| aiortc | WebRTC implementation |
| OpenCV | Image processing |
| PyAV | Video decoding |
| NumPy | Numerical operations |

---

## Artificial Intelligence

| Technology | Purpose |
|------------|---------|
| YOLO | Object detection |
| PyTorch | Deep learning framework |
| ONNX | Model conversion format |
| TensorRT | Optimized inference engine |

---

## GPU Computing

| Technology | Purpose |
|------------|---------|
| CUDA | GPU acceleration |
| NVDEC | Hardware video decoding |
| CuPy | GPU array processing |

---

## Frontend

| Technology | Purpose |
|------------|---------|
| React | User interface |
| WebRTC | Real-time video communication |
| Vite | Frontend development server |

---

---

# Performance Goals

| Metric | Target |
|--------|--------|
| Streaming Latency | <100ms |
| Video Processing | Real-time |
| Inference | TensorRT optimized |
| Decode | GPU accelerated |
| CPU Usage | Reduced |
| Camera Support | Multiple streams |

---

# Installation

## Backend Setup

Clone repository:

```bash
git clone <repository-url>

cd VisionEdge/backend

Author
H.A. Uday Bhasker

B.Tech Electronics and Communication Engineering

Areas of Interest:

Artificial Intelligence
Computer Vision
Deep Learning
GPU Accelerated Computing
Real-Time Video Processing
