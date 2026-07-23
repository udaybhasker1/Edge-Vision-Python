#!/usr/bin/env python3
"""
server.py — VisionEdge Week 1, Track B deliverable.

Minimal aiortc server that streams a static/unedited video file to a
browser peer over WebRTC. Signaling (SDP offer/answer) happens over a
simple HTTP endpoint served by aiohttp — no external signaling service
needed for this milestone.

Flow:
  1. Browser POSTs its SDP offer to /offer.
  2. Server creates an RTCPeerConnection, attaches a video track read
     from a local file via aiortc's MediaPlayer (which decodes the file
     with PyAV/FFmpeg — this is the plain-CPU path; NVDEC hardware
     decode is Week 2's Track A, not this milestone).
  3. Server generates an SDP answer and returns it in the HTTP response.
  4. Browser renders the incoming track in a <video> element.

Usage:
    python server.py --video sample.mp4 --host 0.0.0.0 --port 8080

Requires (on the target machine):
    pip install aiortc aiohttp av
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

from aiohttp import web # type: ignore
from aiortc import RTCPeerConnection, RTCSessionDescription # type: ignore
from aiortc.contrib.media import MediaPlayer, MediaRelay # type: ignore

logger = logging.getLogger("visionedge.streaming")

# Track every open peer connection so we can clean them up on shutdown.
pcs: set[RTCPeerConnection] = set()
relay: MediaRelay | None = None
player: MediaPlayer | None = None


async def index(request: web.Request) -> web.Response:
    """Serves a minimal static HTML page for manual/browser testing
    without the full React app (see frontend/ for the real client)."""
    html_path = Path(__file__).parent / "static" / "index.html"
    return web.Response(content_type="text/html", text=html_path.read_text())


async def offer(request: web.Request) -> web.Response:
    """Handles an SDP offer from a browser peer, returns an SDP answer."""
    params = await request.json()
    offer_desc = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state is %s", pc.connectionState)
        if pc.connectionState in ("failed", "closed"):
            await pc.close()
            pcs.discard(pc)

    # Attach the looping static video track. MediaRelay lets multiple
    # peers subscribe to the same decoded source without re-reading the
    # file per connection.
    assert relay is not None and player is not None
    video_track = relay.subscribe(player.video)
    pc.addTrack(video_track)

    await pc.setRemoteDescription(offer_desc)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}),
    )


async def on_shutdown(app: web.Application) -> None:
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


def build_app(video_path: str) -> web.Application:
    global relay, player

    relay = MediaRelay()
    # loop=True keeps the static test video playing continuously, which is
    # all this Week 1 milestone needs (a real RTSP source arrives in Week 2).
    player = MediaPlayer(video_path, loop=True)

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.router.add_static("/static/", path=str(Path(__file__).parent / "static"), name="static")

    # Minimal permissive CORS for local dev, since the React dev server
    # typically runs on a different port (e.g. 5173) than this server.
    async def cors_middleware(request, handler):
        if request.method == "OPTIONS":
            resp = web.Response()
        else:
            resp = await handler(request)
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp

    app.middlewares.append(cors_middleware)
    return app


def main():
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="VisionEdge Week 1 WebRTC skeleton server.")
    p.add_argument("--video", type=str, default=str(Path(__file__).parents[2] / "media" / "sample.mp4"),
                   help="Path to the static/unedited video file to stream")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()

    if not Path(args.video).exists():
        raise SystemExit(
            f"Video file not found: {args.video}\n"
            "Provide a --video path to any local .mp4 for this milestone."
        )

    app = build_app(args.video)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
