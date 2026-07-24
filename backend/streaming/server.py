#!/usr/bin/env python3

"""
server.py — VisionEdge Week 1, Track B deliverable.

Minimal aiortc server that streams a static video file over WebRTC.
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

from aiohttp import web  # type: ignore
from aiortc import RTCPeerConnection, RTCSessionDescription  # type: ignore
from aiortc.contrib.media import MediaPlayer, MediaRelay  # type: ignore


logger = logging.getLogger("visionedge.streaming")


pcs: set[RTCPeerConnection] = set()
relay: MediaRelay | None = None
player: MediaPlayer | None = None


async def index(request: web.Request) -> web.Response:
    """
    Serves static test HTML page.
    """

    html_path = Path(__file__).parent / "static" / "index.html"

    return web.Response(
        content_type="text/html",
        text=html_path.read_text()
    )


async def offer(request: web.Request) -> web.Response:
    """
    Handles SDP offer and returns SDP answer.
    """

    params = await request.json()

    offer_desc = RTCSessionDescription(
        sdp=params["sdp"],
        type=params["type"]
    )

    pc = RTCPeerConnection()
    pcs.add(pc)


    @pc.on("connectionstatechange")
    async def on_connectionstatechange():

        logger.info(
            "Connection state is %s",
            pc.connectionState
        )

        if pc.connectionState in ("failed", "closed"):
            await pc.close()
            pcs.discard(pc)


    assert relay is not None
    assert player is not None


    video_track = relay.subscribe(player.video)

    pc.addTrack(video_track)


    await pc.setRemoteDescription(offer_desc)

    answer = await pc.createAnswer()

    await pc.setLocalDescription(answer)


    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
            }
        ),
    )


async def on_shutdown(app: web.Application):

    await asyncio.gather(
        *[pc.close() for pc in pcs]
    )

    pcs.clear()



def build_app(video_path: str) -> web.Application:

    global relay, player


    relay = MediaRelay()

    player = MediaPlayer(
        video_path,
        loop=True
    )


    app = web.Application()


    app.on_shutdown.append(on_shutdown)


    app.router.add_get("/", index)

    app.router.add_post("/offer", offer)


    app.router.add_static(
        "/static/",
        path=str(Path(__file__).parent / "static"),
        name="static"
    )


    # CORS middleware
    @web.middleware
    async def cors_middleware(request, handler):

        if request.method == "OPTIONS":
            response = web.Response()

        else:
            response = await handler(request)


        response.headers["Access-Control-Allow-Origin"] = "*"

        response.headers["Access-Control-Allow-Methods"] = (
            "POST, GET, OPTIONS"
        )

        response.headers["Access-Control-Allow-Headers"] = (
            "Content-Type"
        )


        return response


    app.middlewares.append(cors_middleware)


    return app



def main():

    logging.basicConfig(
        level=logging.INFO
    )


    parser = argparse.ArgumentParser(
        description="VisionEdge WebRTC server"
    )


    parser.add_argument(
        "--video",
        type=str,
        default=str(
            Path(__file__).parents[2]
            / "backend"
            / "videos"
            / "traffic video.mp4"
        ),
        help="Video file path"
    )


    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0"
    )


    parser.add_argument(
        "--port",
        type=int,
        default=8080
    )


    args = parser.parse_args()


    if not Path(args.video).exists():

        raise SystemExit(
            f"Video file not found: {args.video}\n"
            "Provide a valid mp4 path."
        )


    app = build_app(args.video)


    web.run_app(
        app,
        host=args.host,
        port=args.port
    )



if __name__ == "__main__":
    main()