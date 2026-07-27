#!/usr/bin/env python3

"""
server.py — VisionEdge WebRTC Streaming Server

Pipeline:

React Frontend
        |
        ↓
server.py (/offer)
        |
        ↓
peer.py
        |
        ↓
video_track.py
        |
        ↓
services/inference_loop.py
        |
        ↓
decoder.py
        |
        ↓
preprocess.py
        |
        ↓
TensorRT (optional)
"""


import argparse
import json
import logging
import sys


from pathlib import Path


# -------------------------------------------------
# Add backend folder to Python path
# -------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parents[1]

sys.path.append(
    str(BACKEND_DIR)
)



from aiohttp import web  # type: ignore
from aiortc import RTCSessionDescription  # type: ignore


from streaming.peer import PeerManager
from services.inference_loop import VisionPipeline



logger = logging.getLogger(
    "visionedge.streaming"
)



peer_manager = None
pipeline = None



# =================================================
# Serve frontend test page
# =================================================

async def index(request: web.Request):

    html_path = (
        Path(__file__).parent
        / "static"
        / "index.html"
    )


    return web.Response(

        content_type="text/html",

        text=html_path.read_text()

    )



# =================================================
# WebRTC Offer Handler
# =================================================

async def offer(request: web.Request):


    params = await request.json()



    offer_desc = RTCSessionDescription(

        sdp=params["sdp"],

        type=params["type"]

    )



    if peer_manager is None:

        return web.Response(

            status=500,

            text="Peer manager not initialized"

        )



    pc = await peer_manager.create_peer_connection()



    await pc.setRemoteDescription(
        offer_desc
    )


    answer = await pc.createAnswer()


    await pc.setLocalDescription(
        answer
    )



    return web.Response(

        content_type="application/json",

        text=json.dumps(

            {

                "sdp":
                    pc.localDescription.sdp,

                "type":
                    pc.localDescription.type

            }

        )

    )



# =================================================
# Shutdown
# =================================================

async def on_shutdown(
    app: web.Application
):

    if peer_manager:

        await peer_manager.close_all()



# =================================================
# Build Application
# =================================================

def build_app(
    video_source,
    engine_path=None
):

    global peer_manager
    global pipeline



    logger.info(
        "Initializing pipeline: %s",
        video_source
    )



    #
    # AI Pipeline
    #

    pipeline = VisionPipeline(

        video_source,

        engine_path

    )



    #
    # WebRTC Manager
    #

    peer_manager = PeerManager(
        pipeline
    )



    app = web.Application()



    app.on_shutdown.append(
        on_shutdown
    )



    app.router.add_get(
        "/",
        index
    )



    app.router.add_post(
        "/offer",
        offer
    )



    app.router.add_static(

        "/static/",

        path=str(

            Path(__file__).parent
            /
            "static"

        ),

        name="static"

    )



    # =================================================
    # CORS
    # =================================================

    @web.middleware
    async def cors_middleware(
        request,
        handler
    ):


        if request.method == "OPTIONS":

            response = web.Response()

        else:

            response = await handler(request)



        response.headers[

            "Access-Control-Allow-Origin"

        ] = "*"



        response.headers[

            "Access-Control-Allow-Methods"

        ] = "POST, GET, OPTIONS"



        response.headers[

            "Access-Control-Allow-Headers"

        ] = "Content-Type"



        return response



    app.middlewares.append(
        cors_middleware
    )



    return app



# =================================================
# Main
# =================================================

def main():


    logging.basicConfig(

        level=logging.INFO,

        format="%(asctime)s %(levelname)s %(message)s"

    )



    parser = argparse.ArgumentParser(

        description="VisionEdge WebRTC server"

    )



    parser.add_argument(

        "--video",

        type=str,

        default=str(

            BACKEND_DIR

            /

            "videos"

            /

            "traffic video.mp4"

        ),

        help="Video file path or RTSP URL"

    )



    parser.add_argument(

        "--engine",

        type=str,

        default=None,

        help="TensorRT engine path"

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



    #
    # Validate source
    #

    if (

        not args.video.startswith("rtsp://")

        and

        not Path(args.video).exists()

    ):


        raise SystemExit(

            f"Video source not found: {args.video}"

        )



    app = build_app(

        args.video,

        args.engine

    )



    logger.info(

        "Starting WebRTC server on port %s",

        args.port

    )



    web.run_app(

        app,

        host=args.host,

        port=args.port

    )




if __name__ == "__main__":

    main()