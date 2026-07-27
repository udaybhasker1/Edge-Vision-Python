#!/usr/bin/env python3

"""
decoder.py — VisionEdge Decoder

Supports:

MP4 file:
    OpenCV VideoCapture

RTSP:
    PyAV (future NVIDIA/NVDEC support)

Current development:
CPU decoding using OpenCV
"""


import logging
import cv2 # type: ignore
import numpy as np # type: ignore


logger = logging.getLogger(
    "visionedge.decoder"
)



class VideoDecoder:


    def __init__(
        self,
        source: str
    ):

        self.source = source


        self.cap = cv2.VideoCapture(
            source
        )


        if not self.cap.isOpened():

            raise RuntimeError(
                f"Cannot open video source: {source}"
            )


        logger.info(
            "Decoder opened: %s",
            source
        )



    def get_frame(self):
        ret, frame = self.cap.read()

        if not ret:
            # Try to loop the video (for file sources)
            try:
                self.cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    0
                )
                ret, frame = self.cap.read()
            except Exception:
                print("Error occurred while trying to loop the video")
                return None

        if not ret:
            print("No frame is recieved from decoder")
            return None

        print(
            "Decoder frame:",
            frame.shape,
            frame.dtype
        )

        return frame


    def release(self):

        self.cap.release()