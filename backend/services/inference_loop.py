#!/usr/bin/env python3

"""
inference_loop.py — VisionEdge Pipeline

Used by WebRTC streaming.

Flow:

decoder.py
      |
      ↓
preprocess.py
      |
      ↓
TensorRT (optional)
      |
      ↓
processed frame
"""


import logging

from services.decoder import VideoDecoder
from services.preprocess import preprocess_frame


logger = logging.getLogger(
    "visionedge.pipeline"
)



try:

    from services.trt_engine import TRTEngine


except Exception:

    TRTEngine = None



    logger.warning(
        "TensorRT unavailable. CPU mode."
    )




class VisionPipeline:


    def __init__(
        self,
        video_path: str,
        engine_path: str | None = None
    ):


        self.decoder = VideoDecoder(
            video_path
        )


        self.engine = None



        if (
            engine_path
            and
            TRTEngine
        ):


            try:

                self.engine = TRTEngine(

                    engine_path,

                    (
                        1,
                        3,
                        640,
                        640
                    )

                )


                logger.info(
                    "TensorRT loaded"
                )


            except Exception as e:


                logger.warning(
                    "TensorRT failed: %s",
                    e
                )



        else:

            logger.info(
                "Running CPU pipeline"
            )



    def get_frame(self):

        """
        Called by video_track.py

        Returns:
            BGR frame
        """

        print("Pipeline frame recieved")
        frame = self.decoder.get_frame()
        


        if frame is None:
            print("No frame is recieved from decoder")
            return None



        #
        # TensorRT path
        #

        if self.engine:


            tensor = preprocess_frame(

                frame,

                (
                    640,
                    640
                )

            )


            output = self.engine.infer(
                tensor
            )


            frame = self.process_detections(
                frame,
                output
            )
        print(
        "Inference received frame:",
        frame.shape
    )


        return frame



    def process_detections(
        self,
        frame,
        output
    ):

        """
        Temporary placeholder.

        Later:
        - YOLO decoding
        - NMS
        - bounding boxes
        - labels
        """


        return frame