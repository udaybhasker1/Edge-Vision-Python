"""
video_track.py

Custom WebRTC video track
"""

import av # type: ignore

from aiortc import VideoStreamTrack # type: ignore


class VisionVideoTrack(VideoStreamTrack):

    """
    Video track that receives frames
    from AI inference pipeline
    """


    def __init__(self, pipeline):

        super().__init__()

        self.pipeline = pipeline



    async def recv(self):

        """
        Called automatically by aiortc
        whenever a new frame is required
        """

        # Get processed frame
        frame = self.pipeline.get_frame()


        if frame is None:
            return await super().recv()



        # Convert numpy frame to AV frame

        video_frame = av.VideoFrame.from_ndarray(
            frame,
            format="bgr24"
        )


        # Timestamp handling
        video_frame.pts, video_frame.time_base = (
            await self.next_timestamp()
        )


        return video_frame