"""
peer.py

Handles WebRTC PeerConnection lifecycle
"""

from aiortc import RTCPeerConnection # pyright: ignore[reportMissingImports]
from streaming.video_track import VisionVideoTrack # type: ignore


class PeerManager:
    """
    Manages WebRTC peer connections
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.connections = set()


    async def create_peer_connection(self):
        """
        Creates a new WebRTC peer connection
        """

        pc = RTCPeerConnection()

        self.connections.add(pc)


        # Add processed video track
        video_track = VisionVideoTrack(
            pipeline=self.pipeline
        )

        pc.addTrack(video_track)


        @pc.on("connectionstatechange")
        async def on_connection_state_change():

            print(
                "WebRTC state:",
                pc.connectionState
            )

            if pc.connectionState in [
                "failed",
                "closed"
            ]:
                await pc.close()
                self.connections.discard(pc)


        return pc



    async def close_all(self):
        """
        Close all active WebRTC connections
        """

        coroutines = [
            pc.close()
            for pc in self.connections
        ]

        for c in coroutines:
            await c

        self.connections.clear()