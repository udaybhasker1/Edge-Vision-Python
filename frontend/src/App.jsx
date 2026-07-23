import { useWebRTC } from "./hooks/webrtc/useWebRTC";
import { VideoPlayer } from "./components/VideoPlayer";

const SIGNALING_URL = import.meta.env.VITE_SIGNALING_URL || "http://localhost:8080/offer";

/**
 * App — VisionEdge Week 1, Track B deliverable.
 *
 * Deliverable check: opening this app should show the test video
 * (streamed from backend/streaming/server.py) playing live via WebRTC,
 * end to end.
 */
export default function App() {
  const { stream, connectionState } = useWebRTC(SIGNALING_URL);

  return (
    <div style={{ padding: 24, fontFamily: "sans-serif" }}>
      <h1>VisionEdge — Week 1 WebRTC Skeleton</h1>
      <p>Streaming a static test video from the aiortc backend over WebRTC.</p>
      <VideoPlayer stream={stream} connectionState={connectionState} />
    </div>
  );
}
