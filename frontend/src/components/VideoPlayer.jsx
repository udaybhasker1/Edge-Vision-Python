import { useEffect, useRef } from "react";

/**
 * VideoPlayer — VisionEdge Week 1, Track B deliverable.
 * Renders a MediaStream (the incoming WebRTC video track) in a <video> element.
 */
export function VideoPlayer({ stream, connectionState }) {
  const videoRef = useRef(null);

  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;
    }
  }, [stream]);

  return (
    <div style={{ fontFamily: "sans-serif" }}>
      <p>
        Connection state: <strong>{connectionState}</strong>
      </p>
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{ width: "100%", maxWidth: 960, background: "#000" }}
      />
    </div>
  );
}
