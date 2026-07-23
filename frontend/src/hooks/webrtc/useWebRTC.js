import { useEffect, useRef, useState } from "react";

/**
 * useWebRTC — VisionEdge Week 1, Track B.
 *
 * Handles the SDP offer/answer signaling handshake against the aiortc
 * backend's /offer endpoint and exposes the resulting remote MediaStream
 * so a <video> element can render it.
 *
 * @param {string} signalingUrl - e.g. "http://localhost:8000/offer"
 */
export function useWebRTC(signalingUrl) {
  const [stream, setStream] = useState(null);
  const [connectionState, setConnectionState] = useState("new");
  const pcRef = useRef(null);

  useEffect(() => {
    let cancelled = false;

    async function connect() {
      const pc = new RTCPeerConnection();
      pcRef.current = pc;

      pc.addTransceiver("video", { direction: "recvonly" });

      pc.ontrack = (event) => {
        if (!cancelled) {
          setStream(event.streams[0]);
        }
      };

      pc.onconnectionstatechange = () => {
        setConnectionState(pc.connectionState);
      };

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      const response = await fetch(signalingUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sdp: pc.localDescription.sdp,
          type: pc.localDescription.type,
        }),
      });

      if (!response.ok) {
        throw new Error(`Signaling request failed: ${response.status}`);
      }

      const answer = await response.json();
      if (!cancelled) {
        await pc.setRemoteDescription(answer);
      }
    }

    connect().catch((err) => {
      console.error("WebRTC connection failed:", err);
      setConnectionState("failed");
    });

    return () => {
      cancelled = true;
      pcRef.current?.close();
    };
  }, [signalingUrl]);

  return { stream, connectionState };
}
