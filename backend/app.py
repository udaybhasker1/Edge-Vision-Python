from fastapi import FastAPI  # type: ignore[import]


app = FastAPI(
    title="VisionEdge Backend",
    version="1.0.0",
    description="AI-powered real-time video analytics backend"
)


@app.get("/")
def home():

    return {
        "message": "Welcome to VisionEdge Backend"
    }



@app.get("/health")
def health():

    return {
        "status": "Running",
        "service": "VisionEdge Backend"
    }



@app.get("/stream/status")
def stream_status():

    return {
        "streaming_service": "WebRTC",
        "status": "Available",
        "endpoint": "/offer"
    }