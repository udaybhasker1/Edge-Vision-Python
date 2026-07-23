try:
    from fastapi import FastAPI # type: ignore
except Exception:  # pragma: no cover - provide a lightweight fallback when FastAPI isn't available
    # Minimal stub for FastAPI to avoid import errors in environments without fastapi installed.
    class FastAPI:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, path):
            def decorator(func):
                return func

            return decorator

app = FastAPI(
    title="VisionEdge Backend",
    version="1.0.0"
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