"""Browser API for the Decidef deepfake audio detector."""

import asyncio
import os
import shutil
import subprocess
import tempfile
import threading
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from main1 import analyze_live_recording
from model_utils import (
    DeepfakeAudioDetector,
    SAMPLE_RATE,
    SILENCE_RMS,
    WINDOW_SIZE,
    aggregate_fake_scores,
    display_confidence,
    get_ffmpeg_executable,
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR
FRONTEND_FILE = BASE_DIR / "index (17).html"
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".wma", ".webm"}

app = FastAPI(title="Decidef API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["null"],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")
_detector: DeepfakeAudioDetector | None = None
_live_inference_lock = threading.Lock()


def infer_live_window(detector: DeepfakeAudioDetector, audio: np.ndarray):
    """Run live inference serially so concurrent streams cannot share spectral cache state."""
    with _live_inference_lock:
        return detector.predict_window_fused(audio, allow_cached_spectral=False)


def get_detector() -> DeepfakeAudioDetector:
    """Reuse one detector, applying a measured profile when one is available."""
    global _detector
    if _detector is None:
        if (BASE_DIR / "calibration_result.json").is_file():
            _detector = DeepfakeAudioDetector(verbose=False)
        else:
            _detector = DeepfakeAudioDetector(threshold=0.42, verbose=False)
    return _detector


def json_safe(value: Any) -> Any:
    """Convert numpy scalar values nested in detector output to JSON values."""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def frontend_result(result: dict[str, Any], filename: str | None = None) -> dict[str, Any]:
    """Adapt detector output to the self-contained frontend's result schema."""
    fake_prob = float(result.get("fake_prob", 0.0))
    real_prob = float(result.get("real_prob", 1.0 - fake_prob))
    windows = result.get("windows") or []
    status = result.get("status", "REAL")
    no_speech = status in {"SILENCE", "SILENCE_OR_EMPTY"}
    return {
        "status": "success",
        "file_name": filename,
        "verdict": "NO SPEECH" if no_speech else ("AI-GENERATED" if status == "FAKE" else "HUMAN"),
        "verdict_label": "Unknown" if no_speech else ("AI" if status == "FAKE" else "Human"),
        "fake_prob": fake_prob,
        "human_prob": real_prob,
        "neural_fake_prob": fake_prob,
        "windows_analyzed": len(windows),
        "windows_total": len(windows),
        "acoustic": result.get("spectral") or {},
        "duration": result.get("duration", 0.0),
        "raw_result": result,
    }


def prepare_audio_file(content: bytes, extension: str) -> tuple[str, str | None]:
    """Write an upload to disk and decode browser containers to WAV when needed."""
    source_path: str | None = None
    converted_path: str | None = None
    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as source_file:
        source_file.write(content)
        source_path = source_file.name

    if extension != ".webm":
        return source_path, None

    ffmpeg = get_ffmpeg_executable()
    if not ffmpeg:
        os.unlink(source_path)
        raise RuntimeError(
            "Microphone recordings require FFmpeg. Install the project "
            "dependencies, including imageio-ffmpeg, or add FFmpeg to PATH."
        )

    converted_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    converted_path = converted_file.name
    converted_file.close()
    try:
        completed = subprocess.run(
            [
                ffmpeg, "-y", "-i", source_path,
                "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                converted_path,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip().splitlines()[-1] if completed.stderr else "unknown decoder error"
            raise RuntimeError(f"Could not decode the microphone recording: {detail}")
        return converted_path, source_path
    except Exception:
        if converted_path and os.path.exists(converted_path):
            os.unlink(converted_path)
        if source_path and os.path.exists(source_path):
            os.unlink(source_path)
        raise


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_FILE)


@app.get("/api/status")
def status() -> dict[str, Any]:
    try:
        detector = get_detector()
    except Exception as error:
        return {
            "loaded": False,
            "message": f"Detector unavailable: {error}",
        }
    return {
        "loaded": True,
        "message": "Detector ready.",
        "threshold": detector.threshold,
    }


@app.get("/api/config")
def config() -> dict[str, Any]:
    calibrated = (BASE_DIR / "calibration_result.json").is_file()
    return {
        "engine": "main1.py",
        "threshold": (
            _detector.threshold if _detector is not None
            else (None if calibrated else 0.42)
        ),
        "mode": "calibrated" if calibrated else "standard",
    }


@app.post("/api/analyze")
async def analyze(
    audio: UploadFile = File(...),
) -> dict[str, Any]:
    filename = audio.filename or "audio"
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio format. Use: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    content = await audio.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    detector = get_detector()
    temp_path: str | None = None
    source_path: str | None = None
    try:
        temp_path, source_path = prepare_audio_file(content, extension)
        # File uploads share the same detector configuration as main1.py.
        result = detector.predict_file(temp_path)
        result["filename"] = filename
        result["engine"] = "model_utils.py"
        return json_safe(frontend_result(result, filename))
    except Exception as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        if source_path and os.path.exists(source_path):
            os.unlink(source_path)


@app.post("/api/analyze-batch")
async def analyze_batch(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """Analyze a folder selection using the same detector as single uploads."""
    results: list[dict[str, Any]] = []
    for audio in files:
        try:
            filename = audio.filename or "audio"
            extension = Path(filename).suffix.lower()
            if extension not in ALLOWED_EXTENSIONS:
                raise ValueError(f"Unsupported audio format: {extension or 'unknown'}")
            content = await audio.read()
            if not content:
                raise ValueError("The uploaded file is empty.")
            temp_path: str | None = None
            source_path: str | None = None
            try:
                temp_path, source_path = prepare_audio_file(content, extension)
                result = get_detector().predict_file(temp_path)
                results.append(frontend_result(result, filename))
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
                if source_path and os.path.exists(source_path):
                    os.unlink(source_path)
        except Exception as error:
            results.append({
                "status": "error",
                "file_name": audio.filename or "audio",
                "message": str(error),
            })

    successful = [item for item in results if item["status"] == "success"]
    summary = {
        "total": len(results),
        "human": sum(item["verdict_label"] == "Human" for item in successful),
        "ai": sum(item["verdict_label"] == "AI" for item in successful),
        "borderline": sum(item["verdict_label"] == "Unknown" for item in successful),
        "errors": len(results) - len(successful),
    }
    return {"status": "success", "summary": summary, "results": results}


@app.post("/api/live")
async def live_chunk(audio: UploadFile = File(...)) -> dict[str, Any]:
    """Analyze a browser microphone recording using main1.py's detector."""
    content = await audio.read()
    if not content:
        raise HTTPException(status_code=400, detail="The live audio chunk is empty.")

    temp_path: str | None = None
    source_path: str | None = None
    try:
        temp_path, source_path = prepare_audio_file(content, ".webm")
        result = analyze_live_recording(get_detector(), temp_path)
        result["filename"] = audio.filename or "live-recording.webm"
        result["engine"] = "main1.py"
        return json_safe(frontend_result(result, audio.filename or "Live recording"))
    except Exception as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        if source_path and os.path.exists(source_path):
            os.unlink(source_path)


@app.websocket("/api/live-stream")
async def live_stream(websocket: WebSocket) -> None:
    """Receive 16 kHz PCM chunks and return rolling live voice classifications."""
    await websocket.accept()
    try:
        detector = await asyncio.to_thread(get_detector)
        await websocket.send_json({
            "type": "ready",
            "sample_rate": SAMPLE_RATE,
            "window_seconds": WINDOW_SIZE / SAMPLE_RATE,
        })
        audio_buffer = np.empty(0, dtype=np.float32)
        recent_scores: deque[float] = deque(maxlen=5)
        received_samples = 0
        speech_samples = 0
        next_inference_sample = WINDOW_SIZE

        while True:
            chunk_bytes = await websocket.receive_bytes()
            if not chunk_bytes or len(chunk_bytes) % 2 or len(chunk_bytes) > SAMPLE_RATE * 4:
                await websocket.send_json({
                    "type": "error",
                    "message": "Expected a 16-bit PCM audio chunk of at most two seconds.",
                })
                continue

            chunk = np.frombuffer(chunk_bytes, dtype="<i2").astype(np.float32) / 32768.0
            received_samples += len(chunk)
            chunk_rms = float(np.sqrt(np.mean(chunk ** 2)))

            if chunk_rms < SILENCE_RMS:
                audio_buffer = np.empty(0, dtype=np.float32)
                recent_scores.clear()
                speech_samples = 0
                next_inference_sample = WINDOW_SIZE
                await websocket.send_json({
                    "type": "waiting",
                    "duration": received_samples / SAMPLE_RATE,
                })
                continue

            audio_buffer = np.concatenate((audio_buffer, chunk))[-WINDOW_SIZE:]
            speech_samples += len(chunk)
            if speech_samples < next_inference_sample:
                await websocket.send_json({
                    "type": "buffering",
                    "duration": received_samples / SAMPLE_RATE,
                })
                continue

            next_inference_sample += SAMPLE_RATE
            try:
                _, fake_score, _ = await asyncio.to_thread(
                    infer_live_window,
                    detector,
                    audio_buffer.copy(),
                )
                recent_scores.append(float(fake_score))
                aggregate_fake = aggregate_fake_scores(np.asarray(recent_scores))
                threshold = detector.dynamic_threshold(aggregate_fake)
                verdict = "FAKE" if aggregate_fake >= threshold else "REAL"
                display_fake, display_real = display_confidence(
                    aggregate_fake, verdict, threshold
                )
                await websocket.send_json({
                    "type": "result",
                    "status": "success",
                    "verdict": "AI-GENERATED" if verdict == "FAKE" else "HUMAN",
                    "verdict_label": "AI" if verdict == "FAKE" else "Human",
                    "fake_prob": aggregate_fake,
                    "human_prob": 1.0 - aggregate_fake,
                    "neural_fake_prob": float(fake_score),
                    "display_fake_confidence": display_fake,
                    "display_real_confidence": display_real,
                    "decision_threshold": threshold,
                    "windows_analyzed": len(recent_scores),
                    "windows_total": len(recent_scores),
                    "duration": received_samples / SAMPLE_RATE,
                })
            except Exception as error:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Live inference failed: {error}",
                })
    except WebSocketDisconnect:
        return
    except Exception as error:
        try:
            await websocket.send_json({"type": "error", "message": str(error)})
        except WebSocketDisconnect:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
