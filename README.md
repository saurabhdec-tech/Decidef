# Decidef

Decidef analyzes audio for signs of AI-generated speech using the Dhwani ONNX
classifier and a supporting spectral-artifact analyzer. It provides a browser
interface for audio uploads, batch uploads, and microphone recordings, plus a
command-line microphone/file/folder launcher.

> **Limitations:** Results are model estimates, not proof of how a voice was
> produced. Background noise, short clips, compression, language, and newer
> speech generators can affect results. Use labeled, representative audio to
> calibrate and independently evaluate the detector before relying on it.

## Requirements

- Python 3.10 or newer
- The packages listed in `requirements.txt`
- The ONNX model `best_model.onnx` (about 1.17 GB). If it is missing, the
  detector attempts to download it automatically; you can also run
  `python model_utils.py --download`.

`imageio-ffmpeg` supplies FFmpeg for decoding browser microphone recordings
and audio formats that the Python audio readers cannot decode. A system FFmpeg
installation on `PATH` is also supported.

## Install

From the project directory, create and activate a virtual environment:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the browser app

```powershell
python app.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Keep the backend running
while using the page. The first detector initialization loads the ONNX model
and may take a while. Browser microphone access is available on localhost;
allow the permission request when prompted.

The browser microphone mode streams mono 16 kHz PCM audio to the backend over
a WebSocket. Inference starts after the first three seconds and then refreshes
on rolling windows while you speak. Use **Stop live detection** or tap the
microphone card again to close the stream. This is live inference rather than
record-then-upload; browser audio is not saved as a recording.

### API endpoints

- `GET /api/status` — initialize/check the detector
- `GET /api/config` — report the detector mode and threshold
- `POST /api/analyze` — analyze one audio file (`audio` form field)
- `POST /api/analyze-batch` — analyze multiple files (`files` form fields)
- `POST /api/live` — analyze a browser microphone recording (`audio` form field)
- `WS /api/live-stream` — stream 16 kHz signed 16-bit PCM chunks for live results

Supported file uploads: WAV, MP3, FLAC, M4A, OGG, AAC, WMA, and WebM. Live
microphone audio is streamed as PCM and is not wrapped in a recording file.

## Run the desktop/CLI launcher

```powershell
python run.py
```

Choose file analysis, folder analysis, or microphone detection from the menu.
You can also use `main1.py` directly:

```powershell
python main1.py --help
python main1.py --file "C:\path\to\audio.wav"
python main1.py --folder "C:\path\to\audio-folder"
python main1.py --list-devices
```

The CLI microphone mode uses the system's default input device unless a device
index is specified.

## Calibrate with labeled audio

The default settings are not calibrated to every microphone, language, or
generator. To fit the fusion weights and decision threshold to your data,
place labeled examples in separate folders and run:

```powershell
python model_utils.py --calibrate ".\samples\human" ".\samples\ai"
```

Calibration searches folders recursively and writes `calibration_result.json`
in the project directory. The browser app uses this profile on its next
detector initialization. Restart the app after changing the profile if the
detector is already loaded.

Use a varied set of clean, correctly labeled human recordings and AI-generated
recordings representative of the voices and tools you expect. Avoid using
near-duplicates or recordings from the same speakers/generator in both tuning
and evaluation. The calibration score is measured on the examples used to fit
the settings; evaluate the result on a separate held-out set before treating
it as an accuracy estimate. If you do not have labeled audio, leave the
calibration profile absent and the app will use its standard settings.

## Project files

- `app.py` — FastAPI web backend and audio upload endpoints
- `index (17).html` — browser interface served by the backend
- `main1.py` — command-line microphone and file/folder analysis
- `model_utils.py` — model loading, audio processing, scoring, and calibration
- `run.py` — interactive desktop launcher
- `best_model.onnx` — Dhwani model
- `requirements.txt` — Python dependencies
