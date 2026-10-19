"""
Model Management & Inference Utilities for Multilingual Deepfake Audio Detection
Powered by Dhwani (Wav2Vec2 XLS-R + AASIST) via ONNX Runtime.

Enhanced with Spectral Artifact Analysis for detecting modern AI TTS voices
(ElevenLabs, OpenAI TTS, Google WaveNet, Bark, XTTS, etc.)

IMPORTANT: The spectral heuristics below use configurable thresholds/weights.
Out of the box they are reasonable starting points, but they are NOT a
substitute for calibrating against your own labeled real/fake samples.
Use `calibrate_thresholds()` at the bottom of this file with a folder of
known-real and known-fake clips to tune SPECTRAL_PARAMS and the sensitivity
weights/threshold for your actual use case. Detectors that rely on guessed
constants will always lag behind the newest TTS releases, and will keep
swinging between "too many false fakes" and "too many false reals" every
time the constants are hand-tuned without labeled data to check against.
"""

import os
import sys
import time
import json
import io
import shutil
import subprocess
import urllib.request
import numpy as np
import onnxruntime as ort
import librosa
import soundfile as sf
from tqdm import tqdm

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

MODEL_FILENAME = "best_model.onnx"
MODEL_URL = "https://huggingface.co/ayush2635/Dhwani-Multilingual-Deepfake-Audio-Detection-Model/resolve/main/best_model.onnx"
EXPECTED_SIZE = 1262760439  # ~1.17 GB
SAMPLE_RATE = 16000
# Dhwani was exported and documented for a maximum 3-second window.
# Keeping this at 48,000 samples is important: shorter windows produce
# systematically overconfident "real" logits from the ONNX classifier.
WINDOW_SIZE = 48000
SILENCE_RMS = 0.005

# Default fusion weights (used when no sensitivity preset / explicit
# weights are given).
ONNX_WEIGHT = 0.50
SPECTRAL_WEIGHT = 0.50

# Sensitivity presets: (decision threshold, onnx_weight, spectral_weight).
# Rebalanced back toward trusting the spectral analyzer roughly equally
# with the ONNX model — the ONNX model alone tends to under-score modern
# TTS systems it wasn't trained on, so leaning too heavily on it causes
# fakes to pass as real; leaning too heavily on the spectral analyzer
# (which uses uncalibrated heuristic thresholds) causes real audio to get
# flagged fake. "balanced" aims for a middle ground pending real
# calibration against labeled data.
SENSITIVITY_PRESETS = {
    "strict": {
        "threshold": 0.60,
        "onnx_weight": 0.35,
        "spectral_weight": 0.65
    },
    "balanced": {
        "threshold": 0.50,
        "onnx_weight": 0.25,
        "spectral_weight": 0.75
    },
    "aggressive": {
        "threshold": 0.42,
        "onnx_weight": 0.15,
        "spectral_weight": 0.85
    }
}
DEFAULT_SENSITIVITY = "balanced"

# Spectral sub-features considered strong, independent evidence of
# synthetic speech on their own.
STRONG_SPECTRAL_SIGNALS = [
    "pitch_jitter", "amplitude_shimmer", "f0_smoothness", "hf_artifact", "hnr"
]
STRONG_SIGNAL_THRESHOLD = 0.85

# Spectral sub-feature weights and mapping constants.
SPECTRAL_PARAMS = {
    "spectral_flatness": {"weight": 0.08, "center": 0.02, "scale": 0.10},
    "rolloff_variance": {"weight": 0.04, "scale": 0.35},
    "zcr_consistency": {"weight": 0.04, "scale": 0.70},

    "pitch_jitter": {"weight": 0.22, "scale": 0.035},
    "amplitude_shimmer": {"weight": 0.16, "scale": 0.08},

    "hnr": {"weight": 0.04, "center": 10.0, "scale": 15.0},
    "pause_irregularity": {"weight": 0.10, "scale": 0.35},
    "f0_smoothness": {"weight": 0.16, "scale": 0.02},

    "hf_artifact": {"weight": 0.16}
}


def download_model(dest_path: str = None) -> str:
    if dest_path is None:
        dest_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME)

    if os.path.exists(dest_path) and os.path.getsize(dest_path) == EXPECTED_SIZE:
        print(f"[✓] Model already exists and verified: {dest_path}")
        return dest_path

    temp_path = dest_path + ".part"
    downloaded_bytes = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    if downloaded_bytes > 0:
        headers['Range'] = f"bytes={downloaded_bytes}-"
        print(f"[*] Attempting resume from {downloaded_bytes / (1024 * 1024):.1f} MB...")
    else:
        print(f"[*] Starting download of Dhwani model (1.17 GB) from Hugging Face...")

    req = urllib.request.Request(MODEL_URL, headers=headers)

    try:
        resp = urllib.request.urlopen(req)
        is_resume = (resp.status == 206)
        if not is_resume and downloaded_bytes > 0:
            downloaded_bytes = 0

        write_mode = "ab" if is_resume else "wb"
        total_size = downloaded_bytes + int(resp.headers.get('Content-Length', EXPECTED_SIZE - downloaded_bytes))

        with open(temp_path, write_mode) as f:
            with tqdm(
                total=total_size, initial=downloaded_bytes, unit='B',
                unit_scale=True, unit_divisor=1024, desc="Dhwani Model", ncols=80
            ) as pbar:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))

        resp.close()
        time.sleep(0.5)

        if os.path.getsize(temp_path) > EXPECTED_SIZE:
            with open(temp_path, "r+b") as tf:
                tf.truncate(EXPECTED_SIZE)

        if os.path.exists(dest_path):
            os.remove(dest_path)
        os.rename(temp_path, dest_path)
        print(f"\n[✓] Model successfully downloaded and saved to: {dest_path}")
        return dest_path

    except Exception as e:
        print(f"\n[!] Download error: {e}")
        print("    You can resume the download by re-running the script.")
        raise


def softmax(logits: np.ndarray) -> np.ndarray:
    exp_l = np.exp(logits - np.max(logits))
    return exp_l / np.sum(exp_l)


def aggregate_fake_scores(fake_scores: np.ndarray) -> float:
    """Aggregate rolling window scores without letting one spike decide."""
    scores = np.asarray(fake_scores, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        return 0.0
    scores = np.clip(scores, 0.0, 1.0)
    top_count = max(1, int(np.ceil(scores.size * 0.33)))
    top_mean = float(np.mean(np.sort(scores)[-top_count:]))
    median = float(np.median(scores))
    maximum = float(np.max(scores))
    return float(np.clip(0.35 * top_mean + 0.20 * maximum + 0.45 * median, 0.0, 1.0))


def fuse_fake_scores(onnx_fake: float, spectral_fake: float,
                     onnx_weight: float, spectral_weight: float) -> float:
    """Combine model and spectral scores using the production inference rules."""
    weighted_fake = float(np.clip(
        onnx_weight * onnx_fake + spectral_weight * spectral_fake,
        0.0,
        1.0,
    ))
    disagreement = abs(onnx_fake - spectral_fake)
    if disagreement > 0.70:
        weighted_fake *= float(np.clip(1.0 - (disagreement - 0.70) * 0.35, 0.75, 1.0))

    strongest = max(onnx_fake, spectral_fake)
    if strongest >= STRONG_SIGNAL_THRESHOLD:
        weighted_fake = max(weighted_fake, 0.30 * strongest + 0.70 * weighted_fake)
    return float(np.clip(weighted_fake, 0.0, 1.0))


def get_ffmpeg_executable() -> str | None:
    """Find system FFmpeg or the executable bundled by imageio-ffmpeg."""
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    return imageio_ffmpeg.get_ffmpeg_exe()


def _load_calibration() -> dict:
    """Load optional local calibration without changing the CLI contract."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "calibration_result.json")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        required = {"onnx_weight", "spectral_weight", "threshold"}
        if required.issubset(value):
            return {key: float(value[key]) for key in required}
    except (OSError, ValueError, TypeError):
        pass
    return {}


# ============================================================
# SPECTRAL ARTIFACT ANALYZER
# ============================================================

class SpectralArtifactAnalyzer:
    """
    Detects spectral/prosodic artifacts modern AI TTS systems leave behind.

    pitch_jitter and f0_smoothness share one YIN pitch-contour computation
    (previously computed twice independently, which was the single most
    expensive call in this class run redundantly on every analysis).
    """

    def _hf_artifact_score(self, audio: np.ndarray) -> float:
        try:
            spec = np.abs(
                librosa.stft(
                    audio,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )
            )
            freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
            hf_mask = freqs > 7000
            mf_mask = (freqs > 1000) & (freqs < 4000)
            hf_energy = float(np.mean(spec[hf_mask]))
            mf_energy = float(np.mean(spec[mf_mask]))
            ratio = hf_energy / (mf_energy + 1e-8)
            score = np.clip((ratio - 0.08) / 0.12, 0.0, 1.0)
            return float(score)
        except Exception:
            return 0.0

    def __init__(self, sr: int = SAMPLE_RATE, verbose: bool = False, params: dict = None):
        self.sr = sr
        self.verbose = verbose
        self.n_fft = 1024
        self.hop_length = 256
        self.params = params or SPECTRAL_PARAMS

    def _compute_voiced_f0(self, audio: np.ndarray) -> np.ndarray:
        """Computed once per analyze() call, shared by jitter + smoothness."""
        try:
            f0 = librosa.yin(
                audio, fmin=60, fmax=500, sr=self.sr,
                frame_length=1024, hop_length=self.hop_length,
            )
            return f0[np.isfinite(f0) & (f0 > 0.0)]
        except Exception:
            return np.array([])

    def analyze(self, audio: np.ndarray) -> dict:
        if len(audio) < self.n_fft:
            return {"artifact_score": 0.0, "detail": "audio_too_short"}

        voiced_f0 = self._compute_voiced_f0(audio)

        scores = {
            "spectral_flatness": self._spectral_flatness_score(audio),
            "rolloff_variance": self._rolloff_variance_score(audio),
            "zcr_consistency": self._zcr_consistency_score(audio),
            "pitch_jitter": self._pitch_jitter_score(voiced_f0),
            "amplitude_shimmer": self._shimmer_score(audio),
            "hnr": self._hnr_score(audio),
            "pause_irregularity": self._pause_irregularity_score(audio),
            "f0_smoothness": self._f0_smoothness_score(voiced_f0),
            "hf_artifact": self._hf_artifact_score(audio)
        }

        total_weight = sum(self.params[k]["weight"] for k in scores)
        artifact_score = sum(scores[k] * self.params[k]["weight"] for k in scores) / total_weight
        artifact_score = float(np.clip(artifact_score, 0.0, 1.0))

        # --- Anti-dilution nudge ---
        # If a reliable individual sub-feature is confidently high, don't
        # let the weighted average erase it. Blend 45% toward the
        # strongest confident signal.
        strong_vals = [scores[k] for k in STRONG_SPECTRAL_SIGNALS
                       if k in scores and scores[k] >= STRONG_SIGNAL_THRESHOLD]
        if strong_vals:
            top_signal = max(strong_vals)
            nudged = max(artifact_score, 0.55 * artifact_score + 0.45 * top_signal)
            artifact_score = float(np.clip(nudged, 0.0, 1.0))

        if self.verbose:
            print(f"\n[Spectral Analysis]")
            for k, v in scores.items():
                flag = "  ⚠ strong" if (k in STRONG_SPECTRAL_SIGNALS and v >= STRONG_SIGNAL_THRESHOLD) else ""
                print(f"  {k:25s}: {v:.4f}{flag}")
            print(f"  {'FUSED artifact_score':25s}: {artifact_score:.4f}")

        return {"artifact_score": artifact_score, **scores}

    def _spectral_flatness_score(self, audio: np.ndarray) -> float:
        try:
            p = self.params["spectral_flatness"]
            flatness = librosa.feature.spectral_flatness(
                y=audio, n_fft=self.n_fft, hop_length=self.hop_length
            )
            mean_flat = float(np.mean(flatness))
            return float(np.clip((mean_flat - p["center"]) / p["scale"], 0.0, 1.0))
        except Exception:
            return 0.0

    def _rolloff_variance_score(self, audio: np.ndarray) -> float:
        try:
            p = self.params["rolloff_variance"]
            rolloff = librosa.feature.spectral_rolloff(
                y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
            )
            mean_r = float(np.mean(rolloff))
            if mean_r < 1e-6:
                return 0.5
            cv = float(np.std(rolloff)) / mean_r
            return float(np.clip(1.0 - (cv / p["scale"]), 0.0, 1.0))
        except Exception:
            return 0.0

    def _zcr_consistency_score(self, audio: np.ndarray) -> float:
        try:
            p = self.params["zcr_consistency"]
            zcr = librosa.feature.zero_crossing_rate(
                y=audio, frame_length=self.n_fft, hop_length=self.hop_length
            )
            if zcr.shape[1] < 3:
                return 0.0
            mean_zcr = float(np.mean(zcr))
            if mean_zcr < 1e-8:
                return 0.0
            cv = float(np.std(zcr)) / mean_zcr
            return float(np.clip(1.0 - (cv / p["scale"]), 0.0, 1.0))
        except Exception:
            return 0.0

    def _pitch_jitter_score(self, voiced_f0: np.ndarray) -> float:
        try:
            p = self.params["pitch_jitter"]
            if len(voiced_f0) < 5:
                return 0.0
            f0_diffs = np.abs(np.diff(voiced_f0))
            mean_f0 = float(np.mean(voiced_f0))
            if mean_f0 < 1e-6:
                return 0.0
            jitter = float(np.mean(f0_diffs)) / mean_f0
            return float(np.clip(1.0 - (jitter / p["scale"]), 0.0, 1.0))
        except Exception:
            return 0.0

    def _shimmer_score(self, audio: np.ndarray) -> float:
        try:
            p = self.params["amplitude_shimmer"]
            rms = librosa.feature.rms(
                y=audio, frame_length=self.n_fft, hop_length=self.hop_length
            )[0]
            if len(rms) < 5:
                return 0.0
            active_mask = rms > 0.005
            active_rms = rms[active_mask]
            if len(active_rms) < 5:
                return 0.0
            rms_diffs = np.abs(np.diff(active_rms))
            mean_rms = float(np.mean(active_rms))
            if mean_rms < 1e-8:
                return 0.0
            shimmer = float(np.mean(rms_diffs)) / mean_rms
            return float(np.clip(1.0 - (shimmer / p["scale"]), 0.0, 1.0))
        except Exception:
            return 0.0

    def _hnr_score(self, audio: np.ndarray) -> float:
        try:
            p = self.params["hnr"]
            n = len(audio)
            if n < self.sr // 4:
                return 0.0
            mid = n // 2
            seg_len = min(self.sr, n)
            start = max(0, mid - seg_len // 2)
            segment = audio[start:start + seg_len]
            seg_rms = float(np.sqrt(np.mean(segment ** 2)))
            if seg_rms < 0.005:
                return 0.0
            corr = np.correlate(segment, segment, mode='full')
            corr = corr[len(corr) // 2:]
            corr = corr / (corr[0] + 1e-10)
            min_lag = self.sr // 500
            max_lag = self.sr // 60
            if max_lag >= len(corr):
                max_lag = len(corr) - 1
            if min_lag >= max_lag:
                return 0.0
            search_region = corr[min_lag:max_lag]
            if len(search_region) == 0:
                return 0.0
            peak_corr = float(np.clip(np.max(search_region), 0.01, 0.999))
            hnr_db = 10.0 * np.log10(peak_corr / (1.0 - peak_corr))
            return float(np.clip((hnr_db - p["center"]) / p["scale"], 0.0, 1.0))
        except Exception:
            return 0.0

    def _pause_irregularity_score(self, audio: np.ndarray) -> float:
        try:
            p = self.params["pause_irregularity"]
            intervals = librosa.effects.split(audio, top_db=30,
                                               frame_length=self.n_fft,
                                               hop_length=self.hop_length)
            if len(intervals) < 3:
                return 0.0
            gaps = [
                (intervals[i + 1][0] - intervals[i][1]) / self.sr
                for i in range(len(intervals) - 1)
            ]
            gaps = [g for g in gaps if g > 0.02]
            if len(gaps) < 2:
                return 0.0
            mean_gap = float(np.mean(gaps))
            if mean_gap < 1e-6:
                return 0.0
            cv = float(np.std(gaps)) / mean_gap
            return float(np.clip(1.0 - (cv / p["scale"]), 0.0, 1.0))
        except Exception:
            return 0.0

    def _f0_smoothness_score(self, voiced_f0: np.ndarray) -> float:
        try:
            p = self.params["f0_smoothness"]
            if len(voiced_f0) < 8:
                return 0.0
            mean_f0 = float(np.mean(voiced_f0))
            if mean_f0 < 1e-6:
                return 0.0
            norm_f0 = voiced_f0 / mean_f0
            second_deriv = np.diff(norm_f0, n=2)
            curvature = float(np.mean(np.abs(second_deriv)))
            return float(np.clip(1.0 - (curvature / p["scale"]), 0.0, 1.0))
        except Exception:
            return 0.0


# ============================================================
# MAIN DETECTOR CLASS
# ============================================================

class dynamic_threshold:
    """Calculate an adaptive, bounded decision threshold for fake scores."""

    def __init__(self, base: float = 0.48, minimum: float = 0.15,
                 maximum: float = 0.95):
        if minimum > maximum:
            raise ValueError("minimum threshold cannot exceed maximum threshold")
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.base = float(np.clip(base, self.minimum, self.maximum))

    def calculate(self, aggregate_fake: float) -> float:
        """Return a lower threshold for stronger aggregate fake evidence."""
        score = float(np.clip(aggregate_fake, 0.0, 1.0))
        adjustment = 0.0
        if score > 0.55:
            adjustment -= 0.03
        if score > 0.70:
            adjustment -= 0.05
        return float(np.clip(self.base + adjustment,
                             self.minimum, self.maximum))

    def is_fake(self, aggregate_fake: float) -> bool:
        score = float(np.clip(aggregate_fake, 0.0, 1.0))
        return score >= self.calculate(score)

    def __call__(self, aggregate_fake: float) -> float:
        return self.calculate(aggregate_fake)


def display_confidence(fake_score: float, verdict: str, threshold: float) -> tuple[float, float]:
    """Return stable, verdict-aligned display confidence values.

    This is presentation-only: detection decisions continue to use the raw
    score and threshold. Five-point bands prevent near-threshold jitter from
    looking like a changing verdict.
    """
    score = float(np.clip(fake_score, 0.0, 1.0))
    limit = float(np.clip(threshold, 0.001, 0.999))
    if verdict == "FAKE":
        # Keep a close positive result modestly above the boundary, while
        # allowing a strong score to communicate stronger confidence.
        margin = max(0.0, score - limit) / max(0.001, 1.0 - limit)
        confidence = 0.50 + 0.50 * (margin ** 0.75)
        # Keep the presentation calibrated and readable rather than making
        # confirmed fakes appear artificially near 100%.
        confidence = float(np.clip(confidence, 0.55, 0.65))
    else:
        # Compress near-threshold real results. This avoids presenting a
        # normal voice with a raw score around .40 as 40% "fake probability".
        ratio = score / max(0.001, limit)
        confidence = 0.50 * (ratio ** 1.8)
        confidence = min(0.45, confidence)
    confidence = float(np.clip(np.round(confidence * 20.0) / 20.0, 0.05, 0.95))
    return confidence, 1.0 - confidence


def load_audio_file(audio_path: str) -> tuple[np.ndarray, int]:
    """Read an audio file without silently dropping channels or sample rate."""
    try:
        audio, sample_rate = sf.read(
            audio_path,
            dtype="float32",
            always_2d=True,
        )
        y = np.mean(audio, axis=1, dtype=np.float32)
    except (RuntimeError, OSError, ValueError):
        try:
            # Librosa remains the first compatibility fallback for formats
            # supported by the installed audioread backend.
            y, sample_rate = librosa.load(audio_path, sr=None, mono=True)
            y = np.asarray(y, dtype=np.float32)
        except Exception as librosa_error:
            # Decode through FFmpeg to canonical PCM when a codec/container is
            # not handled by SoundFile or audioread.
            ffmpeg = get_ffmpeg_executable()
            if not ffmpeg:
                raise RuntimeError(
                    f"Could not decode '{audio_path}' with SoundFile or librosa: "
                    f"{librosa_error}"
                ) from librosa_error
            completed = subprocess.run(
                [
                    ffmpeg, "-v", "error", "-i", audio_path,
                    "-vn", "-f", "wav", "pipe:1",
                ],
                capture_output=True,
                check=False,
            )
            if completed.returncode != 0 or not completed.stdout:
                detail = completed.stderr.decode(errors="replace").strip()
                raise RuntimeError(
                    f"Could not decode '{audio_path}' with FFmpeg: "
                    f"{detail or 'unknown decoder error'}"
                ) from librosa_error
            audio, sample_rate = sf.read(
                io.BytesIO(completed.stdout),
                dtype="float32",
                always_2d=True,
            )
            y = np.mean(audio, axis=1, dtype=np.float32)

    if y.size == 0:
        raise ValueError(f"Audio file contains no samples: {audio_path}")
    if not np.isfinite(y).all():
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y, int(sample_rate)


class DeepfakeAudioDetector:
    # Minimum time between full spectral re-analyses during rapid repeated
    # calls (e.g. the live mic loop calling predict_window_fused every
    # ~150ms). Spectral characteristics don't meaningfully change within a
    # few hundred ms, so between refreshes the last computed spectral
    # result is reused instead of recomputing STFT/YIN/etc. every tick.
    # ONNX inference (cheap, fast) still runs fresh every call.
    SPECTRAL_REFRESH_INTERVAL = 0.35

    def __init__(self, model_path: str = None, threshold: float = None,
                 sensitivity: str = DEFAULT_SENSITIVITY,
                 onnx_weight: float = None, spectral_weight: float = None,
                 verbose: bool = False):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME)

        preset = SENSITIVITY_PRESETS.get(sensitivity, SENSITIVITY_PRESETS[DEFAULT_SENSITIVITY]).copy()
        if threshold is None and onnx_weight is None and spectral_weight is None:
            preset.update(_load_calibration())

        self.sensitivity = sensitivity if sensitivity in SENSITIVITY_PRESETS else DEFAULT_SENSITIVITY
        self.onnx_weight = float(onnx_weight) if onnx_weight is not None else preset["onnx_weight"]
        self.spectral_weight = float(spectral_weight) if spectral_weight is not None else preset["spectral_weight"]
        self.threshold = float(threshold) if threshold is not None else preset["threshold"]

        self.model_path = model_path
        self.dynamic_threshold = dynamic_threshold(base=self.threshold)
        self.verbose = verbose

        # Spectral result cache (see SPECTRAL_REFRESH_INTERVAL above)
        self._cached_spectral_detail = None
        self._cached_spectral_time = 0.0

        if not os.path.exists(self.model_path):
            print(f"[!] Model not found at '{self.model_path}'. Initiating automatic download...")
            download_model(self.model_path)

        print(f"[*] Loading ONNX model from: {self.model_path}")
        self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(f"[✓] Model loaded. Input: '{self.input_name}', Output: '{self.output_name}'")

        self.spectral_analyzer = SpectralArtifactAnalyzer(sr=SAMPLE_RATE, verbose=verbose)
        print(f"[✓] Spectral Artifact Analyzer initialized (modern AI TTS detection)")
        print(f"[✓] Sensitivity: {self.sensitivity}  |  "
              f"ONNX weight: {self.onnx_weight:.0%}  |  Spectral weight: {self.spectral_weight:.0%}  |  "
              f"Threshold: {self.threshold:.2f}")

    def preprocess_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        if len(audio_chunk) < WINDOW_SIZE:
            audio_chunk = np.pad(audio_chunk, (0, WINDOW_SIZE - len(audio_chunk)), mode='constant')
        elif len(audio_chunk) > WINDOW_SIZE:
            audio_chunk = audio_chunk[:WINDOW_SIZE]

        mean = np.mean(audio_chunk)
        std = np.sqrt(np.var(audio_chunk) + 1e-5)
        norm_audio = (audio_chunk - mean) / std
        return norm_audio.astype(np.float32).reshape(1, WINDOW_SIZE)

    def predict_window(self, audio_chunk: np.ndarray) -> tuple[float, float]:
        model_input = self.preprocess_chunk(audio_chunk)
        raw_outputs = self.session.run([self.output_name], {self.input_name: model_input})
        logits = raw_outputs[0][0]

        if self.verbose:
            print(f"\n[ONNX] Raw logits: {logits}")

        probs = softmax(logits)

        if self.verbose:
            print(f"[ONNX] Real: {float(probs[0]):.4f}, Fake: {float(probs[1]):.4f}")

        return float(probs[0]), float(probs[1])

    def predict_window_fused(self, audio_chunk: np.ndarray, allow_cached_spectral: bool = True):
        """
        Fuse ONNX and spectral artifact scores for one audio window.

        allow_cached_spectral=True (default; used by the live mic loop)
        lets a recent spectral analysis be reused within
        SPECTRAL_REFRESH_INTERVAL for latency. File/folder analysis calls
        this with allow_cached_spectral=False so every window gets a
        fully fresh spectral score — correctness over speed there.
        """
        audio_chunk = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)
        _, onnx_fake = self.predict_window(audio_chunk)

        analysis_chunk = audio_chunk[:WINDOW_SIZE]
        if len(analysis_chunk) < WINDOW_SIZE:
            # Keep the ONNX input at its required fixed size, but do not send
            # the artificial zero padding through the spectral heuristics.
            # During live capture that padding can be most of the window and
            # forces the fused score toward an unhelpful midpoint.
            spectral_chunk = analysis_chunk
            analysis_chunk = np.pad(
                analysis_chunk,
                (0, WINDOW_SIZE - len(analysis_chunk)),
                mode="constant",
            )
        else:
            spectral_chunk = analysis_chunk
        if len(spectral_chunk) < self.spectral_analyzer.n_fft:
            spectral_chunk = np.pad(
                spectral_chunk,
                (0, self.spectral_analyzer.n_fft - len(spectral_chunk)),
                mode="constant",
            )

        now = time.time()
        use_cache = (
            allow_cached_spectral
            and self._cached_spectral_detail is not None
            and (now - self._cached_spectral_time) < self.SPECTRAL_REFRESH_INTERVAL
        )

        if use_cache:
            spectral_detail = self._cached_spectral_detail
        else:
            spectral_detail = self.spectral_analyzer.analyze(spectral_chunk)
            self._cached_spectral_detail = spectral_detail
            self._cached_spectral_time = now

        spectral_fake = float(spectral_detail.get("artifact_score", 0.0))

        # Keep calibration and live/file inference on the exact same fusion path.
        fake_prob = fuse_fake_scores(
            onnx_fake,
            spectral_fake,
            self.onnx_weight,
            self.spectral_weight,
        )

        real_prob = 1.0 - fake_prob

        if self.verbose:
            print(
                f"[FUSION] Real: {real_prob:.4f}, Fake: {fake_prob:.4f} "
                f"(ONNX: {onnx_fake:.4f}, Spectral: {spectral_fake:.4f}, "
                f"weighted: {weighted_fake:.4f}, cached_spectral: {use_cache})"
            )

        return real_prob, fake_prob, spectral_detail

    def predict_audio(self, y: np.ndarray, sr: int = SAMPLE_RATE, hop_sec: float = 1.0) -> dict:
        if sr != SAMPLE_RATE:
            y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE

        total_samples = len(y)
        duration_sec = total_samples / sr
        hop_samples = int(hop_sec * sr)

        if total_samples <= WINDOW_SIZE:
            rms = float(np.sqrt(np.mean(y ** 2)))
            # VAD is an activity aid only. Keep the original RMS condition as
            # an OR fallback so quiet multilingual speech is not discarded.
            if rms < SILENCE_RMS:
                return {
                    "is_fake": False, "fake_prob": 0.0, "real_prob": 1.0,
                    "duration": duration_sec, "status": "SILENCE", "windows": []
                }
            real_p, fake_p, spectral_detail = self.predict_window_fused(y, allow_cached_spectral=False)
            status = "FAKE" if fake_p >= self.threshold else "REAL"
            display_fake, display_real = display_confidence(
                fake_p, status, self.threshold
            )
            return {
                "is_fake": fake_p >= self.threshold,
                "fake_prob": fake_p, "real_prob": real_p, "duration": duration_sec,
                "decision_threshold": self.threshold,
                "display_fake_confidence": display_fake,
                "display_real_confidence": display_real,
                "status": status,
                "spectral": spectral_detail,
                "windows": [{"start": 0.0, "end": duration_sec, "fake_prob": fake_p, "real_prob": real_p, "rms": rms}]
            }

        windows = []
        fake_probs = []
        active_speech_windows = 0

        for start_idx in range(0, total_samples, hop_samples):
            end_idx = min(start_idx + WINDOW_SIZE, total_samples)
            chunk = y[start_idx:end_idx]

            chunk_rms = float(np.sqrt(np.mean(chunk ** 2)))
            t_start = start_idx / sr
            t_end = end_idx / sr

            # Do not let a tiny padded tail influence a full-file verdict.
            # Keep it for genuinely short recordings, which are handled
            # above, but ignore sub-half-second fragments on longer files.
            if len(chunk) < int(0.5 * sr):
                if end_idx >= total_samples:
                    break
                continue

            if chunk_rms >= SILENCE_RMS:
                active_speech_windows += 1
                real_p, fake_p, _ = self.predict_window_fused(chunk, allow_cached_spectral=False)
                fake_probs.append(fake_p)
                windows.append({
                    "start": round(t_start, 2), "end": round(t_end, 2),
                    "fake_prob": round(fake_p, 4), "real_prob": round(real_p, 4),
                    "rms": round(chunk_rms, 4)
                })

            if end_idx >= total_samples:
                break

        if not fake_probs:
            return {
                "is_fake": False, "fake_prob": 0.0, "real_prob": 1.0,
                "duration": duration_sec, "status": "SILENCE_OR_EMPTY", "windows": []
            }

        fake_arr = np.array(fake_probs)
        aggregate_fake = aggregate_fake_scores(fake_arr)
        aggregate_real = 1.0 - aggregate_fake

        dynamic_threshold_value = self.dynamic_threshold(aggregate_fake)
        is_fake = aggregate_fake >= dynamic_threshold_value
        full_spectral = self.spectral_analyzer.analyze(y[:SAMPLE_RATE * 10])

        status = "FAKE" if is_fake else "REAL"
        display_fake, display_real = display_confidence(
            aggregate_fake, status, dynamic_threshold_value
        )
        return {
            "is_fake": is_fake,
            "fake_prob": aggregate_fake, "real_prob": aggregate_real,
            "decision_threshold": dynamic_threshold_value,
            "display_fake_confidence": display_fake,
            "display_real_confidence": display_real,
            "duration": round(duration_sec, 2),
            "active_windows": active_speech_windows,
            "status": status,
            "spectral": full_spectral,
            "windows": windows
        }

    def predict_file(self, audio_path: str) -> dict:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        y, sr = load_audio_file(audio_path)
        return self.predict_audio(y, sr=sr, hop_sec=1.0)


# ============================================================
# CALIBRATION HELPER
# ============================================================

def calibrate_thresholds(real_folder: str, fake_folder: str, model_path: str = None,
                          save_path: str = None) -> dict:
    """
    Empirically tunes onnx_weight/spectral_weight and the decision threshold
    against labeled examples, instead of relying on guessed constants.

    real_folder: directory of known-genuine audio clips
    fake_folder: directory of known-AI-generated audio clips (ideally
                 including the specific modern TTS systems you want to
                 catch)

    This is the only way to stop the back-and-forth of manually tweaking
    weights every time a false positive or false negative shows up — run
    this with even a small set of labeled clips (10-15 per folder is
    enough to start) and use the returned onnx_weight/spectral_weight/
    threshold directly instead of guessed constants.
    """
    import glob

    detector = DeepfakeAudioDetector(
        model_path=model_path,
        sensitivity=DEFAULT_SENSITIVITY,
        verbose=False,
    )

    def load_scores(folder):
        file_scores = []
        paths = glob.glob(os.path.join(folder, "**", "*"), recursive=True)
        for path in paths:
            if not os.path.isfile(path):
                continue
            try:
                y, sr = load_audio_file(path)
            except Exception:
                continue

            if sr != SAMPLE_RATE:
                y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
            if len(y) == 0:
                continue

            chunks = []
            if len(y) <= WINDOW_SIZE:
                if float(np.sqrt(np.mean(y ** 2))) >= SILENCE_RMS:
                    chunks.append(y)
            else:
                for start_idx in range(0, len(y), SAMPLE_RATE):
                    end_idx = min(start_idx + WINDOW_SIZE, len(y))
                    chunk = y[start_idx:end_idx]
                    if len(chunk) < int(0.5 * SAMPLE_RATE) and end_idx >= len(y):
                        break
                    if float(np.sqrt(np.mean(chunk ** 2))) >= SILENCE_RMS:
                        chunks.append(chunk)

            onnx_scores, spectral_scores = [], []
            for chunk in chunks:
                onnx_scores.append(detector.predict_window(chunk)[1])
                spectral_chunk = chunk
                if len(spectral_chunk) < detector.spectral_analyzer.n_fft:
                    spectral_chunk = np.pad(
                        spectral_chunk,
                        (0, detector.spectral_analyzer.n_fft - len(spectral_chunk)),
                        mode="constant",
                    )
                spectral_scores.append(
                    detector.spectral_analyzer.analyze(spectral_chunk)["artifact_score"]
                )
            if onnx_scores:
                file_scores.append((
                    np.asarray(onnx_scores),
                    np.asarray(spectral_scores),
                    len(y) <= WINDOW_SIZE,
                ))
        return file_scores

    real_scores = load_scores(real_folder)
    fake_scores = load_scores(fake_folder)

    if not real_scores or not fake_scores:
        raise ValueError("Need at least one loadable file in each folder.")

    best = {"balanced_acc": -1}
    for w in np.arange(0.2, 0.95, 0.05):
        def aggregate_files(file_scores):
            values = []
            for onnx_scores, spectral_scores, _ in file_scores:
                fused = [
                    fuse_fake_scores(onnx, spectral, w, 1 - w)
                    for onnx, spectral in zip(onnx_scores, spectral_scores)
                ]
                values.append(aggregate_fake_scores(np.asarray(fused)))
            return np.asarray(values)

        real_fused = aggregate_files(real_scores)
        fake_fused = aggregate_files(fake_scores)
        for t in np.arange(0.15, 0.8, 0.02):
            real_predictions = [
                score >= (t if real_scores[index][2] else dynamic_threshold(base=t)(score))
                for index, score in enumerate(real_fused)
            ]
            fake_predictions = [
                score >= (t if fake_scores[index][2] else dynamic_threshold(base=t)(score))
                for index, score in enumerate(fake_fused)
            ]
            tpr = float(np.mean(fake_predictions))
            tnr = float(np.mean(np.logical_not(real_predictions)))
            balanced_acc = (tpr + tnr) / 2
            if balanced_acc > best["balanced_acc"]:
                best = {
                    "onnx_weight": round(float(w), 2),
                    "spectral_weight": round(float(1 - w), 2),
                    "threshold": round(float(t), 2),
                    "balanced_acc": round(balanced_acc, 4),
                    "fake_recall": round(tpr, 4),
                    "real_recall": round(tnr, 4),
                }

    if save_path is None:
        save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "calibration_result.json",
        )
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print(f"[✓] Calibration complete. Best config saved to {save_path}:")
    print(json.dumps(best, indent=2))
    print("\nThis profile is loaded automatically by DeepfakeAudioDetector.")

    return best


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dhwani Deepfake Audio Model Tool")
    parser.add_argument("--download", action="store_true", help="Download the Dhwani ONNX model from Hugging Face")
    parser.add_argument("--calibrate", nargs=2, metavar=("REAL_FOLDER", "FAKE_FOLDER"),
                         help="Calibrate detection weights/threshold against labeled folders")
    args = parser.parse_args()

    if args.download:
        download_model()
    elif args.calibrate:
        calibrate_thresholds(args.calibrate[0], args.calibrate[1])
    else:
        parser.print_help()