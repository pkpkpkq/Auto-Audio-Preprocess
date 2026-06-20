from __future__ import annotations

import copy
import os
import sys
import wave

import librosa

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from audio_normalizer import trim_silence  # noqa: E402


EVIDENCE_DIR = os.path.join(REPO_ROOT, ".sisyphus", "evidence")


def _trim_config(test_config, *, top_db=30, min_trim_ms=100, keep_padding_ms=50):
    cfg = copy.deepcopy(test_config)
    cfg["audio"]["trim_silence"].update(
        {
            "top_db": top_db,
            "min_trim_ms": min_trim_ms,
            "keep_padding_ms": keep_padding_ms,
        }
    )
    return cfg


def _duration(path):
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def _write_evidence(name, content):
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    with open(os.path.join(EVIDENCE_DIR, name), "w", encoding="utf-8") as f:
        f.write(content)


def test_wav_with_leading_and_trailing_silence_is_trimmed(sample_wav_with_silence, tmp_path, test_config):
    cfg = _trim_config(test_config, min_trim_ms=100, keep_padding_ms=50)
    output = tmp_path / "trimmed.wav"

    trim_silence(sample_wav_with_silence, output, cfg)

    original_duration = _duration(sample_wav_with_silence)
    output_duration = _duration(output)
    _write_evidence(
        "task-6-silence-trimmed.txt",
        f"original_duration={original_duration:.6f}\noutput_duration={output_duration:.6f}\n",
    )
    assert output.exists()
    assert output_duration < original_duration
    assert 1.05 <= output_duration <= 1.20


def test_already_clean_wav_is_copied_without_retrim(sample_wav_22k_mono, tmp_path, test_config):
    cfg = _trim_config(test_config, min_trim_ms=100, keep_padding_ms=50)
    output = tmp_path / "clean_output.wav"

    trim_silence(sample_wav_22k_mono, output, cfg)

    original_duration = _duration(sample_wav_22k_mono)
    output_duration = _duration(output)
    _write_evidence(
        "task-6-no-trim-needed.txt",
        f"original_duration={original_duration:.6f}\noutput_duration={output_duration:.6f}\n",
    )
    assert output.exists()
    assert output_duration == original_duration


def test_trimmed_output_is_valid_16bit_pcm_wav(sample_wav_with_silence, tmp_path, test_config):
    cfg = _trim_config(test_config, min_trim_ms=100, keep_padding_ms=50)
    output = tmp_path / "valid_trimmed.wav"

    trim_silence(sample_wav_with_silence, output, cfg)

    with wave.open(str(output), "rb") as wf:
        params = {
            "framerate": wf.getframerate(),
            "channels": wf.getnchannels(),
            "sampwidth": wf.getsampwidth(),
            "frames": wf.getnframes(),
        }

    y, sr = librosa.load(str(output), sr=None)
    _write_evidence(
        "task-6-valid-wav.txt",
        "\n".join(f"{key}={value}" for key, value in params.items()) + f"\nlibrosa_sr={sr}\nsamples={len(y)}\n",
    )
    assert params == {
        "framerate": 22050,
        "channels": 1,
        "sampwidth": 2,
        "frames": params["frames"],
    }
    assert params["frames"] > 0
    assert sr == 22050
    assert len(y) > 0
