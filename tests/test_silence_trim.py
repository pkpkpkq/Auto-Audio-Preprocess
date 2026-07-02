from __future__ import annotations

import os
import sys
import wave

import librosa

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from audio_normalizer import trim_silence  # noqa: E402
from config_types import NormalizerConfig  # noqa: E402


EVIDENCE_DIR = os.path.join(REPO_ROOT, ".sisyphus", "evidence")


def _trim_config(test_config, *, top_db=30, min_trim_ms=100, keep_padding_ms=50):
    base = test_config.normalizer
    return NormalizerConfig(
        target_sample_rate=base.target_sample_rate, target_channels=base.target_channels,
        target_bit_depth=base.target_bit_depth, input_formats=base.input_formats,
        trim_enabled=base.trim_enabled, trim_top_db=top_db,
        trim_frame_length=base.trim_frame_length, trim_hop_length=base.trim_hop_length,
        trim_min_trim_ms=min_trim_ms, trim_keep_padding_ms=keep_padding_ms,
        loudness_enabled=base.loudness_enabled, loudness_target_lufs=base.loudness_target_lufs,
        loudness_true_peak=base.loudness_true_peak, loudness_lra=base.loudness_lra,
        loudness_dual_pass=base.loudness_dual_pass,
        dc_offset_removal_enabled=base.dc_offset_removal_enabled,
        dc_offset_threshold=base.dc_offset_threshold,
    )


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
