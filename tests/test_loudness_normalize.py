from __future__ import annotations

import copy
import filecmp
import os
import sys

import numpy as np
import soundfile as sf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from audio_normalizer import _measure_loudness, normalize_loudness  # noqa: E402
from config_types import NormalizerConfig, build_normalizer_config  # noqa: E402


EVIDENCE_DIR = os.path.join(REPO_ROOT, ".sisyphus", "evidence")


def _loudness_config(test_config, *, enable=True, target_lufs=-16, dual_pass=False):
    cfg = copy.deepcopy(test_config)
    cfg["audio"]["loudness"].update(
        {
            "enable": enable,
            "target_lufs": target_lufs,
            "true_peak": -1.5,
            "lra": 7,
            "dual_pass": dual_pass,
        }
    )
    return build_normalizer_config(cfg)


def _write_evidence(name, content):
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    with open(os.path.join(EVIDENCE_DIR, name), "w", encoding="utf-8") as f:
        f.write(content)


def test_quiet_wav_normalized_close_to_target_lufs(sample_wav_quiet, tmp_path, test_config):
    cfg = _loudness_config(test_config, target_lufs=-16, dual_pass=False)
    output = tmp_path / "quiet_loudnorm.wav"

    normalize_loudness(sample_wav_quiet, output, cfg)

    measured = _measure_loudness(output, cfg)
    output_lufs = float(measured["input_i"])
    _write_evidence(
        "task-5-lufs-normalize.txt",
        f"target_lufs=-16\noutput_lufs={output_lufs:.3f}\noutput={output}\n",
    )
    assert output.exists()
    assert output.stat().st_size > 0
    assert -18 <= output_lufs <= -14


def test_loudness_disabled_copies_input_without_ffmpeg(sample_wav_22k_mono, tmp_path, test_config, monkeypatch):
    cfg = _loudness_config(test_config, enable=False)
    output = tmp_path / "disabled_copy.wav"

    def fail_if_called(*args, **kwargs):
        raise AssertionError("ffmpeg should not be called when loudness is disabled")

    monkeypatch.setattr("audio_normalizer.subprocess.run", fail_if_called)

    normalize_loudness(sample_wav_22k_mono, output, cfg)

    _write_evidence(
        "task-5-loudness-disabled.txt",
        f"input={sample_wav_22k_mono}\noutput={output}\nidentical={filecmp.cmp(sample_wav_22k_mono, output, shallow=False)}\n",
    )
    assert output.exists()
    assert filecmp.cmp(sample_wav_22k_mono, output, shallow=False)


def test_very_short_file_loudnorm_does_not_crash(tmp_path, test_config):
    sr = 22050
    t = np.linspace(0.0, 0.1, int(sr * 0.1), endpoint=False)
    data = (np.sin(2.0 * np.pi * 440.0 * t) * 0.05).astype(np.float32)
    short_wav = tmp_path / "short.wav"
    output = tmp_path / "short_loudnorm.wav"
    sf.write(str(short_wav), data, sr, subtype="PCM_16")
    cfg = _loudness_config(test_config, target_lufs=-16, dual_pass=False)

    normalize_loudness(short_wav, output, cfg)

    _write_evidence(
        "task-5-short-file-loudnorm.txt",
        f"input={short_wav}\noutput={output}\noutput_exists={output.exists()}\noutput_size={output.stat().st_size}\n",
    )
    assert output.exists()
    assert output.stat().st_size > 0


def test_loudness_output_sample_rate_pinned(tmp_path, sample_wav_22k_mono, test_config):
    cfg = build_normalizer_config(test_config)
    out_path = str(tmp_path / "loud_out.wav")
    normalize_loudness(str(sample_wav_22k_mono), out_path, cfg)
    import wave
    with wave.open(out_path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
    assert sr == test_config["audio"]["target_sample_rate"]
    assert ch == test_config["audio"]["target_channels"]
    assert sw == test_config["audio"]["target_bit_depth"] // 8
