from __future__ import annotations

import filecmp
import os
import sys

import numpy as np
import soundfile as sf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from audio_normalizer import _measure_loudness, normalize_loudness  # noqa: E402
from config_types import NormalizerConfig  # noqa: E402


EVIDENCE_DIR = os.path.join(REPO_ROOT, ".sisyphus", "evidence")


def _loudness_config(test_config, *, enabled=True, target_lufs=-16, dual_pass=False):
    base = test_config.normalizer
    return NormalizerConfig(
        target_sample_rate=base.target_sample_rate, target_channels=base.target_channels,
        target_bit_depth=base.target_bit_depth, input_formats=base.input_formats,
        trim_enabled=base.trim_enabled, trim_top_db=base.trim_top_db,
        trim_frame_length=base.trim_frame_length, trim_hop_length=base.trim_hop_length,
        trim_min_trim_ms=base.trim_min_trim_ms, trim_keep_padding_ms=base.trim_keep_padding_ms,
        loudness_enabled=enabled, loudness_target_lufs=target_lufs,
        loudness_true_peak=-1.5, loudness_lra=7, loudness_dual_pass=dual_pass,
        dc_offset_removal_enabled=base.dc_offset_removal_enabled,
        dc_offset_threshold=base.dc_offset_threshold,
    )


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
    cfg = _loudness_config(test_config, enabled=False)
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
    cfg = test_config.normalizer
    out_path = str(tmp_path / "loud_out.wav")
    normalize_loudness(str(sample_wav_22k_mono), out_path, cfg)
    import wave
    with wave.open(out_path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
    assert sr == cfg.target_sample_rate
    assert ch == cfg.target_channels
    assert sw == cfg.target_bit_depth // 8


def test_very_short_file_dual_pass_with_padding(tmp_path, test_config, caplog):
    """测试极短音频 (<400ms) 在双 pass 模式下通过 padding 正常完成响度归一化。"""
    import logging

    sr = 44100
    duration = 0.2  # 200ms, 远低于 400ms EBU R128 最小窗口
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    data = (np.sin(2.0 * np.pi * 440.0 * t) * 0.3).astype(np.float32)
    short_wav = tmp_path / "very_short.wav"
    output = tmp_path / "very_short_loud.wav"
    sf.write(str(short_wav), data, sr, subtype="PCM_16")

    cfg = _loudness_config(test_config, target_lufs=-16, dual_pass=True)

    with caplog.at_level(logging.WARNING, logger="audio_normalizer"):
        normalize_loudness(short_wav, output, cfg)

    assert output.exists()
    assert output.stat().st_size > 0

    # 验证记录了 padding 警告
    pad_warnings = [r for r in caplog.records if "pad" in r.message.lower() or "pad" in r.message]
    assert len(pad_warnings) >= 1, "应记录短音频 padding 警告"

    _write_evidence(
        "task-5-dual-pass-short-padding.txt",
        f"input={short_wav}\noutput={output}\n"
        f"output_exists={output.exists()}\noutput_size={output.stat().st_size}\n"
        f"warning_logged={len(pad_warnings) >= 1}\n",
    )
