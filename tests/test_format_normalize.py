from __future__ import annotations

import copy
import os
import sys
import wave

import librosa
import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from audio_normalizer import needs_normalization, normalize_format  # noqa: E402
from audio_utils import get_audio_duration_seconds as get_audio_duration_any  # noqa: E402
from config_types import NormalizerConfig, build_normalizer_config  # noqa: E402


def _audio_config(test_config, *, sample_rate=44100, channels=1, bit_depth=16, dc_offset=False):
    cfg = copy.deepcopy(test_config)
    cfg["audio"]["target_sample_rate"] = sample_rate
    cfg["audio"]["target_channels"] = channels
    cfg["audio"]["target_bit_depth"] = bit_depth
    cfg["audio"]["dc_offset_removal"]["enable"] = dc_offset
    return build_normalizer_config(cfg)


def _wav_params(path):
    with wave.open(str(path), "rb") as wf:
        return wf.getframerate(), wf.getnchannels(), wf.getsampwidth()


def test_flac_converted_to_target_wav_format(sample_flac, tmp_path, test_config):
    cfg = _audio_config(test_config, sample_rate=44100, channels=1, bit_depth=16)
    output = tmp_path / "normalized.wav"

    normalize_format(sample_flac, output, cfg)

    assert output.exists()
    assert output.stat().st_size > 0
    assert _wav_params(output) == (44100, 1, 2)
    assert get_audio_duration_any(output) > 0
    assert needs_normalization(sample_flac, cfg) is True


def test_48k_24bit_wav_converted_to_44k_16bit_mono(sample_wav_48k_24bit, tmp_path, test_config):
    cfg = _audio_config(test_config, sample_rate=44100, channels=1, bit_depth=16)
    output = tmp_path / "normalized_48k.wav"
    original_size = sample_wav_48k_24bit.stat().st_size

    assert needs_normalization(sample_wav_48k_24bit, cfg) is True
    normalize_format(sample_wav_48k_24bit, output, cfg)

    assert output.exists()
    assert output.stat().st_size > 0
    assert _wav_params(output) == (44100, 1, 2)
    assert sample_wav_48k_24bit.stat().st_size == original_size


def test_already_normalized_wav_is_skipped_by_needs_normalization(sample_wav_22k_mono, tmp_path, test_config):
    cfg = _audio_config(test_config, sample_rate=22050, channels=1, bit_depth=16)
    output = tmp_path / "should_not_exist.wav"

    assert needs_normalization(sample_wav_22k_mono, cfg) is False
    assert not output.exists()


def test_dc_offset_removal_highpass_reduces_mean(sample_wav_dc_offset, tmp_path, test_config):
    cfg = _audio_config(test_config, sample_rate=22050, channels=1, bit_depth=16, dc_offset=True)
    output = tmp_path / "dc_removed.wav"

    before, _ = librosa.load(str(sample_wav_dc_offset), sr=None, mono=True)
    assert abs(float(np.mean(before))) > 0.05

    normalize_format(sample_wav_dc_offset, output, cfg)

    after, _ = librosa.load(str(output), sr=None, mono=True)
    assert output.stat().st_size > 0
    assert abs(float(np.mean(after))) < 0.005


def test_ffmpeg_failure_raises_runtime_error(tmp_path, test_config):
    cfg = _audio_config(test_config)
    output = tmp_path / "missing.wav"

    with pytest.raises(RuntimeError, match="ffmpeg.*失败"):
        normalize_format(tmp_path / "does_not_exist.flac", output, cfg)

    assert not output.exists()
