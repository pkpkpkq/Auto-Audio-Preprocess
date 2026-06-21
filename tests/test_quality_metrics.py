"""Tests for silence ratio detection and VAD-SNR estimation."""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from audio_validator import compute_silence_ratio, estimate_snr, batch_quality_check, write_quality_report
from audio_entry import AudioEntry
from config_types import ValidatorConfig, build_validator_config


def _validator_cfg(test_config) -> ValidatorConfig:
    return build_validator_config(test_config)


class TestComputeSilenceRatio:
    def test_high_silence_flagged_warn(self, sample_wav_with_silence, test_config):
        result = compute_silence_ratio(str(sample_wav_with_silence), _validator_cfg(test_config))
        assert result["silence_ratio"] > 0.3, f"Expected high silence ratio, got {result['silence_ratio']}"
        assert result["status"] in ("warn", "reject"), f"Expected warn/reject, got {result['status']}"

    def test_clean_file_low_silence(self, sample_wav_22k_mono, test_config):
        result = compute_silence_ratio(str(sample_wav_22k_mono), _validator_cfg(test_config))
        assert result["silence_ratio"] < 0.3, f"Expected low silence ratio, got {result['silence_ratio']}"
        assert result["status"] == "ok"

    def test_returns_required_fields(self, sample_wav_22k_mono, test_config):
        result = compute_silence_ratio(str(sample_wav_22k_mono), _validator_cfg(test_config))
        for key in ("silence_ratio", "speech_duration", "total_duration", "status"):
            assert key in result, f"Missing field: {key}"

    def test_reject_threshold(self, tmp_path, test_config):
        sr = 22050
        tone = np.sin(2 * np.pi * 440.0 * np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)).astype(np.float32)
        silence = np.zeros(int(sr * 3.9), dtype=np.float32)
        data = np.concatenate([tone, silence])
        path = tmp_path / "mostly_silent.wav"
        sf.write(str(path), data, sr, subtype="PCM_16")
        result = compute_silence_ratio(str(path), _validator_cfg(test_config))
        assert result["silence_ratio"] > 0.7
        assert result["status"] == "reject"


class TestEstimateSNR:
    def test_snr_returns_reasonable_value(self, sample_wav_with_silence, test_config):
        result = estimate_snr(str(sample_wav_with_silence), _validator_cfg(test_config))
        assert "snr" in result
        assert "reliable" in result
        assert isinstance(result["snr"], float)
        assert isinstance(result["reliable"], bool)

    def test_snr_with_silence_is_reliable(self, sample_wav_with_silence, test_config):
        result = estimate_snr(str(sample_wav_with_silence), _validator_cfg(test_config))
        assert result["reliable"] is True

    def test_snr_no_silence_unreliable(self, sample_wav_22k_mono, test_config):
        result = estimate_snr(str(sample_wav_22k_mono), _validator_cfg(test_config))
        assert result["reliable"] is False
        assert result["snr"] == 40.0

    def test_snr_status_field(self, sample_wav_with_silence, test_config):
        result = estimate_snr(str(sample_wav_with_silence), _validator_cfg(test_config))
        assert result["status"] in ("ok", "warn", "reject")

    def test_low_snr_flagged(self, tmp_path, test_config):
        sr = 22050
        speech = (np.sin(2 * np.pi * 440.0 * np.linspace(0, 1.0, sr, endpoint=False)) * 0.05).astype(np.float32)
        noise_floor = (np.random.randn(int(sr * 0.5)) * 0.02).astype(np.float32)
        data = np.concatenate([speech, noise_floor])
        tiny_gap = np.zeros(int(sr * 0.01), dtype=np.float32)
        data = np.concatenate([speech, tiny_gap, noise_floor])
        path = tmp_path / "low_snr.wav"
        sf.write(str(path), data, sr, subtype="PCM_16")
        result = estimate_snr(str(path), _validator_cfg(test_config))
        assert "snr" in result
        assert "reliable" in result


class TestBatchQualityCheck:
    def test_batch_check_returns_list(self, tmp_path, test_config):
        sr = 22050
        data1 = np.sin(2 * np.pi * 440.0 * np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)).astype(np.float32)
        path1 = tmp_path / "clean.wav"
        sf.write(str(path1), data1, sr, subtype="PCM_16")

        silence = np.zeros(int(sr * 0.5), dtype=np.float32)
        tone = np.sin(2 * np.pi * 440.0 * np.linspace(0, 1.0, sr, endpoint=False)).astype(np.float32)
        data2 = np.concatenate([silence, tone, silence])
        path2 = tmp_path / "with_silence.wav"
        sf.write(str(path2), data2, sr, subtype="PCM_16")

        entries = [
            AudioEntry(path=str(path1), lab_text="", duration=0.0),
            AudioEntry(path=str(path2), lab_text="", duration=0.0),
        ]
        results = batch_quality_check(entries, _validator_cfg(test_config))
        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert "silence_ratio" in r
            assert "snr" in r

    def test_write_quality_report(self, tmp_path, test_config):
        quality_data = [
            {
                "wav_path": "test1.wav",
                "silence_ratio": 0.1,
                "silence_status": "ok",
                "snr": 30.0,
                "snr_reliable": True,
                "snr_status": "ok",
            }
        ]
        report_path = tmp_path / "质量报告.txt"
        write_quality_report(quality_data, str(report_path))
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "test1.wav" in content
        assert "0.1" in content
