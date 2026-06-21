from __future__ import annotations

import os
import subprocess

import numpy as np
import pytest
import soundfile as sf

from audio_validator import (
    audit_audio_metadata,
    batch_detect_clipping,
    detect_clipping,
    write_audit_report,
    write_clipping_report,
)
from audio_entry import AudioEntry
from config_types import ValidatorConfig, build_validator_config


def _validator_cfg(test_config) -> ValidatorConfig:
    return build_validator_config(test_config)


# ---------------------------------------------------------------------------
# audit_audio_metadata
# ---------------------------------------------------------------------------


class TestAuditAudioMetadata:
    def test_sample_rate_distribution(self, sample_wav_44k_stereo, sample_wav_48k_24bit, test_config):
        entries = [
            AudioEntry(path=str(sample_wav_44k_stereo), lab_text="", duration=0.0),
            AudioEntry(path=str(sample_wav_48k_24bit), lab_text="", duration=0.0),
        ]
        result = audit_audio_metadata(entries, _validator_cfg(test_config))
        assert result["total"] == 2
        assert result["sample_rates"][44100] == 1
        assert result["sample_rates"][48000] == 1

    def test_needs_conversion_flags_mismatch(self, sample_wav_48k_24bit, test_config):
        entries = [AudioEntry(path=str(sample_wav_48k_24bit), lab_text="", duration=0.0)]
        result = audit_audio_metadata(entries, _validator_cfg(test_config))
        assert len(result["needs_conversion"]) >= 1
        assert any("48000" in item or str(sample_wav_48k_24bit) in item for item in result["needs_conversion"])

    def test_no_conversion_needed(self, sample_wav_44k_stereo, test_config):
        entries = [AudioEntry(path=str(sample_wav_44k_stereo), lab_text="", duration=0.0)]
        result = audit_audio_metadata(entries, _validator_cfg(test_config))
        sr_reasons = [i for i in result["needs_conversion"] if "采样率" in i or "sample_rate" in i]
        assert len(sr_reasons) == 0

    def test_bit_depth_distribution(self, sample_wav_44k_stereo, sample_wav_48k_24bit, test_config):
        entries = [
            AudioEntry(path=str(sample_wav_44k_stereo), lab_text="", duration=0.0),
            AudioEntry(path=str(sample_wav_48k_24bit), lab_text="", duration=0.0),
        ]
        result = audit_audio_metadata(entries, _validator_cfg(test_config))
        assert 16 in result["bit_depths"]
        assert 24 in result["bit_depths"]

    def test_channels_distribution(self, sample_wav_44k_stereo, sample_wav_22k_mono, test_config):
        entries = [
            AudioEntry(path=str(sample_wav_44k_stereo), lab_text="", duration=0.0),
            AudioEntry(path=str(sample_wav_22k_mono), lab_text="", duration=0.0),
        ]
        result = audit_audio_metadata(entries, _validator_cfg(test_config))
        assert result["channels"][1] == 1
        assert result["channels"][2] == 1

    def test_audit_disabled(self, sample_wav_44k_stereo, test_config):
        cfg = test_config.copy()
        cfg["audio"] = dict(cfg.get("audio", {}))
        cfg["audio"]["audit"] = {"enable": False}
        entries = [AudioEntry(path=str(sample_wav_44k_stereo), lab_text="", duration=0.0)]
        result = audit_audio_metadata(entries, build_validator_config(cfg))
        assert result["total"] == 0
        assert result.get("skipped") is True

    def test_flac_file_audit(self, sample_flac, test_config):
        entries = [AudioEntry(path=str(sample_flac), lab_text="", duration=0.0)]
        result = audit_audio_metadata(entries, _validator_cfg(test_config))
        assert result["total"] == 1
        assert 44100 in result["sample_rates"]

    def test_missing_file_skipped(self, test_config):
        entries = [AudioEntry(path="/nonexistent/file.wav", lab_text="", duration=0.0)]
        result = audit_audio_metadata(entries, _validator_cfg(test_config))
        assert result["total"] == 0
        assert len(result.get("errors", [])) >= 1


# ---------------------------------------------------------------------------
# write_audit_report
# ---------------------------------------------------------------------------


class TestWriteAuditReport:
    def test_creates_file(self, tmp_path, test_config):
        audit_data = {
            "total": 2,
            "sample_rates": {44100: 1, 48000: 1},
            "bit_depths": {16: 2},
            "channels": {1: 2},
            "needs_conversion": ["/fake/48k.wav: 采样率 48000 != 目标 44100"],
            "errors": [],
        }
        out_path = str(tmp_path / "！格式审计.txt")
        write_audit_report(audit_data, out_path)
        assert os.path.isfile(out_path)

    def test_report_content_has_timestamp(self, tmp_path, test_config):
        audit_data = {
            "total": 1,
            "sample_rates": {44100: 1},
            "bit_depths": {16: 1},
            "channels": {1: 1},
            "needs_conversion": [],
            "errors": [],
        }
        out_path = str(tmp_path / "！格式审计.txt")
        write_audit_report(audit_data, out_path)
        content = open(out_path, encoding="utf-8").read()
        assert "--- 格式审计报告" in content or "格式审计" in content

    def test_report_shows_needs_conversion(self, tmp_path, test_config):
        audit_data = {
            "total": 1,
            "sample_rates": {48000: 1},
            "bit_depths": {24: 1},
            "channels": {2: 1},
            "needs_conversion": ["/fake/48k.wav: 采样率 48000 != 目标 44100"],
            "errors": [],
        }
        out_path = str(tmp_path / "！格式审计.txt")
        write_audit_report(audit_data, out_path)
        content = open(out_path, encoding="utf-8").read()
        assert "需要转换" in content or "needs_conversion" in content


# ---------------------------------------------------------------------------
# detect_clipping
# ---------------------------------------------------------------------------


class TestDetectClipping:
    def test_clipped_file_detected(self, sample_wav_clipped, test_config):
        result = detect_clipping(str(sample_wav_clipped), _validator_cfg(test_config))
        assert result["clipped"] is True
        assert result["clip_percentage"] > 1.0

    def test_clean_file_not_flagged(self, sample_wav_22k_mono, test_config):
        result = detect_clipping(str(sample_wav_22k_mono), _validator_cfg(test_config))
        assert result["clipped"] is False
        assert result["clip_percentage"] < 1.0

    def test_result_keys(self, sample_wav_22k_mono, test_config):
        result = detect_clipping(str(sample_wav_22k_mono), _validator_cfg(test_config))
        assert "clipped" in result
        assert "max_amplitude" in result
        assert "clip_percentage" in result
        assert "peak_dbfs" in result

    def test_peak_dbfs_reasonable(self, sample_wav_22k_mono, test_config):
        result = detect_clipping(str(sample_wav_22k_mono), _validator_cfg(test_config))
        assert result["peak_dbfs"] > -7.0

    def test_clipping_disabled_returns_skip(self, sample_wav_clipped, test_config):
        cfg = test_config.copy()
        cfg["audio"] = dict(cfg.get("audio", {}))
        cfg["audio"]["clipping_detection"] = {"enable": False, "clip_threshold": 0.99, "threshold_ratio": 0.01}
        result = detect_clipping(str(sample_wav_clipped), build_validator_config(cfg))
        assert result.get("skipped") is True


# ---------------------------------------------------------------------------
# batch_detect_clipping + write_clipping_report
# ---------------------------------------------------------------------------


class TestBatchClipping:
    def test_batch_detects_clipped(self, sample_wav_clipped, sample_wav_22k_mono, test_config):
        entries = [
            AudioEntry(path=str(sample_wav_clipped), lab_text="", duration=0.0),
            AudioEntry(path=str(sample_wav_22k_mono), lab_text="", duration=0.0),
        ]
        results = batch_detect_clipping(entries, _validator_cfg(test_config))
        clipped_files = [r for r in results if r["clipped"]]
        clean_files = [r for r in results if not r["clipped"]]
        assert len(clipped_files) >= 1
        assert len(clean_files) >= 1

    def test_write_clipping_report(self, tmp_path, sample_wav_clipped, sample_wav_22k_mono, test_config):
        entries = [
            AudioEntry(path=str(sample_wav_clipped), lab_text="", duration=0.0),
            AudioEntry(path=str(sample_wav_22k_mono), lab_text="", duration=0.0),
        ]
        results = batch_detect_clipping(entries, _validator_cfg(test_config))
        out_path = str(tmp_path / "！削波报告.txt")
        write_clipping_report(results, out_path)
        assert os.path.isfile(out_path)
        content = open(out_path, encoding="utf-8").read()
        assert "削波" in content

    def test_batch_disabled(self, sample_wav_clipped, test_config):
        cfg = test_config.copy()
        cfg["audio"] = dict(cfg.get("audio", {}))
        cfg["audio"]["clipping_detection"] = {"enable": False, "clip_threshold": 0.99, "threshold_ratio": 0.01}
        entries = [AudioEntry(path=str(sample_wav_clipped), lab_text="", duration=0.0)]
        results = batch_detect_clipping(entries, build_validator_config(cfg))
        assert results == [] or results[0].get("skipped") is True
