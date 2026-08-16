from __future__ import annotations

import os

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
from config_types import AudioEntry
from config_types import AuditCheckConfig, ClippingCheckConfig


def _audit_cfg(test_config) -> AuditCheckConfig:
    return test_config.checks.audit


def _clipping_cfg(test_config) -> ClippingCheckConfig:
    return test_config.checks.clipping


class TestAuditAudioMetadata:
    def test_sample_rate_distribution(self, sample_wav_44k_stereo, sample_wav_48k_24bit, test_config):
        entries = [
            AudioEntry(path=str(sample_wav_44k_stereo), lab_text="", duration=0.0),
            AudioEntry(path=str(sample_wav_48k_24bit), lab_text="", duration=0.0),
        ]
        result = audit_audio_metadata(entries, _audit_cfg(test_config))
        assert result["total"] == 2
        assert result["sample_rates"][44100] == 1
        assert result["sample_rates"][48000] == 1

    def test_needs_conversion_flags_mismatch(self, sample_wav_48k_24bit, test_config):
        entries = [AudioEntry(path=str(sample_wav_48k_24bit), lab_text="", duration=0.0)]
        result = audit_audio_metadata(entries, _audit_cfg(test_config))
        assert len(result["needs_conversion"]) >= 1
        assert any("48000" in item or str(sample_wav_48k_24bit) in item for item in result["needs_conversion"])

    def test_no_conversion_needed(self, sample_wav_44k_stereo, test_config):
        entries = [AudioEntry(path=str(sample_wav_44k_stereo), lab_text="", duration=0.0)]
        result = audit_audio_metadata(entries, _audit_cfg(test_config))
        sr_reasons = [i for i in result["needs_conversion"] if "采样率" in i or "sample_rate" in i]
        assert len(sr_reasons) == 0

    def test_bit_depth_distribution(self, sample_wav_44k_stereo, sample_wav_48k_24bit, test_config):
        entries = [
            AudioEntry(path=str(sample_wav_44k_stereo), lab_text="", duration=0.0),
            AudioEntry(path=str(sample_wav_48k_24bit), lab_text="", duration=0.0),
        ]
        result = audit_audio_metadata(entries, _audit_cfg(test_config))
        assert 16 in result["bit_depths"]
        assert 24 in result["bit_depths"]

    def test_channels_distribution(self, sample_wav_44k_stereo, sample_wav_22k_mono, test_config):
        entries = [
            AudioEntry(path=str(sample_wav_44k_stereo), lab_text="", duration=0.0),
            AudioEntry(path=str(sample_wav_22k_mono), lab_text="", duration=0.0),
        ]
        result = audit_audio_metadata(entries, _audit_cfg(test_config))
        assert result["channels"][1] == 1
        assert result["channels"][2] == 1

    def test_audit_disabled(self, sample_wav_44k_stereo):
        cfg = AuditCheckConfig(enable=False)
        entries = [AudioEntry(path=str(sample_wav_44k_stereo), lab_text="", duration=0.0)]
        result = audit_audio_metadata(entries, cfg)
        assert result["total"] == 0
        assert result.get("skipped") is True

    def test_flac_file_audit(self, sample_flac, test_config):
        entries = [AudioEntry(path=str(sample_flac), lab_text="", duration=0.0)]
        result = audit_audio_metadata(entries, _audit_cfg(test_config))
        assert result["total"] == 1
        assert 44100 in result["sample_rates"]

    def test_missing_file_skipped(self, test_config):
        entries = [AudioEntry(path="/nonexistent/file.wav", lab_text="", duration=0.0)]
        result = audit_audio_metadata(entries, _audit_cfg(test_config))
        assert result["total"] == 0
        assert len(result.get("errors", [])) >= 1


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

    def test_report_content_has_summary(self, tmp_path, test_config):
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
        assert "格式审计" in content
        assert "汇总" in content

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


class TestDetectClipping:
    def test_clipped_file_detected(self, sample_wav_clipped, test_config):
        result = detect_clipping(str(sample_wav_clipped), _clipping_cfg(test_config))
        assert result["clipped"] is True
        assert result["clip_percentage"] > 1.0

    def test_clean_file_not_flagged(self, sample_wav_22k_mono, test_config):
        result = detect_clipping(str(sample_wav_22k_mono), _clipping_cfg(test_config))
        assert result["clipped"] is False
        assert result["clip_percentage"] < 1.0

    def test_result_keys(self, sample_wav_22k_mono, test_config):
        result = detect_clipping(str(sample_wav_22k_mono), _clipping_cfg(test_config))
        assert "clipped" in result
        assert "max_amplitude" in result
        assert "clip_percentage" in result
        assert "peak_dbfs" in result

    def test_peak_dbfs_reasonable(self, sample_wav_22k_mono, test_config):
        result = detect_clipping(str(sample_wav_22k_mono), _clipping_cfg(test_config))
        assert result["peak_dbfs"] > -7.0

    def test_clipping_disabled_returns_skip(self, sample_wav_clipped):
        cfg = ClippingCheckConfig(enable=False)
        result = detect_clipping(str(sample_wav_clipped), cfg)
        assert result.get("skipped") is True


class TestBatchClipping:
    def test_batch_detects_clipped(self, sample_wav_clipped, sample_wav_22k_mono, test_config):
        entries = [
            AudioEntry(path=str(sample_wav_clipped), lab_text="", duration=0.0),
            AudioEntry(path=str(sample_wav_22k_mono), lab_text="", duration=0.0),
        ]
        results = batch_detect_clipping(entries, _clipping_cfg(test_config))
        clipped_files = [r for r in results if r["clipped"]]
        clean_files = [r for r in results if not r["clipped"]]
        assert len(clipped_files) >= 1
        assert len(clean_files) >= 1

    def test_write_clipping_report(self, tmp_path, sample_wav_clipped, sample_wav_22k_mono, test_config):
        entries = [
            AudioEntry(path=str(sample_wav_clipped), lab_text="", duration=0.0),
            AudioEntry(path=str(sample_wav_22k_mono), lab_text="", duration=0.0),
        ]
        results = batch_detect_clipping(entries, _clipping_cfg(test_config))
        out_path = str(tmp_path / "！削波报告.txt")
        write_clipping_report(results, out_path)
        assert os.path.isfile(out_path)
        content = open(out_path, encoding="utf-8").read()
        assert "削波" in content
        assert "汇总" in content

    def test_write_clipping_report_sorting_and_format(self, tmp_path):
        data = [
            {"wav_path": "/path/to/ok_b.wav", "clip_percentage": 0.0, "peak_dbfs": -3.0, "clipping_status": "ok"},
            {"wav_path": "/path/to/reject_b.wav", "clip_percentage": 10.0, "peak_dbfs": 0.0, "clipping_status": "reject"},
            {"wav_path": "/path/to/warn_b.wav", "clip_percentage": 2.0, "peak_dbfs": -0.1, "clipping_status": "warn"},
            {"wav_path": "/path/to/ok_a.wav", "clip_percentage": 0.0, "peak_dbfs": -6.0, "clipping_status": "ok"},
            {"wav_path": "/path/to/reject_a.wav", "clip_percentage": 8.0, "peak_dbfs": 0.0, "clipping_status": "reject"},
            {"wav_path": "/path/to/warn_a.wav", "clip_percentage": 1.5, "peak_dbfs": -0.5, "clipping_status": "warn"},
        ]
        out_path = str(tmp_path / "！削波报告.txt")
        write_clipping_report(data, out_path)
        content = open(out_path, encoding="utf-8").read()
        lines = [line for line in content.splitlines() if line]
        detail_idx = lines.index("--- 明细 ---")
        detail_lines = lines[detail_idx + 1:]

        expected = [
            "reject_a.wav  削波比例=8.0%  峰值=0.0dBFS (拒绝)",
            "reject_b.wav  削波比例=10.0%  峰值=0.0dBFS (拒绝)",
            "warn_a.wav  削波比例=1.5%  峰值=-0.5dBFS (警告)",
            "warn_b.wav  削波比例=2.0%  峰值=-0.1dBFS (警告)",
            "ok_a.wav  削波比例=0.0%  峰值=-6.0dBFS (正常)",
            "ok_b.wav  削波比例=0.0%  峰值=-3.0dBFS (正常)",
        ]
        assert detail_lines == expected

    def test_batch_disabled(self, sample_wav_clipped):
        cfg = ClippingCheckConfig(enable=False)
        entries = [AudioEntry(path=str(sample_wav_clipped), lab_text="", duration=0.0)]
        results = batch_detect_clipping(entries, cfg)
        assert results == [] or results[0].get("skipped") is True
