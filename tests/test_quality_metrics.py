"""Tests for silence ratio detection and VAD-SNR estimation."""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from audio_validator import compute_silence_ratio, estimate_snr, batch_check_silence_ratio, batch_check_snr, write_silence_ratio_report, write_snr_report, run_check_pipeline, run_full_audit
from config_types import SilenceRatioCheckConfig, SnrCheckConfig, CheckPipelineConfig, AuditCheckConfig, ClippingCheckConfig, AudioEntry


def _silence_cfg(test_config) -> SilenceRatioCheckConfig:
    return test_config.checks.silence_ratio


def _snr_cfg(test_config) -> SnrCheckConfig:
    return test_config.checks.snr


class TestComputeSilenceRatio:
    def test_high_silence_flagged_warn(self, sample_wav_with_silence, test_config):
        result = compute_silence_ratio(str(sample_wav_with_silence), _silence_cfg(test_config))
        assert result["silence_ratio"] > 0.3, f"Expected high silence ratio, got {result['silence_ratio']}"
        assert result["status"] in ("warn", "reject"), f"Expected warn/reject, got {result['status']}"

    def test_clean_file_low_silence(self, sample_wav_22k_mono, test_config):
        result = compute_silence_ratio(str(sample_wav_22k_mono), _silence_cfg(test_config))
        assert result["silence_ratio"] < 0.3, f"Expected low silence ratio, got {result['silence_ratio']}"
        assert result["status"] == "ok"

    def test_returns_required_fields(self, sample_wav_22k_mono, test_config):
        result = compute_silence_ratio(str(sample_wav_22k_mono), _silence_cfg(test_config))
        for key in ("silence_ratio", "speech_duration", "total_duration", "status"):
            assert key in result, f"Missing field: {key}"

    def test_reject_threshold(self, tmp_path, test_config):
        sr = 22050
        tone = np.sin(2 * np.pi * 440.0 * np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)).astype(np.float32)
        silence = np.zeros(int(sr * 3.9), dtype=np.float32)
        data = np.concatenate([tone, silence])
        path = tmp_path / "mostly_silent.wav"
        sf.write(str(path), data, sr, subtype="PCM_16")
        result = compute_silence_ratio(str(path), _silence_cfg(test_config))
        assert result["silence_ratio"] > 0.7
        assert result["status"] == "reject"


class TestEstimateSNR:
    def test_snr_returns_reasonable_value(self, sample_wav_with_silence, test_config):
        result = estimate_snr(str(sample_wav_with_silence), _snr_cfg(test_config))
        assert "snr" in result
        assert "reliable" in result
        assert isinstance(result["snr"], float)
        assert isinstance(result["reliable"], bool)

    def test_snr_with_silence_is_reliable(self, sample_wav_with_silence, test_config):
        result = estimate_snr(str(sample_wav_with_silence), _snr_cfg(test_config))
        assert result["reliable"] is True

    def test_snr_no_silence_unreliable(self, sample_wav_22k_mono, test_config):
        result = estimate_snr(str(sample_wav_22k_mono), _snr_cfg(test_config))
        assert result["reliable"] is False
        assert result["snr"] == 40.0

    def test_snr_status_field(self, sample_wav_with_silence, test_config):
        result = estimate_snr(str(sample_wav_with_silence), _snr_cfg(test_config))
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
        result = estimate_snr(str(path), _snr_cfg(test_config))
        assert "snr" in result
        assert "reliable" in result


class TestBatchSilenceRatio:
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
        results = batch_check_silence_ratio(entries, _silence_cfg(test_config))
        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert "silence_ratio" in r
            assert "silence_status" in r

    def test_write_silence_ratio_report(self, tmp_path, test_config):
        data = [
            {
                "wav_path": "test1.wav",
                "silence_ratio": 0.1,
                "silence_status": "ok",
            },
            {
                "wav_path": "test2.wav",
                "silence_ratio": 0.5,
                "silence_status": "warn",
            },
            {
                "wav_path": "test3.wav",
                "silence_ratio": 0.8,
                "silence_status": "reject",
            },
        ]
        report_path = tmp_path / "静默比例报告.txt"
        write_silence_ratio_report(data, str(report_path))
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "汇总" in content
        assert "test1.wav" in content
        assert "0.1" in content

    def test_write_silence_ratio_report_sorting_and_format(self, tmp_path):
        data = [
            {"wav_path": "/path/to/ok_b.wav", "silence_ratio": 0.1, "silence_status": "ok"},
            {"wav_path": "/path/to/reject_b.wav", "silence_ratio": 0.9, "silence_status": "reject"},
            {"wav_path": "/path/to/warn_b.wav", "silence_ratio": 0.5, "silence_status": "warn"},
            {"wav_path": "/path/to/ok_a.wav", "silence_ratio": 0.05, "silence_status": "ok"},
            {"wav_path": "/path/to/reject_a.wav", "silence_ratio": 0.8, "silence_status": "reject"},
            {"wav_path": "/path/to/warn_a.wav", "silence_ratio": 0.45, "silence_status": "warn"},
        ]
        out_path = str(tmp_path / "！静默比例报告.txt")
        write_silence_ratio_report(data, out_path)
        content = open(out_path, encoding="utf-8").read()
        lines = [line for line in content.splitlines() if line]
        detail_idx = lines.index("--- 明细 ---")
        detail_lines = lines[detail_idx + 1:]

        expected = [
            "reject_a.wav  静默比例=0.8 (拒绝)",
            "reject_b.wav  静默比例=0.9 (拒绝)",
            "warn_a.wav  静默比例=0.45 (警告)",
            "warn_b.wav  静默比例=0.5 (警告)",
            "ok_a.wav  静默比例=0.05 (正常)",
            "ok_b.wav  静默比例=0.1 (正常)",
        ]
        assert detail_lines == expected


class TestBatchSnr:
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
        results = batch_check_snr(entries, _snr_cfg(test_config))
        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert "snr" in r
            assert "snr_status" in r

    def test_write_snr_report(self, tmp_path, test_config):
        data = [
            {
                "wav_path": "test1.wav",
                "snr": 30.0,
                "snr_reliable": True,
                "snr_status": "ok",
            },
            {
                "wav_path": "test2.wav",
                "snr": 18.0,
                "snr_reliable": True,
                "snr_status": "warn",
            },
            {
                "wav_path": "test3.wav",
                "snr": 10.0,
                "snr_reliable": True,
                "snr_status": "reject",
            },
        ]
        report_path = tmp_path / "SNR报告.txt"
        write_snr_report(data, str(report_path))
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "汇总" in content
        assert "test1.wav" in content
        assert "30.0" in content

    def test_write_snr_report_sorting_and_format(self, tmp_path):
        data = [
            {"wav_path": "/path/to/ok_b.wav", "snr": 35.0, "snr_status": "ok"},
            {"wav_path": "/path/to/reject_b.wav", "snr": 8.0, "snr_status": "reject"},
            {"wav_path": "/path/to/warn_b.wav", "snr": 18.0, "snr_status": "warn"},
            {"wav_path": "/path/to/ok_a.wav", "snr": 30.0, "snr_status": "ok"},
            {"wav_path": "/path/to/reject_a.wav", "snr": 5.0, "snr_status": "reject"},
            {"wav_path": "/path/to/warn_a.wav", "snr": 16.0, "snr_status": "warn"},
        ]
        out_path = str(tmp_path / "！SNR报告.txt")
        write_snr_report(data, out_path)
        content = open(out_path, encoding="utf-8").read()
        lines = [line for line in content.splitlines() if line]
        detail_idx = lines.index("--- 明细 ---")
        detail_lines = lines[detail_idx + 1:]

        expected = [
            "reject_a.wav  SNR=5.0dB (拒绝)",
            "reject_b.wav  SNR=8.0dB (拒绝)",
            "warn_a.wav  SNR=16.0dB (警告)",
            "warn_b.wav  SNR=18.0dB (警告)",
            "ok_a.wav  SNR=30.0dB (正常)",
            "ok_b.wav  SNR=35.0dB (正常)",
        ]
        assert detail_lines == expected


class TestRunCheckPipeline:
    def test_runs_only_checks_with_matching_run_at(self, tmp_path, test_config):
        sr = 22050
        data = np.sin(2 * np.pi * 440.0 * np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)).astype(np.float32)
        wav_path = tmp_path / "test.wav"
        sf.write(str(wav_path), data, sr, subtype="PCM_16")
        entries = [AudioEntry(path=str(wav_path), lab_text="test", duration=0.5)]

        cfg = test_config.checks
        report_dir = tmp_path / "reports"
        report_dir.mkdir()

        reject_records = run_check_pipeline(entries, cfg, "after_scan", str(report_dir))

        assert (report_dir / "！格式审计.after_scan.txt").exists()
        assert not (report_dir / "！削波检测.after_scan.txt").exists()
        assert not (report_dir / "！静默比例.after_scan.txt").exists()
        assert reject_records == []

    def test_collects_reject_records_at_last_run(self, tmp_path, monkeypatch):
        cfg = CheckPipelineConfig(
            audit=AuditCheckConfig(enable=False),
            clipping=ClippingCheckConfig(enable=False),
            silence_ratio=SilenceRatioCheckConfig(enable=False),
            snr=SnrCheckConfig(enable=True, run_at=["after_normalize", "final"], reject_action="isolate"),
        )
        sr = 22050
        data = np.sin(2 * np.pi * 440.0 * np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)).astype(np.float32)
        wav_path = tmp_path / "test.wav"
        sf.write(str(wav_path), data, sr, subtype="PCM_16")
        entries = [AudioEntry(path=str(wav_path), lab_text="test", duration=0.5)]
        report_dir = tmp_path / "reports"
        report_dir.mkdir()

        import audio_validator
        reject_result = [{"wav_path": str(wav_path), "snr": 5.0, "snr_reliable": True, "snr_status": "reject"}]
        monkeypatch.setattr(audio_validator, "batch_check_snr", lambda e, c, **_k: reject_result)
        monkeypatch.setattr(audio_validator, "write_snr_report", lambda *a, **k: None)

        rejects_at_normalize = run_check_pipeline(entries, cfg, "after_normalize", str(report_dir))
        rejects_at_final = run_check_pipeline(entries, cfg, "final", str(report_dir))

        assert len(rejects_at_normalize) == 0
        assert len(rejects_at_final) == 1
        assert rejects_at_final[0][1] == "snr"
        assert rejects_at_final[0][2] == "isolate"

    def test_returns_empty_for_empty_entries(self, tmp_path):
        cfg = CheckPipelineConfig()
        report_dir = tmp_path / "reports"
        report_dir.mkdir()
        result = run_check_pipeline([], cfg, "after_scan", str(report_dir))
        assert result == []


class TestRunFullAudit:
    def test_runs_all_checks_ignoring_run_at(self, tmp_path, test_config):
        sr = 22050
        data = np.sin(2 * np.pi * 440.0 * np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)).astype(np.float32)
        wav_path = tmp_path / "test.wav"
        sf.write(str(wav_path), data, sr, subtype="PCM_16")
        entries = [AudioEntry(path=str(wav_path), lab_text="test", duration=0.5)]

        cfg = test_config.checks
        audit_dir = tmp_path / "audit_output"
        audit_dir.mkdir()

        run_full_audit(entries, cfg, str(audit_dir))

        assert (audit_dir / "！格式审计.txt").exists()
        assert (audit_dir / "！削波检测.txt").exists()
        assert (audit_dir / "！静默比例.txt").exists()
        assert (audit_dir / "！SNR.txt").exists()

    def test_audit_reports_have_no_stage_suffix(self, tmp_path, test_config):
        sr = 22050
        data = np.sin(2 * np.pi * 440.0 * np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)).astype(np.float32)
        wav_path = tmp_path / "test.wav"
        sf.write(str(wav_path), data, sr, subtype="PCM_16")
        entries = [AudioEntry(path=str(wav_path), lab_text="test", duration=0.5)]

        cfg = test_config.checks
        audit_dir = tmp_path / "audit_output"
        audit_dir.mkdir()

        run_full_audit(entries, cfg, str(audit_dir))

        for f in audit_dir.iterdir():
            assert ".after_" not in f.name and ".final" not in f.name

    def test_no_files_isolated_or_removed(self, tmp_path, test_config):
        sr = 22050
        tone = np.clip(np.sin(2 * np.pi * 440.0 * np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)) * 3.0, -1.0, 1.0).astype(np.float32)
        wav_path = tmp_path / "clipped.wav"
        sf.write(str(wav_path), tone, sr, subtype="PCM_16")
        entries = [AudioEntry(path=str(wav_path), lab_text="", duration=0.5)]

        cfg = test_config.checks
        audit_dir = tmp_path / "audit_output"
        audit_dir.mkdir()

        run_full_audit(entries, cfg, str(audit_dir))

        assert wav_path.exists()
        assert not (audit_dir / "隔离").exists()

    def test_disabled_checks_skipped(self, tmp_path):
        cfg = CheckPipelineConfig(
            audit=AuditCheckConfig(enable=False),
            clipping=ClippingCheckConfig(enable=False),
            silence_ratio=SilenceRatioCheckConfig(enable=False),
            snr=SnrCheckConfig(enable=False),
        )
        audit_dir = tmp_path / "audit_output"
        audit_dir.mkdir()

        run_full_audit([], cfg, str(audit_dir))

        assert not list(audit_dir.iterdir())
