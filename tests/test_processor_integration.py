from __future__ import annotations

import hashlib
import os
import sys
import types
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def _install_heavy_import_stubs():
    def _mkstub(modname, **attrs):
        module = types.ModuleType(modname)
        for name, value in attrs.items():
            setattr(module, name, value)
        sys.modules[modname] = module
        return module

    _mkstub("asr", transcribe_audio=lambda *a, **k: [])
    _mkstub("funasr", AutoModel=None)
    _mkstub("audio_splitter", AudioSplitter=lambda *a, **k: type("Stub", (), {"split": lambda self, pairs, outdir: {}})())
    _mkstub("ser", SpeechEmotionRecognizer=lambda *a, **k: None)
    _mkstub(
        "whisperx",
        load_model=lambda *a, **k: None,
        align=lambda *a, **k: None,
        transcribe=lambda *a, **k: {"segments": []},
        DiarizationSegment=lambda *a, **k: None,
    )
    opencc_pkg = types.ModuleType("opencc")

    class _OpenCCStub:
        def __init__(self, *_a, **_k):
            pass

        def convert(self, text, *_a, **_k):
            return text

    opencc_pkg.OpenCC = _OpenCCStub
    sys.modules["opencc"] = opencc_pkg


_install_heavy_import_stubs()

import processor  # noqa: E402
from config_types import (  # noqa: E402
    PipelineConfig, NormalizerConfig, SerConfig, AsrConfig, AsrWhisperxAlignConfig,
    SplitConfig, CheckPipelineConfig, AuditCheckConfig, ClippingCheckConfig,
    SilenceRatioCheckConfig, SnrCheckConfig,
)


EVIDENCE_DIR = Path(REPO_ROOT) / ".sisyphus" / "evidence"


def _write_evidence(name: str, text: str) -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / name).write_text(text, encoding="utf-8")


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _make_audio(path: Path, duration: float = 0.5, sr: int = 44100) -> None:
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    data = (0.25 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    if path.suffix.lower() == ".flac":
        sf.write(str(path), data, sr, format="FLAC", subtype="PCM_16")
    else:
        sf.write(str(path), data, sr, subtype="PCM_16")


def _write_pair(root: Path, stem: str, ext: str = ".wav", sr: int = 44100, text: str = "你好世界") -> Path:
    audio_path = root / f"{stem}{ext}"
    _make_audio(audio_path, sr=sr)
    (root / f"{stem}.lab").write_text(text, encoding="utf-8")
    return audio_path


def _base_config(root: Path, audio: NormalizerConfig | None = None, checks: CheckPipelineConfig | None = None, audio_enabled: bool = True) -> PipelineConfig:
    return PipelineConfig(
        root_dir=str(root),
        recursive=False,
        use_lab_text_as_filename=False,
        short_threshold=0.1,
        long_threshold=99.0,
        output_to_split_folder=False,
        ser=SerConfig(enabled=False),
        text_t2s=True,
        normalizer=audio if audio is not None else NormalizerConfig(),
        normalizer_enabled=audio_enabled,
        checks=checks if checks is not None else CheckPipelineConfig(),
    )


def _audio_config(enabled: bool = True) -> NormalizerConfig:
    return NormalizerConfig(
        target_sample_rate=44100, target_channels=1, target_bit_depth=16,
        input_formats=["wav", "flac"],
        trim_enabled=enabled, trim_top_db=30, trim_min_trim_ms=100, trim_keep_padding_ms=50,
        loudness_enabled=enabled, loudness_target_lufs=-16, loudness_true_peak=-1.5, loudness_lra=7,
        dc_offset_removal_enabled=enabled,
    )


def _checks_config(enabled: bool = True) -> CheckPipelineConfig:
    return CheckPipelineConfig(
        audit=AuditCheckConfig(enable=enabled, run_at=["after_scan"],
                               target_sample_rate=44100, target_channels=1, target_bit_depth=16),
        clipping=ClippingCheckConfig(enable=enabled, run_at=["after_normalize"],
                                      reject_action="warn_only", clip_threshold=0.99,
                                      warn_threshold=0.01, reject_threshold=0.05),
        silence_ratio=SilenceRatioCheckConfig(enable=enabled, run_at=["after_normalize"],
                                                reject_action="isolate", top_db=30,
                                                warn_threshold=0.4, reject_threshold=0.7),
        snr=SnrCheckConfig(enable=enabled, run_at=["after_normalize"],
                            reject_action="isolate", top_db=30, warn_db=20, reject_db=15),
    )


def _output_dir_for(cwd: Path, root: Path) -> Path:
    return cwd / "output" / root.name / root.name


def _patch_fast_enabled_audio(monkeypatch, calls: list[str]) -> None:
    def duration(path):
        data, sr = sf.read(str(path), always_2d=False)
        return len(data) / float(sr)

    def write_target(input_path, output_path, _config):
        data, _sr = sf.read(str(input_path), always_2d=False)
        sf.write(str(output_path), data, 44100, subtype="PCM_16")
        return output_path

    monkeypatch.setattr(processor.norm, "needs_normalization", lambda *_a, **_k: True)
    monkeypatch.setattr(processor.au, "get_audio_duration_seconds", duration)
    monkeypatch.setattr(processor.norm, "normalize_format", lambda i, o, c: calls.append("normalize_format") or write_target(i, o, c))
    monkeypatch.setattr(processor.norm, "trim_silence", lambda i, o, c: calls.append("trim_silence") or write_target(i, o, c))
    monkeypatch.setattr(processor.norm, "normalize_loudness", lambda i, o, c: calls.append("normalize_loudness") or write_target(i, o, c))
    monkeypatch.setattr(processor.val, "audit_audio_metadata", lambda entries, config: calls.append("audit") or {"total": len(entries)})
    monkeypatch.setattr(processor.val, "batch_detect_clipping", lambda entries, config: calls.append("clipping") or [{"wav_path": e.path, "clipped": False} for e in entries])
    monkeypatch.setattr(processor.val, "batch_check_silence_ratio", lambda entries, config: calls.append("silence_ratio") or [{"wav_path": e.path, "silence_status": "ok"} for e in entries])
    monkeypatch.setattr(processor.val, "batch_check_snr", lambda entries, config: calls.append("snr") or [{"wav_path": e.path, "snr_status": "ok"} for e in entries])
    monkeypatch.setattr(processor.val, "write_audit_report", lambda data, path: Path(path).write_text("audit ok", encoding="utf-8"))
    monkeypatch.setattr(processor.val, "write_clipping_report", lambda data, path, **_k: Path(path).write_text("clipping ok", encoding="utf-8"))
    monkeypatch.setattr(processor.val, "write_silence_ratio_report", lambda data, path, **_k: Path(path).write_text("silence_ratio ok", encoding="utf-8"))
    monkeypatch.setattr(processor.val, "write_snr_report", lambda data, path, **_k: Path(path).write_text("snr ok", encoding="utf-8"))


def test_full_pipeline_with_audio_preprocessing_enabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "enabled_input"
    input_dir.mkdir()
    source = _write_pair(input_dir, "sample", sr=22050)
    before = _md5(source)
    calls = []
    _patch_fast_enabled_audio(monkeypatch, calls)

    processor.DatasetProcessor(_base_config(input_dir, _audio_config(enabled=True), _checks_config(enabled=True))).run()

    out_dir = _output_dir_for(tmp_path, input_dir)
    wavs = [p for p in out_dir.glob("*.wav")]
    assert len(wavs) == 1
    with wave.open(str(wavs[0]), "rb") as wf:
        assert wf.getframerate() == 44100
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
    assert (out_dir / "！格式审计.after_scan.txt").exists()
    assert (out_dir / "！削波检测.after_normalize.txt").exists()
    assert (out_dir / "！静默比例.after_normalize.txt").exists()
    assert (tmp_path / "output" / input_dir.name / f"{input_dir.name}.list").exists()
    assert _md5(source) == before
    assert calls == ["audit", "normalize_format", "trim_silence", "normalize_loudness", "clipping", "silence_ratio", "snr"]
    _write_evidence("task-9-full-pipeline.txt", f"PASS full pipeline: {wavs[0]} calls={calls}\n")


def test_pipeline_with_all_audio_features_disabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "disabled_input"
    input_dir.mkdir()
    source = _write_pair(input_dir, "sample", sr=44100)
    before = _md5(source)
    monkeypatch.setattr(processor.norm, "normalize_format", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("normalization disabled")))

    processor.DatasetProcessor(_base_config(input_dir, _audio_config(enabled=False), _checks_config(enabled=False))).run()

    out_dir = _output_dir_for(tmp_path, input_dir)
    output_wav = out_dir / source.name
    assert output_wav.exists()
    assert _md5(source) == before
    assert _md5(output_wav) == before
    assert not (out_dir / "！格式审计.after_scan.txt").exists()
    assert not (out_dir / "！削波检测.after_normalize.txt").exists()
    assert not (out_dir / "！静默比例.after_normalize.txt").exists()
    _write_evidence("task-9-disabled-pipeline.txt", f"PASS disabled pipeline: {output_wav}\n")


def test_no_audio_section_keeps_original_input_unmodified(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "legacy_input"
    input_dir.mkdir()
    source = _write_pair(input_dir, "legacy", sr=44100)
    before = _md5(source)
    monkeypatch.setattr(processor.norm, "normalize_format", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("audio config absent")))

    processor.DatasetProcessor(_base_config(input_dir, audio=None, audio_enabled=False)).run()

    assert _md5(source) == before
    assert (_output_dir_for(tmp_path, input_dir) / source.name).exists()
    _write_evidence("task-9-input-unmodified.txt", f"PASS md5 unchanged: {source} {before}\n")


def test_non_wav_input_converted_and_list_references_wav(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "flac_input"
    input_dir.mkdir()
    _write_pair(input_dir, "sample", ext=".flac", sr=44100)
    calls = []
    _patch_fast_enabled_audio(monkeypatch, calls)

    processor.DatasetProcessor(_base_config(input_dir, _audio_config(enabled=True), _checks_config(enabled=True))).run()

    out_dir = _output_dir_for(tmp_path, input_dir)
    wavs = list(out_dir.glob("*.wav"))
    assert len(wavs) == 1
    assert not list(out_dir.glob("*.flac"))
    list_path = tmp_path / "output" / input_dir.name / f"{input_dir.name}.list"
    list_text = list_path.read_text(encoding="utf-8")
    assert ".wav|" in list_text
    assert ".flac|" not in list_text
    _write_evidence("task-9-flac-pipeline.txt", f"PASS flac converted: {wavs[0]}\n")
