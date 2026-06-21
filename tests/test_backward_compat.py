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


EVIDENCE_DIR = Path(REPO_ROOT) / ".sisyphus" / "evidence"


def _write_evidence(name: str, text: str) -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / name).write_text(text, encoding="utf-8")


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _make_wav(path: Path, duration: float = 0.5, sr: int = 44100) -> None:
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    data = (0.25 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    sf.write(str(path), data, sr, subtype="PCM_16")


def _write_pair(root: Path, stem: str, text: str = "你好世界", sr: int = 44100) -> Path:
    wav = root / f"{stem}.wav"
    _make_wav(wav, sr=sr)
    (root / f"{stem}.lab").write_text(text, encoding="utf-8")
    return wav


def _output_dir_for(cwd: Path, root: Path) -> Path:
    return cwd / "output" / root.name / root.name


def _legacy_config(root: Path) -> dict:
    return {
        "root_dir": str(root),
        "recursive": False,
        "use_lab_text_as_filename": False,
        "split_and_merge": {
            "short_threshold": 0.1,
            "long_threshold": 99.0,
            "merge_silence_duration": 0.0,
            "output_to_split_folder": False,
        },
        "split": {},
        "ser": {"enable": False},
        "asr": {},
        "text": {"t2s": True, "replace_map": {}},
        "output": {"jsonl_output": False},
    }


def _all_disabled_audio_config(root: Path) -> dict:
    cfg = _legacy_config(root)
    cfg["audio"] = {
        "target_sample_rate": 44100,
        "target_channels": 1,
        "target_bit_depth": 16,
        "input_formats": ["wav"],
        "trim_silence": {"enable": False, "top_db": 30, "min_trim_ms": 100, "keep_padding_ms": 50},
        "loudness": {"enable": False, "target_lufs": -16, "true_peak": -1.5, "lra": 7},
        "dc_offset_removal": {"enable": False},
        "audit": {"enable": False},
        "clipping_detection": {"enable": False, "clip_threshold": 0.99, "threshold_ratio": 0.01},
        "silence_ratio": {"enable": False, "top_db": 30, "warn_threshold": 0.4, "reject_threshold": 0.7},
        "snr_estimation": {"enable": False, "top_db": 30, "warn_db": 20, "reject_db": 15},
    }
    return cfg


def test_config_without_audio_section_runs_unchanged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "legacy_no_audio"
    input_dir.mkdir()
    source = _write_pair(input_dir, "sample", text="旧配置测试")
    before = _md5(source)

    monkeypatch.setattr(
        processor.norm,
        "normalize_format",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("normalize_format called without audio config")),
    )

    processor.DatasetProcessor(_legacy_config(input_dir)).run()

    out_dir = _output_dir_for(tmp_path, input_dir)
    output_wav = out_dir / source.name
    assert output_wav.exists(), f"Output WAV missing: {output_wav}"
    assert _md5(source) == before, "Source WAV was modified"
    assert _md5(output_wav) == before, f"Output WAV differs from input: output={_md5(output_wav)} input={before}"
    assert not (out_dir / "！格式审计.txt").exists(), "Audit report created without audio config"
    assert not (out_dir / "！削波检测.txt").exists(), "Clipping report created without audio config"
    assert not (out_dir / "！质量报告.txt").exists(), "Quality report created without audio config"
    list_path = tmp_path / "output" / input_dir.name / f"{input_dir.name}.list"
    assert list_path.exists(), ".list file not generated"
    _write_evidence("task-11-backward-compat.txt", f"PASS no-audio-section: src={source} md5={before} out={output_wav}\n")


def test_config_with_audio_all_disabled_runs_unchanged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "all_disabled"
    input_dir.mkdir()
    source = _write_pair(input_dir, "sample", text="全禁用测试")
    before = _md5(source)

    monkeypatch.setattr(
        processor.norm,
        "normalize_format",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("normalize_format called with all disabled")),
    )

    processor.DatasetProcessor(_all_disabled_audio_config(input_dir)).run()

    out_dir = _output_dir_for(tmp_path, input_dir)
    output_wav = out_dir / source.name
    assert output_wav.exists(), f"Output WAV missing: {output_wav}"
    assert _md5(source) == before, "Source WAV was modified"
    assert _md5(output_wav) == before, f"Output WAV differs from input: output={_md5(output_wav)} input={before}"
    assert not (out_dir / "！格式审计.txt").exists(), "Audit report created with all disabled"
    assert not (out_dir / "！削波检测.txt").exists(), "Clipping report created with all disabled"
    assert not (out_dir / "！质量报告.txt").exists(), "Quality report created with all disabled"
    _write_evidence("task-11-all-disabled.txt", f"PASS all-disabled: src={source} md5={before} out={output_wav}\n")


def test_legacy_config_keys_still_work(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    input_dir = tmp_path / "legacy_keys"
    input_dir.mkdir()
    _write_pair(input_dir, "a", text="第一句话")
    _write_pair(input_dir, "b", text="第二句话")
    _write_pair(input_dir, "c", text="{SKIP}")
    _write_pair(input_dir, "d", text="第一句话")

    config = {
        "root_dir": str(input_dir),
        "recursive": False,
        "use_lab_text_as_filename": False,
        "split_and_merge": {
            "short_threshold": 0.1,
            "long_threshold": 99.0,
            "merge_silence_duration": 0.0,
            "output_to_split_folder": False,
        },
        "split": {},
        "ser": {"enable": False},
        "asr": {},
        "text": {"t2s": True, "replace_map": {"话": "語"}},
        "output": {"jsonl_output": True},
    }

    monkeypatch.setattr(
        processor.norm,
        "normalize_format",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("normalize_format called")),
    )

    processor.DatasetProcessor(config).run()

    out_dir = _output_dir_for(tmp_path, input_dir)

    assert out_dir.exists(), "Output directory not created via root_dir"

    wavs = list(out_dir.glob("*.wav"))
    assert len(wavs) >= 1, f"No WAV output files found in {out_dir}"

    list_path = tmp_path / "output" / input_dir.name / f"{input_dir.name}.list"
    assert list_path.exists(), ".list file not generated"
    list_text = list_path.read_text(encoding="utf-8")
    assert "語" in list_text, f"replace_map not applied: {list_text[:200]}"

    lines_with_content = [l for l in list_text.strip().splitlines() if l.strip()]
    assert len(lines_with_content) == 2, f"Expected 2 entries (1 skip + 1 dedup removed), got {len(lines_with_content)}: {lines_with_content}"

    jsonl_path = tmp_path / "output" / input_dir.name / f"{input_dir.name}.jsonl"
    assert jsonl_path.exists(), ".jsonl file not generated with jsonl_output=True"

    jsonl_text = jsonl_path.read_text(encoding="utf-8")
    assert "語" in jsonl_text, f"replace_map not applied in JSONL: {jsonl_text[:200]}"
