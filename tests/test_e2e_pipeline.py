from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
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
from config_types import PipelineConfig, NormalizerConfig, SerConfig, SplitConfig  # noqa: E402


EVIDENCE_DIR = Path(REPO_ROOT) / ".sisyphus" / "evidence"


def _write_evidence(name: str, text: str) -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / name).write_text(text, encoding="utf-8")


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _copy_pair(input_dir: Path, source: Path, stem: str, text: str) -> Path:
    target = input_dir / f"{stem}{source.suffix.lower()}"
    shutil.copy2(source, target)
    (input_dir / f"{stem}.lab").write_text(text, encoding="utf-8")
    return target


def _make_wav(path: Path, duration: float, sr: int = 44100, amplitude: float = 0.35) -> Path:
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    data = (amplitude * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    sf.write(str(path), data, sr, subtype="PCM_16")
    return path


def _make_mp3(source_wav: Path, output_path: Path, sr: int = 22050) -> Path:
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_wav),
        "-ar",
        str(sr),
        "-ac",
        "1",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, timeout=300)
    return output_path


def _base_config(test_config: PipelineConfig, input_dir: Path) -> PipelineConfig:
    return PipelineConfig(
        root_dir=str(input_dir),
        recursive=False,
        use_lab_text_as_filename=False,
        short_threshold=0.25,
        long_threshold=99.0,
        output_to_split_folder=False,
        output_jsonl=True,
        text_t2s=test_config.text_t2s,
        text_replace_map=test_config.text_replace_map,
        ser=SerConfig(enabled=False),
        asr=test_config.asr,
        normalizer=NormalizerConfig(
            target_sample_rate=44100, target_channels=1, target_bit_depth=16,
            input_formats=test_config.normalizer.input_formats,
            trim_enabled=test_config.normalizer.trim_enabled,
            trim_top_db=test_config.normalizer.trim_top_db,
            trim_frame_length=test_config.normalizer.trim_frame_length,
            trim_hop_length=test_config.normalizer.trim_hop_length,
            trim_min_trim_ms=test_config.normalizer.trim_min_trim_ms,
            trim_keep_padding_ms=test_config.normalizer.trim_keep_padding_ms,
            loudness_enabled=False, loudness_target_lufs=-16,
            loudness_true_peak=-1.5, loudness_lra=7, loudness_dual_pass=False,
            dc_offset_removal_enabled=test_config.normalizer.dc_offset_removal_enabled,
            dc_offset_threshold=test_config.normalizer.dc_offset_threshold,
        ),
        normalizer_enabled=True,
        checks=test_config.checks,
        split=test_config.split,
    )


def _output_dir_for(cwd: Path, input_dir: Path) -> Path:
    return cwd / "output" / input_dir.name / input_dir.name


def _list_path_for(cwd: Path, input_dir: Path) -> Path:
    return cwd / "output" / input_dir.name / f"{input_dir.name}.list"


def _assert_target_wav(path: Path) -> None:
    with wave.open(str(path), "rb") as wf:
        assert wf.getframerate() == 44100
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2


def _assert_all_output_wavs_are_target(out_dir: Path) -> list[Path]:
    wavs = sorted(out_dir.rglob("*.wav"))
    assert wavs
    for wav_path in wavs:
        _assert_target_wav(wav_path)
    return wavs


def _assert_list_references_wavs_only(list_path: Path) -> None:
    lines = [line for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines
    for line in lines:
        audio_path = line.split("|", 1)[0]
        assert audio_path.lower().endswith(".wav")
        assert ".flac|" not in line.lower()
        assert ".mp3|" not in line.lower()


def test_mixed_format_dataset_end_to_end(
    tmp_path,
    monkeypatch,
    sample_wav_48k_24bit,
    sample_flac,
    sample_wav_22k_mono,
    test_config,
):
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "mixed_input"
    input_dir.mkdir()
    wav_path = _copy_pair(input_dir, sample_wav_48k_24bit, "wav_48k", "四八千赫兹")
    flac_path = _copy_pair(input_dir, sample_flac, "flac_44k", "无损音频")
    mp3_source = input_dir / "mp3_source.wav"
    shutil.copy2(sample_wav_22k_mono, mp3_source)
    mp3_path = _make_mp3(mp3_source, input_dir / "mp3_22k.mp3")
    mp3_source.unlink()
    (input_dir / "mp3_22k.lab").write_text("压缩音频", encoding="utf-8")
    before = {p: _md5(p) for p in [wav_path, flac_path, mp3_path]}

    processor.DatasetProcessor(_base_config(test_config, input_dir)).run()

    out_dir = _output_dir_for(tmp_path, input_dir)
    wavs = _assert_all_output_wavs_are_target(out_dir)
    assert not list(out_dir.rglob("*.flac"))
    assert not list(out_dir.rglob("*.mp3"))
    list_path = _list_path_for(tmp_path, input_dir)
    _assert_list_references_wavs_only(list_path)
    audit_text = (out_dir / "！格式审计.after_scan.txt").read_text(encoding="utf-8")
    assert "采样率 48000Hz" in audit_text
    assert "采样率 44100Hz" in audit_text
    assert "采样率 22050Hz" in audit_text
    assert {p: _md5(p) for p in before} == before
    _write_evidence(
        "task-10-e2e-mixed-format.txt",
        f"PASS mixed formats: outputs={len(wavs)} list={list_path}\n{audit_text}\n",
    )


def test_edge_case_files_generate_clipping_and_quality_reports(
    tmp_path,
    monkeypatch,
    sample_wav_clipped,
    sample_wav_with_silence,
    test_config,
):
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "edge_input"
    input_dir.mkdir()
    ultra_short = _make_wav(input_dir / "ultra_short.wav", 0.1, sr=44100)
    (input_dir / "ultra_short.lab").write_text("极短音频", encoding="utf-8")
    clipped = _copy_pair(input_dir, sample_wav_clipped, "sample_clipped", "削波音频")
    silence = _copy_pair(input_dir, sample_wav_with_silence, "sample_with_silence", "高静默音频")
    before = {p: _md5(p) for p in [ultra_short, clipped, silence]}
    config = _base_config(test_config, input_dir)
    config = PipelineConfig(**{**config.__dict__, "short_threshold": 0.05})

    processor.DatasetProcessor(config).run()

    out_dir = _output_dir_for(tmp_path, input_dir)
    _assert_all_output_wavs_are_target(out_dir)
    clipping_text = (out_dir / "！削波检测.after_normalize.txt").read_text(encoding="utf-8")
    silence_text = (out_dir / "！静默比例.after_normalize.txt").read_text(encoding="utf-8")
    assert "sample_clipped" in clipping_text
    assert "sample_with_silence" in silence_text
    assert "静默比例" in silence_text
    assert {p: _md5(p) for p in before} == before
    _write_evidence(
        "task-10-e2e-edge-cases.txt",
        f"PASS edge cases: clipping report\n{clipping_text}\nsilence report\n{silence_text}\n",
    )


def test_original_inputs_unchanged_after_dedup_and_normalization(
    tmp_path,
    monkeypatch,
    sample_wav_48k_24bit,
    sample_flac,
    test_config,
):
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "immutability_input"
    input_dir.mkdir()
    wav_path = _copy_pair(input_dir, sample_wav_48k_24bit, "source_wav", "同一句话")
    flac_path = _copy_pair(input_dir, sample_flac, "source_flac", "另一句话")
    duplicate_path = _make_wav(input_dir / "duplicate.wav", 0.5, sr=22050)
    (input_dir / "duplicate.lab").write_text("同一句话", encoding="utf-8")
    before = {p: _md5(p) for p in [wav_path, flac_path, duplicate_path]}

    processor.DatasetProcessor(_base_config(test_config, input_dir)).run()

    out_dir = _output_dir_for(tmp_path, input_dir)
    assert (out_dir / "！去重记录.txt").exists()
    dedup_text = (out_dir / "！去重记录.txt").read_text(encoding="utf-8")
    assert "source_wav.wav" in dedup_text or "duplicate.wav" in dedup_text
    assert {p: _md5(p) for p in before} == before
    _assert_all_output_wavs_are_target(out_dir)


def test_list_file_references_wav_outputs_for_non_wav_inputs(
    tmp_path,
    monkeypatch,
    sample_flac,
    sample_wav_22k_mono,
    test_config,
):
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "non_wav_list_input"
    input_dir.mkdir()
    _copy_pair(input_dir, sample_flac, "only_flac", "列表无损")
    mp3_source = input_dir / "source.wav"
    shutil.copy2(sample_wav_22k_mono, mp3_source)
    _make_mp3(mp3_source, input_dir / "only_mp3.mp3")
    mp3_source.unlink()
    (input_dir / "only_mp3.lab").write_text("列表压缩", encoding="utf-8")

    processor.DatasetProcessor(_base_config(test_config, input_dir)).run()

    _assert_list_references_wavs_only(_list_path_for(tmp_path, input_dir))
    _assert_all_output_wavs_are_target(_output_dir_for(tmp_path, input_dir))


def test_short_merge_and_long_split_finalize_in_one_pipeline(tmp_path, monkeypatch, test_config):
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "split_merge_input"
    input_dir.mkdir()
    for idx in range(2):
        _make_wav(input_dir / f"short_{idx}.wav", 0.2, sr=44100)
        (input_dir / f"short_{idx}.lab").write_text(f"短音频{idx}", encoding="utf-8")
    _make_wav(input_dir / "long.wav", 1.2, sr=44100)
    (input_dir / "long.lab").write_text("长音频", encoding="utf-8")

    class FakeSplitter:
        def __init__(self, _config, logger_arg=None):
            pass

        def split(self, pairs, out_dir):
            results = {}
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            for wav_path, lab_text in pairs:
                split_path = Path(out_dir) / f"{Path(wav_path).stem}_part1.wav"
                shutil.copy2(wav_path, split_path)
                split_path.with_suffix(".lab").write_text(lab_text + "切分", encoding="utf-8")
                results[wav_path] = [str(split_path)]
            return results

    monkeypatch.setattr(processor, "AudioSplitter", FakeSplitter)
    config = _base_config(test_config, input_dir)
    config = PipelineConfig(**{**config.__dict__,
        "short_threshold": 0.3, "long_threshold": 1.0,
        "output_to_split_folder": False,
    })

    processor.DatasetProcessor(config).run()

    out_dir = _output_dir_for(tmp_path, input_dir)
    wavs = _assert_all_output_wavs_are_target(out_dir)
    assert any("merge" in path.name for path in wavs)
    assert any("part1" in path.name for path in wavs)
    assert (out_dir / "！合并记录.txt").exists()
    _assert_list_references_wavs_only(_list_path_for(tmp_path, input_dir))
