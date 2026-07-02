"""Tests for multi-format audio scanning in processor._find_audio_entries() (Task #3, TDD-RED)."""
import logging
import os
import sys
import types
import wave

import numpy as np
import soundfile as sf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def _install_heavy_import_stubs():
    """Stub heavy ML deps so processor import doesn't pull torch/funasr/transformers.

    These tests exercise pure-Python scan logic — we don't need real ASR/SER,
    we only need `processor.DatasetProcessor` to be importable as a class.
    Each real module is replaced with a tiny stub exposing the symbol processor.py
    imports at module load time.
    """
    def _mkstub(modname, **attrs):
        m = types.ModuleType(modname)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[modname] = m
        return m

    _mkstub("asr", transcribe_audio=lambda *a, **k: [])
    _mkstub("funasr", AutoModel=None)
    _mkstub("audio_splitter", AudioSplitter=lambda *a, **k: type("Stub", (), {"split": lambda self, pairs, outdir: {}})())
    _mkstub("ser", SpeechEmotionRecognizer=lambda *a, **k: None)
    _mkstub("whisperx", load_model=lambda *a, **k: None,
                          align=lambda *a, **k: None,
                          transcribe=lambda *a, **k: {"segments": []},
                          DiarizationSegment=lambda *a, **k: None)
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


def _make_wav(path, duration=0.5, freq=440, sr=16000):
    """Write a minimal mono 16-bit PCM WAV file via the stdlib wave module."""
    nframes = int(sr * duration)
    t = np.linspace(0, duration, nframes, False)
    data = (0.3 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())


def _make_flac(path, duration=0.5, freq=440, sr=16000):
    """Write a minimal FLAC file via soundfile (used by librosa fallback)."""
    nframes = int(sr * duration)
    t = np.linspace(0, duration, nframes, False)
    data = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(path, data, sr, format="FLAC")


def _make_lab(path, text="你好世界"):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


class _StubProcessor(processor.DatasetProcessor):
    """Bypass __init__ — assign only the attributes _find_audio_entries() touches."""

    def __init__(self, tmpdir, recursive=False, input_formats=None):
        self.root_dir = os.path.abspath(str(tmpdir))
        self.recursive = recursive
        self.folder_name = os.path.basename(self.root_dir.rstrip(os.sep))
        from config_types import PipelineConfig, NormalizerConfig
        self._cfg = PipelineConfig(
            normalizer=NormalizerConfig(
                input_formats=input_formats if input_formats is not None else ["wav", "flac", "mp3", "ogg", "aac", "m4a"]
            ),
        )
        self.ser_enabled = False
        self.ser_recognizer = None
        self.output_to_split_folder = True
        self.replace_map = {}
        self.t2s_enabled = True
        self.logger = logging.getLogger("tts_preprocess_test")
        self.logger.addHandler(logging.NullHandler())

    def _process_lab_with_asr(self, lab_path, wav_path):
        # Stub ASR-backed lab processing — tests focus on scan + duration
        with open(lab_path, "r", encoding="utf-8") as f:
            return f.read()


class TestPickPrimaryAudio:

    def test_wav_wins_over_flac(self, tmp_path):
        d = tmp_path / "stem"
        d.mkdir()
        _make_wav(str(d / "test.wav"))
        _make_flac(str(d / "test.flac"))

        proc = _StubProcessor(tmp_path)
        extmap = {
            ".wav": "test.wav",
            ".flac": "test.flac",
            ".lab": "test.lab",
            "dirpath": str(d),
        }
        chosen = processor.DatasetProcessor._pick_primary_audio(
            "test", str(d), extmap, proc._cfg.normalizer.input_formats
        )
        assert chosen is not None and chosen.endswith(".wav"), chosen

    def test_flac_when_no_wav(self, tmp_path):
        d = tmp_path / "stem"
        d.mkdir()
        _make_flac(str(d / "test.flac"))

        proc = _StubProcessor(tmp_path)
        extmap = {
            ".flac": "test.flac",
            ".lab": "test.lab",
            "dirpath": str(d),
        }
        chosen = processor.DatasetProcessor._pick_primary_audio(
            "test", str(d), extmap, proc._cfg.normalizer.input_formats
        )
        assert chosen is not None and chosen.endswith(".flac"), chosen

    def test_returns_none_when_no_audio(self, tmp_path):
        d = tmp_path / "stem"
        d.mkdir()
        proc = _StubProcessor(tmp_path)
        extmap = {".lab": "test.lab", "dirpath": str(d)}
        chosen = processor.DatasetProcessor._pick_primary_audio(
            "test", str(d), extmap, proc._cfg.normalizer.input_formats
        )
        assert chosen is None


class TestFindAudioEntriesMultiFormat:

    def test_flac_plus_lab_pair_discovered(self, tmp_path):
        """QA#1 — FLAC + LAB pair is discovered as a single entry."""
        _make_flac(str(tmp_path / "test.flac"))
        _make_lab(str(tmp_path / "test.lab"), "flac test entry")
        proc = _StubProcessor(tmp_path)
        entries = proc._find_audio_entries()
        assert len(entries) == 1, f"Expected 1 entry, got {len(entries)}"
        e = entries[0]
        assert e.path.endswith(".flac"), e.path
        assert e.lab_path.endswith(".lab")
        assert e.lab_text == "flac test entry"
        assert e.duration > 0, f"Duration should be > 0, got {e.duration}"

    def test_wav_priority_over_flac_same_stem(self, tmp_path):
        """QA#2 — When stem has BOTH .wav and .flac, .wav wins."""
        _make_wav(str(tmp_path / "test.wav"))
        _make_flac(str(tmp_path / "test.flac"))
        _make_lab(str(tmp_path / "test.lab"), "wav priority test")
        proc = _StubProcessor(tmp_path)
        entries = proc._find_audio_entries()
        assert len(entries) == 1, f"Expected 1 entry (no dup), got {len(entries)}"
        assert entries[0].path.endswith(".wav"), entries[0].path

    def test_unsupported_format_skipped_silently(self, tmp_path):
        """QA#3 — .xyz not in input_formats is skipped without raising."""
        (tmp_path / "test.xyz").write_bytes(b"bogus bytes for fake format")
        _make_lab(str(tmp_path / "test.lab"), "unsupported format test")
        proc = _StubProcessor(tmp_path)
        entries = proc._find_audio_entries()
        assert entries == [], f"Expected no entries, got {entries}"

    def test_wav_only_input_formats_ignores_flac(self, tmp_path):
        """Backward-compat: legacy config with input_formats=['wav'] skips flac."""
        _make_wav(str(tmp_path / "legacy.wav"))
        _make_flac(str(tmp_path / "modern.flac"))
        _make_lab(str(tmp_path / "legacy.lab"), "legacy wav only")
        _make_lab(str(tmp_path / "modern.lab"), "modern flac should be ignored")
        proc = _StubProcessor(tmp_path, input_formats=["wav"])
        entries = proc._find_audio_entries()
        assert len(entries) == 1, f"Expected 1 entry, got {len(entries)}"
        assert entries[0].path.endswith("legacy.wav"), entries[0].path

    def test_duration_librosa_for_flac(self, tmp_path):
        """QA#4 — non-WAV duration comes from librosa.get_duration (not wave module)."""
        _make_flac(str(tmp_path / "flacdur.flac"), duration=0.7)
        _make_lab(str(tmp_path / "flacdur.lab"), "flac duration sample")
        proc = _StubProcessor(tmp_path)
        entries = proc._find_audio_entries()
        assert len(entries) == 1
        d = entries[0].duration
        assert 0.65 < d < 0.75, f"Expected duration ~0.7s, got {d}"
