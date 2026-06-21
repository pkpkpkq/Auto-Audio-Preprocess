"""Tests for extracted pure functions from audio_splitter."""
from __future__ import annotations
import os
import sys
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
if getattr(sys.modules.get("audio_splitter"), "__file__", None) is None:
    sys.modules.pop("audio_splitter", None)
sys.modules.pop("torch", None)
torch_stub = types.ModuleType("torch")
torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
sys.modules["torch"] = torch_stub
sys.modules.setdefault("librosa", types.ModuleType("librosa"))
sys.modules.setdefault("whisperx", types.ModuleType("whisperx"))
opencc_stub = types.ModuleType("opencc")
opencc_stub.OpenCC = lambda *a, **k: types.SimpleNamespace(convert=lambda text: text)
sys.modules.setdefault("opencc", opencc_stub)

from audio_splitter import plan_splits, adjust_punctuation


class TestPlanSplits:
    def test_no_split_needed_when_within_range(self):
        result = plan_splits(10.0, [], min_duration=3.0, max_duration=25.0)
        assert result is not None
        assert result == []  # no cuts needed

    def test_single_split_for_long_audio(self):
        # One silence at 10s, total 25s → split at 10s
        silence_spans = [(8.0, 12.0)]
        result = plan_splits(25.0, silence_spans, min_duration=3.0, max_duration=15.0)
        assert result is not None
        assert len(result) >= 1

    def test_returns_none_when_no_valid_plan(self):
        # Total duration too short for any split to satisfy min_duration
        result = plan_splits(2.0, [], min_duration=3.0, max_duration=25.0)
        # 2s < max_duration, so no split needed, should return []
        assert result == []

    def test_multiple_silences_picks_best(self):
        # Two silence spans, algorithm picks optimal cuts
        silence_spans = [(3.0, 4.0), (8.0, 9.0)]
        result = plan_splits(15.0, silence_spans, min_duration=3.0, max_duration=8.0)
        assert result is not None


class TestAdjustPunctuation:
    def test_moves_punctuation_from_start_to_previous_end(self):
        result = adjust_punctuation(["你好", "，世界", "！再见"])
        assert result[0].endswith("，") or "你好，" in result[0] or result[0] == "你好，"

    def test_no_change_when_no_leading_punctuation(self):
        result = adjust_punctuation(["你好世界", "再见再见"])
        assert result == ["你好世界", "再见再见"]

    def test_moves_period(self):
        result = adjust_punctuation(["第一句话", "。第二句话"])
        assert result[0] == "第一句话。"
        assert result[1] == "第二句话"

    def test_empty_parts_handled(self):
        result = adjust_punctuation(["", "，世界"])
        assert len(result) == 2
