from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field, fields as _dc_fields
from typing import Optional


PIPELINE_STAGES = ["after_scan", "after_normalize", "after_classify", "after_merge", "final"]


# ── Generic helpers ────────────────────────────────────────────────

_TYPE_COERCE = {"int": int, "float": float}


def _from_dict(cls, data: dict, /, **overrides):
    """Construct a dataclass from a dict, falling back to field defaults.

    Only keys present in data are used; missing keys use the dataclass default.
    YAML key must match dataclass field name — no implicit key remapping.
    overrides: explicit values that take priority over data.
    """
    kwargs = {}
    for f in _dc_fields(cls):
        if f.name in overrides:
            continue
        if f.name in data:
            val = data[f.name]
            coerce = _TYPE_COERCE.get(f.type)
            if coerce and not isinstance(val, bool):
                val = coerce(val)
            kwargs[f.name] = val
    kwargs.update(overrides)
    return cls(**kwargs)


def _prefix_dict(d: dict, prefix: str) -> dict:
    """Flatten a sub-section dict by prefixing keys.

    Example: _prefix_dict({"enabled": True, "top_db": 30}, "trim_")
    → {"trim_enabled": True, "trim_top_db": 30}
    """
    return {f"{prefix}{k}": v for k, v in d.items()}


@dataclass
class NormalizerConfig:
    """Config for audio_normalizer module."""
    target_sample_rate: int = 44100
    target_channels: int = 1
    target_bit_depth: int = 16
    input_formats: list[str] = field(default_factory=lambda: ["wav", "flac", "mp3", "ogg", "aac", "m4a"])

    # Trim silence sub-section
    trim_enabled: bool = True
    trim_top_db: float = 30.0
    trim_frame_length: int = 2048
    trim_hop_length: int = 512
    trim_min_trim_ms: float = 100.0
    trim_keep_padding_ms: float = 50.0

    # Loudness sub-section
    loudness_enabled: bool = True
    loudness_target_lufs: float = -16.0
    loudness_true_peak: float = -1.5
    loudness_lra: float = 7.0
    loudness_dual_pass: bool = True

    # DC offset removal sub-section
    dc_offset_removal_enabled: bool = True
    dc_offset_threshold: float = 0.001


@dataclass
class AuditCheckConfig:
    enable: bool = True
    run_at: list[str] = field(default_factory=lambda: ["after_scan", "final"])
    target_sample_rate: int = 44100
    target_channels: int = 1
    target_bit_depth: int = 16


@dataclass
class ClippingCheckConfig:
    enable: bool = True
    run_at: list[str] = field(default_factory=lambda: ["after_normalize", "final"])
    reject_action: str = "isolate"
    clip_threshold: float = 0.99
    warn_threshold: float = 0.01
    reject_threshold: float = 0.05


@dataclass
class SilenceRatioCheckConfig:
    enable: bool = True
    run_at: list[str] = field(default_factory=lambda: ["after_normalize", "final"])
    reject_action: str = "isolate"
    top_db: float = 30.0
    warn_threshold: float = 0.4
    reject_threshold: float = 0.7


@dataclass
class SnrCheckConfig:
    enable: bool = True
    run_at: list[str] = field(default_factory=lambda: ["after_normalize", "final"])
    reject_action: str = "isolate"
    top_db: float = 30.0
    warn_db: float = 20.0
    reject_db: float = 15.0


@dataclass
class CheckPipelineConfig:
    audit: AuditCheckConfig = field(default_factory=AuditCheckConfig)
    clipping: ClippingCheckConfig = field(default_factory=ClippingCheckConfig)
    silence_ratio: SilenceRatioCheckConfig = field(default_factory=SilenceRatioCheckConfig)
    snr: SnrCheckConfig = field(default_factory=SnrCheckConfig)


@dataclass
class SerConfig:
    """Config for SER (Speech Emotion Recognition) module."""
    enabled: bool = True
    backend: str = "hubert"  # "hubert" | "emotion2vec" | "sensevoice"
    device: str = "auto"
    on_failure: str = "default"  # "default" | "truncate_and_retry" | "isolate"

    hubert_model_id: str = "xmj2002/hubert-base-ch-speech-emotion-recognition"

    emotion2vec_model_id: str = "emotion2vec/emotion2vec_plus_large"
    emotion2vec_granularity: str = "utterance"  # "utterance" | "frame"
    emotion2vec_safe_max_duration: float = 30.0  # seconds

    sensevoice_model_id: str = "FunAudioLLM/SenseVoiceSmall"
    sensevoice_language: str = "auto"  # "zh"|"en"|"yue"|"ja"|"ko"|"auto"
    sensevoice_use_itn: bool = True
    sensevoice_ban_emo_unk: bool = False
    sensevoice_vad_max_single_segment_time: int = 30000  # ms


@dataclass
class AsrWhisperxAlignConfig:
    """Config for ASR-level WhisperX forced alignment (separate from splitter's SplitConfig)."""
    device: str = "auto"  # "cuda" | "cpu" | "auto"
    language: str = "zh"  # language code for whisperx.load_align_model


@dataclass
class AsrConfig:
    backend: str = "funasr"  # "funasr" | "whisperx"
    language: str = "zh"
    force_align: bool = False  # true: return char-level timestamped segments; false: return plain text
    funasr_model: str = "paraformer-zh"
    funasr_use_vad: bool = False
    funasr_vad_max_single_segment_time: int = 30000  # ms, passed as vad_kwargs
    funasr_itn: bool = True  # SenseVoiceSmall: use_itn; Fun-ASR-Nano: itn; paraformer-zh: 忽略
    whisperx_model: str = "large-v2"
    whisperx_batch_size: int = 16
    whisperx_align: AsrWhisperxAlignConfig = field(default_factory=AsrWhisperxAlignConfig)


@dataclass
class SplitConfig:
    """Config for AudioSplitter."""
    device: str = "auto"
    language: str = "zh"
    min_duration: float = 3.0
    max_duration: float = 25.0
    max_segments: int = 5
    split_min_silence_len_ms: int = 500
    split_offset_from_silence_end_ms: int = 0
    split_silence_top_db: int = 40
    merge_silence_duration: float = 0.3
    on_split_failure: str = "ignore"  # "isolate" | "ignore"


def build_normalizer_config(raw_config: dict) -> NormalizerConfig:
    audio = raw_config.get("audio", {})
    if not audio:
        return NormalizerConfig(trim_enabled=False, loudness_enabled=False, dc_offset_removal_enabled=False)
    flat = {
        **audio,
        **_prefix_dict(audio.get("trim_silence", {}), "trim_"),
        **_prefix_dict(audio.get("loudness", {}), "loudness_"),
        **_prefix_dict(audio.get("dc_offset_removal", {}), "dc_offset_removal_"),
    }
    return _from_dict(NormalizerConfig, flat)


def build_asr_config(raw_config: dict) -> AsrConfig:
    asr = raw_config.get("asr", {})
    funasr = asr.get("funasr", {})
    wx = asr.get("whisperx", {})
    wxa = asr.get("whisperx_align", {})
    lang = asr.get("language", "zh")
    flat = {
        **asr,
        **_prefix_dict(funasr, "funasr_"),
        **_prefix_dict(wx, "whisperx_"),
    }
    return _from_dict(AsrConfig, flat,
        whisperx_align=AsrWhisperxAlignConfig(
            device=wxa.get("device", "auto"),
            language=wxa.get("language", lang),
        ),
    )


def _validate_run_at(value, default: list[str]) -> list[str]:
    """Filter run_at list to only valid pipeline stages."""
    stages = value if isinstance(value, list) else default
    return [stage for stage in stages if stage in PIPELINE_STAGES]


def build_check_pipeline_config(raw_config: dict) -> CheckPipelineConfig:
    checks = raw_config.get("checks", {})
    if not checks:
        return CheckPipelineConfig(
            audit=AuditCheckConfig(enable=False),
            clipping=ClippingCheckConfig(enable=False),
            silence_ratio=SilenceRatioCheckConfig(enable=False),
            snr=SnrCheckConfig(enable=False),
        )
    audit = checks.get("audit", {})
    clip = checks.get("clipping", {})
    sr = checks.get("silence_ratio", {})
    snr = checks.get("snr", {})
    return CheckPipelineConfig(
        audit=_from_dict(AuditCheckConfig, audit,
            run_at=_validate_run_at(audit.get("run_at"), ["after_scan", "final"])),
        clipping=_from_dict(ClippingCheckConfig, clip,
            run_at=_validate_run_at(clip.get("run_at"), ["after_normalize", "final"])),
        silence_ratio=_from_dict(SilenceRatioCheckConfig, sr,
            run_at=_validate_run_at(sr.get("run_at"), ["after_normalize", "final"])),
        snr=_from_dict(SnrCheckConfig, snr,
            run_at=_validate_run_at(snr.get("run_at"), ["after_normalize", "final"])),
    )


def build_split_config(raw_config: dict) -> SplitConfig:
    sam = raw_config.get("split_and_merge", {})
    split = sam.get("split", raw_config.get("split", {}))
    overrides = {k: sam[k] for k in ("merge_silence_duration", "on_split_failure") if k in sam}
    return _from_dict(SplitConfig, split, **overrides)


def build_ser_config(raw_config: dict) -> SerConfig:
    ser = raw_config.get("ser", {})
    if not ser:
        return SerConfig()
    flat = {
        **ser,
        **_prefix_dict(ser.get("hubert", {}), "hubert_"),
        **_prefix_dict(ser.get("emotion2vec", {}), "emotion2vec_"),
        **_prefix_dict(ser.get("sensevoice", {}), "sensevoice_"),
    }
    return _from_dict(SerConfig, flat)


@dataclass
class PipelineConfig:
    """Top-level config — the only type passed to DatasetProcessor."""
    root_dir: str = ""
    recursive: bool = False
    use_lab_text_as_filename: bool = False

    text_t2s: bool = True
    text_replace_map: dict = field(default_factory=dict)

    output_jsonl: bool = False
    output_to_split_folder: bool = False
    short_threshold: float = 2.0
    long_threshold: float = 15.0

    asr: AsrConfig = field(default_factory=AsrConfig)
    ser: SerConfig = field(default_factory=SerConfig)
    normalizer: NormalizerConfig = field(default_factory=NormalizerConfig)
    normalizer_enabled: bool = True  # False when audio: section is absent
    checks: CheckPipelineConfig = field(default_factory=CheckPipelineConfig)
    split: SplitConfig = field(default_factory=SplitConfig)


def _build_pipeline_config(raw: dict) -> PipelineConfig:
    """Internal: build PipelineConfig from raw YAML dict."""
    text = raw.get("text", {})
    sam = raw.get("split_and_merge", {})
    output = raw.get("output", {})

    return PipelineConfig(
        root_dir=raw.get("root_dir", ""),
        recursive=raw.get("recursive", False),
        use_lab_text_as_filename=raw.get("use_lab_text_as_filename", False),
        text_t2s=text.get("t2s", True),
        text_replace_map=text.get("replace_map", {}),
        output_jsonl=output.get("jsonl_output", False),
        output_to_split_folder=sam.get("output_to_split_folder", False),
        short_threshold=sam.get("short_threshold", 2.0),
        long_threshold=sam.get("long_threshold", 15.0),
        asr=build_asr_config(raw),
        ser=build_ser_config(raw),
        normalizer=build_normalizer_config(raw),
        normalizer_enabled="audio" in raw and bool(raw["audio"]),
        checks=build_check_pipeline_config(raw),
        split=build_split_config(raw),
    )


def _read_yaml(config_path: str) -> dict:
    """Read a YAML config file and return raw dict."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw or {}


def load_normalizer_config(config_path: str = "config.yaml") -> NormalizerConfig:
    """Read YAML and build NormalizerConfig."""
    return build_normalizer_config(_read_yaml(config_path))


def load_asr_config(config_path: str = "config.yaml") -> AsrConfig:
    """Read YAML and build AsrConfig."""
    return build_asr_config(_read_yaml(config_path))


def load_check_pipeline_config(config_path: str = "config.yaml") -> CheckPipelineConfig:
    """Read YAML and build CheckPipelineConfig."""
    return build_check_pipeline_config(_read_yaml(config_path))


def load_split_config(config_path: str = "config.yaml") -> SplitConfig:
    """Read YAML and build SplitConfig."""
    return build_split_config(_read_yaml(config_path))


def load_ser_config(config_path: str = "config.yaml") -> SerConfig:
    """Read YAML and build SerConfig."""
    return build_ser_config(_read_yaml(config_path))


def load_pipeline_config(config_path: str = "config.yaml") -> PipelineConfig:
    """Read YAML config file and build PipelineConfig — production entry point."""
    return _build_pipeline_config(_read_yaml(config_path))


@dataclass
class AudioEntry:
    """Represents a single audio entry in the preprocessing pipeline."""
    path: str
    lab_text: str
    duration: float
    emotion: str = "neutral"
    stem: Optional[str] = None
    lab_path: Optional[str] = None
    origin: str = "source"  # "source", "split", or "merge"

    @property
    def absolute_path(self) -> str:
        return os.path.abspath(self.path)
