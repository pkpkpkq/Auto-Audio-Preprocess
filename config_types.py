from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NormalizerConfig:
    """Config for audio_normalizer module."""
    target_sample_rate: int = 44100
    target_channels: int = 1
    target_bit_depth: int = 16
    input_formats: list[str] = field(default_factory=lambda: ["wav"])

    # Trim silence sub-section
    trim_enabled: bool = False
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
    loudness_dual_pass: bool = False

    # DC offset removal sub-section
    dc_offset_removal_enabled: bool = True
    dc_offset_threshold: float = 0.001


@dataclass
class ValidatorConfig:
    """Config for audio_validator module."""
    # Audit
    audit_enabled: bool = True
    target_sample_rate: int = 44100
    target_channels: int = 1
    target_bit_depth: int = 16

    # Clipping
    clipping_enabled: bool = True
    clip_threshold: float = 0.99
    threshold_ratio: float = 0.01

    # Silence ratio
    silence_ratio_enabled: bool = True
    silence_ratio_top_db: float = 30.0
    silence_warn_threshold: float = 0.4
    silence_reject_threshold: float = 0.7

    # SNR
    snr_enabled: bool = True
    snr_top_db: float = 30.0
    snr_warn_db: float = 20.0
    snr_reject_db: float = 15.0


@dataclass
class SerConfig:
    """Config for SER (Speech Emotion Recognition) module."""
    enabled: bool = False
    backend: str = "hubert"  # "hubert" | "emotion2vec" | "sensevoice"
    device: str = "auto"
    on_failure: str = "default"  # "default" | "truncate_and_retry" | "isolate"

    hubert_model_id: str = "xmj2002/hubert-base-ch-speech-emotion-recognition"

    emotion2vec_model_id: str = "iic/emotion2vec_plus_large"
    emotion2vec_granularity: str = "utterance"  # "utterance" | "frame"
    emotion2vec_safe_max_duration: float = 30.0  # seconds

    sensevoice_model_id: str = "iic/SenseVoiceSmall"
    sensevoice_language: str = "auto"  # "zh"|"en"|"yue"|"ja"|"ko"|"auto"
    sensevoice_use_itn: bool = True
    sensevoice_ban_emo_unk: bool = False
    sensevoice_vad_model: str = "fsmn-vad"
    sensevoice_vad_max_single_segment_time: int = 30000  # ms


@dataclass
class SplitConfig:
    """Config for AudioSplitter."""
    device: str = "auto"
    language: str = "zh"
    min_duration: float = 3.0
    max_duration: float = 25.0
    max_segments: int = 8
    split_min_silence_len_ms: int = 300
    split_offset_from_silence_end_ms: int = 0
    split_silence_top_db: int = 40
    merge_silence_duration: float = 0.3
    on_split_failure: str = "ignore"  # "isolate" | "ignore"


def build_normalizer_config(raw_config: dict) -> NormalizerConfig:
    audio = raw_config.get("audio", {})
    trim = audio.get("trim_silence", {})
    loud = audio.get("loudness", {})
    dc = audio.get("dc_offset_removal", {})
    # If no audio section, return defaults with everything disabled
    if not audio:
        return NormalizerConfig()
    return NormalizerConfig(
        target_sample_rate=int(audio.get("target_sample_rate", 44100)),
        target_channels=int(audio.get("target_channels", 1)),
        target_bit_depth=int(audio.get("target_bit_depth", 16)),
        input_formats=audio.get("input_formats", ["wav"]),
        trim_enabled=trim.get("enable", False),
        trim_top_db=float(trim.get("top_db", 30.0)),
        trim_frame_length=int(trim.get("frame_length", 2048)),
        trim_hop_length=int(trim.get("hop_length", 512)),
        trim_min_trim_ms=float(trim.get("min_trim_ms", 100.0)),
        trim_keep_padding_ms=float(trim.get("keep_padding_ms", 50.0)),
        loudness_enabled=loud.get("enable", True),
        loudness_target_lufs=float(loud.get("target_lufs", -16.0)),
        loudness_true_peak=float(loud.get("true_peak", -1.5)),
        loudness_lra=float(loud.get("lra", 7.0)),
        loudness_dual_pass=loud.get("dual_pass", False),
        dc_offset_removal_enabled=dc.get("enable", True),
        dc_offset_threshold=float(dc.get("threshold", 0.001)),
    )


def build_validator_config(raw_config: dict) -> ValidatorConfig:
    audio = raw_config.get("audio", {})
    if not audio:
        return ValidatorConfig(audit_enabled=False, clipping_enabled=False, silence_ratio_enabled=False, snr_enabled=False)
    audit = audio.get("audit", {})
    clip = audio.get("clipping_detection", {})
    sr = audio.get("silence_ratio", {})
    snr = audio.get("snr_estimation", {})
    return ValidatorConfig(
        audit_enabled=audit.get("enable", True),
        target_sample_rate=int(audio.get("target_sample_rate", 44100)),
        target_channels=int(audio.get("target_channels", 1)),
        target_bit_depth=int(audio.get("target_bit_depth", 16)),
        clipping_enabled=clip.get("enable", True),
        clip_threshold=float(clip.get("clip_threshold", 0.99)),
        threshold_ratio=float(clip.get("threshold_ratio", 0.01)),
        silence_ratio_enabled=sr.get("enable", True),
        silence_ratio_top_db=float(sr.get("top_db", 30.0)),
        silence_warn_threshold=float(sr.get("warn_threshold", 0.4)),
        silence_reject_threshold=float(sr.get("reject_threshold", 0.7)),
        snr_enabled=snr.get("enable", True),
        snr_top_db=float(snr.get("top_db", 30.0)),
        snr_warn_db=float(snr.get("warn_db", 20.0)),
        snr_reject_db=float(snr.get("reject_db", 15.0)),
    )


def build_split_config(raw_config: dict) -> SplitConfig:
    split = raw_config.get("split_and_merge", {}).get("split", raw_config.get("split", {}))
    sam = raw_config.get("split_and_merge", {})
    return SplitConfig(
        device=split.get("device", "auto"),
        language=split.get("language", "zh"),
        min_duration=float(split.get("min_duration", 3.0)),
        max_duration=float(split.get("max_duration", 25.0)),
        max_segments=int(split.get("max_segments", 8)),
        split_min_silence_len_ms=int(split.get("split_min_silence_len_ms", 300)),
        split_offset_from_silence_end_ms=int(split.get("split_offset_from_silence_end_ms", 0)),
        split_silence_top_db=int(split.get("split_silence_top_db", 40)),
        merge_silence_duration=float(sam.get("merge_silence_duration", 0.3)),
        on_split_failure=sam.get("on_split_failure", "ignore"),
    )


def build_ser_config(raw_config: dict) -> SerConfig:
    ser = raw_config.get("ser", {})
    if not ser:
        return SerConfig()
    return SerConfig(
        enabled=ser.get("enable", False),
        backend=ser.get("backend", "hubert"),
        device=ser.get("device", "auto"),
        on_failure=ser.get("on_failure", "default"),
        hubert_model_id=ser.get("hubert", {}).get("model_id", "xmj2002/hubert-base-ch-speech-emotion-recognition"),
        emotion2vec_model_id=ser.get("emotion2vec", {}).get("model_id", "iic/emotion2vec_plus_large"),
        emotion2vec_granularity=ser.get("emotion2vec", {}).get("granularity", "utterance"),
        emotion2vec_safe_max_duration=float(ser.get("emotion2vec", {}).get("safe_max_duration", 30.0)),
        sensevoice_model_id=ser.get("sensevoice", {}).get("model_id", "iic/SenseVoiceSmall"),
        sensevoice_language=ser.get("sensevoice", {}).get("language", "auto"),
        sensevoice_use_itn=ser.get("sensevoice", {}).get("use_itn", True),
        sensevoice_ban_emo_unk=ser.get("sensevoice", {}).get("ban_emo_unk", False),
        sensevoice_vad_model=ser.get("sensevoice", {}).get("vad_model", "fsmn-vad"),
        sensevoice_vad_max_single_segment_time=int(ser.get("sensevoice", {}).get("vad_max_single_segment_time", 30000)),
    )
