from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from config_types import (
    PipelineConfig, NormalizerConfig, SerConfig, AsrConfig, AsrWhisperxAlignConfig,
    SplitConfig, CheckPipelineConfig, AuditCheckConfig, ClippingCheckConfig,
    SilenceRatioCheckConfig, SnrCheckConfig,
)


def _sine(freq_hz: float, duration_s: float, sample_rate: int) -> np.ndarray:
    t = np.linspace(0.0, duration_s, int(sample_rate * duration_s), endpoint=False)
    return (0.5 * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


@pytest.fixture
def tmp_audio_dir(tmp_path):
    audio_dir = tmp_path / "audio_data"
    audio_dir.mkdir()
    wav_path = audio_dir / "sample1.wav"
    lab_path = audio_dir / "sample1.lab"
    sr = 22050
    data = _sine(440.0, 0.5, sr)
    sf.write(str(wav_path), data, sr, subtype="PCM_16")
    lab_path.write_text("你好世界", encoding="utf-8")
    return audio_dir


@pytest.fixture
def sample_wav_44k_stereo(tmp_path):
    sr = 44100
    mono = _sine(440.0, 0.5, sr)
    stereo = np.stack([mono, mono], axis=1)
    path = tmp_path / "sample_44k_stereo.wav"
    sf.write(str(path), stereo, sr, subtype="PCM_16")
    return path


@pytest.fixture
def sample_wav_22k_mono(tmp_path):
    sr = 22050
    data = _sine(440.0, 0.5, sr)
    path = tmp_path / "sample_22k_mono.wav"
    sf.write(str(path), data, sr, subtype="PCM_16")
    return path


@pytest.fixture
def sample_wav_48k_24bit(tmp_path):
    sr = 48000
    data = _sine(440.0, 0.5, sr)
    path = tmp_path / "sample_48k_24bit.wav"
    sf.write(str(path), data, sr, subtype="PCM_24")
    return path


@pytest.fixture
def sample_flac(tmp_path):
    sr = 44100
    data = _sine(440.0, 0.5, sr)
    path = tmp_path / "sample.flac"
    sf.write(str(path), data, sr, format="FLAC", subtype="PCM_16")
    return path


@pytest.fixture
def sample_wav_with_silence(tmp_path):
    sr = 22050
    silence_pre = np.zeros(int(sr * 0.5), dtype=np.float32)
    tone = _sine(440.0, 1.0, sr)
    silence_post = np.zeros(int(sr * 0.5), dtype=np.float32)
    data = np.concatenate([silence_pre, tone, silence_post])
    path = tmp_path / "sample_with_silence.wav"
    sf.write(str(path), data, sr, subtype="PCM_16")
    return path


@pytest.fixture
def sample_wav_clipped(tmp_path):
    sr = 22050
    t = np.linspace(0.0, 0.5, int(sr * 0.5), endpoint=False)
    tone = np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    over = np.clip(tone * 3.0, -1.0, 1.0).astype(np.float32)
    path = tmp_path / "sample_clipped.wav"
    sf.write(str(path), over, sr, subtype="PCM_16")
    return path


@pytest.fixture
def sample_wav_dc_offset(tmp_path):
    sr = 22050
    tone = _sine(440.0, 0.5, sr) + 0.1
    path = tmp_path / "sample_dc_offset.wav"
    sf.write(str(path), tone.astype(np.float32), sr, subtype="PCM_16")
    return path


@pytest.fixture
def sample_wav_quiet(tmp_path):
    sr = 22050
    tone = _sine(440.0, 0.5, sr) * 0.05
    path = tmp_path / "sample_quiet.wav"
    sf.write(str(path), tone.astype(np.float32), sr, subtype="PCM_16")
    return path


@pytest.fixture
def test_config():
    return PipelineConfig(
        root_dir="",
        recursive=True,
        use_lab_text_as_filename=False,
        short_threshold=3.0,
        long_threshold=25.0,
        output_to_split_folder=False,
        output_jsonl=True,
        text_t2s=True,
        text_replace_map={},
        split=SplitConfig(
            device="auto", language="zh", min_duration=3.0, max_duration=25.0,
            max_segments=5, split_silence_top_db=40, split_min_silence_len_ms=500,
            split_offset_from_silence_end_ms=0, merge_silence_duration=0.3,
        ),
        ser=SerConfig(
            enabled=True,
            hubert_model_id="xmj2002/hubert-base-ch-speech-emotion-recognition",
            device="auto",
        ),
        asr=AsrConfig(
            language="zh", backend="funasr", funasr_model="paraformer-zh",
            funasr_use_vad=False, whisperx_model="large-v2", whisperx_batch_size=16,
        ),
        normalizer=NormalizerConfig(
            target_sample_rate=44100, target_channels=1, target_bit_depth=16,
            input_formats=["wav", "flac", "mp3", "ogg", "aac", "m4a"],
            trim_enabled=True, trim_top_db=30, trim_frame_length=2048, trim_hop_length=512,
            trim_min_trim_ms=100, trim_keep_padding_ms=50,
            loudness_enabled=True, loudness_target_lufs=-16, loudness_true_peak=-1.5,
            loudness_lra=7, loudness_dual_pass=False,
            dc_offset_removal_enabled=True, dc_offset_threshold=0.001,
        ),
        normalizer_enabled=True,
        checks=CheckPipelineConfig(
            audit=AuditCheckConfig(enable=True, run_at=["after_scan"],
                                   target_sample_rate=44100, target_channels=1, target_bit_depth=16),
            clipping=ClippingCheckConfig(enable=True, run_at=["after_normalize"],
                                         reject_action="warn_only", clip_threshold=0.99,
                                         warn_threshold=0.01, reject_threshold=0.05),
            silence_ratio=SilenceRatioCheckConfig(enable=True, run_at=["after_normalize"],
                                                    reject_action="isolate", top_db=30,
                                                    warn_threshold=0.4, reject_threshold=0.7),
            snr=SnrCheckConfig(enable=True, run_at=["after_normalize"],
                                reject_action="isolate", top_db=30, warn_db=20, reject_db=15),
        ),
    )
