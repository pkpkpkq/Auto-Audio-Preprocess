from __future__ import annotations

import soundfile as sf


def test_tmp_audio_dir_creates_wav_lab_pair(tmp_audio_dir):
    wav_files = list(tmp_audio_dir.glob("*.wav"))
    lab_files = list(tmp_audio_dir.glob("*.lab"))
    assert len(wav_files) == 1
    assert len(lab_files) == 1
    assert wav_files[0].stat().st_size > 0
    assert lab_files[0].read_text(encoding="utf-8").strip() == "你好世界"


def test_sample_wav_44k_stereo(sample_wav_44k_stereo):
    assert sample_wav_44k_stereo.exists()
    assert sample_wav_44k_stereo.stat().st_size > 0
    data, sr = sf.read(str(sample_wav_44k_stereo))
    assert sr == 44100
    assert data.ndim == 2 and data.shape[1] == 2


def test_sample_wav_22k_mono(sample_wav_22k_mono):
    assert sample_wav_22k_mono.exists()
    assert sample_wav_22k_mono.stat().st_size > 0
    data, sr = sf.read(str(sample_wav_22k_mono))
    assert sr == 22050
    assert data.ndim == 1


def test_sample_wav_48k_24bit(sample_wav_48k_24bit):
    assert sample_wav_48k_24bit.exists()
    assert sample_wav_48k_24bit.stat().st_size > 0
    info = sf.info(str(sample_wav_48k_24bit))
    assert info.samplerate == 48000
    assert info.subtype == "PCM_24"


def test_sample_flac(sample_flac):
    assert sample_flac.exists()
    assert sample_flac.stat().st_size > 0
    info = sf.info(str(sample_flac))
    assert info.format == "FLAC"
    assert info.samplerate == 44100


def test_sample_wav_with_silence(sample_wav_with_silence):
    assert sample_wav_with_silence.exists()
    assert sample_wav_with_silence.stat().st_size > 0
    data, sr = sf.read(str(sample_wav_with_silence))
    assert abs(len(data) / sr - 2.0) < 0.05


def test_sample_wav_clipped(sample_wav_clipped):
    assert sample_wav_clipped.exists()
    assert sample_wav_clipped.stat().st_size > 0
    data, _ = sf.read(str(sample_wav_clipped))
    assert float(abs(data).max()) >= 0.99


def test_sample_wav_dc_offset(sample_wav_dc_offset):
    assert sample_wav_dc_offset.exists()
    assert sample_wav_dc_offset.stat().st_size > 0
    data, _ = sf.read(str(sample_wav_dc_offset))
    assert abs(float(data.mean()) - 0.1) < 0.05


def test_sample_wav_quiet(sample_wav_quiet):
    assert sample_wav_quiet.exists()
    assert sample_wav_quiet.stat().st_size > 0
    data, _ = sf.read(str(sample_wav_quiet))
    assert float(abs(data).max()) < 0.2


def test_test_config_has_full_audio_section(test_config):
    cfg = test_config
    for key in (
        "root_dir",
        "recursive",
        "use_lab_text_as_filename",
        "split_and_merge",
        "split",
        "ser",
        "asr",
        "text",
        "output",
        "audio",
    ):
        assert key in cfg, f"missing top-level key: {key}"

    assert "audio" in cfg
    audio = cfg["audio"]
    for key in (
        "target_sample_rate",
        "target_channels",
        "target_bit_depth",
        "input_formats",
        "trim_silence",
        "loudness",
        "dc_offset_removal",
        "audit",
        "clipping_detection",
        "silence_ratio",
        "snr_estimation",
    ):
        assert key in audio, f"missing audio key: {key}"
    assert audio["target_sample_rate"] == 44100
    assert audio["target_channels"] == 1
    assert audio["target_bit_depth"] == 16
    assert "wav" in audio["input_formats"]

    for sub in ("trim_silence", "loudness", "dc_offset_removal", "audit",
                "clipping_detection", "silence_ratio", "snr_estimation"):
        assert "enable" in audio[sub], f"missing 'enable' in audio.{sub}"
    assert audio["loudness"]["target_lufs"] == -16
    assert audio["trim_silence"]["top_db"] == 30
    assert audio["dc_offset_removal"]["threshold"] == 0.001
    assert audio["clipping_detection"]["clip_threshold"] == 0.99
    assert audio["silence_ratio"]["warn_threshold"] == 0.4
    assert audio["snr_estimation"]["warn_db"] == 20

    assert cfg["split_and_merge"]["short_threshold"] == 3.0
    assert cfg["split"]["language"] == "zh"
    assert cfg["ser"]["enable"] is True
    assert cfg["asr"]["funasr"]["model"] == "paraformer-zh"
    assert cfg["text"]["t2s"] is True
    assert cfg["output"]["jsonl_output"] is True
