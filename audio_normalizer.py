from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import wave


def _audio_config(config):
    return config.get("audio", {})


def _loudness_config(config):
    return _audio_config(config).get("loudness", {}) if config else {}


def _loudnorm_values(config):
    loudness_cfg = _loudness_config(config)
    return {
        "target": float(loudness_cfg.get("target_lufs", -16)),
        "lra": float(loudness_cfg.get("lra", 7)),
        "tp": float(loudness_cfg.get("true_peak", -1.5)),
    }


def _format_number(value):
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return str(value)


def _parse_loudnorm_json(output):
    start = output.rfind("{")
    end = output.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise RuntimeError("ffmpeg loudnorm 未返回JSON测量结果")
    try:
        return json.loads(output[start : end + 1])
    except json.JSONDecodeError as e:
        raise RuntimeError(f"ffmpeg loudnorm JSON解析失败: {e}") from e


def get_audio_duration_any(path):
    """获取任意 librosa 支持的音频时长；WAV 优先使用 stdlib wave。"""
    path = os.fspath(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        with contextlib.closing(wave.open(path, "rb")) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate == 0:
                raise RuntimeError("采样率为0")
            return frames / float(rate)

    try:
        import librosa

        duration = float(librosa.get_duration(path=path))
        if duration <= 0:
            raise RuntimeError(f"librosa 返回非正时长: {duration}")
        return duration
    except Exception as e:
        raise RuntimeError(f"librosa 获取 {path} 时长失败: {e}") from e


def needs_normalization(input_path, config):
    """判断输入音频是否需要归一化到目标 WAV 参数。"""
    input_path = os.fspath(input_path)
    if os.path.splitext(input_path)[1].lower() != ".wav":
        return True

    audio_cfg = _audio_config(config)
    target_sample_rate = int(audio_cfg.get("target_sample_rate", 44100))
    target_channels = int(audio_cfg.get("target_channels", 1))
    target_bit_depth = int(audio_cfg.get("target_bit_depth", 16))
    target_sampwidth = target_bit_depth // 8

    with contextlib.closing(wave.open(input_path, "rb")) as wf:
        return not (
            wf.getframerate() == target_sample_rate
            and wf.getnchannels() == target_channels
            and wf.getsampwidth() == target_sampwidth
        )


def normalize_format(input_path, output_path, config):
    """用 ffmpeg 将任意支持格式转为目标 WAV PCM，不修改原始输入。"""
    input_path = os.fspath(input_path)
    output_path = os.fspath(output_path)
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError("normalize_format 不允许覆盖原始输入文件")

    audio_cfg = _audio_config(config)
    target_sample_rate = int(audio_cfg.get("target_sample_rate", 44100))
    target_channels = int(audio_cfg.get("target_channels", 1))
    dc_cfg = audio_cfg.get("dc_offset_removal", {})

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-ar", str(target_sample_rate),
        "-ac", str(target_channels),
        "-c:a", "pcm_s16le",
    ]
    if dc_cfg.get("enable", False):
        cmd.extend(["-af", "highpass=f=5"])
    cmd.append(output_path)

    try:
        subprocess.run(cmd, check=True, timeout=300)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"ffmpeg 格式归一化超时: {input_path}") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg 格式归一化失败: {input_path} -> {output_path}: {e}") from e

    if not os.path.exists(output_path) or os.path.getsize(output_path) <= 0:
        raise RuntimeError(f"ffmpeg 格式归一化失败: 输出文件为空或不存在: {output_path}")

    return output_path


def trim_silence(input_path, output_path, config):
    input_path = os.fspath(input_path)
    output_path = os.fspath(output_path)
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError("trim_silence 不允许覆盖原始输入文件")

    try:
        import librosa
        import soundfile as sf
    except ImportError as e:
        raise RuntimeError(f"静音修剪依赖导入失败: {e}") from e

    trim_cfg = _audio_config(config).get("trim_silence", {})
    top_db = float(trim_cfg.get("top_db", 30))
    keep_padding_ms = float(trim_cfg.get("keep_padding_ms", 50))
    min_trim_ms = float(trim_cfg.get("min_trim_ms", 100))

    try:
        y, sr = librosa.load(input_path, sr=None)
        _, trim_idx = librosa.effects.trim(y, top_db=top_db)
    except Exception as e:
        raise RuntimeError(f"静音修剪分析失败: {input_path}: {e}") from e

    input_len = len(y)
    start_idx = int(trim_idx[0])
    end_idx = int(trim_idx[1])
    trimmed_samples = start_idx + max(0, input_len - end_idx)
    min_trim_samples = int(min_trim_ms * sr / 1000)

    if trimmed_samples < min_trim_samples:
        shutil.copy2(input_path, output_path)
        return output_path

    padding_samples = int(keep_padding_ms * sr / 1000)
    start = max(0, start_idx - padding_samples)
    end = min(input_len, end_idx + padding_samples)

    try:
        sf.write(output_path, y[start:end], sr, subtype="PCM_16")
    except Exception as e:
        raise RuntimeError(f"静音修剪写出失败: {input_path} -> {output_path}: {e}") from e

    if not os.path.exists(output_path) or os.path.getsize(output_path) <= 0:
        raise RuntimeError(f"静音修剪失败: 输出文件为空或不存在: {output_path}")

    return output_path


def _measure_loudness(input_path, config=None):
    input_path = os.fspath(input_path)
    values = _loudnorm_values(config)
    loudnorm_filter = (
        f"loudnorm=I={_format_number(values['target'])}:"
        f"LRA={_format_number(values['lra'])}:"
        f"TP={_format_number(values['tp'])}:print_format=json"
    )
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-loglevel", "info",
        "-i", input_path,
        "-af", loudnorm_filter,
        "-f", "null", "-",
    ]

    try:
        result = subprocess.run(cmd, check=True, timeout=300, capture_output=True, text=True)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"ffmpeg loudnorm 测量超时: {input_path}") from e
    except subprocess.CalledProcessError as e:
        detail = e.stderr or e.stdout or str(e)
        raise RuntimeError(f"ffmpeg loudnorm 测量失败: {input_path}: {detail}") from e

    return _parse_loudnorm_json((result.stderr or "") + "\n" + (result.stdout or ""))


def normalize_loudness(input_path, output_path, config):
    input_path = os.fspath(input_path)
    output_path = os.fspath(output_path)
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError("normalize_loudness 不允许覆盖原始输入文件")

    loudness_cfg = _loudness_config(config)
    if not loudness_cfg.get("enable", True):
        shutil.copy2(input_path, output_path)
        return output_path

    values = _loudnorm_values(config)
    target = _format_number(values["target"])
    lra = _format_number(values["lra"])
    tp = _format_number(values["tp"])
    loudnorm_parts = [f"I={target}", f"LRA={lra}", f"TP={tp}"]

    if loudness_cfg.get("dual_pass", False):
        measured = _measure_loudness(input_path, config)
        loudnorm_parts.extend(
            [
                f"measured_I={measured['input_i']}",
                f"measured_LRA={measured['input_lra']}",
                f"measured_TP={measured['input_tp']}",
                f"measured_thresh={measured['input_thresh']}",
                f"offset={measured['target_offset']}",
                "linear=true",
            ]
        )

    audio_cfg = _audio_config(config)
    target_sample_rate = str(int(audio_cfg.get("target_sample_rate", 44100)))
    target_channels = str(int(audio_cfg.get("target_channels", 1)))

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-af", "loudnorm=" + ":".join(loudnorm_parts),
        "-ar", target_sample_rate,
        "-ac", target_channels,
        output_path,
    ]

    try:
        subprocess.run(cmd, check=True, timeout=300)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"ffmpeg loudnorm 归一化超时: {input_path}") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg loudnorm 归一化失败: {input_path} -> {output_path}: {e}") from e

    if not os.path.exists(output_path) or os.path.getsize(output_path) <= 0:
        raise RuntimeError(f"ffmpeg loudnorm 归一化失败: 输出文件为空或不存在: {output_path}")

    return output_path
