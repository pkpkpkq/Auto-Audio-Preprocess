from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import wave

from config_types import NormalizerConfig
from audio_utils import bit_depth_to_ffmpeg_codec


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


def needs_normalization(input_path, config: NormalizerConfig):
    """判断输入音频是否需要归一化到目标 WAV 参数。"""
    input_path = os.fspath(input_path)
    if os.path.splitext(input_path)[1].lower() != ".wav":
        return True

    target_sample_rate = config.target_sample_rate
    target_channels = config.target_channels
    target_bit_depth = config.target_bit_depth
    target_sampwidth = target_bit_depth // 8

    with contextlib.closing(wave.open(input_path, "rb")) as wf:
        return not (
            wf.getframerate() == target_sample_rate
            and wf.getnchannels() == target_channels
            and wf.getsampwidth() == target_sampwidth
        )


def normalize_format(input_path, output_path, config: NormalizerConfig):
    """用 ffmpeg 将任意支持格式转为目标 WAV PCM，不修改原始输入。"""
    input_path = os.fspath(input_path)
    output_path = os.fspath(output_path)
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError("normalize_format 不允许覆盖原始输入文件")

    target_sample_rate = config.target_sample_rate
    target_channels = config.target_channels
    target_bit_depth = config.target_bit_depth

    ffmpeg_codec, _ = bit_depth_to_ffmpeg_codec(target_bit_depth)

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-ar", str(target_sample_rate),
        "-ac", str(target_channels),
        "-c:a", ffmpeg_codec,
    ]
    if config.dc_offset_removal_enabled:
        should_remove_dc = True
        if config.dc_offset_threshold > 0:
            try:
                import numpy as np
                import soundfile as sf
                data, _ = sf.read(input_path, dtype="float64")
                if data.ndim > 1:
                    data = data.mean(axis=1)
                dc_offset = float(np.mean(data))
                should_remove_dc = abs(dc_offset) > config.dc_offset_threshold
            except Exception:
                pass  # 测量失败时保守执行，避免遗漏
        if should_remove_dc:
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


def trim_silence(input_path, output_path, config: NormalizerConfig):
    input_path = os.fspath(input_path)
    output_path = os.fspath(output_path)
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError("trim_silence 不允许覆盖原始输入文件")

    try:
        import librosa
        import soundfile as sf
    except ImportError as e:
        raise RuntimeError(f"静音修剪依赖导入失败: {e}") from e

    top_db = config.trim_top_db
    keep_padding_ms = config.trim_keep_padding_ms
    min_trim_ms = config.trim_min_trim_ms
    frame_length = config.trim_frame_length
    hop_length = config.trim_hop_length

    try:
        y, sr = librosa.load(input_path, sr=None)
        _, trim_idx = librosa.effects.trim(y, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
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
        _, sf_subtype = bit_depth_to_ffmpeg_codec(config.target_bit_depth)
        sf.write(output_path, y[start:end], sr, subtype=sf_subtype)
    except Exception as e:
        raise RuntimeError(f"静音修剪写出失败: {input_path} -> {output_path}: {e}") from e

    if not os.path.exists(output_path) or os.path.getsize(output_path) <= 0:
        raise RuntimeError(f"静音修剪失败: 输出文件为空或不存在: {output_path}")

    return output_path


def _measure_loudness(input_path, config: NormalizerConfig):
    input_path = os.fspath(input_path)
    values = {
        "target": config.loudness_target_lufs,
        "lra": config.loudness_lra,
        "tp": config.loudness_true_peak,
    }
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


def normalize_loudness(input_path, output_path, config: NormalizerConfig):
    input_path = os.fspath(input_path)
    output_path = os.fspath(output_path)
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError("normalize_loudness 不允许覆盖原始输入文件")

    if not config.loudness_enabled:
        shutil.copy2(input_path, output_path)
        return output_path

    values = {
        "target": config.loudness_target_lufs,
        "lra": config.loudness_lra,
        "tp": config.loudness_true_peak,
    }
    target = _format_number(values["target"])
    lra = _format_number(values["lra"])
    tp = _format_number(values["tp"])
    loudnorm_parts = [f"I={target}", f"LRA={lra}", f"TP={tp}"]

    if config.loudness_dual_pass:
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

    target_sample_rate = str(config.target_sample_rate)
    target_channels = str(config.target_channels)

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
