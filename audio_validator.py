"""Audio quality validation: silence ratio, VAD-SNR estimation, batch checks."""
from __future__ import annotations

import logging
import os
import subprocess

import librosa
import numpy as np

from audio_entry import AudioEntry
from config_types import ValidatorConfig

logger = logging.getLogger(__name__)


def _write_report(lines: list[str], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def compute_silence_ratio(input_path: str, config: ValidatorConfig) -> dict:
    top_db = config.silence_ratio_top_db
    warn_threshold = config.silence_warn_threshold
    reject_threshold = config.silence_reject_threshold

    y, sr = librosa.load(input_path, sr=None, mono=True)
    intervals = librosa.effects.split(y, top_db=top_db)

    speech_samples = sum(end - start for start, end in intervals)
    total_samples = len(y)
    speech_ratio = speech_samples / total_samples if total_samples > 0 else 0.0
    silence_ratio = 1.0 - speech_ratio

    if silence_ratio >= reject_threshold:
        status = "reject"
    elif silence_ratio >= warn_threshold:
        status = "warn"
    else:
        status = "ok"

    return {
        "silence_ratio": round(silence_ratio, 4),
        "speech_duration": round(speech_samples / sr, 4),
        "total_duration": round(total_samples / sr, 4),
        "status": status,
    }


def estimate_snr(input_path: str, config: ValidatorConfig) -> dict:
    top_db = config.snr_top_db
    warn_db = config.snr_warn_db
    reject_db = config.snr_reject_db

    y, sr = librosa.load(input_path, sr=None, mono=True)
    intervals = librosa.effects.split(y, top_db=top_db)

    speech_mask = np.zeros(len(y), dtype=bool)
    for start, end in intervals:
        speech_mask[start:end] = True
    noise_mask = ~speech_mask

    noise_samples = int(noise_mask.sum())
    if noise_samples == 0:
        return {"snr": 40.0, "reliable": False, "status": "ok"}

    p_speech = float(np.mean(y[speech_mask] ** 2)) if speech_mask.any() else 0.0
    p_noise = float(np.mean(y[noise_mask] ** 2))

    if p_noise == 0:
        return {"snr": 40.0, "reliable": True, "status": "ok"}

    snr = 10.0 * np.log10(p_speech / p_noise)

    if snr <= reject_db:
        status = "reject"
    elif snr <= warn_db:
        status = "warn"
    else:
        status = "ok"

    return {"snr": round(float(snr), 2), "reliable": True, "status": status}


def batch_quality_check(entries: list[AudioEntry], config: ValidatorConfig) -> list[dict]:
    sr_enable = config.silence_ratio_enabled
    snr_enable = config.snr_enabled

    results = []
    for entry in entries:
        wav_path = entry.path
        if not wav_path or not os.path.isfile(wav_path):
            logger.warning(f"跳过无效路径: {wav_path}")
            results.append({
                "wav_path": wav_path,
                "silence_ratio": None,
                "silence_status": "skip",
                "snr": None,
                "snr_reliable": None,
                "snr_status": "skip",
            })
            continue

        record = {"wav_path": wav_path}

        if sr_enable:
            try:
                sr_result = compute_silence_ratio(wav_path, config)
                record["silence_ratio"] = sr_result["silence_ratio"]
                record["silence_status"] = sr_result["status"]
            except Exception as e:
                logger.error(f"静默比例检测失败 {wav_path}: {e}")
                record["silence_ratio"] = None
                record["silence_status"] = "error"

        if snr_enable:
            try:
                snr_result = estimate_snr(wav_path, config)
                record["snr"] = snr_result["snr"]
                record["snr_reliable"] = snr_result["reliable"]
                record["snr_status"] = snr_result["status"]
            except Exception as e:
                logger.error(f"SNR估计失败 {wav_path}: {e}")
                record["snr"] = None
                record["snr_reliable"] = None
                record["snr_status"] = "error"

        results.append(record)
    return results


def write_quality_report(quality_data: list[dict], output_path: str) -> None:
    lines = []
    lines.append("=== 音频质量报告 ===")
    lines.append("")

    warn_count = 0
    reject_count = 0

    for item in quality_data:
        wav = os.path.basename(item.get("wav_path", "unknown"))
        sr_val = item.get("silence_ratio", "N/A")
        sr_st = item.get("silence_status", "N/A")
        snr_val = item.get("snr", "N/A")
        snr_rel = item.get("snr_reliable", "N/A")
        snr_st = item.get("snr_status", "N/A")

        if sr_st == "warn" or snr_st == "warn":
            warn_count += 1
        if sr_st == "reject" or snr_st == "reject":
            reject_count += 1

        reliable_str = "" if snr_rel is True else (" [不可靠]" if snr_rel is False else "")
        lines.append(f"{wav}  静默比例={sr_val} ({sr_st})  SNR={snr_val}dB{reliable_str} ({snr_st})")

    lines.append("")
    total = len(quality_data)
    lines.append(f"总计: {total} 个文件, 警告: {warn_count}, 拒绝: {reject_count}")

    _write_report(lines, output_path)
    logger.info(f"质量报告已写入: {output_path}")


def audit_audio_metadata(entries: list[AudioEntry], config: ValidatorConfig) -> dict:
    if not config.audit_enabled:
        return {"total": 0, "sample_rates": {}, "bit_depths": {}, "channels": {},
                "needs_conversion": [], "errors": [], "skipped": True}

    target_sr = config.target_sample_rate
    target_ch = config.target_channels
    target_bd = config.target_bit_depth

    from collections import Counter
    import wave

    sample_rates: Counter = Counter()
    bit_depths: Counter = Counter()
    channels: Counter = Counter()
    needs_conversion: list[str] = []
    errors: list[str] = []

    for entry in entries:
        wav_path = entry.path
        if not wav_path or not isinstance(wav_path, str):
            errors.append(f"missing: {wav_path}")
            continue
        if not os.path.isfile(wav_path):
            errors.append(f"missing: {wav_path}")
            logger.warning("审计跳过不存在的文件: %s", wav_path)
            continue

        ext = os.path.splitext(wav_path)[1].lower()
        try:
            if ext == ".wav":
                with wave.open(wav_path, "rb") as wf:
                    sr = wf.getframerate()
                    ch = wf.getnchannels()
                    bd = wf.getsampwidth() * 8
            else:
                result = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_streams", "-of", "json", wav_path],
                    capture_output=True, text=True, timeout=30,
                )
                import json
                info = json.loads(result.stdout)
                stream = None
                for s in info.get("streams", []):
                    if s.get("codec_type") == "audio":
                        stream = s
                        break
                if stream is None:
                    raise ValueError("未找到音频流")
                sr = int(stream.get("sample_rate", 0))
                ch = int(stream.get("channels", 0))
                bd = int(stream.get("bits_per_raw_sample", 0) or stream.get("bits_per_sample", 0))
                if bd == 0:
                    bd = 16
        except Exception as e:
            errors.append(f"{wav_path}: 读取元数据失败 - {e}")
            logger.warning("读取 %s 元数据失败: %s", wav_path, e)
            continue

        sample_rates[sr] += 1
        bit_depths[bd] += 1
        channels[ch] += 1

        reasons = []
        if sr != target_sr:
            reasons.append(f"采样率 {sr} != 目标 {target_sr}")
        if ch != target_ch:
            reasons.append(f"声道数 {ch} != 目标 {target_ch}")
        if bd != target_bd:
            reasons.append(f"位深 {bd} != 目标 {target_bd}")
        if reasons:
            needs_conversion.append(f"{wav_path}: {', '.join(reasons)}")

    return {
        "total": sum(sample_rates.values()),
        "sample_rates": dict(sample_rates),
        "bit_depths": dict(bit_depths),
        "channels": dict(channels),
        "needs_conversion": needs_conversion,
        "errors": errors,
        "skipped": False,
    }


def detect_clipping(input_path: str, config: ValidatorConfig) -> dict:
    if not config.clipping_enabled:
        return {"skipped": True}
    import soundfile as sf
    data, sr = sf.read(input_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    clip_threshold = config.clip_threshold
    threshold_ratio = config.threshold_ratio
    clipped_samples = int(np.sum(np.abs(data) >= clip_threshold))
    total = len(data)
    clip_percentage = (clipped_samples / total) * 100.0 if total > 0 else 0.0
    max_amp = float(np.max(np.abs(data)))
    peak_dbfs = 20.0 * np.log10(max_amp) if max_amp > 0 else -np.inf
    return {
        "clipped": clip_percentage > threshold_ratio,
        "max_amplitude": round(max_amp, 6),
        "clip_percentage": round(clip_percentage, 4),
        "peak_dbfs": round(float(peak_dbfs), 2),
    }


def batch_detect_clipping(entries: list[AudioEntry], config: ValidatorConfig) -> list[dict]:
    if not config.clipping_enabled:
        return [{"skipped": True}]
    results = []
    for entry in entries:
        wav_path = entry.path
        if not wav_path or not os.path.isfile(wav_path):
            continue
        try:
            r = detect_clipping(wav_path, config)
            r["wav_path"] = wav_path
            results.append(r)
        except Exception as e:
            logger.error(f"削波检测失败 {wav_path}: {e}")
    return results


def write_audit_report(audit_data: dict, output_path: str) -> None:
    import datetime
    lines = []
    lines.append(f"--- 格式审计报告 ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
    lines.append("")
    lines.append(f"总计: {audit_data.get('total', 0)} 个文件")
    for sr, cnt in audit_data.get("sample_rates", {}).items():
        lines.append(f"  采样率 {sr}Hz: {cnt} 个文件")
    for bd, cnt in audit_data.get("bit_depths", {}).items():
        lines.append(f"  位深 {bd}bit: {cnt} 个文件")
    for ch, cnt in audit_data.get("channels", {}).items():
        lines.append(f"  声道 {ch}: {cnt} 个文件")
    nc = audit_data.get("needs_conversion", [])
    if nc:
        lines.append("")
        lines.append("需要转换:")
        for item in nc:
            lines.append(f"  {item}")
    errors = audit_data.get("errors", [])
    if errors:
        lines.append("")
        lines.append("错误:")
        for item in errors:
            lines.append(f"  {item}")
    _write_report(lines, output_path)


def write_clipping_report(clipping_data: list[dict], output_path: str) -> None:
    lines = ["=== 削波检测报告 ===", ""]
    for item in clipping_data:
        wav = os.path.basename(item.get("wav_path", "unknown"))
        clipped = item.get("clipped", False)
        pct = item.get("clip_percentage", 0)
        peak = item.get("peak_dbfs", "N/A")
        status = "削波" if clipped else "正常"
        lines.append(f"{wav}  {status}  削波比例={pct}%  峰值={peak}dBFS")
    _write_report(lines, output_path)
