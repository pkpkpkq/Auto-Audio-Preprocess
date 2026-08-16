"""Audio quality validation: silence ratio, VAD-SNR estimation, batch checks."""
from __future__ import annotations

import logging
import os
import subprocess

import librosa
import numpy as np

from config_types import AuditCheckConfig, ClippingCheckConfig, SilenceRatioCheckConfig, SnrCheckConfig, CheckPipelineConfig, AudioEntry

logger = logging.getLogger(__name__)


def _format_progress(label: str, current: int, total: int, width: int = 20) -> str:
    """Format a single-line progress bar string (meant for \\r overwrite)."""
    pct = current / total if total > 0 else 0
    filled = int(pct * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"\r[{label}] {current}/{total} ({pct:5.1%}) |{bar}|"


def _write_report(lines: list[str], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


_STATUS_SORT_ORDER = {"reject": 0, "warn": 1, "ok": 2}


def _sort_report_items(items: list[dict], status_key: str) -> list[dict]:
    return sorted(
        items,
        key=lambda x: (
            _STATUS_SORT_ORDER.get(x.get(status_key, ""), 99),
            os.path.basename(x.get("wav_path", "") or "unknown"),
        ),
    )


def compute_silence_ratio(input_path: str, config: SilenceRatioCheckConfig) -> dict:
    top_db = config.top_db
    warn_threshold = config.warn_threshold
    reject_threshold = config.reject_threshold

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


def estimate_snr(input_path: str, config: SnrCheckConfig) -> dict:
    top_db = config.top_db
    warn_db = config.warn_db
    reject_db = config.reject_db

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


def batch_check_silence_ratio(entries: list[AudioEntry], config: SilenceRatioCheckConfig,
                              progress_callback=None) -> list[dict]:
    results = []
    total = len(entries)
    for i, entry in enumerate(entries, 1):
        wav_path = entry.path
        if not wav_path or not os.path.isfile(wav_path):
            logger.warning(f"跳过无效路径: {wav_path}")
            results.append({
                "wav_path": wav_path,
                "silence_ratio": None,
                "speech_duration": None,
                "total_duration": None,
                "silence_status": "skip",
            })
            continue

        record = {"wav_path": wav_path}
        try:
            sr_result = compute_silence_ratio(wav_path, config)
            record["silence_ratio"] = sr_result["silence_ratio"]
            record["speech_duration"] = sr_result["speech_duration"]
            record["total_duration"] = sr_result["total_duration"]
            record["silence_status"] = sr_result["status"]
        except Exception as e:
            logger.error(f"静默比例检测失败 {wav_path}: {e}")
            record["silence_ratio"] = None
            record["speech_duration"] = None
            record["total_duration"] = None
            record["silence_status"] = "error"

        results.append(record)
        if progress_callback:
            progress_callback(i, total)
    return results


def write_silence_ratio_report(data: list[dict], output_path: str, warn_threshold: float = 0.4, reject_threshold: float = 0.7) -> None:
    lines = ["=== 静默比例检测报告 ===", ""]

    ok_count = 0
    warn_count = 0
    reject_count = 0

    for item in data:
        status = item.get("silence_status", "N/A")
        if status == "reject":
            reject_count += 1
        elif status == "warn":
            warn_count += 1
        elif status == "ok":
            ok_count += 1

    warn_pct = int(warn_threshold * 100)
    reject_pct = int(reject_threshold * 100)
    total = len(data)
    lines.append("--- 汇总 ---")
    lines.append(f"总计: {total} 个文件")
    lines.append(f"正常: {ok_count} 个")
    lines.append(f"警告: {warn_count} 个 (静默比例 > {warn_pct}%)")
    lines.append(f"拒绝: {reject_count} 个 (静默比例 > {reject_pct}%)")
    lines.append("")
    lines.append("--- 明细 ---")

    for item in _sort_report_items(data, "silence_status"):
        wav = os.path.basename(item.get("wav_path", "unknown") or "unknown")
        sr_val = item.get("silence_ratio", "N/A")
        status = item.get("silence_status", "N/A")
        if status == "ok":
            status_str = "正常"
        elif status == "warn":
            status_str = "警告"
        elif status == "reject":
            status_str = "拒绝"
        else:
            status_str = status
        lines.append(f"{wav}  静默比例={sr_val} ({status_str})")

    _write_report(lines, output_path)
    logger.info(f"静默比例报告已写入: {output_path}")


def batch_check_snr(entries: list[AudioEntry], config: SnrCheckConfig,
                    progress_callback=None) -> list[dict]:
    results = []
    total = len(entries)
    for i, entry in enumerate(entries, 1):
        wav_path = entry.path
        if not wav_path or not os.path.isfile(wav_path):
            logger.warning(f"跳过无效路径: {wav_path}")
            results.append({
                "wav_path": wav_path,
                "snr": None,
                "snr_reliable": None,
                "snr_status": "skip",
            })
            continue

        record = {"wav_path": wav_path}
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
        if progress_callback:
            progress_callback(i, total)
    return results


def write_snr_report(data: list[dict], output_path: str, warn_db: float = 20.0, reject_db: float = 15.0) -> None:
    lines = ["=== SNR 检测报告 ===", ""]

    ok_count = 0
    warn_count = 0
    reject_count = 0

    for item in data:
        status = item.get("snr_status", "N/A")
        if status == "reject":
            reject_count += 1
        elif status == "warn":
            warn_count += 1
        elif status == "ok":
            ok_count += 1

    total = len(data)
    lines.append("--- 汇总 ---")
    lines.append(f"总计: {total} 个文件")
    lines.append(f"正常: {ok_count} 个")
    lines.append(f"警告: {warn_count} 个 (SNR < {warn_db}dB)")
    lines.append(f"拒绝: {reject_count} 个 (SNR < {reject_db}dB)")
    lines.append("")
    lines.append("--- 明细 ---")

    for item in _sort_report_items(data, "snr_status"):
        wav = os.path.basename(item.get("wav_path", "unknown") or "unknown")
        snr_val = item.get("snr", "N/A")
        status = item.get("snr_status", "N/A")
        if status == "ok":
            status_str = "正常"
        elif status == "warn":
            status_str = "警告"
        elif status == "reject":
            status_str = "拒绝"
        else:
            status_str = status
        lines.append(f"{wav}  SNR={snr_val}dB ({status_str})")

    _write_report(lines, output_path)
    logger.info(f"SNR报告已写入: {output_path}")


def audit_audio_metadata(entries: list[AudioEntry], config: AuditCheckConfig,
                         progress_callback=None) -> dict:
    if not config.enable:
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
    total = len(entries)

    for i, entry in enumerate(entries, 1):
        wav_path = entry.path
        if not wav_path or not isinstance(wav_path, str):
            errors.append(f"missing: {wav_path}")
            if progress_callback:
                progress_callback(i, total)
            continue
        if not os.path.isfile(wav_path):
            errors.append(f"missing: {wav_path}")
            logger.warning("审计跳过不存在的文件: %s", wav_path)
            if progress_callback:
                progress_callback(i, total)
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
                    capture_output=True, text=True, timeout=30, encoding="utf-8",
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
            if progress_callback:
                progress_callback(i, total)
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

        if progress_callback:
            progress_callback(i, total)

    return {
        "total": sum(sample_rates.values()),
        "sample_rates": dict(sample_rates),
        "bit_depths": dict(bit_depths),
        "channels": dict(channels),
        "needs_conversion": needs_conversion,
        "errors": errors,
        "skipped": False,
    }


def write_audit_report(audit_data: dict, output_path: str) -> None:
    import datetime
    lines = ["=== 格式审计报告 ===", ""]

    total = audit_data.get("total", 0)
    needs_conv = audit_data.get("needs_conversion", [])
    errors = audit_data.get("errors", [])

    lines.append("--- 汇总 ---")
    lines.append(f"总计: {total} 个文件")
    lines.append(f"需转换: {len(needs_conv)} 个文件")
    lines.append(f"错误: {len(errors)} 个文件")
    lines.append("")
    lines.append("--- 明细 ---")
    lines.append(f"({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    lines.append("")

    for sr, cnt in audit_data.get("sample_rates", {}).items():
        lines.append(f"  采样率 {sr}Hz: {cnt} 个文件")
    for bd, cnt in audit_data.get("bit_depths", {}).items():
        lines.append(f"  位深 {bd}bit: {cnt} 个文件")
    for ch, cnt in audit_data.get("channels", {}).items():
        lines.append(f"  声道 {ch}: {cnt} 个文件")
    if needs_conv:
        lines.append("")
        lines.append("需要转换:")
        for item in needs_conv:
            lines.append(f"  {item}")
    if errors:
        lines.append("")
        lines.append("错误:")
        for item in errors:
            lines.append(f"  {item}")
    _write_report(lines, output_path)


def detect_clipping(input_path: str, config: ClippingCheckConfig) -> dict:
    if not config.enable:
        return {"skipped": True}
    import soundfile as sf
    data, sr = sf.read(input_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    clip_threshold = config.clip_threshold
    clipped_samples = int(np.sum(np.abs(data) >= clip_threshold))
    total = len(data)
    clip_ratio = clipped_samples / total if total > 0 else 0.0
    max_amp = float(np.max(np.abs(data)))
    peak_dbfs = 20.0 * np.log10(max_amp) if max_amp > 0 else -np.inf
    clip_percentage = clip_ratio * 100.0

    warn_threshold = config.warn_threshold
    reject_threshold = config.reject_threshold
    if clip_ratio >= reject_threshold:
        status = "reject"
    elif clip_ratio >= warn_threshold:
        status = "warn"
    else:
        status = "ok"

    return {
        "clipped": clip_ratio > warn_threshold,
        "clipping_status": status,
        "max_amplitude": round(max_amp, 6),
        "clip_percentage": round(clip_percentage, 4),
        "peak_dbfs": round(float(peak_dbfs), 2),
    }


def batch_detect_clipping(entries: list[AudioEntry], config: ClippingCheckConfig,
                         progress_callback=None) -> list[dict]:
    if not config.enable:
        return [{"skipped": True}]
    results = []
    total = len(entries)
    for i, entry in enumerate(entries, 1):
        wav_path = entry.path
        if not wav_path or not os.path.isfile(wav_path):
            if progress_callback:
                progress_callback(i, total)
            continue
        try:
            r = detect_clipping(wav_path, config)
            r["wav_path"] = wav_path
            results.append(r)
        except Exception as e:
            logger.error(f"削波检测失败 {wav_path}: {e}")
        if progress_callback:
            progress_callback(i, total)
    return results


def write_clipping_report(clipping_data: list[dict], output_path: str, warn_threshold: float = 0.01, reject_threshold: float = 0.05) -> None:
    lines = ["=== 削波检测报告 ===", ""]

    ok_count = 0
    warn_count = 0
    reject_count = 0
    for item in clipping_data:
        status = item.get("clipping_status", "N/A")
        if status == "reject":
            reject_count += 1
        elif status == "warn":
            warn_count += 1
        elif status == "ok":
            ok_count += 1
    total = len(clipping_data)

    warn_pct = int(warn_threshold * 100)
    reject_pct = int(reject_threshold * 100)
    lines.append("--- 汇总 ---")
    lines.append(f"总计: {total} 个文件")
    lines.append(f"正常: {ok_count} 个")
    lines.append(f"警告: {warn_count} 个 (削波比例 > {warn_pct}%)")
    lines.append(f"拒绝: {reject_count} 个 (削波比例 > {reject_pct}%)")
    lines.append("")
    lines.append("--- 明细 ---")

    for item in _sort_report_items(clipping_data, "clipping_status"):
        wav = os.path.basename(item.get("wav_path", "unknown") or "unknown")
        pct = item.get("clip_percentage", 0)
        peak = item.get("peak_dbfs", "N/A")
        status = item.get("clipping_status", "N/A")
        if status == "ok":
            status_str = "正常"
        elif status == "warn":
            status_str = "警告"
        elif status == "reject":
            status_str = "拒绝"
        else:
            status_str = status
        lines.append(f"{wav}  削波比例={pct}%  峰值={peak}dBFS ({status_str})")
    _write_report(lines, output_path)


def run_check_pipeline(
    entries: list[AudioEntry],
    check_config: CheckPipelineConfig,
    stage: str,
    output_dir: str,
) -> list[tuple[str, str, str, str]]:
    """Run all applicable checks for a pipeline stage, write stage-suffixed reports.

    Returns list of (wav_path, check_name, reject_action, stage_name) for rejected entries.
    Only checks whose run_at includes `stage` will run.
    Reject records are only collected at the LAST run_at stage for each check.
    """
    reject_records: list[tuple[str, str, str, str]] = []

    check_specs = [
        # audit — 返回 dict 而非 list[dict]，无 reject_action，单独处理
        ("audit", check_config.audit, audit_audio_metadata, "格式审计", None, None, None, None),
        # 统一定义: (name, cfg, batch_fn, report_name, status_key, write_fn, warn_key, reject_key)
        ("clipping", check_config.clipping, batch_detect_clipping, "削波检测", "clipping_status",
         write_clipping_report, "warn_threshold", "reject_threshold"),
        ("silence_ratio", check_config.silence_ratio, batch_check_silence_ratio, "静默比例", "silence_status",
         write_silence_ratio_report, "warn_threshold", "reject_threshold"),
        ("snr", check_config.snr, batch_check_snr, "SNR", "snr_status",
         write_snr_report, "warn_db", "reject_db"),
    ]

    for check_name, check_cfg, batch_func, report_name, status_key, write_fn, warn_key, reject_key in check_specs:
        if not check_cfg.enable:
            continue
        if stage not in check_cfg.run_at:
            continue

        is_last_run = (stage == check_cfg.run_at[-1])
        report_path = os.path.join(output_dir, f"！{report_name}.{stage}.txt")

        if check_name == "audit":
            result = batch_func(entries, check_cfg)
            write_audit_report(result, report_path)
        else:
            results = batch_func(entries, check_cfg)
            kwargs = {}
            if warn_key:
                kwargs[warn_key] = getattr(check_cfg, warn_key)
            if reject_key:
                kwargs[reject_key] = getattr(check_cfg, reject_key)
            write_fn(results, report_path, **kwargs)
            if is_last_run:
                for r in results:
                    if r.get(status_key) == "reject":
                        reject_records.append((r.get("wav_path", ""), check_name, check_cfg.reject_action, stage))

    return reject_records


def run_full_audit(
    entries: list[AudioEntry],
    check_config: CheckPipelineConfig,
    output_dir: str,
    show_progress: bool = False,
) -> None:
    """Run ALL enabled checks on entries, write reports without stage suffix.

    Runs every check regardless of run_at. No reject_action is applied.
    Reports are written as ！<report_name>.txt (no stage suffix).
    Thresholds still classify entries as ok/warn/reject in reports for informational purposes.
    When show_progress is True, prints a single-line progress bar per check to stdout.
    """
    total = len(entries)

    def _make_cb(label: str):
        if not show_progress:
            return None
        def _cb(current: int, _total: int):
            print(_format_progress(label, current, _total), end="", flush=True)
        return _cb

    if check_config.audit.enable:
        result = audit_audio_metadata(entries, check_config.audit,
                                      progress_callback=_make_cb("格式审计"))
        if show_progress:
            print()
        write_audit_report(result, os.path.join(output_dir, "！格式审计.txt"))

    if check_config.clipping.enable:
        results = batch_detect_clipping(entries, check_config.clipping,
                                        progress_callback=_make_cb("削波检测"))
        if show_progress:
            print()
        write_clipping_report(results, os.path.join(output_dir, "！削波检测.txt"),
                              warn_threshold=check_config.clipping.warn_threshold,
                              reject_threshold=check_config.clipping.reject_threshold)

    if check_config.silence_ratio.enable:
        results = batch_check_silence_ratio(entries, check_config.silence_ratio,
                                            progress_callback=_make_cb("静默比例"))
        if show_progress:
            print()
        write_silence_ratio_report(results, os.path.join(output_dir, "！静默比例.txt"),
                                   warn_threshold=check_config.silence_ratio.warn_threshold,
                                   reject_threshold=check_config.silence_ratio.reject_threshold)

    if check_config.snr.enable:
        results = batch_check_snr(entries, check_config.snr,
                                  progress_callback=_make_cb("SNR"))
        if show_progress:
            print()
        write_snr_report(results, os.path.join(output_dir, "！SNR.txt"),
                         warn_db=check_config.snr.warn_db,
                         reject_db=check_config.snr.reject_db)

    logger.info("独立审计完成，报告已写入: %s", output_dir)


if __name__ == "__main__":
    import sys

    input_dir = sys.argv[1] if len(sys.argv) > 1 else None
    config_path = sys.argv[2] if len(sys.argv) > 2 else None

    from config_types import PipelineConfig, load_pipeline_config

    config_file = config_path or ("config.yaml" if os.path.isfile("config.yaml") else None)
    if config_file:
        cfg = load_pipeline_config(config_file)
    else:
        cfg = PipelineConfig()

    if not input_dir:
        input_dir = cfg.root_dir or os.getcwd()

    if not os.path.isdir(input_dir):
        print(f"目录不存在: {input_dir}")
        sys.exit(1)

    _check_config = cfg.checks

    import glob as _glob
    from audio_utils import get_audio_duration_seconds

    audio_exts = ["wav", "flac", "mp3", "ogg", "aac", "m4a"]
    input_formats = cfg.normalizer.input_formats or audio_exts
    audio_files: list[str] = []
    for ext in input_formats:
        pattern = os.path.join(input_dir, "**", f"*.{ext}")
        audio_files.extend(_glob.glob(pattern, recursive=True))

    audit_entries: list[AudioEntry] = []
    for f in sorted(set(audio_files)):
        try:
            duration = get_audio_duration_seconds(f)
        except Exception:
            duration = 0.0
        audit_entries.append(AudioEntry(path=f, lab_text="", duration=duration))

    if not audit_entries:
        print(f"在 {input_dir} 中未找到音频文件")
        sys.exit(1)

    _folder_name = os.path.basename(os.path.abspath(input_dir).rstrip(os.sep))
    _base_output_dir = os.path.join(os.path.abspath("output"), _folder_name)
    _output_dir = os.path.join(_base_output_dir, _folder_name)

    from audio_utils import confirm_output_dir
    if not confirm_output_dir(_base_output_dir):
        sys.exit(0)

    os.makedirs(_output_dir, exist_ok=True)
    print(f"找到 {len(audit_entries)} 个音频文件，开始独立审计...")
    run_full_audit(audit_entries, _check_config, _output_dir, show_progress=True)
    print(f"独立审计完成，报告已写入: {_output_dir}")
