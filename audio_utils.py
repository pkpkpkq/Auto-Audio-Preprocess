import contextlib
import re
import subprocess
import tempfile
import wave
import sys
import os

from opencc import OpenCC

_t2s_converter = OpenCC('t2s')  # 繁体转简体

def traditional_to_simplified(text, enabled=True):
    if not enabled:
        return text
    try:
        return _t2s_converter.convert(text)
    except Exception as e:
        print(f"[WARN] 繁简转换失败: {e}", file=sys.stderr)
        return text

def sanitize_filename(filename):
    """清理文件名中的无效字符，并限制长度"""
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    return sanitized.strip(' .')[:200]

def read_lab(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return content

def get_wav_duration_seconds(path):
    with contextlib.closing(wave.open(path, "rb")) as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        if rate == 0:
            raise RuntimeError("采样率为0")
        duration = frames / float(rate)
    return duration

def format_hms(total_seconds):
    total_secs = int(round(total_seconds))
    hours = total_secs // 3600
    minutes = (total_secs % 3600) // 60
    seconds = total_secs % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def format_hms_filename(total_seconds):
    return format_hms(total_seconds).replace(":", "-")

def read_wav_params_and_frames(path):
    with contextlib.closing(wave.open(path, "rb")) as wf:
        params = wf.getparams()
        frames = wf.readframes(params.nframes)
    return {
        "nchannels": params.nchannels,
        "sampwidth": params.sampwidth,
        "framerate": params.framerate,
        "comptype": params.comptype,
        "nframes": params.nframes
    }, frames

def merge_wavs(output_path, wav_infos, config):
    """
    使用 ffmpeg concat 来合并音频，避免一次性加载大量 PCM 到内存中。
    """
    if not wav_infos:
        raise ValueError("没有 wav 可合并")
    
    silence_duration = config.get('split_and_merge', {}).get('merge_silence_duration', 0.5)

    base = wav_infos[0]["params"]
    nch = base["nchannels"]
    fr = base["framerate"]

    file_list_lines = []
    tmp_files_to_remove = []
    try:
        if silence_duration > 0:
            silence_path = None
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as sf:
                silence_path = sf.name
            tmp_files_to_remove.append(silence_path)
            cmd_silence = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", f"anullsrc=channel_layout=mono:sample_rate={fr}",
                "-t", f"{silence_duration:.3f}",
                "-ar", str(fr), "-ac", str(nch), "-c:a", "pcm_s16le", silence_path
            ]
            subprocess.run(cmd_silence, check=True)

        for idx, info in enumerate(wav_infos):
            safe_path = info['path'].replace("'", "'\\'")
            file_list_lines.append("file '{}'".format(safe_path))
            if silence_duration > 0 and idx != len(wav_infos)-1:
                safe_silence = silence_path.replace("'", "'\\'")
                file_list_lines.append("file '{}'".format(safe_silence))

        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt", encoding="utf-8") as lf:
            list_path = lf.name
            lf.write("\n".join(file_list_lines))
        tmp_files_to_remove.append(list_path)

        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", list_path,
            "-ar", str(fr), "-ac", str(nch), "-c:a", "pcm_s16le", output_path
        ]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg 合并失败: {e}")
    finally:
        for p in tmp_files_to_remove:
            try:
                os.remove(p)
            except Exception:
                pass
