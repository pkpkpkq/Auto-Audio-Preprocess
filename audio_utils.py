import contextlib
import re
import subprocess
import tempfile
import wave
import sys
import os

from opencc import OpenCC


def setup_model_cache() -> str:
    """将 HuggingFace 和 ModelScope 模型缓存统一重定向到项目根目录下的 local_model/。

    设置 HF_HOME 和 MODELSCOPE_CACHE 两个环境变量。
    幂等操作 — 多次调用安全。返回 local_model 目录的绝对路径。
    """
    project_dir = os.path.dirname(os.path.abspath(__file__))
    local_model_dir = os.path.join(project_dir, "local_model")
    os.makedirs(local_model_dir, exist_ok=True)
    os.environ['HF_HOME'] = local_model_dir
    os.environ['MODELSCOPE_CACHE'] = local_model_dir
    return local_model_dir

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
    with contextlib.closing(wave.open(os.fspath(path), "rb")) as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        if rate == 0:
            raise RuntimeError("采样率为0")
        duration = frames / float(rate)
    return duration

def get_audio_duration_seconds(path):
    """获取任意 ffmpeg/librosa 支持的音频文件时长（秒）。

    - 对 .wav 使用 stdlib wave 模块（零依赖、快速）
    - 对其它格式回退到 librosa.get_duration (librosa 已在项目依赖中)
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        return get_wav_duration_seconds(path)
    try:
        import librosa
        duration = float(librosa.get_duration(path=path))
        if duration <= 0:
            raise RuntimeError(f"librosa 返回非正时长: {duration}")
        return duration
    except Exception as e:
        raise RuntimeError(f"librosa 获取 {path} 时长失败: {e}")

def format_hms(total_seconds):
    total_secs = int(round(total_seconds))
    hours = total_secs // 3600
    minutes = (total_secs % 3600) // 60
    seconds = total_secs % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def format_hms_filename(total_seconds):
    return format_hms(total_seconds).replace(":", "-")

def _sampwidth_to_ffmpeg_codec(sampwidth: int) -> str:
    """将 WAV sampwidth (字节数) 映射到 ffmpeg PCM 编码器名。

    Args:
        sampwidth: 每样本字节数 (2=16-bit, 3=24-bit, 4=32-bit float)

    Returns:
        ffmpeg 编码器名 (如 'pcm_s16le', 'pcm_s24le', 'pcm_f32le')

    Raises:
        ValueError: 不支持的 sampwidth
    """
    mapping = {
        2: "pcm_s16le",
        3: "pcm_s24le",
        4: "pcm_f32le",
    }
    codec = mapping.get(sampwidth)
    if codec is None:
        raise ValueError(f"不支持的样本宽度: {sampwidth} 字节 (支持: {list(mapping.keys())})")
    return codec


def bit_depth_to_ffmpeg_codec(bit_depth: int) -> tuple[str, str]:
    """将目标位深映射到 ffmpeg PCM 编码器名和 soundfile subtype 名。

    Args:
        bit_depth: 位深 (16, 24, 32)

    Returns:
        (ffmpeg_codec, soundfile_subtype) 元组

    Raises:
        ValueError: 不支持的位深
    """
    mapping = {
        16: ("pcm_s16le", "PCM_16"),
        24: ("pcm_s24le", "PCM_24"),
        32: ("pcm_f32le", "FLOAT"),
    }
    entry = mapping.get(bit_depth)
    if entry is None:
        raise ValueError(f"不支持的目标位深: {bit_depth} (支持: {list(mapping.keys())})")
    return entry


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

def merge_wavs(output_path, wav_infos, silence_duration: float):
    """
    使用 ffmpeg concat 来合并音频，避免一次性加载大量 PCM 到内存中。
    """
    if not wav_infos:
        raise ValueError("没有 wav 可合并")

    base = wav_infos[0]["params"]
    nch = base["nchannels"]
    fr = base["framerate"]
    codec = _sampwidth_to_ffmpeg_codec(base["sampwidth"])

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
                "-ar", str(fr), "-ac", str(nch), "-c:a", codec, silence_path
            ]
            try:
                subprocess.run(cmd_silence, check=True, timeout=300)
            except subprocess.TimeoutExpired as e:
                import logging
                logging.getLogger(__name__).warning(f"生成静音音频超时: {e}")
                raise RuntimeError("生成静音音频超时") from e

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
            "-ar", str(fr), "-ac", str(nch), "-c:a", codec, output_path
        ]
        try:
            subprocess.run(cmd, check=True, timeout=300)
        except subprocess.TimeoutExpired as e:
            import logging
            logging.getLogger(__name__).warning(f"ffmpeg 合并超时: {e}")
            raise RuntimeError("ffmpeg 合并超时") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg 合并失败: {e}")
    finally:
        for p in tmp_files_to_remove:
            try:
                os.remove(p)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"清理临时文件失败: {p}, 错误: {e}")


def confirm_output_dir(base_output_dir: str) -> bool:
    """Check if base_output_dir exists; if so, ask whether to overwrite.

    Returns True if safe to proceed (dir doesn't exist or user chose to overwrite).
    Returns False if user declined to overwrite — caller should exit.
    When overwriting, deletes the entire base_output_dir with shutil.rmtree.
    """
    import shutil

    if not os.path.isdir(base_output_dir):
        return True

    print(f"输出目录已存在: {base_output_dir}")
    answer = input("是否覆盖已有输出? (y/N): ").strip().lower()
    if answer != "y":
        print("请更改输入文件夹名以避免输出冲突，然后重新运行。")
        return False

    shutil.rmtree(base_output_dir)
    print(f"已删除旧输出目录: {base_output_dir}")
    return True
