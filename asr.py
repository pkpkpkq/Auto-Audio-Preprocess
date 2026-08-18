import json
import logging
import os
import re
import sys
from functools import lru_cache
import torch
from funasr import AutoModel
from audio_utils import setup_model_cache
from config_types import AsrConfig, load_asr_config

logger = logging.getLogger(__name__)

setup_model_cache()


# ── FunASR sub-backend detection ───────────────────────────────────

def _detect_funasr_model_type(model_name: str) -> str:
    """根据模型名检测 FunASR 子后端类型。

    返回: "paraformer" | "sensevoice" | "nano"
    """
    name_lower = model_name.lower()
    if "sensevoice" in name_lower:
        return "sensevoice"
    if "fun-asr-nano" in name_lower:
        return "nano"
    return "paraformer"


# ── FunASR model loading ───────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_funasr_model(device, model_name="paraformer-zh", use_vad=False, vad_max_single_segment_time=30000,
                      trust_remote_code=False):
    model_kwargs = {"model": model_name, "device": device, "hub": "hf"}
    if trust_remote_code:
        model_kwargs["trust_remote_code"] = True
    else:
        model_kwargs["punc_model"] = "ct-punc"
    if use_vad:
        model_kwargs["vad_model"] = "fsmn-vad"
        model_kwargs["vad_kwargs"] = {"max_single_segment_time": vad_max_single_segment_time}
    return AutoModel(**model_kwargs)


@lru_cache(maxsize=1)
def _get_punc_model():
    return AutoModel(model="ct-punc")


# ── FunASR transcription ──────────────────────────────────────────

_SENSEVOICE_TAG_RE = re.compile(r'<\|[^|]*\|>')


_NANO_LANG_MAP = {
    "zh": "中文", "en": "英文", "ja": "日文",
    "yue": "粤语", "ko": "韩文",
}


def _resolve_language(language: str, model_type: str) -> str | None:
    """将 config 中的通用语言代码转为模型特定值。

    paraformer-zh: 不接受 language 参数 → 返回 None
    SenseVoice: 直接使用 ISO 代码 → 原样返回
    Nano: 转为中文全称 → 返回映射值 (auto/nospeech 不传 language)
    """
    if model_type == "paraformer":
        return None
    if model_type == "nano":
        if language in ("auto", "nospeech"):
            return None
        return _NANO_LANG_MAP.get(language)
    return language


def _transcribe_funasr(wav_path: str, asr_config: AsrConfig) -> list[dict]:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_type = _detect_funasr_model_type(asr_config.funasr_model)

    trust_remote = model_type in ("sensevoice", "nano")
    model = _get_funasr_model(device, asr_config.funasr_model,
                              asr_config.funasr_use_vad,
                              asr_config.funasr_vad_max_single_segment_time,
                              trust_remote)

    gen_kwargs = {"batch_size_s": 300}
    lang = _resolve_language(asr_config.language, model_type)

    if model_type == "sensevoice":
        if lang is not None:
            gen_kwargs["language"] = lang
        gen_kwargs["use_itn"] = asr_config.funasr_itn
        if asr_config.force_align:
            gen_kwargs["output_timestamp"] = True
    elif model_type == "nano":
        gen_kwargs.pop("batch_size_s", None)
        gen_kwargs["batch_size"] = 1
        if lang is not None:
            gen_kwargs["language"] = lang
        gen_kwargs["itn"] = asr_config.funasr_itn

    res = model.generate(wav_path, **gen_kwargs)

    if not (res and isinstance(res, list) and "text" in res[0]):
        return []

    # Clean text per model type
    if model_type == "sensevoice":
        full_text = _SENSEVOICE_TAG_RE.sub('', res[0]["text"]).strip().replace(" ", "")
    else:
        full_text = res[0]["text"].replace(" ", "")

    if not full_text:
        return []

    if not asr_config.force_align:
        return [{"start": 0.0, "end": 0.0, "text": full_text}]

    # force_align=true
    char_timestamps = res[0].get("timestamp", [])
    if not char_timestamps:
        # paraformer / Nano 不生成 timestamp → 用 WhisperX forced alignment 补全
        aligned = _force_align_with_whisperx(wav_path, full_text, asr_config)
        if aligned is not None:
            return aligned
        return [{"start": 0.0, "end": 0.0, "text": full_text}]

    result = []
    for i, char in enumerate(full_text):
        if i >= len(char_timestamps):
            break
        start_ms, end_ms = char_timestamps[i]
        result.append({
            "start": float(start_ms) / 1000.0,
            "end": float(end_ms) / 1000.0,
            "text": char,
        })
    return result


# ── WhisperX backend ──────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_whisperx_model(model_name: str, device: str, language: str):
    import whisperx
    return whisperx.load_model(model_name, device, language=language)


@lru_cache(maxsize=1)
def get_whisperx_align_model(language: str, device: str):
    import whisperx
    return whisperx.load_align_model(language_code=language, device=device)

def _add_punctuation(text: str) -> str:
    if not text.strip():
        return text
    model = _get_punc_model()
    res = model.generate(text)
    if res and "text" in res[0]:
        return res[0]["text"]
    return text


def _force_align_with_whisperx(wav_path: str, text: str, asr_config: AsrConfig) -> list[dict] | None:
    """用 WhisperX forced alignment 对已有文本做字符级对齐。

    返回字符级 segment 列表，对齐失败时返回 None。
    """
    import whisperx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    align_device = asr_config.whisperx_align.device
    if align_device == "auto":
        align_device = device
    align_lang = asr_config.whisperx_align.language

    try:
        align_model, align_metadata = get_whisperx_align_model(align_lang, align_device)
    except Exception as e:
        logger.error(f"加载 WhisperX 对齐模型失败: {e}", exc_info=True)
        return None

    audio = whisperx.load_audio(wav_path)
    audio_duration = len(audio) / whisperx.audio.SAMPLE_RATE
    segments_to_align = [{"text": text, "start": 0.0, "end": audio_duration}]

    try:
        aligned = whisperx.align(
            segments_to_align, align_model, align_metadata, audio, align_device,
            return_char_alignments=True,
        )
    except Exception as e:
        logger.warning(f"WhisperX 对齐失败: {e}")
        return None

    char_segments = []
    for seg in aligned.get("segments", []):
        for char_info in seg.get("chars", []):
            char = char_info.get("char", "")
            if not char.strip():
                continue
            char_segments.append({
                "start": float(char_info.get("start", 0.0)),
                "end": float(char_info.get("end", 0.0)),
                "text": char,
            })

    return char_segments if char_segments else None


def _transcribe_whisperx(wav_path: str, asr_config: AsrConfig) -> list[dict]:
    import whisperx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _get_whisperx_model(asr_config.whisperx_model, device, asr_config.language)
    audio = whisperx.load_audio(wav_path)
    result = model.transcribe(audio, batch_size=asr_config.whisperx_batch_size,
                              language=asr_config.language)

    raw_text = "".join(seg.get("text", "").strip() for seg in result.get("segments", []))
    text = _add_punctuation(raw_text)
    text = text.replace(" ", "")

    if not text:
        return []

    if not asr_config.force_align:
        return [{"start": 0.0, "end": 0.0, "text": text}]

    aligned = _force_align_with_whisperx(wav_path, text, asr_config)
    if aligned is not None:
        return aligned
    return [{"start": 0.0, "end": 0.0, "text": text}]


# ── Public dispatch ──────────────────────────────────────────────

def transcribe_audio(wav_path: str, asr_config: AsrConfig) -> list[dict]:
    match asr_config.backend:
        case "funasr":
            return _transcribe_funasr(wav_path, asr_config)
        case "whisperx":
            return _transcribe_whisperx(wav_path, asr_config)
        case _:
            raise ValueError(f"未知的 ASR 后端: {asr_config.backend}")


# ── Model lifecycle (unload) ──────────────────────────────────────
# ponytail: lru_cache cache_clear 一刀全清；本项目单进程串行 + 单 backend active 时无副作用。
# 切多 backend 并发 / hot-reload 时升级到 refcount registry。

def unload_asr_models():
    """释放 ASR 主体模型 (funasr/punc/whisperx_model)。
    保留 align 模型 — split 阶段 (audio_splitter) 会复用。
    """
    _get_funasr_model.cache_clear()
    _get_punc_model.cache_clear()
    _get_whisperx_model.cache_clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unload_align_model():
    """释放 WhisperX 对齐模型 — asr force_align 与 audio_splitter 共享。"""
    get_whisperx_align_model.cache_clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unload_models():
    """释放 ASR 全部模型 (含 align)。"""
    unload_asr_models()
    unload_align_model()


# ── Standalone test ───────────────────────────────────────────────

if __name__ == "__main__":
    # Ponytail smoke check: unload API surface must be callable + work on empty caches.
    assert all(callable(fn) for fn in (unload_models, unload_asr_models, unload_align_model))
    unload_models()  # no-op on empty caches; fails if any underlying cache_clear breaks

    if len(sys.argv) < 3:
        print("Usage: python asr.py <wav_path> <output_json>")
        sys.exit(1)

    wav_path = sys.argv[1]
    output_json = sys.argv[2]

    try:
        result = transcribe_audio(wav_path, load_asr_config())
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception as e:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump({"error": str(e)}, f, ensure_ascii=False)
        sys.exit(1)
