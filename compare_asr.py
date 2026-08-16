#!/usr/bin/env python3
"""临时跨模型 ASR 对比脚本 — paraformer-zh / Nano / SenseVoice / WhisperX v2 / v3

沿用 config.yaml 配置，仅切换模型。对每个模型：
  - 跑 ASR (force_align=True, 统一返回 char 级 timestamp)
  - 将 lab.txt 用 WhisperX 对齐到音频 → 标准时间戳 (ground truth)
  - 计算 CER + 错字列表
  - 对比模型时间戳 vs 标准时间戳
  - 输出控制台报告 + 文本文件 + JSON
"""

import sys
import os
import json
import math
import copy
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_types import load_pipeline_config, AsrConfig
from asr import transcribe_audio, _force_align_with_whisperx, unload_models

# ── paths ──────────────────────────────────────────────────────────
WAV = r"D:\Justin\programs\AudioPreProcess\input\芙宁娜-emo\vo_furina_dialog_pendant_01.wav"
LAB = r"D:\Justin\programs\AudioPreProcess\input\芙宁娜-emo\vo_furina_dialog_pendant_01.lab"
OUT_REPORT = r"D:\Justin\programs\AudioPreProcess\output\asr_compare\report.txt"
OUT_JSON   = r"D:\Justin\programs\AudioPreProcess\output\asr_compare\report.json"

# ── helpers ────────────────────────────────────────────────────────

def levenshtein_align(ref: str, hyp: str):
    """Levenshtein distance, 返回:
    - distance: int
    - ops: list of ("M"|"S"|"I"|"D", ref_char_or_None, hyp_char_or_None)
    """
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # delete
                dp[i][j - 1] + 1,      # insert
                dp[i - 1][j - 1] + cost  # match/substitute
            )
    # backtrack
    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ((ref[i - 1] == hyp[j - 1]) or dp[i][j] == dp[i - 1][j - 1] + 1):
            kind = "M" if ref[i - 1] == hyp[j - 1] else "S"
            ops.append((kind, ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("D", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("I", None, hyp[j - 1]))
            j -= 1
    ops.reverse()
    return dp[m][n], ops


def segs_to_text(segs: list[dict]) -> str:
    return "".join(s.get("text", "") for s in segs)


# ── main ───────────────────────────────────────────────────────────

def main():
    cfg = load_pipeline_config()
    base_asr = cfg.asr

    # read lab
    with open(LAB, "r", encoding="utf-8") as f:
        lab_text = f.read().strip()

    print("=== 加载配置 & lab ===")
    print(f"  lab 长度: {len(lab_text)} 字符")
    print(f"  lab 内容: {lab_text[:80]}...")

    # ── Ground truth: WhisperX align(lab) → 标准时间戳 ──────────────
    print("\n=== Ground Truth: WhisperX 对齐 lab 到音频 ===")
    align_cfg = copy.copy(base_asr)
    align_cfg = replace(align_cfg,
        whisperx_align=replace(align_cfg.whisperx_align, device="auto", language="zh")
    )
    gt_segs = _force_align_with_whisperx(WAV, lab_text, align_cfg)
    if not gt_segs:
        print("  [FATAL] Ground truth 对齐失败，退出。")
        sys.exit(1)
    gt_text = segs_to_text(gt_segs)
    print(f"  GT 对齐后字符数: {len(gt_text)}  (lab={len(lab_text)})")
    unload_models()

    # ── 5 个模型 ────────────────────────────────────────────────────
    model_defs = [
        {"name": "paraformer-zh",       "backend": "funasr", "funasr_model": "paraformer-zh"},
        {"name": "Fun-ASR-Nano-2512",   "backend": "funasr", "funasr_model": "FunAudioLLM/Fun-ASR-Nano-2512"},
        {"name": "SenseVoiceSmall",     "backend": "funasr", "funasr_model": "FunAudioLLM/SenseVoiceSmall"},
        {"name": "WhisperX large-v2",   "backend": "whisperx", "whisperx_model": "large-v2", "whisperx_batch_size": 16},
        {"name": "WhisperX large-v3",   "backend": "whisperx", "whisperx_model": "large-v3", "whisperx_batch_size": 16},
    ]

    results = []
    lines = []

    def emit(*args):
        text = " ".join(str(a) for a in args)
        print(text)
        lines.append(text)

    emit(f"音频: {WAV}")
    emit(f"lab : {LAB}")
    emit(f"lab 文本 ({len(lab_text)} 字): {lab_text}")
    emit(f"lab 文本 (Ground Truth 对齐后 {len(gt_text)} 字): {gt_text}")
    emit("")

    for mdef in model_defs:
        name = mdef["name"]
        emit(f"{'='*60}")
        emit(f"  模型: {name}")
        emit(f"{'='*60}")

        # Build model-specific AsrConfig
        asr_cfg = copy.copy(base_asr)
        asr_cfg.backend = mdef["backend"]
        asr_cfg.funasr_model = mdef.get("funasr_model", base_asr.funasr_model)
        if "whisperx_model" in mdef:
            asr_cfg.whisperx_model = mdef["whisperx_model"]
        if "whisperx_batch_size" in mdef:
            asr_cfg.whisperx_batch_size = mdef["whisperx_batch_size"]

        # Run ASR with force_align=True
        emit("  正在转写 (force_align=True)...")
        try:
            segs = transcribe_audio(WAV, asr_cfg)
        except Exception as e:
            emit(f"  [ERROR] ASR 失败: {e}")
            results.append({"name": name, "error": str(e)})
            emit("")
            continue
        unload_models()
        asr_text = segs_to_text(segs)
        emit(f"  转写结果: {len(segs)} segments, {len(asr_text)} 字符")
        emit(f"  ASR 文本: {asr_text[:200]}...")

        # CER
        dist, ops = levenshtein_align(lab_text, asr_text)
        cer = dist / len(lab_text) if len(lab_text) > 0 else 1.0
        subs = [(r, h) for kind, r, h in ops if kind == "S"]
        ins  = [h for kind, r, h in ops if kind == "I" and h is not None]
        dels = [r for kind, r, h in ops if kind == "D" and r is not None]
        emit(f"  CER = {dist}/{len(lab_text)} = {cer:.4f} ({cer*100:.2f}%)")
        emit(f"    替换 (S): {len(subs)} 字 → {', '.join(f'{r}→{h}' for r, h in subs[:10])}{'...' if len(subs)>10 else ''}")
        emit(f"    插入 (I): {len(ins)} 字 → {', '.join(ins[:10])}{'...' if len(ins)>10 else ''}")
        emit(f"    删除 (D): {len(dels)} 字 → {', '.join(dels[:10])}{'...' if len(dels)>10 else ''}")

        # Timestamp comparison: align ASR_segs ↔ GT_segs via Levenshtein
        # Levenshtein 对齐 asr_text 到 gt_text, 然后把 op 序列映射回 segment 索引
        asr_chars = []
        for si, seg in enumerate(segs):
            for ch in seg.get("text", ""):
                asr_chars.append((ch, seg["start"], seg["end"], si))

        gt_chars = []
        for si, seg in enumerate(gt_segs):
            for ch in seg.get("text", ""):
                gt_chars.append((ch, seg["start"], seg["end"], si))

        # Align asr_text to gt_text
        dist_gt, ops_gt = levenshtein_align(gt_text, asr_text)

        offsets_ms = []
        aligned_pairs = 0
        gt_idx = 0
        asr_idx = 0
        for kind, r_ch, h_ch in ops_gt:
            if kind == "M" or kind == "S":
                if gt_idx < len(gt_chars) and asr_idx < len(asr_chars):
                    gt_mid = (gt_chars[gt_idx][1] + gt_chars[gt_idx][2]) / 2
                    asr_mid = (asr_chars[asr_idx][1] + asr_chars[asr_idx][2]) / 2
                    offset = abs(gt_mid - asr_mid) * 1000  # ms
                    offsets_ms.append(offset)
                    aligned_pairs += 1
                gt_idx += 1
                asr_idx += 1
            elif kind == "D":
                gt_idx += 1
            elif kind == "I":
                asr_idx += 1

        if offsets_ms:
            mean_off = sum(offsets_ms) / len(offsets_ms)
            max_off = max(offsets_ms)
            emit(f"  时间戳对齐: {aligned_pairs} 对齐字符, 平均偏移 {mean_off:.1f}ms, 最大偏移 {max_off:.1f}ms")
        else:
            mean_off = max_off = 0.0
            emit(f"  时间戳对齐: 无对齐字符 (可能 ASR 文本完全不同)")

        result_entry = {
            "name": name,
            "backend": mdef["backend"],
            "asr_text": asr_text,
            "segments": segs,
            "cer": cer,
            "edit_distance": dist,
            "ref_length": len(lab_text),
            "substitutions": [{"ref": r, "hyp": h} for r, h in subs],
            "insertions": ins,
            "deletions": dels,
            "timestamp_alignment": {
                "aligned_chars": aligned_pairs,
                "mean_offset_ms": round(mean_off, 1),
                "max_offset_ms": round(max_off, 1),
            },
        }
        results.append(result_entry)
        emit("")

    # ── Summary ─────────────────────────────────────────────────────
    emit(f"{'='*60}")
    emit("  汇总")
    emit(f"{'='*60}")
    emit(f"  {'模型':<25} {'CER':>8} {'字错':>6} {'替换':>5} {'插入':>5} {'删除':>5} {'时间偏移(ms) avg/max':>25}")
    emit(f"  {'-'*25} {'-'*8} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*25}")
    for r in results:
        if "error" in r:
            emit(f"  {r['name']:<25} [ERROR: {r['error'][:40]}]")
        else:
            ta = r["timestamp_alignment"]
            emit(f"  {r['name']:<25} {r['cer']*100:>7.2f}% {r['edit_distance']:>5}  "
                 f"{len(r['substitutions']):>4}  {len(r['insertions']):>4}  {len(r['deletions']):>4}  "
                 f"{ta['mean_offset_ms']:>6.1f} / {ta['max_offset_ms']:>6.1f}")

    # ── Write outputs ───────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT_REPORT), exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n文本报告已写入: {OUT_REPORT}")

    # JSON: exclude verbose segments from root, save separately
    json_out = {
        "audio": WAV,
        "lab_text": lab_text,
        "ground_truth_segments": gt_segs,
        "models": [],
    }
    for r in results:
        entry = {k: v for k, v in r.items() if k != "segments"}
        entry["segments_count"] = len(r.get("segments", []))
        json_out["models"].append(entry)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(json_out, f, ensure_ascii=False, indent=2)
    print(f"JSON 已写入: {OUT_JSON}")


if __name__ == "__main__":
    main()