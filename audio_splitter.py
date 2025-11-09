import os
import sys
import contextlib
import math
import shutil
import wave
import logging
import time
import subprocess
import json

import torch
import librosa
import numpy as np

import whisperx

import audio_utils as au

logger = logging.getLogger(__name__)

def split_wav_by_times(src_wav, split_times, outdir, base_name):
    with contextlib.closing(wave.open(src_wav, "rb")) as wf:
        nch = wf.getnchannels()
        fr = wf.getframerate()
    split_times = sorted(split_times or [])
    split_times = [0.0] + split_times + [None]
    out_wav_paths = []
    for i in range(len(split_times) - 1):
        start, end = split_times[i], split_times[i+1]
        out_path = os.path.join(outdir, f"{base_name}_part{i+1}.wav")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
        if start and start > 0: cmd += ["-ss", f"{start:.3f}"]
        cmd += ["-i", src_wav]
        if end is not None: cmd += ["-to", f"{end:.3f}"]
        cmd += ["-ar", str(fr), "-ac", str(nch), "-c:a", "pcm_s16le", out_path]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg 切分失败: {e}")
        out_wav_paths.append(out_path)
    return out_wav_paths


class AudioSplitter:
    def __init__(self, config, logger_arg=None):
        self.config = config
        self.logger = logger_arg or logger
        
        split_config = config.get('split_and_merge', {}).get('split', {})
        
        device_setting = split_config.get('device', 'auto')
        if device_setting == 'auto':
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device_setting
        
        self.lang = split_config.get('language', 'zh')
        self.max_duration = split_config.get('max_duration', 15.0)
        self.min_duration = split_config.get('min_duration', 2.0)
        self.max_segments = split_config.get('max_segments', 8)
        
        self.split_min_silence_len_ms = split_config.get('split_min_silence_len_ms', 300)
        self.split_offset_from_silence_end_ms = split_config.get('split_offset_from_silence_end_ms', 0)
        self.split_silence_top_db = split_config.get('split_silence_top_db', 40)

        self.align_model = None
        self.align_metadata = None

    def _load_align_model(self):
        if self.align_model is None:
            self.logger.info(f"Loading whisperX alignment model for language: {self.lang}...")
            print(f"[INFO] Loading whisperX alignment model for '{self.lang}' on device '{self.device}'.")
            try:
                self.align_model, self.align_metadata = whisperx.load_align_model(language_code=self.lang, device=self.device)
            except Exception as e:
                self.logger.error(f"Failed to load whisperX alignment model: {e}", exc_info=True)
                print(f"[ERROR] Failed to load whisperX alignment model: {e}", file=sys.stderr)
                raise

    def _align(self, audio_path, transcript):
        self.logger.info(f"Running whisperX alignment for {os.path.basename(audio_path)}")
        try:
            audio = whisperx.load_audio(audio_path)
            duration = len(audio) / whisperx.audio.SAMPLE_RATE

            # Directly align the provided transcript by creating a single segment that spans the whole audio file
            segments_to_align = [{
                "text": transcript,
                "start": 0.0,
                "end": duration
            }]

            aligned_result = whisperx.align(segments_to_align, self.align_model, self.align_metadata, audio, self.device, return_char_alignments=True)
            return aligned_result
        except Exception as e:
            self.logger.error(f"WhisperX alignment failed for {audio_path}: {e}", exc_info=True)
            print(f"[ERROR] WhisperX alignment failed for {os.path.basename(audio_path)}: {e}. See log for details.", file=sys.stderr)
            return None


    def _find_silence_spans(self, audio_path):
        self.logger.info(f"Finding silence spans in {os.path.basename(audio_path)}")
        try:
            audio, sample_rate = librosa.load(audio_path, sr=None, mono=True)
            
            # Find non-silent chunks
            non_silent_chunks = librosa.effects.split(audio, top_db=self.split_silence_top_db, frame_length=2048, hop_length=512)
            
            # Invert to get silent chunks
            silent_spans = []
            last_end = 0
            for start, end in non_silent_chunks:
                if start > last_end:
                    silent_spans.append([last_end / sample_rate, start / sample_rate])
                last_end = end
            
            total_dur = len(audio) / sample_rate
            if total_dur > last_end / sample_rate:
                silent_spans.append([last_end / sample_rate, total_dur])

            # Filter by min duration
            min_silence_sec = self.split_min_silence_len_ms / 1000.0
            valid_silences = [s for s in silent_spans if (s[1] - s[0]) >= min_silence_sec]
            self.logger.info(f"Found {len(valid_silences)} valid silence spans.")
            return valid_silences
        except Exception as e:
            self.logger.error(f"Failed to find silence spans in {audio_path}: {e}", exc_info=True)
            return []

    def _plan_splits(self, total_dur, silence_spans):
        potential_cuts = sorted(
            [{'time': s[1], 'score': s[1] - s[0]} for s in silence_spans],
            key=lambda x: x['time']
        )

        # dp[t] = (min_cuts, max_score_for_min_cuts, list_of_split_times)
        dp = {0.0: (0, 0.0, [])}
        sorted_dp_keys = [0.0]

        for cut_point in potential_cuts:
            t_end = cut_point['time']
            score = cut_point['score']
            
            for t_start in sorted_dp_keys:
                if t_start >= t_end:
                    continue
                
                segment_duration = t_end - t_start
                if self.min_duration <= segment_duration <= self.max_duration:
                    prev_cuts, prev_score, prev_splits = dp[t_start]
                    new_cuts = prev_cuts + 1
                    new_score = prev_score + score
                    
                    current_cuts, current_score, _ = dp.get(t_end, (float('inf'), -1.0, []))

                    # Primary goal: fewer cuts. Secondary goal: better score.
                    if new_cuts < current_cuts:
                        dp[t_end] = (new_cuts, new_score, prev_splits + [t_end])
                    elif new_cuts == current_cuts and new_score > current_score:
                        dp[t_end] = (new_cuts, new_score, prev_splits + [t_end])

            if t_end in dp and t_end not in sorted_dp_keys:
                sorted_dp_keys.append(t_end)
                sorted_dp_keys.sort()

        # Find the best overall plan
        best_plan = None
        min_total_cuts = float('inf')
        max_final_score = -1.0

        # Check the no-cut scenario first
        if self.min_duration <= total_dur <= self.max_duration:
            min_total_cuts = 0
            max_final_score = 0.0
            best_plan = []

        for t, (cuts, score, splits) in dp.items():
            if self.min_duration <= (total_dur - t) <= self.max_duration:
                if cuts < min_total_cuts:
                    min_total_cuts = cuts
                    max_final_score = score
                    best_plan = splits
                elif cuts == min_total_cuts and score > max_final_score:
                    max_final_score = score
                    best_plan = splits

        if best_plan is not None:
            if best_plan and best_plan[-1] >= total_dur:
                 best_plan = best_plan[:-1]
            self.logger.info(f"Found best plan with {len(best_plan)} cuts.")
        else:
            self.logger.warning("Could not find a valid split plan.")
            
        return best_plan

    def _adjust_punctuation(self, text_parts):
        self.logger.info("Adjusting punctuation between split segments...")
        punctuations = ".,!?;:。，！？；：…"
        for i in range(1, len(text_parts)):
            # If the current sentence is not empty and starts with punctuation,
            # and the previous sentence is not empty.
            if text_parts[i] and text_parts[i][0] in punctuations and text_parts[i-1]:
                punc = text_parts[i][0]
                # Move punctuation from current sentence start to previous sentence end
                text_parts[i-1] += punc
                text_parts[i] = text_parts[i][1:].lstrip()
                self.logger.info(f"Moved punctuation '{punc}' from start of segment {i+1} to end of segment {i}.")
        return text_parts

    def _split_single(self, audio_path, transcript, outdir):
        audio_path = os.path.abspath(audio_path)
        outdir = os.path.abspath(outdir)
        os.makedirs(outdir, exist_ok=True)
        total_dur = au.get_wav_duration_seconds(audio_path)

        if total_dur <= self.max_duration:
            self.logger.info(f"Audio is short enough, no split needed for {audio_path}.")
            print(f"[INFO] Audio is short enough, no split needed for {os.path.basename(audio_path)}.")
            base = os.path.splitext(os.path.basename(audio_path))[0]
            out_wav = os.path.join(outdir, base + ".wav")
            out_lab = os.path.join(outdir, base + ".lab")
            if not os.path.exists(out_wav):
                shutil.copy(audio_path, out_wav)
                with open(out_lab, "w", encoding="utf-8") as f:
                    f.write(transcript)
            return [out_wav]

        split_times = None
        aligned_result = None

        # 优先尝试智能切分
        align_start_time = time.time()
        aligned_result = self._align(audio_path, transcript)
        align_duration = time.time() - align_start_time
        if aligned_result:
            self.logger.info(f"WhisperX Alignment for '{os.path.basename(audio_path)}' took {align_duration:.2f} seconds.")
            silence_spans = self._find_silence_spans(audio_path)
            if silence_spans:
                planned_splits = self._plan_splits(total_dur, silence_spans)
                if planned_splits is not None:
                    offset_s = self.split_offset_from_silence_end_ms / 1000.0
                    split_times = [t + offset_s for t in planned_splits]
                    self.logger.info(f"Final split times: {[round(t, 2) for t in split_times]}")
                else:
                    self.logger.warning(f"Could not find an optimal split plan for {audio_path}. Falling back.")
            else:
                self.logger.warning(f"No silence spans found in {audio_path}. Falling back.")
        else:
            self.logger.error(f"Alignment failed for {audio_path}. Falling back.")

        # 如果智能切分失败，则执行新的后备逻辑
        if split_times is None:
            self.logger.info(f"Smart splitting failed. Falling back to even splitting, prioritizing max_duration.")
            
            # 1. 计算满足 max_duration 所需的最小段数
            num_segs = math.ceil(total_dur / self.max_duration)
            
            # 2. 如果需要，警告用户将超出 max_segments
            if num_segs > self.max_segments:
                self.logger.warning(f"To satisfy max_duration ({self.max_duration}s), audio will be split into {num_segs} segments, exceeding configured max_segments ({self.max_segments}).")

            if num_segs <= 1:
                self.logger.warning(f"Fallback calculation resulted in {num_segs} segments. The audio will not be split.")
                print(f"[WARN] Cannot find a valid even split for {os.path.basename(audio_path)}. The audio will not be split.")
                base = os.path.splitext(os.path.basename(audio_path))[0]
                out_wav = os.path.join(outdir, base + "_part1.wav")
                out_lab = os.path.join(outdir, base + "_part1.lab")
                if not os.path.exists(out_wav):
                    shutil.copy(audio_path, out_wav)
                    with open(out_lab, "w", encoding="utf-8") as f:
                        f.write(transcript)
                return [out_wav]
            
            # 3. 执行均匀切分
            self.logger.info(f"Fallback: Splitting into {num_segs} equal parts.")
            split_times = [(total_dur / num_segs) * (i + 1) for i in range(num_segs - 1)]
            aligned_result = None # 均匀切分时，无法使用之前的对齐结果

        base = os.path.splitext(os.path.basename(audio_path))[0]
        try:
            out_wavs = split_wav_by_times(audio_path, split_times, outdir, base)
        except Exception as e:
            self.logger.error(f"Failed to split audio {audio_path}: {e}", exc_info=True)
            raise

        all_chars = []
        if aligned_result and aligned_result.get("segments"):
            all_chars = [char_info for segment in aligned_result["segments"] for char_info in segment.get("chars", [])]

        split_boundaries = [0] + split_times + [total_dur]
        text_parts = []
        for i in range(len(out_wavs)):
            start_t, end_t = split_boundaries[i], split_boundaries[i+1]
            text_part = ""
            
            if all_chars:
                # Find the first and last character index for the segment
                start_char_idx = -1
                end_char_idx = -1
                
                for idx, char_info in enumerate(all_chars):
                    if start_t <= char_info["start"] and start_char_idx == -1:
                        start_char_idx = idx
                    if char_info["start"] < end_t:
                        end_char_idx = idx

                if start_char_idx != -1:
                    # Find corresponding text in original transcript
                    # This is tricky. Let's just join the characters from alignment.
                    part_chars = [c['char'] for c in all_chars[start_char_idx:end_char_idx+1]]
                    text_part = "".join(part_chars).strip()

            if not text_part:
                self.logger.info("Character-based text split failed. Falling back to proportional text split.")
                start_char_pos = int(len(transcript) * (start_t / total_dur))
                end_char_pos = int(len(transcript) * (end_t / total_dur))
                text_part = transcript[start_char_pos:end_char_pos].strip()
            
            text_parts.append(text_part)

        # Adjust punctuation between segments
        adjusted_text_parts = self._adjust_punctuation(text_parts)

        # Write the final lab files
        for i, text_part in enumerate(adjusted_text_parts):
            out_lab = os.path.splitext(out_wavs[i])[0] + ".lab"
            with open(out_lab, "w", encoding="utf-8") as f:
                f.write(text_part)

        map_path = os.path.join(outdir, "切分记录.txt")
        try:
            with open(map_path, "a", encoding="utf-8") as mf:
                out_basenames = ", ".join(os.path.basename(p) for p in out_wavs)
                split_times_str = ", ".join(f"{t:.3f}s" for t in split_times)
                mf.write(f"{out_basenames} <- {os.path.basename(audio_path)}, splits=[{split_times_str}]\n")
        except Exception as e:
            self.logger.error(f"切分记录写入失败: {e}")
            print("[ERROR] 切分记录写入失败")

        return out_wavs


    def split(self, pairs, outdir):
        self._load_align_model()
        results = {}
        for (audio_path, lab_text) in pairs:
            try:
                out_wavs = self._split_single(audio_path, lab_text, outdir)
                results[audio_path] = out_wavs
            except Exception as e:
                self.logger.error(f"Split failed for {audio_path}: {e}", exc_info=True)
                print(f"[WARN] Split failed for {audio_path}: {e}", file=sys.stderr)
                results[audio_path] = None
        return results

def split_batch(pairs, outdir, config, logger_arg=None):
    splitter = AudioSplitter(config, logger_arg=logger_arg)
    raw_results = splitter.split(pairs, outdir)
    wrapped = {}
    for audio, res in raw_results.items():
        if res is None:
            wrapped[audio] = {"paths": None, "error": "split failed"}
        else:
            wrapped[audio] = {"paths": res, "error": None}
    return wrapped
