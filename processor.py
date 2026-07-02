import os
import shutil
import logging
import sys
import difflib
import re
from collections import Counter, defaultdict

# Local module imports
from asr import transcribe_audio
from audio_splitter import AudioSplitter
import audio_utils as au
import audio_normalizer as norm
import audio_validator as val
from config_types import PipelineConfig, NormalizerConfig, SplitConfig, SerConfig, AsrConfig, CheckPipelineConfig, AuditCheckConfig, ClippingCheckConfig, SilenceRatioCheckConfig, SnrCheckConfig, AudioEntry, PIPELINE_STAGES

class DatasetProcessor:
    """封装整个数据集处理流程的类"""

    def __init__(self, config: PipelineConfig):
        """
        初始化处理器。
        :param config: 完整的 PipelineConfig 实例。
        """
        self._cfg = config
        self.root_dir = os.path.abspath(config.root_dir)
        
        if not os.path.isdir(self.root_dir):
            raise ValueError(f"配置中指定的根目录不存在或不是目录: {self.root_dir}")

        self.recursive = config.recursive
        self.use_lab_text_as_filename = config.use_lab_text_as_filename
        self.output_to_split_folder = config.output_to_split_folder

        self.replace_map = config.text_replace_map
        self.t2s_enabled = config.text_t2s

        self.asr_config = config.asr
        
        self._ser_config = config.ser
        self.ser_enabled = config.ser.enabled
        self.ser_recognizer = None
        # Lazy init: model loaded on first use

        self._norm_config = config.normalizer
        self._check_config = config.checks
        self._split_config = config.split

        self.jsonl_output_enabled = config.output_jsonl

        self.folder_name = os.path.basename(self.root_dir.rstrip(os.sep))
        if not self.folder_name:
            raise ValueError(f"无法从根目录派生文件夹名称: {self.root_dir}")

        # 路径和日志记录器将在 setup 阶段初始化
        self.base_output_dir = None
        self.output_dir = None
        self.split_output_dir = None # 切分文件专属输出目录
        self.temp_dir = None
        self.auto_merge_dir = None
        self.auto_split_dir = None
        self.log_path = None
        self.logger = None

    def _ensure_ser_model(self):
        if self.ser_enabled and self.ser_recognizer is None:
            from ser import SpeechEmotionRecognizer
            self.ser_recognizer = SpeechEmotionRecognizer(self._ser_config)

    def _run_ser_on_entries(self, entries):
        if not self.ser_enabled or not entries:
            return
        self._ensure_ser_model()
        self.ser_recognizer.predict_emotion_batch(entries)

    def _isolate_entries(self, entries, subfolder_name):
        if not entries:
            return
        isolate_dir = os.path.join(self.output_dir, subfolder_name)
        os.makedirs(isolate_dir, exist_ok=True)
        for entry in entries:
            try:
                dest = os.path.join(isolate_dir, os.path.basename(entry.path))
                if os.path.exists(entry.path):
                    shutil.copy2(entry.path, dest)
                self.logger.info(f"已隔离: {os.path.basename(entry.path)} → {subfolder_name}/")
            except Exception as e:
                self.logger.error(f"隔离文件 {entry.path} 失败: {e}")
        self.logger.info(f"共隔离 {len(entries)} 个文件到 {subfolder_name}/")

    # ---- CJK + ASCII 标点与空白字符统一正则 ----
    # norm_to_full_map 构建和 normalize() 必须使用同一套过滤规则，
    # 否则索引映射与归一化文本不对齐（原 Bug #1）。
    _PUNCT_WS_RE = re.compile(
        r'[\s'
        r'，。！？、'
        r',.!?…'
        r'\u201c\u201d\u2018\u2019'
        r'：;；（）()'
        r'《》〈〉【】'
        r'「」『』'
        r'—～·'
        r'｛｝'
        r'\[\]{}'
        r']'
    )

    def _extract_asr_for_placeholder(self, asr_segments, prefix, suffix,
                                     search_start_norm=0):
        """
        基于文本模糊匹配和索引映射的 {ASR} 占位符提取。

        :param asr_segments:      ASR 分段列表，每项含 "text" 键
        :param prefix:           {ASR} 前方的静态文本
        :param suffix:           {ASR} 后方的静态文本
        :param search_start_norm: 规范化文本中的搜索起始索引（多占位符偏移追踪）
        :return: 提取的文本，或 None 表示匹配失败
        """
        if not asr_segments:
            return None

        full_asr_text = "".join(seg["text"] for seg in asr_segments)

        # 1. 构建从规范文本索引到原始文本索引的映射
        #    仅保留不被 _PUNCT_WS_RE 匹配的字符，与 normalize() 保持一致
        norm_to_full_map = []
        for i, char_full in enumerate(full_asr_text):
            if not self._PUNCT_WS_RE.match(char_full):
                norm_to_full_map.append(i)

        # 2. 规范化：移除标点 + 空白，用于模糊匹配
        def normalize(s: str) -> str:
            if not s:
                return ""
            return self._PUNCT_WS_RE.sub('', s)

        norm_full_asr_text = normalize(full_asr_text)
        norm_prefix = normalize(prefix)
        norm_suffix = normalize(suffix)

        if not norm_full_asr_text:
            return None

        MIN_MATCH_RATIO = 0.7  # 模糊匹配的最小匹配率阈值

        # 辅助函数：在 target_norm 的 [lo, hi) 范围内查找 pattern 的最佳匹配
        def find_match_indices_in_norm(target_norm, pattern_norm, lo=0, hi=None):
            """返回 (norm_start, norm_end) 或 (None, None)"""
            if not pattern_norm:
                return None, None
            if hi is None:
                hi = len(target_norm)
            matcher = difflib.SequenceMatcher(None, target_norm, pattern_norm)
            match = matcher.find_longest_match(lo, hi, 0, len(pattern_norm))

            if match.size > 0 and (match.size / len(pattern_norm)) >= MIN_MATCH_RATIO:
                if match.b != 0:
                    self.logger.warning(
                        f"模糊匹配未锚定到模式起始 (offset={match.b}, "
                        f"size={match.size}/{len(pattern_norm)})，提取边界可能偏移")
                return match.a, match.a + match.size
            return None, None

        # 3. 在规范化文本上进行模糊匹配
        #    prefix：从 search_start_norm 开始搜索
        pfx_lo = search_start_norm
        prefix_start_norm, prefix_end_norm = find_match_indices_in_norm(
            norm_full_asr_text, norm_prefix, lo=pfx_lo)

        #    suffix：必须在 prefix 之后搜索 (Bug #3 修复)
        sfx_lo = prefix_end_norm if prefix_end_norm is not None else pfx_lo
        suffix_start_norm, suffix_end_norm = find_match_indices_in_norm(
            norm_full_asr_text, norm_suffix, lo=sfx_lo)

        # 将 suffix 匹配位置中与 prefix 重叠的部分排除
        if suffix_start_norm is not None and prefix_end_norm is not None:
            if suffix_start_norm < prefix_end_norm:
                # suffix 匹配区域与 prefix 重叠 → 尝试在 prefix 结束后重新搜索
                suffix_start_norm, suffix_end_norm = find_match_indices_in_norm(
                    norm_full_asr_text, norm_suffix, lo=prefix_end_norm)

        # 4. 将规范化文本的索引映射回原始文本
        start_idx_full = 0
        if prefix_end_norm is not None:
            if prefix_end_norm < len(norm_to_full_map):
                start_idx_full = norm_to_full_map[prefix_end_norm]
            else:
                start_idx_full = len(full_asr_text)

        end_idx_full = len(full_asr_text)
        if suffix_start_norm is not None:
            if suffix_start_norm < len(norm_to_full_map):
                end_idx_full = norm_to_full_map[suffix_start_norm]
            else:
                end_idx_full = len(full_asr_text)

        # 5. 确保索引有效且顺序正确
        if start_idx_full >= end_idx_full:
            return None

        # 6. 从原始文本中提取内容，并去除首尾边界标点 (Bug #8)
        _BOUNDARY_PUNCT = '，。！？、,.!?…""'':;；（）()《》〈〉【】「」『』—～·\t\n '
        extracted_text = full_asr_text[start_idx_full:end_idx_full].strip(_BOUNDARY_PUNCT)
        return extracted_text if extracted_text else None

    def _process_lab_with_asr(self, lab_path, wav_path):
        """
        支持 {ASR} 占位符的智能替换：仅替换占位符部分，保留上下文。
        支持同一 lab 中多个 {ASR} 占位符，使用预拆分 + 偏移追踪
        避免前一个替换结果污染后续占位符的上下文 (Bug #2 修复)。
        """
        with open(lab_path, "r", encoding="utf-8") as f:
            original_lab = f.read().strip()

        temp_lab = original_lab
        for k, v in self.replace_map.items():
            temp_lab = temp_lab.replace(k, v)

        if "{ASR}" not in temp_lab:
            return temp_lab

        self.logger.info(f"检测到 {{ASR}} 占位符，对 {wav_path} 执行 ASR...")

        # ===== 多 {ASR} 预拆分 (Bug #2 修复) =====
        # 1. 按 {ASR} 分割，得到纯静态文本段
        #    例如 "你好，{ASR}，欢迎来到{ASR}" → ["你好，", "，欢迎来到", ""]
        parts = temp_lab.split("{ASR}")
        n_placeholders = len(parts) - 1

        if n_placeholders == 0:
            # 不应发生（前面已检查 "{ASR}" in temp_lab），但做防御
            return temp_lab

        try:
            segments = transcribe_audio(wav_path, self.asr_config)

            if not segments:
                raise RuntimeError("ASR 未返回任何结果。")

            asr_full_text = "".join(seg["text"] for seg in segments).strip()
            asr_full_text = au.traditional_to_simplified(asr_full_text, self.t2s_enabled)
            self.logger.info(f"ASR 全文结果: {asr_full_text}")

            # 2. 逐个提取，追踪 ASR 文本消耗偏移
            replacements = []
            asr_consumed_norm = 0  # 规范化文本中已消耗的字符偏移

            for i in range(n_placeholders):
                # 每个占位符的上下文：上一段静态文本（prefix）+ 下一段静态文本（suffix）
                static_prefix = parts[i]
                static_suffix = parts[i + 1]

                extracted = self._extract_asr_for_placeholder(
                    segments, static_prefix, static_suffix,
                    search_start_norm=asr_consumed_norm
                )

                if extracted is not None:
                    replacements.append(extracted)
                    self.logger.info(f"第 {i+1} 个占位符提取成功: '{extracted}'")
                    # 更新偏移：推算此提取消耗了多少规范化文本
                    _norm_extracted = self._PUNCT_WS_RE.sub('', extracted)
                    # 静态 suffix 在 norm 空间中的匹配位置即下一次搜索起点
                    # 简化策略：累加提取长度 + suffix 估算偏移
                    norm_suffix = self._PUNCT_WS_RE.sub('', static_suffix)
                    asr_consumed_norm += len(_norm_extracted) + len(norm_suffix)
                else:
                    replacements.append("{ASR_FAILED}")
                    self.logger.warning(f"第 {i+1} 个占位符提取失败，保留失败标记")

            # 3. 重组：静态段 + 替换结果交替拼接
            final_parts = []
            for i, part in enumerate(parts):
                final_parts.append(part)
                if i < n_placeholders:
                    final_parts.append(replacements[i])
            final_lab = "".join(final_parts)

            return final_lab

        except Exception as e:
            error_msg = f"ASR 处理失败: {e}"
            self.logger.warning(error_msg)
            return temp_lab

    def _setup_paths_and_logging(self):
        """设置所有输出路径并配置日志记录。"""
        self.base_output_dir = os.path.join(os.path.abspath("output"), self.folder_name)
        self.output_dir = os.path.join(self.base_output_dir, self.folder_name)

        if not au.confirm_output_dir(self.base_output_dir):
            sys.exit(0)

        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[INFO] 主要音频输出将被保存到: {self.output_dir}")

        if self.output_to_split_folder:
            self.split_output_dir = os.path.join(self.output_dir, "切分后")
            os.makedirs(self.split_output_dir, exist_ok=True)
            print(f"[INFO] 切分后的音频将被保存到: {self.split_output_dir}")
        else:
            self.split_output_dir = self.output_dir # 如果不单独输出，则指向主输出目录
        
        print(f"[INFO] 日志和列表文件将被保存到: {self.base_output_dir}")

        self.temp_dir = os.path.join(self.output_dir, "_temp_processing")
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir)
        self.auto_merge_dir = self.temp_dir
        self.auto_split_dir = self.temp_dir

        self.log_path = os.path.join(self.base_output_dir, f"{self.folder_name}.log")
        # 将去重记录移动到角色文件夹下
        self.dedup_log_path = os.path.join(self.output_dir, "！去重记录.txt")

        self.logger = logging.getLogger("tts_preprocess")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            fh = logging.FileHandler(self.log_path, encoding="utf-8")
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        self.logger.info("="*50)
        self.logger.info("开始新的处理流程")
        self.logger.info("配置信息:")
        self.logger.info(f"  - 输入目录: {self.root_dir}")
        self.logger.info(f"  - 输出目录: {self.output_dir}")
        self.logger.info(f"  - 切分文件输出到 '切分后' 文件夹: {self.output_to_split_folder}")
        self.logger.info(f"  - 使用标注文本命名: {self.use_lab_text_as_filename}")
        self.logger.info(f"  - 情感识别 (SER) 已 {'启用' if self.ser_enabled else '禁用'} (后端: {self._ser_config.backend})")
        self.logger.info(f"  - 替换规则: {self.replace_map}")
        self.logger.info("="*50)

    @staticmethod
    def _pick_primary_audio(stem, dirpath, extmap, input_formats):
        """在同 stem 的多个音频格式中选定主音频。优先级：.wav > 其余按 input_formats 顺序。"""
        supported_exts = [f".{fmt.lower()}" for fmt in input_formats]
        for fmt_ext in (".wav",) + tuple(e for e in supported_exts if e != ".wav"):
            if fmt_ext in extmap:
                return os.path.join(dirpath, extmap[fmt_ext])
        return None

    def _find_audio_entries(self):
        """查找、配对、去重并读取所有音频条目及其元数据。"""
        import datetime
        stems = {}

        input_formats = self._cfg.normalizer.input_formats
        supported_exts = {f".{fmt.lower()}" for fmt in input_formats}
        supported_exts.add(".lab")

        if self.recursive:
            self.logger.info(f"开始递归扫描目录: {self.root_dir}")
            for dirpath, _, filenames in os.walk(self.root_dir):
                for fname in filenames:
                    name, ext = os.path.splitext(fname)
                    ext_lower = ext.lower()
                    if ext_lower in supported_exts:
                        # 使用相对路径和文件名作为唯一键，避免冲突
                        relative_path = os.path.relpath(os.path.join(dirpath, name), self.root_dir)
                        unique_stem_key = os.path.normpath(relative_path)
                        
                        if unique_stem_key not in stems:
                            stems[unique_stem_key] = {"dirpath": dirpath}
                        stems[unique_stem_key][ext_lower] = fname
        else:
            self.logger.info(f"开始扫描单个目录: {self.root_dir}")
            entries = os.listdir(self.root_dir)
            file_entries = [f for f in entries if os.path.isfile(os.path.join(self.root_dir, f))]
            for fname in file_entries:
                name, ext = os.path.splitext(fname)
                ext_lower = ext.lower()
                if ext_lower in supported_exts:
                    if name not in stems:
                        stems[name] = {"dirpath": self.root_dir}
                    stems[name][ext_lower] = fname
        
        self.logger.info(f"找到 {len(stems)} 个带有 {sorted(supported_exts)} 扩展名的唯一文件主干。")

        matched = []
        skipped_pairing = 0
        read_failures = []
        seen_texts = set()
        duplicate_log_lines = []
        skipped_rule_count = 0

        # 按文件名排序以确保去重行为是确定的
        stems_sorted = sorted(stems.items())

        for stem_key, extmap in stems_sorted:
            if ".lab" not in extmap:
                skipped_pairing += 1
                continue

            wav_path = self._pick_primary_audio(stem_key, extmap["dirpath"], extmap, self._cfg.normalizer.input_formats)
            if wav_path is None:
                skipped_pairing += 1
                continue

            dirpath = extmap["dirpath"]
            lab_filename = extmap[".lab"]
            lab_path = os.path.join(dirpath, lab_filename)

            try:
                lab_text = self._process_lab_with_asr(lab_path, wav_path)
            except Exception as e:
                read_failures.append(f"{lab_path} (处理 .lab): {e}")
                self.logger.warning(f"处理 .lab {lab_path} 失败: {e}")
                continue

            # {SKIP} rule check
            if "{SKIP}" in lab_text:
                self.logger.info(f"检测到 {{SKIP}} 规则，跳过文件: {os.path.basename(wav_path)}")
                skipped_rule_count += 1
                continue

            if "{ASR}" in lab_text or "{ASR_FAILED}" in lab_text:
                self.logger.warning(f"ASR 占位符未解析，跳过文件: {os.path.basename(wav_path)}")
                skipped_rule_count += 1
                continue

            # 去重逻辑
            if lab_text in seen_texts:
                log_line = f"丢弃 (文本内容重复): {os.path.basename(wav_path)}"
                duplicate_log_lines.append(log_line)
                self.logger.info(f"发现重复文本，跳过文件: {os.path.basename(wav_path)}")
                continue
            else:
                seen_texts.add(lab_text)

            try:
                duration = au.get_audio_duration_seconds(wav_path)
            except Exception as e:
                read_failures.append(f"{wav_path} (获取时长): {e}")
                self.logger.warning(f"获取 {wav_path} 的时长失败: {e}")
                continue

            emotion = "neutral"

            matched.append(AudioEntry(
                path=wav_path,
                lab_text=lab_text,
                duration=duration,
                emotion=emotion,
                stem=stem_key,
                lab_path=lab_path,
            ))

        if read_failures:
            print(f"[WARN] 文件读取期间遇到 {len(read_failures)} 个错误。详情请查看日志文件: {self.log_path}", file=sys.stderr)

        if duplicate_log_lines:
            try:
                with open(self.dedup_log_path, "a", encoding="utf-8") as f:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"\n--- 去重记录 ({timestamp}) ---\n")
                    f.write("以下文件因 .lab 文本内容重复而被跳过处理：\n")
                    f.write("\n".join(duplicate_log_lines))
                    f.write("\n")
                print(f"[INFO] 发现并跳过了 {len(duplicate_log_lines)} 个重复文件。详情请查看: {self.dedup_log_path}")
                self.logger.info(f"追加 {len(duplicate_log_lines)} 条重复记录到 {self.dedup_log_path}")
            except Exception as e:
                print(f"[WARN] 写入去重日志文件失败: {e}", file=sys.stderr)
                self.logger.error(f"写入去重日志文件失败: {e}")

        total_skipped = skipped_pairing + len(read_failures) + len(duplicate_log_lines) + skipped_rule_count
        self.logger.info(f"匹配到 {len(matched)} 对 .wav 和 .lab 文件。")
        if total_skipped > 0:
            self.logger.warning(f"共跳过了 {total_skipped} 个文件（{skipped_pairing} 个缺少配对, {len(read_failures)} 个读取失败, {len(duplicate_log_lines)} 个内容重复, {skipped_rule_count} 个触发SKIP规则）。")

        return matched

    def _normalize_audio(self, all_entries):
        if not self._cfg.normalizer_enabled:
            return all_entries

        trim_enabled = self._norm_config.trim_enabled
        loudness_enabled = self._norm_config.loudness_enabled

        for index, entry in enumerate(all_entries, start=1):
            current_path = entry.path
            stem = os.path.splitext(os.path.basename(current_path))[0]

            if norm.needs_normalization(current_path, self._norm_config):
                normalized_path = os.path.join(self.temp_dir, f"norm_{index}_{stem}.wav")
                norm.normalize_format(current_path, normalized_path, self._norm_config)
                entry.path = normalized_path
                entry.duration = au.get_audio_duration_seconds(normalized_path)
                current_path = normalized_path

            if trim_enabled:
                trim_stem = os.path.splitext(os.path.basename(current_path))[0]
                trimmed_path = os.path.join(self.temp_dir, f"trim_{index}_{trim_stem}.wav")
                norm.trim_silence(current_path, trimmed_path, self._norm_config)
                if os.path.exists(trimmed_path):
                    entry.path = trimmed_path
                    entry.duration = au.get_audio_duration_seconds(trimmed_path)
                    current_path = trimmed_path

            if loudness_enabled:
                loud_stem = os.path.splitext(os.path.basename(current_path))[0]
                loud_path = os.path.join(self.temp_dir, f"loud_{index}_{loud_stem}.wav")
                norm.normalize_loudness(current_path, loud_path, self._norm_config)
                if os.path.exists(loud_path):
                    entry.path = loud_path
                    entry.duration = au.get_audio_duration_seconds(loud_path)

        return all_entries

    def _process_short_files(self, short_list):
        """处理短音频，按情感分组后进行自动合并，并根据新规则处理剩余的孤立文件。"""
        if not short_list:
            return [], []

        # 1. 按情感对短音频进行分组
        emotion_groups = defaultdict(list)
        for item in short_list:
            emotion_groups[item.emotion].append(item)
        
        self.logger.info(f"短音频按情感分组: { {e: len(v) for e, v in emotion_groups.items()} }")
        print(f"[INFO] Short audios grouped by emotion: { {e: len(v) for e, v in emotion_groups.items()} }")

        processed_entries = []
        merge_map_lines = []
        leftover_items = []

        # 2. 分离可合并的组和剩余的组
        for emotion, items in emotion_groups.items():
            if len(items) > 1:
                # 这些组足够大，可以进行标准合并
                self.logger.info(f"Emotion group '{emotion}' has {len(items)} items and will be merged.")
                
                s = items[:]
                groups = []
                n = len(s)
                if n in [2, 3]:
                    groups = [s]
                elif n > 3:
                    r = n % 3
                    main_count = n - (4 if r == 1 else r)
                    idx = 0
                    while idx < main_count:
                        groups.append(s[idx:idx+3])
                        idx += 3
                    tail = s[main_count:]
                    if r == 1:
                        groups.append(tail[0:2])
                        groups.append(tail[2:4])
                    elif r == 2:
                        groups.append(tail)

                for group_idx, group in enumerate(groups, start=1):
                    wav_infos = []
                    for item in group:
                        try:
                            params, _ = au.read_wav_params_and_frames(item.path)
                            wav_infos.append({"params": params, "path": item.path, "lab_text": item.lab_text, "duration": item.duration, "emotion": item.emotion})
                        except Exception as e:
                            self.logger.warning(f"Reading wav for merge failed {item.path}: {e}")
                            processed_entries.append(item)
                    
                    merged_duration = sum(i['duration'] for i in wav_infos)
                    merge_name = f"{self.folder_name}_{emotion}_merge_{group_idx}_{au.format_hms_filename(merged_duration)}.wav"
                    merge_path = os.path.join(self.auto_merge_dir, merge_name)

                    try:
                        au.merge_wavs(merge_path, wav_infos, self._split_config.merge_silence_duration)
                        merge_lab = "".join(i["lab_text"] for i in wav_infos)
                        processed_entries.append(AudioEntry(path=merge_path, lab_text=merge_lab, duration=merged_duration, emotion=emotion, origin="merge"))
                        
                        source_files_str = ", ".join([f"{os.path.basename(i['path'])} ({i['emotion']})" for i in wav_infos])
                        merge_map_lines.append(f"{os.path.basename(merge_path)} ({emotion}) <- [{source_files_str}]")
                    except Exception as e:
                        self.logger.error(f"Merging group {group_idx} for emotion '{emotion}' failed: {e}")
                        processed_entries.extend(group)

            elif len(items) == 1:
                item = items[0]
                warning_msg = f"无法找到与 {os.path.basename(item.path)} ({item.emotion}) 具有相同情感的音频进行合并，已暂存等待跨情感合并。"
                self.logger.warning(warning_msg)
                merge_map_lines.append(f"[警告] {warning_msg}")
                leftover_items.append(item)

        # 3. 合并所有剩余的孤立文件
        if len(leftover_items) > 1:
            self.logger.warning(f"正在合并 {len(leftover_items)} 个来自不同情感类别的孤立音频。")
            print(f"[WARN] Merging {len(leftover_items)} leftover audios from different emotion groups.")
            
            s = leftover_items[:]
            groups = []
            n = len(s)
            if n in [2, 3]:
                groups = [s]
            elif n > 3:
                r = n % 3
                main_count = n - (4 if r == 1 else r)
                idx = 0
                while idx < main_count:
                    groups.append(s[idx:idx+3])
                    idx += 3
                tail = s[main_count:]
                if r == 1:
                    groups.append(tail[0:2])
                    groups.append(tail[2:4])
                elif r == 2:
                    groups.append(tail)

            for group_idx, group in enumerate(groups, start=1):
                wav_infos = []
                for item in group:
                    try:
                        params, _ = au.read_wav_params_and_frames(item.path)
                        wav_infos.append({"params": params, "path": item.path, "lab_text": item.lab_text, "duration": item.duration, "emotion": item.emotion})
                    except Exception as e:
                        self.logger.warning(f"Reading leftover wav for merge failed {item.path}: {e}")
                        processed_entries.append(item)

                if not wav_infos: continue

                merged_duration = sum(i['duration'] for i in wav_infos)
                emotion_counts = Counter(w['emotion'] for w in wav_infos)
                result_emotion = emotion_counts.most_common(1)[0][0]
                
                merge_name = f"{self.folder_name}_cross-emotion_merge_{group_idx}_{au.format_hms_filename(merged_duration)}.wav"
                merge_path = os.path.join(self.auto_merge_dir, merge_name)

                try:
                    au.merge_wavs(merge_path, wav_infos, self._split_config.merge_silence_duration)
                    merge_lab = "".join(i["lab_text"] for i in wav_infos)
                    processed_entries.append(AudioEntry(path=merge_path, lab_text=merge_lab, duration=merged_duration, emotion=result_emotion, origin="merge"))
                    
                    source_files_str = ", ".join([f"{os.path.basename(i['path'])} ({i['emotion']})" for i in wav_infos])
                    log_msg = f"{os.path.basename(merge_path)} ({result_emotion}) <- [{source_files_str}] (跨情感合并，多数投票)"
                    merge_map_lines.append(log_msg)
                    self.logger.info(log_msg)

                except Exception as e:
                    self.logger.error(f"Merging leftover group {group_idx} failed: {e}")
                    processed_entries.extend(group)
        
        elif len(leftover_items) == 1:
            # 如果最后只剩一个，无法合并，直接加入最终列表
            item = leftover_items[0]
            processed_entries.append(item)

        return processed_entries, merge_map_lines

    def _process_long_files(self, long_list):
        """处理长音频，进行自动切分，并返回切分后的文件信息。"""
        if not long_list:
            return [], []

        print(f"[INFO] 准备切分 {len(long_list)} 个长音频文件。")
        self.logger.info(f"准备对 {len(long_list)} 个长音频进行批量切分。")

        pairs = [(m.path, m.lab_text) for m in long_list]
        unsplit_entries = []
        split_entries = []
        
        try:
            _splitter = AudioSplitter(self._split_config, logger_arg=self.logger)
            _raw_results = _splitter.split(pairs, self.auto_split_dir)
            batch_results = {}
            for audio_path, res in _raw_results.items():
                if res is None:
                    batch_results[audio_path] = {"paths": None, "error": "split failed"}
                else:
                    batch_results[audio_path] = {"paths": res, "error": None}
        except Exception as e:
            self.logger.error(f"批量切分执行失败: {e}")
            # 如果批量切分失败，所有长音频都作为未切分处理，并保留其情感
            unsplit_entries.extend(long_list)
            return unsplit_entries, split_entries

        split_failures = []
        for m in long_list:
            wavp = m.path
            original_emotion = m.emotion # 获取原始情感
            res = batch_results.get(wavp, {"paths": None, "error": "no result"})
            split_wav_paths = res.get("paths")
            
            if split_wav_paths:
                self.logger.info(f"切分成功: {wavp}")
                for split_wav_path in split_wav_paths:
                    try:
                        split_lab_path = os.path.splitext(split_wav_path)[0] + ".lab"
                        lab_text = au.read_lab(split_lab_path).strip()
                        duration = au.get_wav_duration_seconds(split_wav_path)
                        
                        # 切分后的文件继承原始情感
                        split_entries.append(AudioEntry(path=split_wav_path, lab_text=lab_text, duration=duration, emotion=original_emotion, origin="split"))
                    except Exception as e:
                        self.logger.error(f"处理切分后的文件 {split_wav_path} 失败: {e}")
            else:
                split_failures.append(os.path.basename(wavp))
                # 切分失败的文件保留其原始信息
                unsplit_entries.append(m)
        
        if split_failures:
            print(f"[WARN] {len(split_failures)} 个长音频文件自动切分失败。", file=sys.stderr)

        return unsplit_entries, split_entries

    def _finalize_output(self, final_entries, merge_map_lines, initial_count):
        """最终确定输出文件：重命名、复制并写入所有摘要和列表文件。"""
        self.logger.info(f"开始最后的文件整理与输出，共 {len(final_entries)} 个最终音频。")
        
        final_output_entries_by_emotion = defaultdict(list)
        processed_filenames_by_emotion = defaultdict(set)

        for entry in final_entries:
            source_path = entry.absolute_path
            lab_text = entry.lab_text
            is_split = entry.origin == "split"
            emotion = entry.emotion

            # 根据文件类型和情感确定输出目录
            if self.ser_enabled:
                target_dir = os.path.join(self.output_dir, emotion)
            else:
                target_dir = self.output_dir

            if is_split and self.output_to_split_folder:
                # 如果启用了SER，切分文件夹也应该在情感目录下
                target_dir = os.path.join(target_dir, "切分后")

            os.makedirs(target_dir, exist_ok=True)

            if self.use_lab_text_as_filename:
                target_basename = au.sanitize_filename(lab_text) + ".wav"
            else:
                source_filename = os.path.basename(source_path)
                if is_split:
                    target_basename = source_filename
                elif source_path.startswith(os.path.abspath(self.auto_merge_dir)):
                    # 使用更具描述性的合并文件名
                    target_basename = source_filename
                else:
                    target_basename = source_filename

            # 确保在各自的情感目录中文件名唯一
            temp_basename, counter = target_basename, 1
            while temp_basename in processed_filenames_by_emotion[emotion]:
                base, ext = os.path.splitext(target_basename)
                temp_basename = f"{base}_{counter}{ext}"
                counter += 1
            target_basename = temp_basename
            processed_filenames_by_emotion[emotion].add(target_basename)

            target_path = os.path.join(target_dir, target_basename)
            try:
                if source_path.startswith(os.path.abspath(self.temp_dir)):
                    shutil.move(source_path, target_path)
                else:
                    shutil.copy(source_path, target_path)
                
                final_output_entries_by_emotion[emotion].append(AudioEntry(
                    path=os.path.abspath(target_path),
                    lab_text=entry.lab_text,
                    duration=entry.duration,
                    emotion=entry.emotion,
                    stem=entry.stem,
                    lab_path=entry.lab_path,
                    origin=entry.origin,
                ))

            except Exception as e:
                self.logger.error(f"复制或移动 {source_path} 到 {target_path} 失败: {e}")

        if merge_map_lines:
            with open(os.path.join(self.output_dir, "！合并记录.txt"), "a", encoding="utf-8") as mf:
                mf.write("\n".join(merge_map_lines) + "\n")

        split_log_path = os.path.join(self.temp_dir, "切分记录.txt")
        if os.path.exists(split_log_path):
            target_log_path = os.path.join(self.output_dir, "！切分记录.txt")
            shutil.move(split_log_path, target_log_path)

        total_duration_seconds = 0
        self.logger.info("--- 最终结果统计 ---")
        print("\n--- Final Results ---")

        # 如果SER禁用，但仍有条目，则它们都将被视为 "neutral"
        if not self.ser_enabled and any(final_output_entries_by_emotion.values()):
             if "neutral" not in final_output_entries_by_emotion:
                 # This can happen if all files were processed without SER and ended up in other categories
                 # For summary purposes, we can lump them under neutral.
                 all_entries = []
                 for emotion_group in final_output_entries_by_emotion.values():
                     all_entries.extend(emotion_group)
                 final_output_entries_by_emotion.clear()
                 final_output_entries_by_emotion["neutral"] = all_entries

        for emotion, entries in final_output_entries_by_emotion.items():
            emotion_duration = sum(e.duration for e in entries)
            total_duration_seconds += emotion_duration
            summary_text = f"情感 '{emotion}': {len(entries)} 个文件, 总时长: {au.format_hms(emotion_duration)}"
            print(summary_text)
            self.logger.info(summary_text)

            # 为每个情感创建独立的时长文件
            formatted_duration = au.format_hms_filename(emotion_duration)
            duration_filename = f"！{formatted_duration}_{emotion}.txt"
            duration_filepath = os.path.join(self.output_dir, duration_filename)
            with open(duration_filepath, "w", encoding="utf-8") as f:
                pass
            self.logger.info(f"已创建情感时长文件: {duration_filepath}")

            # 为每个情感生成独立的 .list 文件
            if self.ser_enabled:
                list_path = os.path.join(self.base_output_dir, f"{self.folder_name}_{emotion}.list")
            else:
                # 如果SER禁用，则使用旧的单一文件格式
                list_path = os.path.join(self.base_output_dir, f"{self.folder_name}.list")

            with open(list_path, "w", encoding="utf-8") as ol:
                for ent in entries:
                    ol.write(f"{ent.absolute_path}|{self.folder_name}|ZH|{ent.lab_text}\n")
            self.logger.info(f"已生成列表文件: {list_path}")
            print(f"Generated list file: {list_path}")

            # 如果启用了 JSONL 输出，则生成对应的 .jsonl 文件
            if self.jsonl_output_enabled:
                import json
                jsonl_path = os.path.splitext(list_path)[0] + ".jsonl"
                with open(jsonl_path, "w", encoding="utf-8") as f_jsonl:
                    for ent in entries:
                        json_record = {
                            "id": os.path.basename(ent.absolute_path),
                            "audio": ent.absolute_path,
                            "text": ent.lab_text,
                            "speaker": self.folder_name
                        }
                        f_jsonl.write(json.dumps(json_record, ensure_ascii=False) + '\n')
                self.logger.info(f"已生成 JSONL 文件: {jsonl_path}")
                print(f"Generated JSONL file: {jsonl_path}")

        overall_summary = f"最终音频总时长: {au.format_hms(total_duration_seconds)}"
        print(overall_summary)
        self.logger.info(overall_summary)

    def _run_checks_at_stage(self, entries, stage):
        """在指定 pipeline 阶段运行所有适用检查，返回过滤后的条目列表。"""
        if not entries:
            return entries

        reject_records = val.run_check_pipeline(
            entries, self._check_config, stage, self.output_dir
        )

        if not reject_records:
            return entries

        rejected_paths = set()
        path_to_isolate_info = {}
        for wav_path, check_name, action, stage_name in reject_records:
            if wav_path and wav_path not in rejected_paths:
                rejected_paths.add(wav_path)
                folder_name = f"{check_name}_{stage_name}"
                path_to_isolate_info[wav_path] = (folder_name, action)

        for wav_path, (folder_name, action) in path_to_isolate_info.items():
            if action == "isolate":
                isolate_dir = os.path.join(self.output_dir, "隔离", folder_name)
                os.makedirs(isolate_dir, exist_ok=True)
                if os.path.exists(wav_path):
                    dest = os.path.join(isolate_dir, os.path.basename(wav_path))
                    try:
                        shutil.copy2(wav_path, dest)
                        self.logger.info(f"已隔离: {os.path.basename(wav_path)} → 隔离/{folder_name}/")
                    except Exception as e:
                        self.logger.error(f"隔离文件 {wav_path} 失败: {e}")

        isolate_count = sum(1 for v in path_to_isolate_info.values() if v[1] == "isolate")
        remove_count = sum(1 for v in path_to_isolate_info.values() if v[1] == "remove")
        if isolate_count:
            self.logger.info(f"共隔离 {isolate_count} 个文件")
        if remove_count:
            self.logger.info(f"共移除 {remove_count} 个不合格文件")

        remaining = [e for e in entries if e.path not in rejected_paths]
        removed_count = len(entries) - len(remaining)
        if removed_count > 0:
            self.logger.info(f"检测阶段 {stage} 剔除 {removed_count} 个条目，保留 {len(remaining)} 个")

        return remaining

    def run(self):
        """执行完整的数据集处理流程。"""
        self._setup_paths_and_logging()
        all_entries = self._find_audio_entries()

        # Stage: after_scan
        all_entries = self._run_checks_at_stage(all_entries, "after_scan")

        # Normalize
        all_entries = self._normalize_audio(all_entries)

        # Stage: after_normalize
        all_entries = self._run_checks_at_stage(all_entries, "after_normalize")

        # Duration classification
        short_threshold = self._cfg.short_threshold
        long_threshold = self._cfg.long_threshold

        # Stage: after_classify
        all_entries = self._run_checks_at_stage(all_entries, "after_classify")

        short_list = [m for m in all_entries if m.duration <= short_threshold]
        normal_list = [m for m in all_entries if short_threshold < m.duration < long_threshold]
        long_list = [m for m in all_entries if m.duration >= long_threshold]

        # SER(1): on short + normal entries
        if self.ser_enabled:
            self._run_ser_on_entries(short_list)
            self._run_ser_on_entries(normal_list)

        # Merge short
        processed_short_entries, merge_map_lines = self._process_short_files(short_list)

        # Reconstitute combined list: normal + processed_short (long pending split)
        all_entries = normal_list + processed_short_entries

        # Stage: after_merge
        all_entries = self._run_checks_at_stage(all_entries, "after_merge")

        # Split long
        unsplit_long_entries, split_entries = self._process_long_files(long_list)

        # SER(2): on split segments
        if self.ser_enabled:
            self._run_ser_on_entries(split_entries)

        # Handle split failures
        if unsplit_long_entries:
            on_split_failure = self._split_config.on_split_failure
            if on_split_failure == "isolate":
                self._isolate_entries(unsplit_long_entries, "切分失败")
                unsplit_long_entries = []
            else:  # "ignore"
                if self.ser_enabled:
                    self._run_ser_on_entries(unsplit_long_entries)

        # Combine all entries (no after_split stage — merged into final)
        all_entries = all_entries + unsplit_long_entries + split_entries

        # Stage: final
        all_entries = self._run_checks_at_stage(all_entries, "final")

        self._finalize_output(all_entries, merge_map_lines, len(all_entries))
        self._cleanup_temp_files()

    def _cleanup_temp_files(self):
        """清理处理过程中生成的临时目录。"""
        if self.temp_dir and os.path.isdir(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                self.logger.info(f"已清理临时目录: {self.temp_dir}")
            except Exception as e:
                self.logger.error(f"清理临时目录 {self.temp_dir} 失败: {e}")
