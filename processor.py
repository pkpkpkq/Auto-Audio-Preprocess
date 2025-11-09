import os
import shutil
import logging
import sys
import difflib
import re

# Local module imports
from asr import transcribe_audio
from audio_splitter import split_batch
import audio_utils as au

class DatasetProcessor:
    """封装整个数据集处理流程的类"""

    def __init__(self, config):
        """
        初始化处理器。
        :param config: 从 config.yaml 加载的配置字典。
        """
        self.config = config
        self.root_dir = os.path.abspath(config['root_dir'])
        
        if not os.path.isdir(self.root_dir):
            raise ValueError(f"配置中指定的根目录不存在或不是目录: {self.root_dir}")

        # 从配置中提取参数
        self.recursive = self.config.get('recursive', False)
        self.use_lab_text_as_filename = self.config.get('use_lab_text_as_filename', False)
        
        split_merge_config = self.config.get('split_and_merge', {})
        self.output_to_split_folder = split_merge_config.get('output_to_split_folder', True)

        text_config = self.config.get('text', {})
        self.replace_map = text_config.get('replace_map', {})
        self.t2s_enabled = text_config.get('t2s', True)

        self.asr_config = self.config.get('asr', {})

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
        self.out_list_path = None
        self.logger = None

    def _extract_asr_for_placeholder(self, asr_segments, prefix, suffix):
        """
        基于文本模糊匹配和索引映射的 {ASR} 占位符提取。
        """
        if not asr_segments:
            return None

        full_asr_text = "".join(seg["text"] for seg in asr_segments)

        # 1. 创建从规范化文本索引到原始文本索引的映射
        norm_to_full_map = []
        norm_char_idx = 0
        for i, char_full in enumerate(full_asr_text):
            # 只有非标点字符才会计入规范化文本的长度
            if char_full not in '，。！？、,.!?…“”‘’：;；（）()《》〈〉【】':
                norm_to_full_map.append(i) # 存储原始文本中的索引
                norm_char_idx += 1
        
        # 2. 规范化所有文本，用于模糊匹配
        def normalize(s: str) -> str:
            if not s:
                return ""
            # 移除所有标点符号和空白字符
            return re.sub(r'[\s，。！？、,.!?…“”‘’：;；（）()《》〈〉【】]', '', s)

        norm_full_asr_text = normalize(full_asr_text)
        norm_prefix = normalize(prefix)
        norm_suffix = normalize(suffix)

        if not norm_full_asr_text:
            return None

        MIN_MATCH_RATIO = 0.7 # 模糊匹配的最小匹配率阈值

        # 辅助函数：在规范化文本中查找匹配的起始和结束索引
        def find_match_indices_in_norm(target_norm, pattern_norm):
            if not pattern_norm:
                return None, None
            
            matcher = difflib.SequenceMatcher(None, target_norm, pattern_norm)
            match = matcher.find_longest_match(0, len(target_norm), 0, len(pattern_norm))
            
            if match.size > 0 and (match.size / len(pattern_norm)) >= MIN_MATCH_RATIO:
                return match.a, match.a + match.size # 返回在规范化文本中的起始和结束索引
            return None, None

        # 3. 在规范化文本上进行模糊匹配
        prefix_start_norm, prefix_end_norm = find_match_indices_in_norm(norm_full_asr_text, norm_prefix)
        suffix_start_norm, suffix_end_norm = find_match_indices_in_norm(norm_full_asr_text, norm_suffix)

        # 4. 将规范化文本的索引映射回原始文本
        start_idx_full = 0
        if prefix_end_norm is not None:
            # 确保索引在映射范围内
            if prefix_end_norm < len(norm_to_full_map):
                start_idx_full = norm_to_full_map[prefix_end_norm]
            else:
                # 如果前缀匹配到规范化文本的末尾或超出，则从原始文本的末尾开始
                start_idx_full = len(full_asr_text)

        end_idx_full = len(full_asr_text)
        if suffix_start_norm is not None:
            # 确保索引在映射范围内
            if suffix_start_norm < len(norm_to_full_map):
                end_idx_full = norm_to_full_map[suffix_start_norm]
            else:
                # 如果后缀匹配到规范化文本的末尾或超出，则到原始文本的末尾
                end_idx_full = len(full_asr_text)

        # 5. 确保索引有效且顺序正确
        if start_idx_full >= end_idx_full:
            return None

        # 6. 从原始文本中提取内容
        extracted_text = full_asr_text[start_idx_full:end_idx_full].strip()
        return extracted_text if extracted_text else None

    def _process_lab_with_asr(self, lab_path, wav_path):
        """
        支持 {ASR} 占位符的智能替换：仅替换占位符部分，保留上下文。
        """
        with open(lab_path, "r", encoding="utf-8") as f:
            original_lab = f.read().strip()

        temp_lab = original_lab
        for k, v in self.replace_map.items():
            temp_lab = temp_lab.replace(k, v)

        if "{ASR}" not in temp_lab:
            return temp_lab

        if self.logger:
            self.logger.info(f"检测到 {{ASR}} 占位符，对 {wav_path} 执行 ASR...")

        try:
            segments = transcribe_audio(wav_path, self.asr_config)
            
            if not segments:
                raise RuntimeError("ASR 未返回任何结果。")

            asr_full_text = "".join(seg["text"] for seg in segments).strip()
            asr_full_text = au.traditional_to_simplified(asr_full_text, self.t2s_enabled)
            if self.logger:
                self.logger.info(f"ASR 全文结果: {asr_full_text}")

            final_lab = temp_lab
            while "{ASR}" in final_lab:
                parts = final_lab.split("{ASR}", 1)
                prefix = parts[0]
                suffix = parts[1] if len(parts) > 1 else ""

                extracted = self._extract_asr_for_placeholder(segments, prefix, suffix)

                if extracted is not None:
                    replacement = extracted
                    if self.logger:
                        self.logger.info(f"成功提取占位符内容: '{extracted}'")
                else:
                    replacement = "{ASR_FAILED}"
                    if self.logger:
                        self.logger.warning("无法从 ASR 结果中定位占位符内容，保留失败标记")

                final_lab = prefix + replacement + suffix

            return final_lab

        except Exception as e:
            error_msg = f"ASR 处理失败: {e}"
            if self.logger:
                self.logger.warning(error_msg)
            else:
                print(f"[WARN] {error_msg}", file=sys.stderr)
            return temp_lab

    def _setup_paths_and_logging(self):
        """设置所有输出路径并配置日志记录。"""
        self.base_output_dir = os.path.abspath("output")
        self.output_dir = os.path.join(self.base_output_dir, self.folder_name)
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
        self.out_list_path = os.path.join(self.base_output_dir, f"{self.folder_name}.list")
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
        self.logger.info(f"  - 输出列表文件: {self.out_list_path}")
        self.logger.info(f"  - 使用标注文本命名: {self.use_lab_text_as_filename}")
        self.logger.info(f"  - 替换规则: {self.replace_map}")
        self.logger.info("="*50)

    def _find_audio_entries(self):
        """查找、配对、去重并读取所有音频条目及其元数据。"""
        import datetime
        stems = {}

        if self.recursive:
            self.logger.info(f"开始递归扫描目录: {self.root_dir}")
            for dirpath, _, filenames in os.walk(self.root_dir):
                for fname in filenames:
                    name, ext = os.path.splitext(fname)
                    ext_lower = ext.lower()
                    if ext_lower in (".wav", ".lab"):
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
                if ext_lower in (".wav", ".lab"):
                    if name not in stems:
                        stems[name] = {"dirpath": self.root_dir}
                    stems[name][ext_lower] = fname
        
        self.logger.info(f"找到 {len(stems)} 个带有 .wav 或 .lab 扩展名的唯一文件主干。")

        matched = []
        skipped_pairing = 0
        read_failures = []
        seen_texts = set()
        duplicate_log_lines = []
        skipped_rule_count = 0

        # 按文件名排序以确保去重行为是确定的
        stems_sorted = sorted(stems.items())

        for stem_key, extmap in stems_sorted:
            if ".wav" not in extmap or ".lab" not in extmap:
                skipped_pairing += 1
                continue

            dirpath = extmap["dirpath"]
            # 从 extmap 中获取原始文件名，因为 stem_key 可能包含路径
            wav_filename = extmap[".wav"]
            lab_filename = extmap[".lab"]
            
            wav_path = os.path.join(dirpath, wav_filename)
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

            # 去重逻辑
            if lab_text in seen_texts:
                log_line = f"丢弃 (文本内容重复): {os.path.basename(wav_path)}"
                duplicate_log_lines.append(log_line)
                self.logger.info(f"发现重复文本，跳过文件: {os.path.basename(wav_path)}")
                continue
            else:
                seen_texts.add(lab_text)

            try:
                duration = au.get_wav_duration_seconds(wav_path)
            except Exception as e:
                read_failures.append(f"{wav_path} (获取时长): {e}")
                self.logger.warning(f"获取 {wav_path} 的时长失败: {e}")
                continue

            matched.append({
                "stem": stem_key, "wav_path": wav_path, "lab_path": lab_path,
                "lab_text": lab_text, "duration": duration
            })

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

    def _process_short_files(self, short_list):
        """处理短音频，进行自动合并。"""
        if not short_list:
            return [], []

        s = short_list[:]
        groups = []
        n = len(s)
        if n <= 3:
            groups = [s]
        else:
            r = n % 3
            main_count = n - (4 if r == 1 else r)
            idx = 0
            while idx < main_count:
                groups.append([s[idx], s[idx+1], s[idx+2]])
                idx += 3
            tail = s[main_count:]
            if r == 1:
                groups.append([tail[0], tail[1]])
                groups.append([tail[2], tail[3]])
            elif r == 2:
                groups.append([tail[0], tail[1]])

        if groups:
            print(f"[INFO] 准备将 {len(short_list)} 个短音频文件合并成 {len(groups)} 组。")

        processed_entries = []
        merge_map_lines = []
        merge_read_failures, merge_failures = 0, 0

        for group_idx, group in enumerate(groups, start=1):
            wav_infos, ok = [], True
            for item in group:
                try:
                    params, frames = au.read_wav_params_and_frames(item["wav_path"])
                    wav_infos.append({"params": params, "frames": frames, "path": item["wav_path"], "lab_text": item["lab_text"], "duration": item["duration"]})
                except Exception as e:
                    self.logger.warning(f"读取用于合并的 wav 文件失败 {item['wav_path']}: {e}")
                    merge_read_failures += 1
                    ok = False
                    break
            
            if not ok:
                merge_failures += 1
                processed_entries.extend([{"wav_abs": os.path.abspath(item["wav_path"]), "lab_text": item["lab_text"], "duration": item["duration"]}
                                          for item in group])
                continue

            merge_name = f"{self.folder_name}_merge_{group_idx}_{au.format_hms_filename(sum(i['duration'] for i in wav_infos))}.wav"
            merge_path = os.path.join(self.auto_merge_dir, merge_name)
            try:
                au.merge_wavs(merge_path, wav_infos, self.config)
                merge_lab = "".join(i["lab_text"] for i in wav_infos)
                processed_entries.append({"wav_abs": os.path.abspath(merge_path), "lab_text": merge_lab, "duration": sum(i["duration"] for i in wav_infos)})
                merge_map_lines.append(f"{merge_path} <- " + ", ".join(i["path"] for i in wav_infos))
            except Exception as e:
                merge_failures += 1
                self.logger.warning(f"第 {group_idx} 组合并失败: {e}")
                processed_entries.extend([{"wav_abs": os.path.abspath(item["wav_path"]), "lab_text": item["lab_text"], "duration": item["duration"]}
                                          for item in group])

        if merge_read_failures > 0:
            print(f"[WARN] {merge_read_failures} 个短音频文件读取失败，无法合并。", file=sys.stderr)
        if merge_failures > 0:
            print(f"[WARN] {merge_failures} 组短音频文件合并失败。", file=sys.stderr)

        return processed_entries, merge_map_lines

    def _process_long_files(self, long_list):
        """处理长音频，进行自动切分，并返回切分后的文件信息。"""
        if not long_list:
            return [], []

        print(f"[INFO] 准备切分 {len(long_list)} 个长音频文件。")
        self.logger.info(f"准备对 {len(long_list)} 个长音频进行批量切分。")

        pairs = [(m["wav_path"], m["lab_text"]) for m in long_list]
        unsplit_entries = []
        split_entries = []
        
        try:
            batch_results = split_batch(pairs, self.auto_split_dir, self.config, logger_arg=self.logger)
        except Exception as e:
            self.logger.error(f"批量切分函数'split_batch'执行失败: {e}")
            unsplit_entries.extend([{"wav_abs": os.path.abspath(item["wav_path"]), "lab_text": item["lab_text"], "duration": item["duration"]}
                                      for item in long_list])
            return unsplit_entries, split_entries

        split_failures = []
        for m in long_list:
            wavp = m["wav_path"]
            res = batch_results.get(wavp, {"paths": None, "error": "no result"})
            split_wav_paths = res.get("paths")
            
            if split_wav_paths:
                self.logger.info(f"切分成功: {wavp}")
                for split_wav_path in split_wav_paths:
                    try:
                        split_lab_path = os.path.splitext(split_wav_path)[0] + ".lab"
                        lab_text = au.read_lab(split_lab_path).strip()
                        duration = au.get_wav_duration_seconds(split_wav_path)
                        
                        # 为切分文件添加一个标记
                        split_entries.append({
                            "wav_abs": os.path.abspath(split_wav_path),
                            "lab_text": lab_text,
                            "duration": duration,
                            "is_split": True 
                        })
                    except Exception as e:
                        self.logger.error(f"处理切分后的文件 {split_wav_path} 失败: {e}")
            else:
                split_failures.append(os.path.basename(wavp))
                unsplit_entries.append({"wav_abs": os.path.abspath(wavp), "lab_text": m["lab_text"], "duration": m["duration"]})
        
        if split_failures:
            print(f"[WARN] {len(split_failures)} 个长音频文件自动切分失败。", file=sys.stderr)

        return unsplit_entries, split_entries

    def _finalize_output(self, final_entries, merge_map_lines, initial_count):
        """最终确定输出文件：重命名、复制并写入所有摘要和列表文件。"""
        self.logger.info(f"开始最后的文件整理与输出，共 {len(final_entries)} 个最终音频。")
        final_output_entries = []
        processed_filenames = set()
        merge_counter = 1
        split_counter = 1

        for entry in final_entries:
            source_path = entry["wav_abs"]
            lab_text = entry["lab_text"]
            is_split = entry.get("is_split", False)

            # 根据文件类型确定输出目录
            if is_split and self.output_to_split_folder:
                target_dir = self.split_output_dir
            else:
                target_dir = self.output_dir

            if self.use_lab_text_as_filename:
                target_basename = au.sanitize_filename(lab_text) + ".wav"
            else:
                source_filename = os.path.basename(source_path)
                # 对于切分文件，保留切分器生成的更具描述性的文件名 (e.g., original_name_part1.wav)
                if is_split:
                    target_basename = source_filename
                # 对于合并文件，使用更简洁的 `_merged_` 格式
                elif source_path.startswith(os.path.abspath(self.auto_merge_dir)):
                    target_basename = f"{self.folder_name}_merged_{merge_counter}.wav"
                    merge_counter += 1
                # 对于其他所有文件（时长正常的，或处理失败的），保留原始文件名
                else:
                    target_basename = source_filename

            temp_basename, counter = target_basename, 1
            while temp_basename in processed_filenames:
                base, ext = os.path.splitext(target_basename)
                temp_basename = f"{base}_{counter}{ext}"
                counter += 1
            target_basename = temp_basename
            processed_filenames.add(target_basename)

            target_path = os.path.join(target_dir, target_basename)
            try:
                # 临时文件（合并、切分）总是移动，原始文件（正常、未成功处理的长音频）则复制
                if source_path.startswith(os.path.abspath(self.temp_dir)):
                    shutil.move(source_path, target_path)
                else:
                    shutil.copy(source_path, target_path)
                
                final_output_entries.append({"wav_abs": os.path.abspath(target_path), "lab_text": lab_text, "duration": entry["duration"]})
            except Exception as e:
                self.logger.error(f"复制或移动 {source_path} 到 {target_path} 失败: {e}")

        if merge_map_lines:
            # 将合并记录移动到角色文件夹下
            with open(os.path.join(self.output_dir, "！合并记录.txt"), "a", encoding="utf-8") as mf:
                mf.write("\n".join(merge_map_lines) + "\n")

        split_log_path = os.path.join(self.temp_dir, "切分记录.txt")
        if os.path.exists(split_log_path):
            # 将切分记录移动到角色文件夹下
            target_log_path = os.path.join(self.output_dir, "！切分记录.txt")
            shutil.move(split_log_path, target_log_path)

        total_duration_seconds = sum(e['duration'] for e in final_output_entries)
        summary_text = f"最终音频总时长: {au.format_hms(total_duration_seconds)}"
        print(summary_text)
        self.logger.info(summary_text)

        # 创建空的音频总长度文件
        duration_filename = f"！{au.format_hms_filename(total_duration_seconds)}.txt"
        duration_filepath = os.path.join(self.output_dir, duration_filename)
        with open(duration_filepath, "w") as f:
            pass # 创建空文件
        self.logger.info(f"已创建空的音频总时长文件: {duration_filepath}")

        with open(self.out_list_path, "w", encoding="utf-8") as ol:
            for ent in final_output_entries:
                ol.write(f"{ent['wav_abs']}|{self.folder_name}|ZH|{ent['lab_text']}\n")

    def run(self):
        """执行完整的数据集处理流程。"""
        self._setup_paths_and_logging()
        all_entries = self._find_audio_entries()

        thresholds = self.config.get('split_and_merge', {'short_threshold': 2.0, 'long_threshold': 15.0})
        short_threshold = thresholds.get('short_threshold', 2.0)
        long_threshold = thresholds.get('long_threshold', 15.0)

        short_list = [m for m in all_entries if m["duration"] <= short_threshold]
        normal_list = [m for m in all_entries if short_threshold < m["duration"] < long_threshold]
        long_list = [m for m in all_entries if m["duration"] >= long_threshold]
        
        final_entries_before_copy = []
        # 添加正常时长的文件
        final_entries_before_copy.extend([{"wav_abs": os.path.abspath(m["wav_path"]), "lab_text": m["lab_text"], "duration": m["duration"]}
                                          for m in normal_list])
        
        # 处理长音频，同时接收未切分成功和已切分成功的文件
        unsplit_long_entries, split_entries = self._process_long_files(long_list)
        final_entries_before_copy.extend(unsplit_long_entries)
        final_entries_before_copy.extend(split_entries)
        
        # 处理短音频
        processed_short_entries, merge_map_lines = self._process_short_files(short_list)
        final_entries_before_copy.extend(processed_short_entries)
        
        # 最终化输出
        self._finalize_output(final_entries_before_copy, merge_map_lines, len(all_entries))
        self._cleanup_temp_files()

    def _cleanup_temp_files(self):
        """清理处理过程中生成的临时目录。"""
        if self.temp_dir and os.path.isdir(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                self.logger.info(f"已清理临时目录: {self.temp_dir}")
            except Exception as e:
                self.logger.error(f"清理临时目录 {self.temp_dir} 失败: {e}")