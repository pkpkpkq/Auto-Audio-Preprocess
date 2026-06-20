# 音频预处理工具 (Audio Preprocessing Tool for TTS)

这是一个用于语音合成（TTS）训练的音频数据集预处理工具。它可以自动化处理、清洗和格式化原始音频和文本标注文件。

## 主要功能

- **多格式输入支持**: 自动识别 WAV/FLAC/MP3/OGG/AAC/M4A 格式，同文本标注的多格式自动选优。
- **音频格式归一化**: 将任意支持格式统一转为 WAV 16-bit PCM，可配置目标采样率（默认 44100Hz）、声道数、位深。
- **重采样**: 基于 ffmpeg 将输入音频重采样到目标采样率。
- **LUFS 响度归一化**: 基于 EBU R128 标准（ffmpeg loudnorm），默认目标 -16 LUFS，可配置 True Peak、响度范围、单/双 pass。
- **首尾静音修剪**: 基于 librosa.effects.trim + soundfile，自动去除首尾静音段，可保留前后缓冲。
- **DC 偏移消除**: 通过 ffmpeg highpass=f=5 滤波器消除直流偏移。
- **音频质量审计报告**: 自动生成格式审计（！格式审计.txt）、削波检测（！削波检测.txt）和质量报告（！质量报告.txt，含静默比例与 VAD-SNR 估计）。
- **分类处理**: 根据音频时长，自动将文件分为"短音频"、"正常音频"和"长音频"。
- **合并短音频**: 将多个短音频自动合并，并在片段之间插入静音，以达到合适的训练时长。
- **切分长音频**:
    - 基于静音检测找到最佳切分点，使用 `whisperx` 进行语音对齐，以实现自动标注文本切分。
    - 自动调整标点符号，将切分后位于句首的标点符号移动到上一句的句尾。
- **文本规范化**:
    - 可在 `config.yaml` 中自定义灵活的**文本替换规则**，用于清理或规范化标注文本。
    - `{SKIP}` 规则，用于跳过不需要处理的音频文件。
- **去除重复内容**: 自动检测并移除具有重复标注文本的音频文件，避免数据冗余。
- **配置**: 所有关键参数均可通过 `config.yaml` 文件进行配置。
- **输出结构**:
    - 生成结构化的输出目录，包含处理好的音频文件。
    - 生成详细的**合并、切分和去重记录**，方便溯源。
    - 自动生成一个 `.list` 标注文件。

## 如何使用

### 1. 安装依赖

在开始之前，请确保您已安装 Python和ffmpeg。然后，通过以下命令安装所有必需的库：

```bash
pip install -r requirements.txt
```

### 2. 准备数据

将您的原始 `.wav` 音频文件和对应的 `.lab` 文本标注文件放入一个文件夹中。例如：

```
input/
└───Data/
    ├───audio1.wav
    ├───audio1.lab
    ├───audio2.wav
    ├───audio2.lab
    └───...
```

### 3. 配置 `config.yaml`

打开 `config.yaml` 文件，根据您的需求修改以下核心参数：

- `root_dir`: 设置为您存放原始数据的文件夹路径（例如 `D:\data\hutao`）。
- `recursive`: 如果您的数据分布在多个子文件夹中，请设置为 `true`。
- `split_and_merge`:
    - `short_threshold`: 小于此时长（秒）的音频将被视为**短音频**并进行合并。
    - `long_threshold`: 大于此时长（秒）的音频将被视为**长音频**并进行切分。
- `split`:
    - `max_duration` / `min_duration`: 切分后单个音频的最大和最小时长。
- `text`:
    - `replace_map`: 在这里添加您自己的文本替换规则。

### 4. 运行程序

配置完成后，直接运行主程序即可：

```bash
python main.py
```

程序将开始处理，并在 `output` 文件夹中生成结果。

## 音频预处理配置（新增）

在 `config.yaml` 中新增 `audio:` 配置段，控制所有音频预处理行为。缺少此配置段时，管道行为与旧版一致。

### 格式归一化

```yaml
audio:
  target_sample_rate: 44100    # 目标采样率 (Hz)
  target_channels: 1           # 目标声道数 (1=单声道, 2=立体声)
  target_bit_depth: 16         # 目标位深
  input_formats: ["wav", "flac", "mp3", "ogg", "aac", "m4a"]  # 支持的输入格式
```

- 自动检测输入格式，通过 ffmpeg 转为 WAV 16-bit PCM。
- 同一 .lab 对应多种音频格式时，按优先级选择（.wav 优先）。

### DC 偏移消除

```yaml
  dc_offset_removal:
    enable: true
    threshold: 0.001           # DC偏移大于此值才处理
```

- 在格式归一化时附带执行，使用 ffmpeg `highpass=f=5` 滤波器。

### 首尾静音修剪

```yaml
  trim_silence:
    enable: true
    top_db: 30                 # 低于此分贝视为"静音"
    frame_length: 2048
    hop_length: 512
    min_trim_ms: 100           # 至少切100ms才值得处理
    keep_padding_ms: 50        # 前后各保留缓冲
```

- 基于 librosa.effects.trim 检测，soundfile 写出。
- 修剪量低于 `min_trim_ms` 时不处理，前后保留 `keep_padding_ms` 缓冲。

### LUFS 响度归一化

```yaml
  loudness:
    enable: true
    target_lufs: -16           # TTS建议-16, SVC建议-18
    true_peak: -1.5            # True Peak 限制 (dB)
    lra: 7                     # 响度范围限制 (LU)
    dual_pass: false           # true=更精确的双pass, false=快速单pass
```

- 基于 EBU R128 标准，使用 ffmpeg loudnorm 滤波器。
- 双 pass 模式更精确但耗时翻倍。

### 音频质量审计报告

```yaml
  audit:
    enable: true               # 生成格式审计报告 → ！格式审计.txt
  clipping_detection:
    enable: true
    threshold_ratio: 0.01      # 超1%样本削波才报警
    clip_threshold: 0.99       # 振幅≥0.99视为削波
  silence_ratio:
    enable: true
    top_db: 30
    warn_threshold: 0.4        # 静默>40%警告
    reject_threshold: 0.7      # 静默>70%拒绝
  snr_estimation:
    enable: true
    top_db: 30
    warn_db: 20                # SNR<20dB警告
    reject_db: 15              # SNR<15dB拒绝
```

生成的报告文件：
- `！格式审计.txt` — 采样率/位深/声道分布，需转换的文件清单
- `！削波检测.txt` — 各文件削波比例与峰值
- `！质量报告.txt` — 静默比例与 VAD-SNR 估计结果

## 输出文件结构

处理完成后，`output` 文件夹的结构如下：

```
output/
├───Data.log           # 本次处理的详细日志
├───Data.list           # 可直接用于训练的标注列表文件
└───Data/                 # 处理后的音频及相关记录
    ├───！合并记录.txt
    ├───！切分记录.txt
    ├───！去重记录.txt
    ├───！格式审计.txt          # 音频格式元数据审计（新增）
    ├───！削波检测.txt          # 削波检测报告（新增）
    ├───！质量报告.txt          # 静默比例与SNR（新增）
    ├───！00-25-30.txt    # 处理后音频的总时长标记（文件名即时长）
    ├───Data_merged_1.wav
    ├───audio1.wav
    └───切分后/
        ├───audio2_part1.wav
        └───audio2_part2.wav
```
