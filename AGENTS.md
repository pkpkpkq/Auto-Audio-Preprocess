# PROJECT KNOWLEDGE BASE

**Generated:** 2026-06-20
**Commit:** 3ba6b20
**Branch:** main

## OVERVIEW

AudioPreProcess — Python 3 flat-script pipeline for TTS audio dataset preprocessing. Loads WAV/FLAC/MP3/OGG/AAC/M4A + LAB pairs, normalizes format/loudness/silence, runs ASR/SER, deduplicates, merges short segments, splits long audio into training clips. No package structure, not pip-installable.

## STRUCTURE

```
AudioPreProcess/
├── main.py              # Entry point (42 lines)
├── processor.py         # Core orchestration — DatasetProcessor (840 lines)
├── asr.py               # ASR: FunASR paraformer-zh + ct-punc, whisperX (113 lines)
├── ser.py               # SER: HuBERT emotion classification (138 lines)
├── audio_splitter.py     # Split: whisperX align, DP silence, ffmpeg cut (366 lines)
├── audio_utils.py       # Utility: ffmpeg concat, duration, OpenCC t2s (117 lines)
├── audio_normalizer.py  # Normalize: format/resample/DC, LUFS loudnorm, silence trim (253 lines)
├── audio_validator.py   # Validate: audit metadata, clipping, silence-ratio, SNR (331 lines)
├── config.yaml          # Single YAML config — all thresholds, model params, paths
├── requirements.txt     # 9 deps (torch, librosa, numpy, whisperx, opencc, PyYAML, transformers, pytest, soundfile)
├── pytest.ini           # Pytest configuration
├── tests/               # Pytest test suite (conftest + 9 test modules)
├── uv start python.bat   # Windows convenience: uv run main.py
└── README.md            # Chinese docs
```

## ENTRY POINTS

| File | Invoke | Purpose |
|------|--------|---------|
| main.py | `python main.py` / `uv run main.py` | Full pipeline: config → DatasetProcessor.run() |
| asr.py | `python asr.py <wav> <out.json>` | Standalone ASR test on single WAV |
| ser.py | `python ser.py` (needs test.wav) | Standalone SER emotion prediction test |

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Pipeline flow | processor.py:770 `run()` | setup → find_entries → audit → normalize → clipping detect → quality → classify → process_long → process_short → finalize → cleanup |
| Config schema | config.yaml | root_dir, recursive, audio, split_and_merge, split, ser, asr, text.replace_map, output |
| ASR transcription | asr.py `transcribe_audio()` | FunASR + ct-punc; lru_cached model; {ASR} placeholder handler |
| SER emotion | ser.py `SpeechEmotionRecognizer` | HuBERT xmj2002/hubert-base-ch-speech-emotion-recognition |
| Long audio split | audio_splitter.py `AudioSplitter.split()` | whisperX align → DP silence → ffmpeg cut → punctuation adjust |
| Short audio merge | audio_utils.py `merge_wavs()` | ffmpeg concat with silence padding |
| Text processing | processor.py `_process_lab_with_asr()` | {SKIP} skip, {ASR} placeholder, replace_map, OpenCC t2s |
| Dedup logic | processor.py `_find_audio_entries()` | .lab content hash — skip duplicates; multi-format input scan (WAV/FLAC/MP3/OGG/AAC/M4A) |
| Output generation | processor.py `_finalize_output()` | .list + .jsonl per emotion category |
| Format normalization | audio_normalizer.py `normalize_format()` | ffmpeg → WAV 16-bit PCM + resample + DC removal via highpass=f=5 |
| LUFS loudness | audio_normalizer.py `normalize_loudness()` | ffmpeg loudnorm (EBU R128), -ar/-ac pinned, single/dual pass |
| Silence trim | audio_normalizer.py `trim_silence()` | librosa.effects.trim + soundfile write, min_trim_ms + keep_padding_ms |
| Audio normalization flow | processor.py `_normalize_audio()` | format → trim → loudness, per-entry in temp_dir |
| Format audit report | audio_validator.py `audit_audio_metadata()` | ffprobe/wave metadata scan → ！格式审计.txt |
| Clipping detection | audio_validator.py `detect_clipping()` | amplitude ≥ clip_threshold → ！削波检测.txt |
| Quality report | audio_validator.py `batch_quality_check()` | silence-ratio + VAD-SNR → ！质量报告.txt |
| Inline import sites | processor.py:251, 685 | `import datetime`, `import json` inside methods |

## CONVENTIONS

- Input pairs: audio + .lab in input/
- Output per role in output/<role>/
- Config-driven: all behavior from config.yaml, accessed via `.get('key', default)`
- Backward compat: missing `audio:` config section = old behavior (no normalization/validation)
- UTF-8 everywhere, no coding-header comments
- snake_case functions/vars, PascalCase classes, _prefix for private
- Chinese docstrings + log messages; English function/variable names
- `audio_utils` imported as `au`, `audio_normalizer` as `norm`, `audio_validator` as `val` (processor.py)
- `logging.getLogger(__name__)` in ser/audio_splitter/audio_validator; `"tts_preprocess"` logger in processor/main
- `{SKIP}` token in .lab to skip; `{ASR}` for placeholder replacement
- Output metadata prefix `!` (fullwidth exclamation): !00-25-30.txt, !合并记录.txt, !格式审计.txt
- Report file naming: `！<desc>.txt` with fullwidth ！ prefix
- Dual output: .list + .jsonl per emotion category
- Deduplication by .lab content hash
- Never overwrite input audio — all normalization writes to temp_dir
- All audio paths through normalization produce 16-bit PCM WAV only
- ffmpeg invocation pattern: `ffmpeg -y -hide_banner -loglevel error` with `timeout=300`
- DC offset removal via `highpass=f=5` ffmpeg filter (inside normalize_format)
- Loudness via ffmpeg `loudnorm` (EBU R128) with `-ar`/`-ac` pinned to target
- `if __name__ == "__main__":` blocks contain test/demo code (asr.py, ser.py)

## ANTI-PATTERNS (THIS PROJECT)

- No `__init__.py` — flat script project, not importable as package
- No CI/CD
- No linting/formatting setup (no .flake8, .pylintrc, pyproject.toml, .editorconfig)
- Inline imports in processor.py (line 251: datetime, line 685: json)
- `bare except: pass` in audio_utils.py cleanup finally block
- `subprocess.run` without timeout in audio_splitter.py and audio_utils.py
- `lru_cache` on `get_model()` with `os.environ` mutation side effect (asr.py)
- `punc_model: "ct-punc"` hardcoded in asr.py instead of config-driven
- No version pins in requirements.txt — floating dependencies
- ffmpeg is a system dependency (must be on PATH) — not validated at startup

## AUDIO PREPROCESSING

Pipeline order (in `processor.py:run()`):

1. **Audit** — `val.audit_audio_metadata()` → scan all entries' format metadata, write `！格式审计.txt`
2. **Format normalize** — `norm.normalize_format()` → convert any input to WAV 16-bit PCM at target sample rate; DC removal via `highpass=f=5` if enabled
3. **Trim silence** — `norm.trim_silence()` → librosa.effects.trim, keep padding buffer, min trim threshold
4. **Loudness normalize** — `norm.normalize_loudness()` → ffmpeg loudnorm (EBU R128), single or dual pass
5. **Clipping detect** — `val.batch_detect_clipping()` → amplitude threshold scan, write `！削波检测.txt`
6. **Quality check** — `val.batch_quality_check()` → silence-ratio + VAD-SNR, write `！质量报告.txt`

All normalization output goes to `temp_dir`; original input files are never modified. Steps 2–4 are per-entry in `_normalize_audio()`. Steps 1, 5, 6 operate on the full entry list and produce reports only.

Feature flags (all under `audio:` in config.yaml):

| Flag | Key | Default | Effect |
|------|-----|---------|--------|
| Format normalize | (always active if `audio:` present) | — | Convert to target WAV params |
| DC removal | `audio.dc_offset_removal.enable` | true | highpass=f=5 during format step |
| Silence trim | `audio.trim_silence.enable` | true | Trim head/tail silence |
| Loudness | `audio.loudness.enable` | true | EBU R128 loudnorm |
| Audit | `audio.audit.enable` | true | Generate format audit report |
| Clipping | `audio.clipping_detection.enable` | true | Generate clipping detection report |
| Silence ratio | `audio.silence_ratio.enable` | true | Check silence proportion |
| SNR | `audio.snr_estimation.enable` | true | VAD-based SNR estimation |

## COMMANDS

```bash
# Run pipeline
python main.py
# Or with uv
uv run main.py
```

## NOTES

- WhisperX model download on first run — requires network
- FunASR models cached via MODELSCOPE_CACHE env (set by asr.py to ./local_model)
- Config change requires restart — no hot-reload
- MemoryError caught at top level in main.py — large datasets may OOM
