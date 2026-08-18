import os
import re
import tempfile
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from functools import lru_cache
from funasr import AutoModel
from transformers import AutoConfig, Wav2Vec2FeatureExtractor, HubertPreTrainedModel, HubertModel
from audio_utils import setup_model_cache

logger = logging.getLogger(__name__)

setup_model_cache()


# ── Backend models (lru_cache + unload) ─────────────────────────
# ponytail: 三个 backend getter 一刀 cache_clear；本项目单进程串行 + 单 backend active 时无副作用。
# 切多 backend 并发 / hot-reload 时升级到 refcount registry。

@lru_cache(maxsize=1)
def _get_hubert(model_id, device):
    """加载 HuBERT SER 模型三件套 (config, processor, model)。"""
    config = AutoConfig.from_pretrained(model_id)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    model = HubertForSpeechClassification.from_pretrained(model_id, config=config)
    model.to(device)
    model.eval()
    return config, processor, model


@lru_cache(maxsize=1)
def _get_emotion2vec(model_id, device):
    return AutoModel(model=model_id, hub="ms", device=device)


@lru_cache(maxsize=1)
def _get_sensevoice(model_id, device, vad_max_ms):
    return AutoModel(
        model=model_id,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": vad_max_ms},
        hub="ms",
        device=device,
    )


def unload_models():
    """释放所有 SER 后端模型缓存。"""
    _get_hubert.cache_clear()
    _get_emotion2vec.cache_clear()
    _get_sensevoice.cache_clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class HubertClassificationHead(nn.Module):
    """Head for hubert classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_class)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class HubertForSpeechClassification(HubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.hubert = HubertModel(config)
        self.classifier = HubertClassificationHead(config)
        self.init_weights()

    def forward(self, input_values, attention_mask=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        pooled_output = torch.mean(hidden_states, dim=1)
        logits = self.classifier(pooled_output)
        return logits


class _HubertBackend:
    """HuBERT-based SER backend (6s truncation, 6-class emotion)."""

    EMOTION_MAP = {0: "angry", 1: "fear", 2: "happy", 3: "neutral", 4: "sad", 5: "surprise"}

    def __init__(self, config):
        self.model_id = config.hubert_model_id
        self.device = self._resolve_device(config.device)
        self.sample_rate = 16000  # Hz, from model card
        self.max_duration = 6  # seconds, from model card

    @staticmethod
    def _resolve_device(device_setting):
        if device_setting == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device_setting

    def predict(self, audio_path):
        _, processor, model = _get_hubert(self.model_id, self.device)

        speech, sr = librosa.load(audio_path, sr=self.sample_rate)
        input_values = processor(
            speech,
            padding="max_length",
            truncation=True,
            max_length=self.max_duration * self.sample_rate,
            return_tensors="pt",
            sampling_rate=self.sample_rate,
        ).input_values

        input_values = input_values.to(self.device)

        with torch.no_grad():
            logits = model(input_values)

        predicted_id = torch.argmax(logits, dim=1).item()
        return self.EMOTION_MAP.get(predicted_id, "unknown")


class _Emotion2VecBackend:
    """emotion2vec SER backend via FunASR AutoModel (9-class utterance-level)."""

    def __init__(self, config):
        self.model_id = config.emotion2vec_model_id
        self.granularity = config.emotion2vec_granularity
        self.safe_max_duration = config.emotion2vec_safe_max_duration  # seconds
        self.device = self._resolve_device(config.device)

    @staticmethod
    def _resolve_device(device_setting):
        if device_setting == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device_setting

    def predict(self, audio_path):
        model = _get_emotion2vec(self.model_id, self.device)
        res = model.generate(
            audio_path,
            output_dir="./outputs",
            granularity=self.granularity,
            extract_embedding=False,
        )
        return res[0]["text"]

    def truncate_and_retry(self, audio_path):
        logger.info(f"emotion2vec 截断重试: {audio_path}, 最大 {self.safe_max_duration}s")
        speech, sr = librosa.load(audio_path, sr=None)
        max_samples = int(self.safe_max_duration * sr)
        if len(speech) <= max_samples:
            raise RuntimeError("音频时长未超过 safe_max_duration，截断重试无意义")
        truncated = speech[:max_samples]
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_fd)
        try:
            import soundfile as sf
            sf.write(tmp_path, truncated, sr)
            return self.predict(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class _SenseVoiceBackend:
    """SenseVoice SER backend via FunASR AutoModel with VAD (7-class from rich text)."""

    EMO_PATTERN = re.compile(r"<\|(\w+)\|>")
    VALID_EMOTIONS = {"happy", "sad", "angry", "neutral", "fearful", "disgusted", "surprised"}

    def __init__(self, config):
        self.model_id = config.sensevoice_model_id
        self.language = config.sensevoice_language
        self.use_itn = config.sensevoice_use_itn
        self.ban_emo_unk = config.sensevoice_ban_emo_unk
        self.vad_max_single_segment_time = config.sensevoice_vad_max_single_segment_time  # ms
        self.device = self._resolve_device(config.device)

    @staticmethod
    def _resolve_device(device_setting):
        if device_setting == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device_setting

    def predict(self, audio_path):
        model = _get_sensevoice(self.model_id, self.device, self.vad_max_single_segment_time)
        res = model.generate(
            audio_path,
            output_dir="./outputs",
            language=self.language,
            use_itn=self.use_itn,
            ban_emo_unk=self.ban_emo_unk,
        )
        text = res[0]["text"]
        return self._extract_emotion(text)

    def _extract_emotion(self, text):
        """从 SenseVoice 富文本输出中提取情感标签。
        输出格式如: <|HAPPY|><|Speech|>hello world
        """
        for match in self.EMO_PATTERN.finditer(text):
            label = match.group(1).lower()
            if label in self.VALID_EMOTIONS:
                return label
        return "unknown"


class SpeechEmotionRecognizer:
    def __init__(self, config):
        self.config = config
        self.on_failure = config.on_failure
        self._backend = self._create_backend(config)

    @staticmethod
    def _create_backend(config):
        backend_name = config.backend
        if backend_name == "hubert":
            return _HubertBackend(config)
        elif backend_name == "emotion2vec":
            return _Emotion2VecBackend(config)
        elif backend_name == "sensevoice":
            return _SenseVoiceBackend(config)
        else:
            raise ValueError(f"不支持的 SER 后端: {backend_name}")

    def predict_emotion(self, audio_path):
        try:
            emotion = self._backend.predict(audio_path)
            logger.info(f"情感预测结果 {audio_path}: {emotion}")
            return emotion
        except Exception as e:
            return self._handle_failure(audio_path, e)

    def predict_emotion_batch(self, entries):
        for entry in entries:
            try:
                entry.emotion = self.predict_emotion(entry.absolute_path)
            except Exception as e:
                if self.on_failure == "isolate":
                    raise
                logger.warning(f"批量情感预测跳过失败条目 {entry.path}: {e}")
                entry.emotion = "unknown"

    def _handle_failure(self, audio_path, error):
        logger.error(f"情感预测失败 {audio_path}: {error}", exc_info=True)
        if self.on_failure == "default":
            return "unknown"
        elif self.on_failure == "truncate_and_retry":
            return self._truncate_and_retry(audio_path, error)
        elif self.on_failure == "isolate":
            raise
        else:
            logger.warning(f"未知的 on_failure 策略 '{self.on_failure}'，回退到 default")
            return "unknown"

    def _truncate_and_retry(self, audio_path, original_error):
        if isinstance(self._backend, _Emotion2VecBackend):
            try:
                return self._backend.truncate_and_retry(audio_path)
            except Exception as retry_err:
                logger.error(f"emotion2vec 截断重试仍失败 {audio_path}: {retry_err}", exc_info=True)
                return "unknown"
        else:
            try:
                return self._backend.predict(audio_path)
            except Exception as retry_err:
                logger.error(f"截断重试仍失败 {audio_path}: {retry_err}", exc_info=True)
                return "unknown"


if __name__ == '__main__':
    # Ponytail smoke check: unload API surface + cache_clear hooks must be callable + work on empty caches.
    assert callable(unload_models)
    assert all(callable(g.cache_clear) for g in (_get_hubert, _get_emotion2vec, _get_sensevoice))
    unload_models()  # no-op on empty caches; fails if any underlying cache_clear breaks

    from config_types import SerConfig

    test_config = SerConfig(backend="hubert")
    recognizer = SpeechEmotionRecognizer(test_config)
    if os.path.exists('test.wav'):
        emotion = recognizer.predict_emotion('test.wav')
        print(f"预测情感分类结果: {emotion}")
    else:
        print("请放置 test.wav 文件以进行 SER 测试。")
