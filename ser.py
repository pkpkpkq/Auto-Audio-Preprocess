import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, Wav2Vec2FeatureExtractor, HubertPreTrainedModel, HubertModel
import logging

logger = logging.getLogger(__name__)

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
        # Average pooling over the sequence dimension
        pooled_output = torch.mean(hidden_states, dim=1)
        logits = self.classifier(pooled_output)

        return logits

class SpeechEmotionRecognizer:
    def __init__(self, config):
        self.config = config
        self.model_id = config.get('model_id', 'xmj2002/hubert-base-ch-speech-emotion-recognition')
        
        device_setting = config.get('device', 'auto')
        if device_setting == 'auto':
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device_setting

        self.model_config = None
        self.processor = None
        self.model = None
        self.duration = 6  # seconds, from model card
        self.sample_rate = 16000  # Hz, from model card

    def _load_model(self):
        if self.model is None:
            print(f"[INFO] Loading SER model '{self.model_id}' on device '{self.device}'.")
            logger.info(f"Loading SER model '{self.model_id}'...")
            try:
                self.model_config = AutoConfig.from_pretrained(self.model_id)
                self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_id)
                self.model = HubertForSpeechClassification.from_pretrained(self.model_id, config=self.model_config)
                self.model.to(self.device)
                self.model.eval()
                logger.info("SER model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load SER model: {e}", exc_info=True)
                print(f"[ERROR] Failed to load SER model: {e}")
                raise

    def _id2class(self, id_):
        # Based on the code from HuggingFace
        # emotions = ['anger', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        if id_ == 0: return "angry"
        elif id_ == 1: return "fear"
        elif id_ == 2: return "happy"
        elif id_ == 3: return "neutral"
        elif id_ == 4: return "sad"
        else: return "surprise"

    def predict_emotion(self, audio_path):
        self._load_model()
        if self.model is None or self.processor is None:
            raise RuntimeError("SER model is not loaded.")

        try:
            speech, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Process the audio
            input_values = self.processor(
                speech, 
                padding="max_length", 
                truncation=True, 
                max_length=self.duration * self.sample_rate, 
                return_tensors="pt", 
                sampling_rate=self.sample_rate
            ).input_values

            input_values = input_values.to(self.device)

            with torch.no_grad():
                logits = self.model(input_values)
            
            # Get prediction
            predicted_id = torch.argmax(logits, dim=1).item()
            emotion = self._id2class(predicted_id)
            
            logger.info(f"Predicted emotion for {audio_path}: {emotion}")
            return emotion

        except Exception as e:
            logger.error(f"Failed to predict emotion for {audio_path}: {e}", exc_info=True)
            print(f"[WARN] Could not predict emotion for {audio_path}: {e}")
            return "unknown"

if __name__ == '__main__':
    # For testing purposes
    # You would need a sample audio file named 'test.wav'
    if os.path.exists('test.wav'):
        ser_config = {'device': 'auto', 'model_id': 'xmj2002/hubert-base-ch-speech-emotion-recognition'}
        recognizer = SpeechEmotionRecognizer(ser_config)
        emotion = recognizer.predict_emotion('test.wav')
        print(f"The predicted emotion is: {emotion}")

