import torch
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

class ThaiSpeechToText:
    def __init__(self, model_name="airesearch/wav2vec2-large-xlsr-53-th"):
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Running on {'GPU' if self.dev == 'cuda' else 'CPU'}")

        # Load pretrained processor and model
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.model.to(self.dev)
        
    def transcribe_audio(self, audio_data, sample_rate):
        # Preprocess the audio for the model
        inputs = self.processor(audio_data, sampling_rate=sample_rate, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = self.model(inputs.input_values.to(self.dev)).logits.cpu()

        predicted_ids = torch.argmax(logits, dim=-1)

        # Decode the predicted ids to text
        transcriptions = self.processor.batch_decode(predicted_ids)
        return transcriptions[0]
