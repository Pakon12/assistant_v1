import gradio as gr
import numpy as np
import librosa
from service.stt_v4 import ThaiSpeechToText

# Instantiate the ThaiSpeechToText class
thai_stt = ThaiSpeechToText()

def transcribe_audio(audio):
    sr, data = audio  # Gradio provides a tuple (sample rate, numpy array of audio data)
    
    # Convert audio data to floating-point if it's not already
    if data.dtype != np.float32:
        data = data.astype(np.float32) / np.iinfo(data.dtype).max

    # Resample the audio to 16,000 Hz if it's not already
    if sr != 16000:
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    # Transcribe the resampled audio
    text = thai_stt.transcribe_audio(data, sr)
    return text

# Define the input audio component with custom waveform options
input_audio = gr.Audio(
    type="numpy",  # Ensure the input is in the format of numpy arrays
    label="Record your voice",
    show_label=True,
)

# Create the Gradio interface
demo = gr.Interface(
    fn=transcribe_audio,  # Function to call for processing
    inputs=input_audio,  # Input component
    outputs="text",  # Output text component
    live=True,  # Enables live processing
    title="Thai Speech-to-Text Demo",
    description="Record your voice, and the app will transcribe it into Thai text."
)

if __name__ == "__main__":
    demo.launch()
