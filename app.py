import gradio as gr
import numpy as np
import librosa
import threading
import time
from service.transcript.stt_v5 import ThaiSpeechToText
from service.GTTS.tts_v1 import TextToSpeech


thai_stt = ThaiSpeechToText()
thai_tts = TextToSpeech()


# Global variables with a lock for thread safety ตัวแปรทั่วโลกพร้อมล็อคเพื่อความปลอดภัยของเธรด
transcription_lock = threading.Lock()
last_audio_time = None
current_transcription = ""
HELLO_AUDIO_FILE = "service/GTTS/hello.mp3"
is_playing = False
is_first_play = True
file_audio = None


def transcribe_audio(audio):
    global last_audio_time, current_transcription, is_playing, is_first_play ,file_audio

    try:
        sr, data = audio

        with transcription_lock:
            last_audio_time = time.time()

        # Normalize and resample audio if necessary ทำให้เสียงเป็นปกติและทำการสุ่มตัวอย่างใหม่หากจำเป็น
        if data.dtype != np.float32:
            data = data.astype(np.float32) / np.iinfo(data.dtype).max

        if sr != 16000:
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
            sr = 16000

        text = thai_stt.transcribe_audio(data, sr)

        with transcription_lock:
            current_transcription += text

            # Handle first play logic จัดการตรรกะการเล่นครั้งแรก
            if is_first_play:
                file_audio = HELLO_AUDIO_FILE
                is_first_play = False
                is_playing = True  # Ensure the audio plays ให้แน่ใจว่าเสียงเล่น
 
            # Keyword detection for subsequent plays การตรวจจับคำสำคัญสำหรับการเล่นครั้งต่อไป
            elif "สวัสดี" in current_transcription:
                file_audio = thai_tts.speak(text='สวัสดีค่ะ')
                current_transcription = ""  
                is_playing = True

            elif "คำสั่ง" in current_transcription:
                file_audio = thai_tts.speak(text='พร้อมรับคำสั่งค่ะ')
                current_transcription = ""  
                is_playing = True

            # If playback is triggered, return the file_audio หากมีการเรียกใช้การเล่น ให้ส่งคืนไฟล์_เสียง
            if is_playing:
                is_playing = False
                return current_transcription, file_audio

        return current_transcription, file_audio

    except Exception as e:
        return f"An error occurred during transcription: {e}", None

def clear_text_if_no_speech():
    global last_audio_time, current_transcription
    while True:
        with transcription_lock:
            if last_audio_time is not None and (time.time() - last_audio_time) > 30:
                current_transcription = ""
                last_audio_time = None
        time.sleep(1)

# Define the input audio component with custom waveform options
# กำหนดส่วนประกอบเสียงอินพุตด้วยตัวเลือกคลื่นเสียงแบบกำหนดเอง
input_audio = gr.Audio(
    type="numpy",  
    label="Record your voice",
    show_label=True,
    streaming=True, 
)

# Define the output text component
# กำหนดส่วนประกอบข้อความเอาท์พุต
output_textbox = gr.Textbox(label="ถอดเสียง")

# Create the Gradio interface
# สร้างอินเทอร์เฟซ Gradio
app = gr.Interface(
    fn=transcribe_audio,
    inputs=input_audio,
    outputs=[output_textbox, gr.Audio(label="ตอบกลับ", autoplay=True)],
    live=True,
    title="AI KAK v1",
    description="บันทึกเสียงของคุณ และแอปจะคุยกับ ai ชื่อว่า กาก สามารถทดสอบพูด เช่น สวัสดี พร้อมรับคำสั่ง กาก"
)

# Start a thread that checks for silence and clears the text after a timeout
# เริ่มเธรดที่ตรวจสอบความเงียบและล้างข้อความหลังจากหมดเวลา
timeout_thread = threading.Thread(target=clear_text_if_no_speech)
timeout_thread.daemon = True
timeout_thread.start()

if __name__ == "__main__":
    app.launch()
