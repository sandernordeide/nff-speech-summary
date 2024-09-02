# store this as app.py
import datetime
from openai import OpenAI
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from dotenv import load_dotenv, find_dotenv
import io
import wave

import numpy as np
import soundfile as sf
from scipy.io.wavfile import write

load_dotenv(find_dotenv())

st.title("Oppsummering av foredrag")

client = OpenAI()

# def convert_bytearray_to_wav_ndarray(input_bytearray: bytes, sampling_rate=16000):
#     bytes_wav = bytes()
#     byte_io = io.BytesIO(bytes_wav)
#     write(byte_io, sampling_rate, np.frombuffer(input_bytearray, dtype=np.int16))
#     output_wav = byte_io.read()
#     output, _ = sf.read(io.BytesIO(output_wav))
#     return output


def save_audio_file(audio_bytes, file_extension):
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"audio_test.{file_extension}"

    with open(file_name, "wb") as f:
        f.write(audio_bytes)

    return file_name    

audio_bytes = audio_recorder(pause_threshold=2.0, sample_rate=41_000)
if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    save_audio_file(audio_bytes, "wav")
    with open("audio_test.wav", "rb") as audio_file:
        with st.spinner("Tolker det jeg nettopp hørte..."):
            transcription = client.audio.transcriptions.create(
                    model="whisper-1", file=audio_file, response_format="text"
                )
        with st.spinner("Lager oppsummering..."):
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Du kan oppsummere budskap."},
                    {"role": "system", "content": "Du får et foredrag, og skal oppsummere det med en setning."},
                    {"role": "system", "content": "foredraget er transkribert og blir nå levert til deg. Du skal oppsummere det med en setning."},
                    {
                        "role": "user",
                        "content": f"{transcription}"
                    }
                ],
                stream=True 
            )
            
            
        st.write_stream(completion)


   
