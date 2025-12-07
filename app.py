import json
import streamlit as st
from datetime import datetime
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pyttsx3
import os
import tempfile
from streamlit_webrtc import webrtc_streamer
import queue
import soundfile as sf

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Braille Assistant",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stApp {background-color: #1a1a1a;}
.chat-container {height: 60vh; overflow-y: auto; padding: 20px; margin-bottom: 20px;}
.chat-message {padding: 15px; border-radius: 15px; margin-bottom: 15px; max-width: 85%; word-wrap: break-word;}
.chat-message.user {background-color: #2196f3; color: white; margin-left: auto; text-align: right;}
.chat-message.bot {background-color: #2d2d2d; color: #e0e0e0; margin-right: auto;}
.timestamp {font-size: 11px; opacity: 0.7; margin-top: 5px;}
.stTextInput > div > div > input {background-color: #2d2d2d !important; color: white !important; border: none !important; border-radius: 25px !important; padding: 15px 20px !important; font-size: 16px !important;}
.stButton > button {background-color: #2196f3; color: white; border: none; border-radius: 25px; padding: 12px 30px; font-size: 16px; font-weight: bold; width: 100%; margin: 5px 0;}
.stButton > button:hover {background-color: #1976d2;}
.stCheckbox {color: white;}
h1 {color: white !important; text-align: center; padding: 20px 0; margin: 0; font-size: 24px !important;}
.block-container {padding-top: 1rem; padding-bottom: 1rem;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Initialize session state
# -----------------------------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [{
        "sender": "bot",
        "message": "Hello! Ask me anything about Braille.",
        "timestamp": datetime.now().strftime("%H:%M")
    }]

if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

if 'voice_enabled' not in st.session_state:
    st.session_state.voice_enabled = False

if 'last_audio_file' not in st.session_state:
    st.session_state.last_audio_file = None

if 'play_audio' not in st.session_state:
    st.session_state.play_audio = False

# -----------------------------
# Load model & FAISS index
# -----------------------------
@st.cache_resource
def load_model():
    try:
        model = SentenceTransformer("saved_model")
        index = faiss.read_index("faiss_index.index")
        with open("data_saved.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return model, index, data, True
    except Exception as e:
        st.error(f"Error loading model/index: {e}")
        return None, None, None, False

model, index, data, model_loaded = load_model()
st.session_state.model_loaded = model_loaded

# -----------------------------
# Q/A retrieval
# -----------------------------
def retrieve_answer(query):
    if not st.session_state.model_loaded:
        return "Error: Knowledge base not loaded."
    try:
        query_emb = model.encode([query], convert_to_numpy=True)
        D, I = index.search(query_emb.astype("float32"), k=1)
        idx = I[0][0]
        return data[idx]['answer']
    except Exception as e:
        return f"Error: {str(e)}"

# -----------------------------
# Offline TTS with pyttsx3
# -----------------------------
def create_audio_file(text):
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_file.close()
        engine = pyttsx3.init()
        engine.save_to_file(text, temp_file.name)
        engine.runAndWait()
        return temp_file.name
    except Exception as e:
        st.error(f"Audio error: {e}")
        return None

# -----------------------------
# Voice input with streamlit-webrtc
# -----------------------------
audio_queue = queue.Queue()

def callback(frame):
    audio_queue.put(frame.to_ndarray())
    return frame

def speech_to_text_webrtc():
    st.info("🎤 Listening... Speak now!")
    webrtc_ctx = webrtc_streamer(key="voice", audio_receiver_size=256, video_transformer_factory=None)

    if webrtc_ctx.audio_receiver:
        frames = []
        for _ in range(50):  # capture ~5 sec audio
            try:
                frame = webrtc_ctx.audio_receiver.get(timeout=5)
                frames.append(frame.to_ndarray())
            except queue.Empty:
                break

        if frames:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            import numpy as np
            audio_data = np.concatenate(frames, axis=0)
            sf.write(temp_file.name, audio_data, 16000)
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_file.name) as source:
                audio = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio)  # online recognition
                    return text, True
                except sr.UnknownValueError:
                    return "Could not understand audio", False
        else:
            return "No audio captured", False
    else:
        return "Audio not available", False

# -----------------------------
# Add message & generate response
# -----------------------------
def add_message_and_get_response(user_message):
    st.session_state.chat_history.append({
        "sender": "user",
        "message": user_message,
        "timestamp": datetime.now().strftime("%H:%M")
    })
    
    response = retrieve_answer(user_message)
    st.session_state.chat_history.append({
        "sender": "bot",
        "message": response,
        "timestamp": datetime.now().strftime("%H:%M")
    })
    
    if st.session_state.voice_enabled:
        audio_file = create_audio_file(response)
        if audio_file:
            if st.session_state.last_audio_file and os.path.exists(st.session_state.last_audio_file):
                try: os.unlink(st.session_state.last_audio_file)
                except: pass
            st.session_state.last_audio_file = audio_file
            st.session_state.play_audio = True

# -----------------------------
# Streamlit UI
# -----------------------------
st.markdown("# 🤖 Braille Q&A Assistant")

# Voice toggle
st.session_state.voice_enabled = st.checkbox("🔊 Enable Voice Output", value=st.session_state.voice_enabled)

# Audio playback
if st.session_state.play_audio and st.session_state.last_audio_file:
    if os.path.exists(st.session_state.last_audio_file):
        with open(st.session_state.last_audio_file, 'rb') as f:
            st.audio(f.read(), format='audio/mp3', autoplay=True)
        st.session_state.play_audio = False

# Chat display
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for chat in st.session_state.chat_history:
    sender_class = "user" if chat["sender"] == "user" else "bot"
    st.markdown(f"""
        <div class="chat-message {sender_class}">
            <div>{chat["message"]}</div>
            <div class="timestamp">{chat["timestamp"]}</div>
        </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Input & buttons
col1, col2 = st.columns([3,1])
with col1:
    user_input = st.text_input("", placeholder="Type your message...", label_visibility="collapsed", key="input")
with col2:
    voice_btn = st.button("🎤")
send_btn = st.button("📤 Send Message")

# Voice input
if voice_btn:
    voice_text, success = speech_to_text_webrtc()
    if success:
        st.success(f"Heard: {voice_text}")
        add_message_and_get_response(voice_text)
        st.rerun()
    else:
        st.warning(voice_text)

# Text input
if send_btn and user_input:
    add_message_and_get_response(user_input)
    st.rerun()

# Clear chat
if st.button("🗑️ Clear Chat"):
    if st.session_state.last_audio_file and os.path.exists(st.session_state.last_audio_file):
        try: os.unlink(st.session_state.last_audio_file)
        except: pass
    st.session_state.chat_history = [{
        "sender": "bot",
        "message": "Chat cleared. How can I help?",
        "timestamp": datetime.now().strftime("%H:%M")
    }]
    st.session_state.last_audio_file = None
    st.session_state.play_audio = False
    st.rerun()
