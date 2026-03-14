import streamlit as st
import cv2
import numpy as np
import librosa
import sounddevice as sd
from scipy.io.wavfile import write
import joblib
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Load trained model
# ---------------------------

model = joblib.load("multimodal_emotion_model.pkl")

# load scaler if saved
scaler = joblib.load("scaler.pkl")

emotion_map = {
    0: "Normal",
    1: "Depression Stage 1",
    2: "Depression Stage 2"
}

# ---------------------------
# Audio Feature Extraction
# ---------------------------

def extract_audio_features(file_path):

    audio, sr = librosa.load(file_path, duration=5)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    return np.mean(mfcc.T, axis=0)

# ---------------------------
# Video Feature Extraction
# ---------------------------

def extract_video_features():

    cap = cv2.VideoCapture(0)

    frame_features = []

    stframe = st.empty()

    for i in range(100):

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (224,224))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame_features.append(np.mean(gray))

        stframe.image(frame, channels="BGR")

    cap.release()

    mean = np.mean(frame_features)
    std = np.std(frame_features)

    return np.array([mean, std])

# ---------------------------
# Record Audio
# ---------------------------

def record_audio():

    fs = 22050
    duration = 5

    st.write("Recording audio...")

    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    write("live_audio.wav", fs, audio)

    return "live_audio.wav"

# ---------------------------
# Prediction Pipeline
# ---------------------------

def predict_emotion():

    audio_file = record_audio()

    audio_features = extract_audio_features(audio_file)

    video_features = extract_video_features()

    features = np.concatenate((audio_features, video_features))

    features = features.reshape(1,-1)

    features = scaler.transform(features)

    pred = model.predict(features)[0]

    return emotion_map[pred]

# ---------------------------
# Streamlit UI
# ---------------------------

st.title("Multimodal Emotion Recognition")

st.write("Live Camera + Audio Based Emotion Detection")

if st.button("Start Detection"):

    emotion = predict_emotion()

    st.success(f"Predicted Emotion: {emotion}")
