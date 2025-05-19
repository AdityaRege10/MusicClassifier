import joblib
import librosa
import numpy as np


model = joblib.load("C:/Users/getad/OneDrive/Desktop/ML Projects/genre_classifier.pkl")

# Feature extraction function
def extract_features(file_path):
    print(f"Extracting features from: {file_path}")
    y, sr = librosa.load(file_path, duration=90)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
    return np.hstack([mfccs, chroma, zcr, spectral_centroid])

# Prediction function
def predict_genre(audio_file):
    try:
        features = extract_features(audio_file)
        features = features.reshape(1, -1)
        prediction = model.predict(features)
        print(f"\nPredicted Genre: {prediction[0]}")
    except Exception as e:
        print(f"Error during prediction: {e}")

# Get path from user input
file_path = input("Enter the full path to the MP3/WAV file: ").strip()
predict_genre(file_path)
print("Prediction complete!")