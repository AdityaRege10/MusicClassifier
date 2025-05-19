import os
import librosa
import numpy as np
import pandas as pd 
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

DATASET_PATH = r"C:\Users\getad\OneDrive\Desktop\ML Projects\Data\genres_original"

# Get all genres from the dataset
genres = os.listdir(DATASET_PATH)
print("Genres selected: ", genres)

def extract_features(file_path):
    print(f"Extracting features from: {file_path}")
    y, sr = librosa.load(file_path, duration=120)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
    return np.hstack([mfccs, chroma, zcr, spectral_centroid])

# Start feature extraction
print("Starting feature extraction...")

data = []
labels = []
skipped_files = 0
file_count = 0

for genre in genres:
    genre_folder = os.path.join(DATASET_PATH, genre)
    print(f"Processing genre: {genre}")
    for filename in os.listdir(genre_folder):  # Process all files in the folder
        if filename.endswith(".wav"):
            file_path = os.path.join(genre_folder, filename)
            try:
                features = extract_features(file_path)
                data.append(features)
                labels.append(genre)
                file_count += 1
                if file_count % 50 == 0:  # Print progress every 50 files
                    print(f"Processed {file_count} files so far...")
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
                skipped_files += 1

print(f"Feature extraction complete. Processed {file_count} files.")
print(f"Skipped {skipped_files} problematic files.")

# Create a DataFrame to store the features and labels
df = pd.DataFrame(data)
df['label'] = labels
print(f"Dataframe created with {len(df)} samples.")

# Split the data into features (X) and labels (y)
X = df.drop(columns=['label'])
y = df['label']

# Split into training and test sets (80% train, 20% test)
print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest classifier
print("Starting model training...")
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print classification report
print("Classification report:")
print(classification_report(y_test, y_pred))

# Plot the confusion matrix
print("Plotting confusion matrix...")
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Save the model
print("Saving model to genre_classifier.pkl")
joblib.dump(model, "genre_classifier.pkl")
print("Model saved as genre_classifier.pkl")
