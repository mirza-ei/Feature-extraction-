# demo

import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.applications import VGG16
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_distances
from scipy.ndimage import zoom
import gradio as gr

# Define the VGGish model for feature extraction
def build_vggish_model(weights_path='imagenet'):
    base_model = VGG16(weights=weights_path, include_top=False, input_shape=(96, 64, 3))
    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation='relu', name='feature_output')(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

# Convert audio file to mel spectrogram
def audio_to_mel_spec(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=96, hop_length=512)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # Resize to (96, 64) and add color channels
    mel_spec = zoom(mel_spec, (96 / mel_spec.shape[0], 64 / mel_spec.shape[1]), order=1)
    mel_spec = np.expand_dims(mel_spec, axis=-1)  # Add channel dimension
    mel_spec = np.repeat(mel_spec, 3, axis=-1)  # Repeat the channel dimension to make it (96, 64, 3)
    mel_spec = np.expand_dims(mel_spec, axis=0)  # Add batch dimension

    return mel_spec

# Path to the weights file (use 'imagenet' if using pre-trained ImageNet weights)
weights_path = 'imagenet'

# Load the VGGish model
vggish_model = build_vggish_model(weights_path)

# Define the list of training files
train_files = [
    r'C:\Users\User\Voice-Verification-8\Modern method\voice file\1.wav',
    r'C:\Users\User\Voice-Verification-8\Modern method\voice file\2.wav',
    r'C:\Users\User\Voice-Verification-8\Modern method\voice file\3.wav',
    r'C:\Users\User\Voice-Verification-8\Modern method\voice file\4.mp3',
    r'C:\Users\User\Voice-Verification-8\Modern method\voice file\5.wav'
]

# Extract features for training files
train_features = [vggish_model.predict(audio_to_mel_spec(file))[0] for file in train_files]

# Normalize features
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)

# Function to process the uploaded audio and find the most similar file
def verify_speaker(test_audio):
    test_feature = vggish_model.predict(audio_to_mel_spec(test_audio))[0]
    test_feature = scaler.transform(test_feature.reshape(1, -1)).flatten()

    # Compute similarities using cosine distance
    distances = cosine_distances([test_feature], train_features)[0]

    # Find the index of the most similar file
    most_similar_index = np.argmin(distances)
    most_similar_file = train_files[most_similar_index]
    similarity_score = distances[most_similar_index]

    return most_similar_file, similarity_score

# Create a Gradio interface
iface = gr.Interface(
    fn=verify_speaker,
    inputs=gr.Audio(type="filepath"),
    outputs=[gr.Textbox(label="Most Similar File"), gr.Textbox(label="Similarity Score")],
    title="Speaker Verification System",
    description="Upload an audio file to find the most similar file from the training set."
)

# Launch the interface
iface.launch()
