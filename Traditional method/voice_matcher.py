import gradio as gr
import librosa
import numpy as np
from scipy.spatial.distance import euclidean, cityblock, cosine
from scipy.stats import kendalltau
import pandas as pd
import matplotlib.pyplot as plt
import io
from PIL import Image

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    energy_envelope = librosa.feature.rms(y=y)[0]
    features = {
        'mfcc': mfcc,
        'chroma': chroma,
        'zcr': zcr,
        'energy_envelope': energy_envelope
    }
    return features

def calculate_distances(features1, features2):
    distances = {}
    for feature_name in features1:
        feature1 = features1[feature_name].flatten()
        feature2 = features2[feature_name].flatten()
        min_length = min(len(feature1), len(feature2))
        feature1 = feature1[:min_length]
        feature2 = feature2[:min_length]
        distances[f'{feature_name}_euclidean'] = euclidean(feature1, feature2)
        distances[f'{feature_name}_manhattan'] = cityblock(feature1, feature2)
        distances[f'{feature_name}_cosine'] = 1 - cosine(feature1, feature2)
        corr, _ = kendalltau(feature1, feature2)
        distances[f'{feature_name}_kendall'] = 1 - corr
    return distances

def voice_matcher(test_voice):
    voice_files = ['1.wav', '2.wav', '3.wav', '4.wav']
    results_df = pd.DataFrame(columns=['file_name', 'feature_name', 'cosine_similarity', 'kendall_correlation',
                                       'euclidean_distance', 'manhattan_distance', 'match'])
    test_features = extract_features(test_voice)
    if test_features is None:
        return "Error loading test voice file.", "", None

    best_match = None
    best_match_info = "No matches found."
    best_match_distances = {}

    for file in voice_files:
        features = extract_features(file)
        if features is None:
            continue
        distances = calculate_distances(features, test_features)
        for feature_name in features:
            cosine_similarity = distances[f'{feature_name}_cosine']
            match_cosine = "Match" if cosine_similarity > 0.8 else "No Match"
            match_euclidean = "Match" if distances[f'{feature_name}_euclidean'] < 0.5 else "No Match"
            match_manhattan = "Match" if distances[f'{feature_name}_manhattan'] < 2.0 else "No Match"
            match_kendall = "Match" if distances[f'{feature_name}_kendall'] < 0.4 else "No Match"
            new_row = {
                'file_name': file,
                'feature_name': feature_name,
                'cosine_similarity': cosine_similarity,
                'kendall_correlation': distances[f'{feature_name}_kendall'],
                'euclidean_distance': distances[f'{feature_name}_euclidean'],
                'manhattan_distance': distances[f'{feature_name}_manhattan'],
                'match': match_cosine
            }
            if pd.Series(new_row).notna().all():
                results_df = pd.concat([results_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)

        # Determine best match based on cosine similarity
        if best_match is None or cosine_similarity > best_match_distances.get('cosine_similarity', 0):
            best_match = file
            best_match_distances = {
                'cosine_similarity': cosine_similarity,
                'euclidean_distance': distances[f'{feature_name}_euclidean'],
                'manhattan_distance': distances[f'{feature_name}_manhattan'],
                'kendall_correlation': distances[f'{feature_name}_kendall']
            }

    if best_match:
        # Extract features of the best match file
        best_match_features = extract_features(best_match)
        if best_match_features is not None:
            fig, axs = plt.subplots(4, 1, figsize=(10, 20))

            # Plotting test voice features and best match file features
            axs[0].plot(test_features['mfcc'].flatten(), label='Test Voice MFCC', linestyle='dashed')
            axs[0].plot(best_match_features['mfcc'].flatten(), label=f'{best_match} MFCC')
            axs[0].set_title('MFCC Comparison')
            axs[0].legend()
            axs[0].grid(True)

            axs[1].plot(test_features['chroma'].flatten(), label='Test Voice Chroma', linestyle='dashed')
            axs[1].plot(best_match_features['chroma'].flatten(), label=f'{best_match} Chroma')
            axs[1].set_title('Chroma Comparison')
            axs[1].legend()
            axs[1].grid(True)

            axs[2].plot(test_features['zcr'].flatten(), label='Test Voice ZCR', linestyle='dashed')
            axs[2].plot(best_match_features['zcr'].flatten(), label=f'{best_match} ZCR')
            axs[2].set_title('ZCR Comparison')
            axs[2].legend()
            axs[2].grid(True)

            axs[3].plot(test_features['energy_envelope'].flatten(), label='Test Voice Energy Envelope', linestyle='dashed')
            axs[3].plot(best_match_features['energy_envelope'].flatten(), label=f'{best_match} Energy Envelope')
            axs[3].set_title('Energy Envelope Comparison')
            axs[3].legend()
            axs[3].grid(True)

            plt.tight_layout()

            # Save plot to a BytesIO object
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()

             #PIL.Image
            img = Image.open(buf)
            best_match_info = f"Best match: {best_match} (Cosine similarity: {best_match_distances['cosine_similarity']:.2f})"

    return results_df.to_html(), best_match_info, img

iface = gr.Interface(
    fn=voice_matcher,
    inputs=gr.Audio(type="filepath", label="Upload Test Voice"),
    outputs=[gr.HTML(), gr.Textbox(), gr.Image(type="pil")],
    live=True,  
    title="Voice Matcher",
    description="Upload a test voice and compare it with predefined voice files."
)

iface.launch()
