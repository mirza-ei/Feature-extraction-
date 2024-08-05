### **Voice Matcher**

---

##### **Overview**

*Voice Matcher is an application designed to compare a test voice file with a set of predefined voice files. The comparison is performed using various audio features and distance metrics to determine the similarity between the test voice and the predefined voices.*

#### *Features*

- Extracts audio features such as MFCC, chroma, zero-crossing rate, and energy envelope.

- Computes distances between the features of the test voice and predefined voice files using Euclidean, Manhattan, and Cosine distances, as well as Kendall Tau correlation.

- Provides a detailed comparison report with similarity metrics.

- Identifies the best matching voice file based on the computed similarities.


#### *Dependencies*

- `gradio`

- `librosa`

- `numpy`

- `scipy`

- `pandas`

#### *Install the dependencies using:*


pip install gradio librosa numpy scipypandas

#### *Code Explanation*


##### *Feature Extraction*



The `extract_features` function extracts the following features from an audio file:

◾ **MFCC**: Mel Frequency Cepstral  Coefficients

◾ **Chroma**: Chroma feature

◾ **ZCR**: Zero Crossing Rate

◾**Energy Envelope**: Root Mean Square energy


```python
def extract_features(file_path):
    y, sr = librosa.load(file_path)
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
```

#### *Distance Calculation*

The `calculate_distances` function computes the Euclidean, Manhattan, and Cosine distances, and Kendall Tau correlation between the features of two audio files.
```python
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
```

#### *Voice Matching*


The `voice_matcher` function compares the test voice with a predefined list of voice files and generates a comparison report.
```python
def voice_matcher(test_voice):
    voice_files = ['1.wav', '2.wav', '3.wav', '4.wav']
    results_df = pd.DataFrame(columns=['file_name', 'feature_name', 'cosine_similarity', 'kendall_correlation',
                                       'euclidean_distance', 'manhattan_distance', 'match'])
    test_features = extract_features(test_voice)
    for file in voice_files:
        features = extract_features(file)
        distances = calculate_distances(features, test_features)
        for feature_name in features:
            cosine_similarity = distances[f'{feature_name}_cosine']
            match_cosine = "Match" if cosine_similarity > 0.8 else "No Match"
            match_euclidean = "Match" if distances[f'{feature_name}_euclidean'] < 0.5 else "No Match"
            match_manhattan = "Match" if distances[f'{feature_name}_manhattan'] < 2.0 else "No Match"
            match_kendall = "Match" if distances[f'{feature_name}_kendall'] < 0.4 else "No Match"
            results_df = pd.concat([results_df, pd.DataFrame({
                'file_name': file,
                'feature_name': feature_name,
                'cosine_similarity': cosine_similarity,
                'kendall_correlation': distances[f'{feature_name}_kendall'],
                'euclidean_distance': distances[f'{feature_name}_euclidean'],
                'manhattan_distance': distances[f'{feature_name}_manhattan'],
                'match': match_cosine
            }, index=[0])], ignore_index=True)
    if results_df.empty:
        return "No results found.", ""
    else:
        match_counts = results_df[results_df['match'] == 'Match']['file_name'].value_counts()
        if not match_counts.empty:
            best_match = match_counts.index[0]
            match_count = match_counts.iloc[0]
            best_match_info = f"Best match: {best_match} (Number of matches: {match_count})"
        else:
            best_match_info = "No matches found."
        return results_df.to_html(), best_match_info
```

##### *Gradio Interface*


The Gradio interface allows users to upload a test voice file and view the comparison results.
```python
iface = gr.Interface(
    fn=voice_matcher,
    inputs=gr.Audio(type="filepath", label="Upload Test Voice"),
    outputs=[gr.HTML(), gr.Textbox()],
    title="Voice Matcher",
    description="Upload a test voice and compare it with predefined voice files.",
    theme="huggingface"
)

iface.launch()
```

#### *Usage*

*1. Clone the repository.*

*2. Install the required dependencies.*

*3. Run the script to launch the Gradio interface.*

*4. Upload a test voice file to compare it with the predefined voice files.*

*5. View the comparison report and best match information.*




